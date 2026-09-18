// =============================================================================
// ALICE-SDF Sample: Mochi (Dynamic Soft-Body Blobs) — Standalone Edition
// =============================================================================
// Interactive mochi (rice cake) objects that merge, split, and grow.
// SmoothUnion creates organic blob-like blending between mochis.
// Mochi data is sent from UdonSharp via Material.SetVectorArray.
//
// Mechanics:
//   - Mochis near each other blend together (SmoothUnion)
//   - Grab and pull to split mochis apart
//   - Push mochis together to merge (volume conservation)
//   - Merged mochis grow bigger and bigger
//   - Ground contact has a soft "squishy" SmoothUnion feel
//
// SDF formula: SmoothUnion(ground, SmoothUnion(mochi1, mochi2, ..., k), groundK)
//
// Surface material follows the same SmoothUnion: the blend factor of the
// ground/mochi union drives the colour mix, so the "neck" where a mochi
// squishes onto the floor shades continuously instead of switching.
//
// This shader is self-contained — no external include dependencies.
// Works with manual file copy (no UPM package installation required).
// The SDF laws (sdSphere / opSmoothUnion) and the LOD tables are verbatim
// copies of AliceSDF_Include.cginc / AliceSDF_LOD.cginc; keep them in sync.
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Samples/Mochi"
{
    Properties
    {
        [Header(Colors)]
        _MochiColor ("Mochi Color", Color) = (0.96, 0.93, 0.88, 1.0)
        _MochiColor2 ("Mochi Highlight", Color) = (0.99, 0.96, 0.92, 1.0)
        _GroundColor ("Ground Color", Color) = (0.55, 0.46, 0.36, 1.0)
        _GroundColor2 ("Ground Detail", Color) = (0.50, 0.42, 0.33, 1.0)

        [Header(Raymarching)]
        _MaxDist ("Max Distance", Float) = 80.0

        [Header(Mochi Physics)]
        // Driven by SampleMochi_Collider every frame (single source of truth);
        // the Inspector values only matter without the Udon script.
        _BlendK ("Mochi Blend (higher = stickier)", Float) = 0.5
        _GroundK ("Ground Stickiness", Float) = 0.15
        // Player body, driven every frame by the collider: a capsule from
        // A to B (w of A = radius, 0 = no player) pressed into the mochis
        _PlayerCapA ("Player Capsule A (xyz, w = radius)", Vector) = (0, 0, 0, 0)
        _PlayerCapB ("Player Capsule B (xyz)", Vector) = (0, 0, 0, 0)
        _PlayerDentK ("Player Dent Smoothness", Float) = 0.12
        // Ground point, driven every frame by the collider (transform + groundOffset):
        // the ground plane passes through it, so the prefab can sit anywhere
        _Origin ("Origin (ground point)", Vector) = (0, 0, 0, 0)

        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1.0, 1.0, -0.5, 0.0)
        _ShadowEnabled ("Enable Soft Shadow", Int) = 1
        _ShadowSoftness ("Shadow Softness", Range(1, 128)) = 16.0
        _ShadowMaxDist ("Shadow Max Distance", Float) = 10.0

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.83, 0.80, 0.76, 1.0)
        _FogDensity ("Fog Density", Float) = 0.005
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            Cull Off
            ZWrite On
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            // Upper bound of the _MochiData uniform array. Must match
            // SampleMochi_Collider.MaxMochi (Material.SetVectorArray fixes
            // the array length on first use).
            #define MOCHI_MAX 16

            // =================================================================
            // Inlined SDF Primitives (from AliceSDF_Include.cginc)
            // =================================================================

            float sdSphere(float3 p, float radius)
            {
                return length(p) - radius;
            }

            // Smooth Union (Deep Fried: division exorcism, pre-computed inv_k)
            // Guarded: k <= 0 falls back to hard union (no division by zero)
            float opSmoothUnion(float d1, float d2, float k)
            {
                if (k < 0.0001) return min(d1, d2);
                float inv_k = 1.0 / k;
                float h = max(k - abs(d1 - d2), 0.0) * inv_k;
                return min(d1, d2) - h * h * k * 0.25;
            }

            // Capsule: line segment with radius
            float sdCapsule(float3 p, float3 a, float3 b, float radius)
            {
                float3 pa = p - a;
                float3 ba = b - a;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                return length(pa - ba * h) - radius;
            }

            // Smooth Subtraction (d1 minus d2)
            // Guarded: k <= 0 falls back to hard subtraction
            float opSmoothSubtraction(float d1, float d2, float k)
            {
                if (k < 0.0001) return max(d1, -d2);
                float inv_k = 1.0 / k;
                float h = max(k - abs(d1 + d2), 0.0) * inv_k;
                return max(d1, -d2) + h * h * k * 0.25;
            }

            // Smooth Union with blend factor (Inigo Quilez, "smooth minimum
            // with material"). Returns (distance, weight of d2 in [0, 1]).
            // The distance is bit-identical to opSmoothUnion; the weight is
            // 0.5 exactly on the d1 == d2 seam and reaches 0 / 1 at |d1 - d2| >= k.
            float2 opSmoothUnionBlend(float d1, float d2, float k)
            {
                if (k < 0.0001) return (d1 < d2) ? float2(d1, 0.0) : float2(d2, 1.0);
                float inv_k = 1.0 / k;
                float h = max(k - abs(d1 - d2), 0.0) * inv_k;
                float m = h * h * 0.5;
                float s = h * h * k * 0.25;
                return (d1 < d2) ? float2(d1 - s, m) : float2(d2 - s, 1.0 - m);
            }

            // =================================================================
            // Inlined LOD System (from AliceSDF_LOD.cginc)
            // =================================================================

            // Closest-approach acceptance, metres per metre of ray length
            // (~1 px at a 60 deg / 1000 px view)
            #define NEAR_MISS_PER_M 0.002
            #define ALICE_LOD_TIER_HIGH  0
            #define ALICE_LOD_TIER_MED   1
            #define ALICE_LOD_TIER_LOW   2

            int aliceLodTier(float cameraDist)
            {
                if (cameraDist < 20.0) return ALICE_LOD_TIER_HIGH;
                if (cameraDist < 60.0) return ALICE_LOD_TIER_MED;
                return ALICE_LOD_TIER_LOW;
            }

            int aliceLodSteps(int tier)
            {
                if (tier == ALICE_LOD_TIER_HIGH) return 128;
                if (tier == ALICE_LOD_TIER_MED)  return 64;
                return 32;
            }

            float aliceLodEpsilon(int tier)
            {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.0001;
                if (tier == ALICE_LOD_TIER_MED)  return 0.001;
                return 0.005;
            }

            float aliceLodStepScale(int tier)
            {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.9;
                if (tier == ALICE_LOD_TIER_MED)  return 1.0;
                return 1.2;
            }

            // Central-difference half-width for the normal. Wider than the
            // surface epsilon so the normal averages over the hit tolerance
            // instead of resolving the raymarch error as noise.
            float aliceLodNormalEps(int tier)
            {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.001;
                if (tier == ALICE_LOD_TIER_MED)  return 0.003;
                return 0.01;
            }

            // Soft-shadow steps per tier (standalone budget: below the
            // 48 / 24 / 12 of AliceSDF_LOD.cginc, the scene is a few metres)
            int aliceLodShadowSteps(int tier)
            {
                if (tier == ALICE_LOD_TIER_HIGH) return 32;
                if (tier == ALICE_LOD_TIER_MED)  return 16;
                return 8;
            }

            // =================================================================
            // Inspector properties
            // =================================================================

            float4 _MochiColor, _MochiColor2, _GroundColor, _GroundColor2, _FogColor;
            float _MaxDist;
            float _BlendK, _GroundK;
            float4 _LightDir;
            int _ShadowEnabled;
            float _ShadowSoftness;
            float _ShadowMaxDist;
            float _FogDensity;

            // Dynamic mochi data (set from UdonSharp)
            // xyz = world position, w = radius
            float4 _MochiData[MOCHI_MAX];
            float _MochiCount;
            float4 _PlayerCapA, _PlayerCapB;
            float4 _Origin;
            float _PlayerDentK;

            struct appdata {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // =================================================================
            // SDF: Ground + Dynamic Mochi Blobs
            // =================================================================

            // Smooth union of every active mochi sphere (no ground)
            // Note: opSmoothUnion(1e10, d, k) = d, so the fold starts at 1e10
            float mapMochi(float3 p)
            {
                int count = min((int)_MochiCount, MOCHI_MAX);
                float mochi = 1e10;
                for (int i = 0; i < MOCHI_MAX; i++)
                {
                    if (i >= count) break;
                    float d = sdSphere(p - _MochiData[i].xyz, _MochiData[i].w);
                    mochi = opSmoothUnion(mochi, d, _BlendK);
                }
                // The player's body presses into the mochis: the collider keeps
                // the body axis collisionMargin outside the undented surface, so
                // the capsule (radius > margin) carves a dent where the player
                // leans in, and it springs back the moment they step away
                if (_PlayerCapA.w > 0.0)
                {
                    float body = sdCapsule(p, _PlayerCapA.xyz, _PlayerCapB.xyz, _PlayerCapA.w);
                    mochi = opSmoothSubtraction(mochi, body, _PlayerDentK);
                }
                return mochi;
            }

            // Full scene: ground plane at Y=0, mochis "squish" onto it
            float map(float3 p)
            {
                return opSmoothUnion(p.y - _Origin.y, mapMochi(p), _GroundK);
            }

            // Occluder distance for AO: the hard union. The polynomial smooth
            // union under-reports distance inside its blend zone (|grad| < 1,
            // up to k/4 short), which the AO integral (h - d) read as extra
            // occlusion: a sharp dark ring around every mochi's base, ending
            // abruptly where the blend zone ends. The hard union is exact
            // outside the sphere / ground, so only real geometry occludes.
            float mapOccluder(float3 p)
            {
                return min(p.y - _Origin.y, mapMochi(p));
            }

            // Inlined AO with LOD
            float aliceAO_LOD(float3 pos, float3 nor, int tier)
            {
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                              (tier == ALICE_LOD_TIER_MED)  ? 3 : 2;
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < 5; i++)
                {
                    if (i >= aoSteps) break;
                    float h = 0.01 + 0.12 * float(i) / float(max(aoSteps - 1, 1));
                    float d = mapOccluder(pos + h * nor);
                    occ += (h - d) * sca;
                    sca *= 0.95;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // Inlined soft shadow with LOD (penumbra estimate, IQ improved)
            float aliceSoftShadow_LOD(float3 ro, float3 rd, float mint, float maxt, float softness, int tier)
            {
                int maxSteps = aliceLodShadowSteps(tier);
                float res = 1.0;
                float t = mint;
                float ph = 1e20;
                for (int i = 0; i < 32; i++)
                {
                    if (i >= maxSteps) break;
                    float h = map(ro + rd * t);
                    if (h < 0.0001)
                        return 0.0;
                    float y = h * h / (2.0 * ph);
                    float d = sqrt(h * h - y * y);
                    res = min(res, softness * d / max(0.0, t - y));
                    ph = h;
                    t += h;
                    if (t > maxt) break;
                }
                return saturate(res);
            }

            // Normal via central differences (half-width from the LOD tier)
            float3 calcN(float3 p, float e) {
                return normalize(float3(
                    map(p + float3(e,0,0)) - map(p - float3(e,0,0)),
                    map(p + float3(0,e,0)) - map(p - float3(0,e,0)),
                    map(p + float3(0,0,e)) - map(p - float3(0,0,e))
                ));
            }

            // Ground texture: interpolated value noise (a raw hash per pixel
            // aliases into sparkle in VR; the smoothstep lattice does not)
            float hash2d(float2 p) {
                p = frac(p * float2(0.3183, 0.3671));
                p *= 17.0;
                return frac(p.x * p.y * (p.x + p.y));
            }

            float noise2d(float2 p) {
                float2 i = floor(p);
                float2 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);
                return lerp(
                    lerp(hash2d(i + float2(0,0)), hash2d(i + float2(1,0)), f.x),
                    lerp(hash2d(i + float2(0,1)), hash2d(i + float2(1,1)), f.x),
                    f.y);
            }

            v2f vert(appdata v) {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                o.objectCenter = mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz;
                return o;
            }

            struct FragOutput { fixed4 color : SV_Target; float depth : SV_Depth; };

            FragOutput frag(v2f i) {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);
                // LOD: adapt steps based on camera distance
                float camDist = length(i.objectCenter - ro);
                int tier = aliceLodTier(camDist);
                int maxSteps = aliceLodSteps(tier);
                float eps = aliceLodEpsilon(tier);
                float ss = aliceLodStepScale(tier);
                float t = 0.0;
                FragOutput o;

                // Closest approach along the ray. A ray grazing a silhouette
                // takes ever smaller steps (and smaller still inside a smooth
                // union's blend zone, where |grad| < 1) and runs out of budget
                // a hair short of the surface. Dropping it as a miss wrote the
                // far depth, so whatever the world had behind the volume (its
                // floor) showed through as a thin dark line along every
                // mochi / ground contour. If the budget runs out within one
                // pixel footprint of the surface, the closest point is the hit.
                float bestD = 1e10;
                float bestT = 0.0;
                bool hit = false;

                for (int k = 0; k < 128; k++) {
                    if (k >= maxSteps) break;
                    float d = map(ro + rd * t);

                    if (d < eps) { hit = true; break; }
                    if (d < bestD) { bestD = d; bestT = t; }

                    t += d * ss;
                    if (t > _MaxDist) break;
                }
                if (!hit && bestD < max(eps, bestT * NEAR_MISS_PER_M)) {
                    t = bestT;
                    hit = true;
                }

                if (hit) {
                    float3 p = ro + rd * t;
                    float3 n = calcN(p, aliceLodNormalEps(tier));
                    float3 lightDir = normalize(_LightDir.xyz);
                    float ao = aliceAO_LOD(p, n, tier);

                    // Direct-light visibility (mochi -> ground contact shadow)
                    float shadow = 1.0;
                    if (_ShadowEnabled > 0)
                    {
                        shadow = aliceSoftShadow_LOD(p + n * 0.02, lightDir, 0.02,
                                                     _ShadowMaxDist, _ShadowSoftness, tier);
                    }

                    // Material weight from the same union that shaped the
                    // surface: 0 = ground, 1 = mochi, continuous across the neck
                    float mochiW = opSmoothUnionBlend(p.y - _Origin.y, mapMochi(p), _GroundK).y;

                    // === MOCHI SURFACE ===
                    // Warm wrap lighting (subsurface scattering approx)
                    float3 mochiCol = lerp(_MochiColor.rgb, _MochiColor2.rgb,
                                           n.y * 0.5 + 0.5);
                    float wrap = max(dot(n, lightDir) + 0.4, 0.0) / 1.4;

                    // Fresnel rim for soft translucent look
                    float3 viewDir = normalize(ro - p);
                    float fresnel = pow(1.0 - abs(dot(n, viewDir)), 3.0);
                    float3 sss = float3(1.0, 0.88, 0.72) * fresnel * 0.25;

                    float3 mochiShade = mochiCol * (0.3 + wrap * 0.7 * shadow) * ao + sss;

                    // === GROUND SURFACE ===
                    float grain = noise2d(p.xz * 2.0) * 0.7 + noise2d(p.xz * 9.0) * 0.3;
                    float3 groundCol = lerp(_GroundColor.rgb, _GroundColor2.rgb,
                                            grain * 0.6 + 0.2);
                    float diff = max(dot(n, lightDir), 0.0);
                    float3 groundShade = groundCol * (0.2 + diff * 0.8 * shadow) * ao;

                    float3 fc = lerp(groundShade, mochiShade, mochiW);

                    // Fog
                    fc = lerp(_FogColor.rgb, fc, exp(-t * _FogDensity));

                    float4 cp = UnityWorldToClipPos(p);
                    o.color = fixed4(fc, 1.0);
                    #if defined(UNITY_REVERSED_Z)
                        o.depth = cp.z / cp.w;
                    #else
                        o.depth = (cp.z / cp.w) * 0.5 + 0.5;
                    #endif
                    return o;
                }

                o.color = fixed4(_FogColor.rgb, 1.0);
                #if defined(UNITY_REVERSED_Z)
                    o.depth = 0.0;
                #else
                    o.depth = 1.0;
                #endif
                return o;
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}
