// =============================================================================
// ALICE-SDF Sample: DeformableWall
// =============================================================================
// Interactive wall that dents when touched / hit / walked into. Dents
// recover over time. Impact data is sent from UdonSharp via
// Material.SetVectorArray every frame; the collider is the single source
// of truth for the wall size, the dent radius and the blend factor, so
// what is rendered is what the player collides with.
//
// SDF formula: min(ground, SmoothSubtract(SmoothSubtract(wall, dent_i), body))
//   dent_i: sphere of radius _DentRadius * strength_i (strength decays 1 -> 0)
//   body:   the local player's capsule, pressed into the wall (local only)
//
// Frame: the wall and the ground live in the wall frame given by
//   _WorldToWall (set by the collider from transform.position + groundOffset
//   and transform.rotation): ground = local y, wall along local X / Y facing
//   local Z. Dents and the body are world-space spheres / capsules, which
//   need no frame. The march stops where the ray leaves the volume cube, so
//   the ground is drawn only under the cube (two walls in one world do not
//   fight over an infinite plane).
//
// Rendering notes (same fixes as the Mochi sample):
//   - Cull Off: the player walks inside the volume cube
//   - closest-approach acceptance: a ray grazing the wall's edge that runs
//     out of steps a hair short of the surface is a hit, not the far depth
//   - AO samples the hard (k = 0) subtraction: the smooth rim of a dent
//     under-reports distance and read as a dark ring around every dent
//   - far-ground fast path (exact): outside the wall's box the wall
//     distance is positive, so a ray that misses the (slightly inflated)
//     box can only hit the ground plane. Those pixels take the analytic
//     plane hit with the +y normal and AO 1, and march the shadow ray only
//     if it can pass near the wall.
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Samples/DeformableWall"
{
    Properties
    {
        [Header(Colors)]
        _WallColor ("Wall Color", Color) = (0.82, 0.78, 0.72, 1.0)
        _GroundColor ("Ground Color", Color) = (0.35, 0.42, 0.3, 1.0)
        _DentColor ("Dent Glow Color", Color) = (1.0, 0.6, 0.3, 1.0)

        [Header(Textures)]
        // Driven by the collider's Look section; triplanar in the wall frame
        [NoScaleOffset] _WallTex ("Wall Texture", 2D) = "white" {}
        _WallTexScale ("Wall Texture Tiles per Metre", Float) = 1.0
        _WallTexStrength ("Wall Texture Strength", Range(0, 1)) = 0.0
        [NoScaleOffset] _GroundTex ("Ground Texture", 2D) = "white" {}
        _GroundTexScale ("Ground Texture Tiles per Metre", Float) = 0.5
        _GroundTexStrength ("Ground Texture Strength", Range(0, 1)) = 0.0

        [Header(Raymarching)]
        _MaxDist ("Max Distance", Float) = 100.0

        [Header(Wall Dimensions)]
        // Driven every frame by SampleDeformableWall_Collider; the Inspector
        // values only matter without the Udon script
        _WallWidth ("Wall Half-Width", Float) = 5.0
        _WallHeight ("Wall Half-Height", Float) = 2.5
        _WallThick ("Wall Half-Thickness", Float) = 0.2

        [Header(Deformation)]
        _DentRadius ("Dent Radius", Float) = 0.35
        _DentSmooth ("Dent Smoothness", Float) = 0.08
        // Player body, driven every frame by the collider: a capsule from
        // A to B (w of A = radius, 0 = no player) pressed into the wall
        _PlayerCapA ("Player Capsule A (xyz, w = radius)", Vector) = (0, 0, 0, 0)
        _PlayerCapB ("Player Capsule B (xyz)", Vector) = (0, 0, 0, 0)
        _PlayerDentK ("Player Dent Smoothness", Float) = 0.12

        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1.0, 1.0, -0.5, 0.0)
        _ShadowEnabled ("Enable Soft Shadow", Int) = 1
        _ShadowSoftness ("Shadow Softness", Range(1, 128)) = 16.0
        _ShadowMaxDist ("Shadow Max Distance", Float) = 12.0

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.65, 0.7, 0.78, 1.0)
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
            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc"

            // Inspector properties
            float4 _WallColor, _GroundColor, _DentColor, _FogColor;
            sampler2D _WallTex, _GroundTex;
            float _WallTexScale, _WallTexStrength, _GroundTexScale, _GroundTexStrength;
            float _MaxDist;
            float _WallWidth, _WallHeight, _WallThick;
            float _DentRadius, _DentSmooth;
            float4 _PlayerCapA, _PlayerCapB;
            float _PlayerDentK;
            float4 _LightDir;
            int _ShadowEnabled;
            float _ShadowSoftness, _ShadowMaxDist, _FogDensity;

            // Placement (set by the collider): world -> wall frame, rigid
            float4x4 _WorldToWall;

            // Closest-approach acceptance, metres per metre of ray length
            // (~1 px at a 60 deg / 1000 px view)
            #define NEAR_MISS_PER_M 0.002

            // Dynamic impact data (set from UdonSharp)
            // xyz = world position of the impact, w = strength (1 fresh -> 0 recovered)
            float4 _ImpactPoints[16];
            float _ImpactCount;

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

            // A material without the collider has no matrix (all zero): the
            // wall frame is then the world frame, as before
            bool wallFrameUnset()
            {
                return abs(_WorldToWall._m00) + abs(_WorldToWall._m11) + abs(_WorldToWall._m22) < 1e-6;
            }
            float3 toWall(float3 p)
            {
                return wallFrameUnset() ? p : mul(_WorldToWall, float4(p, 1.0)).xyz;
            }
            float3 toWallDir(float3 d)
            {
                return wallFrameUnset() ? d : mul((float3x3)_WorldToWall, d);
            }
            // Wall +y in world (row 1 of the rotation = column 1 of its inverse)
            float3 wallUp()
            {
                return wallFrameUnset() ? float3(0, 1, 0) : normalize(mul(float3(0, 1, 0), (float3x3)_WorldToWall));
            }

            // =================================================================
            // SDF: Ground + Wall with dynamic dents + the player's body
            // =================================================================
            // The dented wall alone (what the collider's EvaluateWallSdf is)
            float mapWall(float3 p)
            {
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                float wall = sdBox(toWall(p) - wc, wh);

                int count = (int)_ImpactCount;
                for (int i = 0; i < 16; i++)
                {
                    if (i >= count) break;
                    float w = _ImpactPoints[i].w;
                    if (w < 0.01) continue;
                    float dent = sdSphere(p - _ImpactPoints[i].xyz, _DentRadius * w);
                    wall = opSmoothSubtraction(wall, dent, _DentSmooth);
                }
                return wall;
            }

            float map(float3 p)
            {
                float wall = mapWall(p);
                // The local player's body presses in (rendered only, the
                // collider pushes the player out of the wall instead)
                if (_PlayerCapA.w > 0.0)
                {
                    float body = sdCapsule(p, _PlayerCapA.xyz, _PlayerCapB.xyz, _PlayerCapA.w);
                    wall = opSmoothSubtraction(wall, body, _PlayerDentK);
                }
                return min(toWall(p).y, wall);
            }

            // Hard (k = 0) version for the ambient occlusion: exact outside
            // the wall and the dents, so only real geometry occludes
            float mapOccluder(float3 p)
            {
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                float3 q = toWall(p);
                float wall = sdBox(q - wc, wh);
                int count = (int)_ImpactCount;
                for (int i = 0; i < 16; i++)
                {
                    if (i >= count) break;
                    float w = _ImpactPoints[i].w;
                    if (w < 0.01) continue;
                    wall = max(wall, -sdSphere(p - _ImpactPoints[i].xyz, _DentRadius * w));
                }
                if (_PlayerCapA.w > 0.0)
                    wall = max(wall, -sdCapsule(p, _PlayerCapA.xyz, _PlayerCapB.xyz, _PlayerCapA.w));
                return min(q.y, wall);
            }

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            // Central-difference half-width for the normal, wider than the
            // surface epsilon so it averages over the hit tolerance
            float normalEps(int tier) {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.001;
                if (tier == ALICE_LOD_TIER_MED)  return 0.003;
                return 0.01;
            }

            float3 calcN(float3 p, float e) {
                return normalize(float3(
                    map(p + float3(e,0,0)) - map(p - float3(e,0,0)),
                    map(p + float3(0,e,0)) - map(p - float3(0,e,0)),
                    map(p + float3(0,0,e)) - map(p - float3(0,0,e))
                ));
            }

            // AO against the hard union, sample count by LOD tier
            float wallAO(float3 p, float3 n, int tier) {
                #ifdef ALICE_MOBILE_BUDGET
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 2 : 1;
                #else
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                              (tier == ALICE_LOD_TIER_MED)  ? 3 : 2;
                #endif
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < 5; i++) {
                    if (i >= aoSteps) break;
                    float h = 0.01 + 0.12 * float(i) / float(max(aoSteps - 1, 1));
                    occ += (h - mapOccluder(p + h * n)) * sca;
                    sca *= 0.95;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // Dent freshness at the hit point (colour glow), the strength of
            // the nearest live dent weighted by distance
            float dentFreshness(float3 p) {
                float f = 0.0;
                int count = (int)_ImpactCount;
                for (int i = 0; i < 16; i++) {
                    if (i >= count) break;
                    float w = _ImpactPoints[i].w;
                    if (w < 0.01) continue;
                    float dist = length(p - _ImpactPoints[i].xyz);
                    f = max(f, w * smoothstep(_DentRadius * 1.5, 0.0, dist));
                }
                return f;
            }

            // Triplanar sample: the three axis projections of q blended by the
            // normal, so a dented surface shows the texture without UVs
            float3 triplanar(sampler2D tex, float3 q, float3 n)
            {
                float3 w = abs(n);
                w = w / max(w.x + w.y + w.z, 1e-4);
                float3 cx = tex2D(tex, q.yz).rgb;
                float3 cy = tex2D(tex, q.xz).rgb;
                float3 cz = tex2D(tex, q.xy).rgb;
                return cx * w.x + cy * w.y + cz * w.z;
            }

            // =================================================================
            // Far-ground fast path (exact, see the header)
            // =================================================================
            // True if the segment q0 + qd * [0, len] (wall frame) enters the
            // wall's box inflated by infl
            bool segmentNearWall(float3 q0, float3 qd, float len, float infl)
            {
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick) + infl;
                float3 inv = 1.0 / (abs(qd) < 1e-6 ? (qd < 0.0 ? -1e-6 : 1e-6) : qd);
                float3 t0 = (wc - wh - q0) * inv;
                float3 t1 = (wc + wh - q0) * inv;
                float3 tmin3 = min(t0, t1);
                float3 tmax3 = max(t0, t1);
                float tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
                float tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
                return tmax >= max(tmin, 0.0) && tmin <= len;
            }

            // Soft shadow of the bare plane: the same loop as
            // aliceSoftShadow_LOD with map replaced by the plane distance
            float planeSoftShadow_LOD(float3 q0, float3 qd, float mint, float maxt, float softness, int tier)
            {
                int maxSteps = (tier == ALICE_LOD_TIER_HIGH) ? ALICE_SHADOW_STEPS_HIGH :
                               (tier == ALICE_LOD_TIER_MED)  ? ALICE_SHADOW_STEPS_MED :
                                                               ALICE_SHADOW_STEPS_LOW;
                if (maxSteps <= 0) return 1.0;
                float res = 1.0;
                float t = mint;
                float ph = 1e20;
                for (int i = 0; i < 48; i++)
                {
                    if (i >= maxSteps) break;
                    float h = q0.y + qd.y * t;
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

            // Distance along the ray to where it leaves this object's unit
            // cube (object space slab test, mapped back to a world distance)
            float exitDistance(float3 ro, float3 rd)
            {
                float3 roObj = mul(unity_WorldToObject, float4(ro, 1.0)).xyz;
                float3 rdObj = mul((float3x3)unity_WorldToObject, rd);
                float3 inv = 1.0 / (abs(rdObj) < 1e-6 ? (rdObj < 0.0 ? -1e-6 : 1e-6) : rdObj);
                float3 t0 = (-0.5 - roObj) * inv;
                float3 t1 = ( 0.5 - roObj) * inv;
                float3 tmax = max(t0, t1);
                float tObj = min(tmax.x, min(tmax.y, tmax.z));
                float3 exitObj = roObj + rdObj * tObj;
                float3 exitWorld = mul(unity_ObjectToWorld, float4(exitObj, 1.0)).xyz;
                return length(exitWorld - ro);
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
                float tExit = exitDistance(ro, rd);
                FragOutput o;

                // Closest approach along the ray (see the header)
                float bestD = 1e10;
                float bestT = 0.0;
                bool hit = false;
                bool farGround = false;

                // Ray in the wall frame (rigid, so lengths are preserved)
                float3 q0 = toWall(ro);
                float3 qd = toWallDir(rd);
                // Inflation covers the closest-approach acceptance and the
                // normal / AO probe reach around a hit
                float infl = 0.3;
                bool nearWall = q0.y < eps
                             || segmentNearWall(q0, qd, min(tExit, _MaxDist), infl);
                if (!nearWall) {
                    if (qd.y < -1e-6) {
                        // Stop eps short of the plane like the march does (its
                        // hit is the first sample with d < eps): a coplanar floor
                        // mesh would otherwise z-fight the exact plane
                        float tp = -q0.y / qd.y - eps;
                        if (tp > 0.0 && tp <= tExit && tp <= _MaxDist) { t = tp; hit = true; farGround = true; }
                    }
                } else {
                for (int k = 0; k < 128; k++) {
                    if (k >= maxSteps) break;
                    float d = map(ro + rd * t);

                    if (d < eps) { hit = true; break; }
                    if (d < bestD) { bestD = d; bestT = t; }

                    t += d * ss;
                    if (t > _MaxDist || t > tExit) break;
                }
                if (!hit && bestD < max(eps, bestT * NEAR_MISS_PER_M)) {
                    t = bestT;
                    hit = true;
                }
                }

                if (hit) {
                    float3 p = ro + rd * t;
                    float3 q = toWall(p);
                    float3 up = wallUp();
                    float3 n = farGround ? up : calcN(p, normalEps(tier));
                    float3 lightDir = normalize(_LightDir.xyz);
                    float ao = farGround ? 1.0 : wallAO(p, n, tier);

                    // Direct-light visibility (the wall's shadow on the ground)
                    float shadow = 1.0;
                    if (_ShadowEnabled > 0)
                    {
                        float3 sro = p + n * 0.02;
                        float3 sq0 = toWall(sro);
                        float3 sqd = toWallDir(lightDir);
                        if (farGround && !segmentNearWall(sq0, sqd, _ShadowMaxDist,
                                                          infl + _DentRadius + _ShadowMaxDist / max(_ShadowSoftness, 1.0)))
                            shadow = planeSoftShadow_LOD(sq0, sqd, 0.02, _ShadowMaxDist, _ShadowSoftness, tier);
                        else
                            shadow = aliceSoftShadow_LOD(sro, lightDir, 0.02,
                                                         _ShadowMaxDist, _ShadowSoftness, tier);
                    }

                    // Surface: ground or wall (whichever the hit belongs to)
                    float3 baseColor;
                    if (farGround || q.y < mapWall(p) + 0.01)
                    {
                        baseColor = _GroundColor.rgb;
                        if (_GroundTexStrength > 0.0)
                            baseColor = lerp(baseColor, triplanar(_GroundTex, q * _GroundTexScale, n), _GroundTexStrength);
                    }
                    else
                    {
                        baseColor = _WallColor.rgb;
                        if (_WallTexStrength > 0.0)
                            baseColor = lerp(baseColor, triplanar(_WallTex, q * _WallTexScale, n), _WallTexStrength);
                        baseColor = lerp(baseColor, _DentColor.rgb, saturate(dentFreshness(p)));
                    }

                    // Lighting
                    float diff = max(dot(n, lightDir), 0.0);
                    float3 fc = baseColor * (0.2 + diff * 0.8 * shadow) * ao;

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
