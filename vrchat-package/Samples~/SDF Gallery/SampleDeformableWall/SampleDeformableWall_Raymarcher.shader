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
// Rendering notes (same fixes as the Mochi sample):
//   - Cull Off: the player walks inside the volume cube
//   - closest-approach acceptance: a ray grazing the wall's edge that runs
//     out of steps a hair short of the surface is a hit, not the far depth
//   - AO samples the hard (k = 0) subtraction: the smooth rim of a dent
//     under-reports distance and read as a dark ring around every dent
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
            float _MaxDist;
            float _WallWidth, _WallHeight, _WallThick;
            float _DentRadius, _DentSmooth;
            float4 _PlayerCapA, _PlayerCapB;
            float _PlayerDentK;
            float4 _LightDir;
            int _ShadowEnabled;
            float _ShadowSoftness, _ShadowMaxDist, _FogDensity;

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

            // =================================================================
            // SDF: Ground + Wall with dynamic dents + the player's body
            // =================================================================
            // The dented wall alone (what the collider's EvaluateWallSdf is)
            float mapWall(float3 p)
            {
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                float wall = sdBox(p - wc, wh);

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
                return min(p.y, wall);
            }

            // Hard (k = 0) version for the ambient occlusion: exact outside
            // the wall and the dents, so only real geometry occludes
            float mapOccluder(float3 p)
            {
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                float wall = sdBox(p - wc, wh);
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
                return min(p.y, wall);
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
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                              (tier == ALICE_LOD_TIER_MED)  ? 3 : 2;
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

                // Closest approach along the ray (see the header)
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
                    float3 n = calcN(p, normalEps(tier));
                    float3 lightDir = normalize(_LightDir.xyz);
                    float ao = wallAO(p, n, tier);

                    // Direct-light visibility (the wall's shadow on the ground)
                    float shadow = 1.0;
                    if (_ShadowEnabled > 0)
                    {
                        shadow = aliceSoftShadow_LOD(p + n * 0.02, lightDir, 0.02,
                                                     _ShadowMaxDist, _ShadowSoftness, tier);
                    }

                    // Surface: ground or wall (whichever the hit belongs to)
                    float3 baseColor;
                    if (p.y < mapWall(p) + 0.01)
                    {
                        baseColor = _GroundColor.rgb;
                    }
                    else
                    {
                        baseColor = lerp(_WallColor.rgb, _DentColor.rgb, saturate(dentFreshness(p)));
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
