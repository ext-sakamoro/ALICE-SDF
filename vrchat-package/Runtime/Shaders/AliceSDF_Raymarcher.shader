// =============================================================================
// AliceSDF_Raymarcher.shader - VRChat Raymarching Surface
// =============================================================================
// Drag & Drop this material onto a mesh (Cube recommended) in your VRChat world.
// The mesh becomes the bounding volume; SDF is rendered inside it.
//
// Features:
//   - Automatic LOD (fewer steps when far from camera)
//   - SV_Depth output (proper occlusion with VRM avatars)
//   - Procedural coloring (infinite resolution textures)
//   - VRChat Performance Rank: Good ~ Excellent
//
// Usage:
//   1. Create Material with this shader
//   2. Assign to a Cube (bounding volume)
//   3. Adjust properties in Inspector
//   4. Replace map() function with your SDF (or use ALICE-Baker)
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Raymarcher"
{
    Properties
    {
        [Header(Colors)]
        _Color ("Primary Color", Color) = (0.2, 0.8, 1.0, 1.0)
        _Color2 ("Secondary Color", Color) = (0.8, 0.3, 0.5, 1.0)

        [Header(Raymarching)]
        _MaxSteps ("Max Steps (Override, 0=Auto)", Int) = 0
        _MaxDist ("Max Distance", Float) = 200.0
        _SurfaceEpsilon ("Surface Epsilon (Override, 0=Auto)", Float) = 0.0

        [Header(Lighting)]
        _AmbientStrength ("Ambient Strength", Range(0, 1)) = 0.2
        _AOEnabled ("Enable AO", Int) = 1

        [Header(Fog)]
        _FogDensity ("Fog Density", Float) = 0.005
        _FogColor ("Fog Color", Color) = (0.01, 0.01, 0.02, 1.0)
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        LOD 200

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"

            // Include ALICE-SDF libraries
            // NOTE: map() must be defined BEFORE including LOD (which calls map)
            //       We forward-declare and define map() below.

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc"

            // =============================================================
            // Properties
            // =============================================================
            float4 _Color;
            float4 _Color2;
            int _MaxSteps;
            float _MaxDist;
            float _SurfaceEpsilon;
            float _AmbientStrength;
            int _AOEnabled;
            float _FogDensity;
            float4 _FogColor;

            // =============================================================
            // Vertex/Fragment Structures
            // =============================================================
            struct appdata
            {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // =============================================================
            // USER SDF DEFINITION
            // =============================================================
            // Replace this function with your own SDF!
            // Or use ALICE-Baker to auto-generate from .asdf files.
            //
            // This default creates: Subtract(Box, Repeat(Cross))
            // = Infinite labyrinth from 1 formula
            // =============================================================

            float map(float3 p)
            {
                // === DEFAULT DEMO: Menger-like labyrinth ===

                // Big bounding box
                float box = sdBox(p, float3(50.0, 50.0, 50.0));

                // Infinite cross via repeat
                float3 rp = opRepeatInfinite(p, float3(15.0, 15.0, 15.0));
                float cross_d = min(
                    sdBox(rp, float3(1000.0, 2.0, 2.0)),
                    min(
                        sdBox(rp, float3(2.0, 1000.0, 2.0)),
                        sdBox(rp, float3(2.0, 2.0, 1000.0))
                    )
                );

                // Subtract = labyrinth
                return opSubtraction(box, cross_d);
            }

            // Now include LOD (it calls map())
            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            // =============================================================
            // Normal Calculation
            // =============================================================
            float3 sdfNormal(float3 p, float eps)
            {
                float3 n;
                n.x = map(p + float3(eps, 0, 0)) - map(p - float3(eps, 0, 0));
                n.y = map(p + float3(0, eps, 0)) - map(p - float3(0, eps, 0));
                n.z = map(p + float3(0, 0, eps)) - map(p - float3(0, 0, eps));
                return normalize(n);
            }

            // =============================================================
            // AO
            // =============================================================
            float sdfAO(float3 pos, float3 nor, int steps)
            {
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < steps; i++)
                {
                    float h = 0.01 + 0.12 * float(i) / float(max(steps - 1, 1));
                    float d = map(pos + h * nor);
                    occ += (h - d) * sca;
                    sca *= 0.95;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // =============================================================
            // Procedural Coloring
            // =============================================================
            float hash3d(float3 p)
            {
                p = frac(p * 0.3183099 + 0.1);
                p *= 17.0;
                return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
            }

            float noise3d(float3 p)
            {
                float3 i = floor(p);
                float3 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);

                return lerp(
                    lerp(lerp(hash3d(i + float3(0,0,0)), hash3d(i + float3(1,0,0)), f.x),
                         lerp(hash3d(i + float3(0,1,0)), hash3d(i + float3(1,1,0)), f.x), f.y),
                    lerp(lerp(hash3d(i + float3(0,0,1)), hash3d(i + float3(1,0,1)), f.x),
                         lerp(hash3d(i + float3(0,1,1)), hash3d(i + float3(1,1,1)), f.x), f.y),
                    f.z
                );
            }

            float3 getColor(float3 p, float3 n)
            {
                float3 baseColor = lerp(_Color.rgb, _Color2.rgb, n.y * 0.5 + 0.5);
                float detail = noise3d(p * 10.0) * 0.3 + noise3d(p * 50.0) * 0.1;
                float3 color = baseColor + detail;
                float fresnel = pow(1.0 - abs(dot(n, normalize(_WorldSpaceCameraPos - p))), 2.0);
                color += fresnel * float3(0.1, 0.2, 0.3);
                return saturate(color);
            }

            // =============================================================
            // Vertex Shader
            // =============================================================
            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                o.objectCenter = mul(unity_ObjectToWorld, float4(0, 0, 0, 1)).xyz;
                return o;
            }

            // =============================================================
            // Fragment Shader (Raymarching + Depth)
            // =============================================================
            struct FragOutput
            {
                fixed4 color : SV_Target;
                float depth : SV_Depth;
            };

            FragOutput frag(v2f i)
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);

                // LOD tier from camera distance to object
                float camDist = length(i.objectCenter - _WorldSpaceCameraPos);
                int tier = aliceLodTier(camDist);

                // Determine step count and epsilon
                int maxSteps = (_MaxSteps > 0) ? _MaxSteps : aliceLodSteps(tier);
                float eps = (_SurfaceEpsilon > 0.0) ? _SurfaceEpsilon : aliceLodEpsilon(tier);
                float stepScale = aliceLodStepScale(tier);

                float t = 0.0;
                FragOutput o;

                // === RAYMARCHING LOOP ===
                for (int k = 0; k < 128; k++) // Compile-time max; runtime exit via maxSteps
                {
                    if (k >= maxSteps) break;

                    float3 p = ro + rd * t;
                    float d = map(p);

                    if (d < eps)
                    {
                        float3 n = sdfNormal(p, eps);
                        float3 color = getColor(p, n);

                        // Lighting
                        float3 lightDir = normalize(float3(1.0, 1.0, -0.5));
                        float diff = max(dot(n, lightDir), 0.0);

                        // AO
                        float ao = 1.0;
                        if (_AOEnabled > 0)
                        {
                            int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                                          (tier == ALICE_LOD_TIER_MED) ? 3 : 2;
                            ao = sdfAO(p, n, aoSteps);
                        }

                        float3 finalColor = color * (_AmbientStrength + diff * (1.0 - _AmbientStrength)) * ao;

                        // Fog
                        float fog = exp(-t * _FogDensity);
                        finalColor = lerp(_FogColor.rgb, finalColor, fog);

                        // Depth output for avatar occlusion
                        float4 clipPos = UnityWorldToClipPos(p);
                        o.color = fixed4(finalColor, 1.0);

                        #if defined(UNITY_REVERSED_Z)
                            o.depth = clipPos.z / clipPos.w;
                        #else
                            o.depth = (clipPos.z / clipPos.w) * 0.5 + 0.5;
                        #endif

                        return o;
                    }

                    t += d * stepScale;
                    if (t > _MaxDist) break;
                }

                // Miss - sky
                float3 sky = lerp(
                    _FogColor.rgb,
                    _FogColor.rgb + float3(0.04, 0.09, 0.18),
                    rd.y * 0.5 + 0.5
                );
                o.color = fixed4(sky, 1.0);

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
