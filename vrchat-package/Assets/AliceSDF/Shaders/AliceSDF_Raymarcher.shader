// =============================================================================
// AliceSDF_Raymarcher.shader - VRChat PBR Raymarching Surface v2.0
// =============================================================================
// Drag & Drop this material onto a mesh (Cube recommended) in your VRChat world.
// The mesh becomes the bounding volume; SDF is rendered inside it.
//
// v2.0 Features:
//   - PBR Lighting (GGX + Fresnel-Schlick + Smith geometry)
//   - Soft Shadows (penumbra estimation via SDF)
//   - Global Illumination approximation (2-bounce)
//   - Volumetric Fog (ray-integrated density)
//   - Reflection (secondary ray with Fresnel)
//   - Multi-material system (up to 4 materials via map2)
//   - Automatic LOD (fewer steps when far from camera)
//   - SV_Depth output (proper occlusion with VRM avatars)
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Raymarcher"
{
    Properties
    {
        [Header(Material 0)]
        _Mat0_Color ("Mat0 Color", Color) = (0.2, 0.8, 1.0, 1.0)
        _Mat0_Roughness ("Mat0 Roughness", Range(0, 1)) = 0.5
        _Mat0_Metallic ("Mat0 Metallic", Range(0, 1)) = 0.0

        [Header(Material 1)]
        _Mat1_Color ("Mat1 Color", Color) = (0.8, 0.3, 0.5, 1.0)
        _Mat1_Roughness ("Mat1 Roughness", Range(0, 1)) = 0.5
        _Mat1_Metallic ("Mat1 Metallic", Range(0, 1)) = 0.0

        [Header(Material 2)]
        _Mat2_Color ("Mat2 Color", Color) = (0.3, 0.9, 0.3, 1.0)
        _Mat2_Roughness ("Mat2 Roughness", Range(0, 1)) = 0.7
        _Mat2_Metallic ("Mat2 Metallic", Range(0, 1)) = 0.0

        [Header(Material 3)]
        _Mat3_Color ("Mat3 Color", Color) = (0.9, 0.9, 0.2, 1.0)
        _Mat3_Roughness ("Mat3 Roughness", Range(0, 1)) = 0.3
        _Mat3_Metallic ("Mat3 Metallic", Range(0, 1)) = 1.0

        [Header(Raymarching)]
        _MaxSteps ("Max Steps (0=Auto)", Int) = 0
        _MaxDist ("Max Distance", Float) = 200.0
        _SurfaceEpsilon ("Surface Epsilon (0=Auto)", Float) = 0.0

        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1, 1, -0.5, 0)
        _LightColor ("Light Color", Color) = (1.0, 0.95, 0.9, 1.0)
        _LightIntensity ("Light Intensity", Range(0, 5)) = 1.5
        _AmbientStrength ("Ambient", Range(0, 1)) = 0.08
        _AOEnabled ("Enable AO", Int) = 1
        _GIEnabled ("Enable GI", Int) = 1

        [Header(Shadow)]
        _ShadowEnabled ("Enable Soft Shadow", Int) = 1
        _ShadowSoftness ("Shadow Softness", Range(1, 128)) = 16.0
        _ShadowMaxDist ("Shadow Max Distance", Float) = 40.0

        [Header(Reflection)]
        _ReflectionEnabled ("Enable Reflection", Int) = 0
        _ReflectionStrength ("Reflection Strength", Range(0, 1)) = 0.5
        _ReflectionMaxSteps ("Reflection Steps", Int) = 32

        [Header(Fog)]
        _FogDensity ("Fog Density", Float) = 0.005
        _FogColor ("Fog Color", Color) = (0.01, 0.01, 0.02, 1.0)

        [Header(Volumetric Fog)]
        _VolFogEnabled ("Enable Volumetric Fog", Int) = 0
        _VolFogDensity ("Vol Fog Density", Range(0, 0.05)) = 0.005
        _VolFogColor ("Vol Fog Color", Color) = (0.4, 0.5, 0.7, 1.0)
        _VolFogHeight ("Vol Fog Height Falloff", Float) = 10.0
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

            #include "AliceSDF_Include.cginc"

            // =============================================================
            // Properties
            // =============================================================
            float4 _Mat0_Color, _Mat1_Color, _Mat2_Color, _Mat3_Color;
            float _Mat0_Roughness, _Mat1_Roughness, _Mat2_Roughness, _Mat3_Roughness;
            float _Mat0_Metallic, _Mat1_Metallic, _Mat2_Metallic, _Mat3_Metallic;

            int _MaxSteps;
            float _MaxDist;
            float _SurfaceEpsilon;

            float4 _LightDir;
            float4 _LightColor;
            float _LightIntensity;
            float _AmbientStrength;
            int _AOEnabled;
            int _GIEnabled;

            int _ShadowEnabled;
            float _ShadowSoftness;
            float _ShadowMaxDist;

            int _ReflectionEnabled;
            float _ReflectionStrength;
            int _ReflectionMaxSteps;

            float _FogDensity;
            float4 _FogColor;

            int _VolFogEnabled;
            float _VolFogDensity;
            float4 _VolFogColor;
            float _VolFogHeight;

            // =============================================================
            // Structures
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
            // SDF DEFINITION
            // =============================================================
            // map2() returns float2(distance, materialID)
            // map() returns distance only (used by LOD, AO, shadow, normal)
            //
            // Replace these functions or use ALICE-Baker to auto-generate.
            // =============================================================

            float2 map2(float3 p)
            {
                // === DEFAULT DEMO: Multi-material Menger labyrinth ===

                // Bounding box (material 0)
                float box = sdBox(p, float3(50.0, 50.0, 50.0));

                // Infinite cross pattern (material 1)
                float3 rp = opRepeatInfinite(p, float3(15.0, 15.0, 15.0));
                float cross_d = min(
                    sdBox(rp, float3(1000.0, 2.0, 2.0)),
                    min(
                        sdBox(rp, float3(2.0, 1000.0, 2.0)),
                        sdBox(rp, float3(2.0, 2.0, 1000.0))
                    )
                );

                // Subtract with material preservation
                float labyrinth = max(box, -cross_d);

                // Ground plane (material 2)
                float ground = p.y + 25.0;

                // Floating sphere (material 3 = metallic gold)
                float sphere = sdSphere(p - float3(0, -15, 0), 5.0);

                // Combine with material selection
                float2 result = float2(labyrinth, 0.0);
                result = opUnionMat(result, float2(ground, 2.0));
                result = opSmoothUnionMat(result, float2(sphere, 3.0), 1.0);
                return result;
            }

            float map(float3 p)
            {
                return map2(p).x;
            }

            // Include LOD (calls map())
            #include "AliceSDF_LOD.cginc"

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
            // Material Lookup
            // =============================================================
            void getMaterial(float matID, out float3 albedo, out float roughness, out float metallic)
            {
                int id = int(matID + 0.5);
                if (id <= 0)      { albedo = _Mat0_Color.rgb; roughness = _Mat0_Roughness; metallic = _Mat0_Metallic; }
                else if (id == 1) { albedo = _Mat1_Color.rgb; roughness = _Mat1_Roughness; metallic = _Mat1_Metallic; }
                else if (id == 2) { albedo = _Mat2_Color.rgb; roughness = _Mat2_Roughness; metallic = _Mat2_Metallic; }
                else              { albedo = _Mat3_Color.rgb; roughness = _Mat3_Roughness; metallic = _Mat3_Metallic; }
            }

            // =============================================================
            // Procedural Surface Detail
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

            // =============================================================
            // PBR Shading
            // =============================================================
            float3 shadePBR(float3 p, float3 n, float3 V, float3 albedo, float roughness, float metallic, int tier)
            {
                float3 L = normalize(_LightDir.xyz);
                float3 H = normalize(V + L);
                float3 lightCol = _LightColor.rgb * _LightIntensity;

                // F0: dialectric = 0.04, metallic = albedo
                float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

                // Cook-Torrance BRDF
                float NDF = distributionGGX(n, H, roughness);
                float G = geometrySmith(n, V, L, roughness);
                float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

                float3 kS = F;
                float3 kD = (1.0 - kS) * (1.0 - metallic);

                float NdotL = max(dot(n, L), 0.0);
                float NdotV = max(dot(n, V), 0.001);
                float3 numerator = NDF * G * F;
                float denominator = 4.0 * NdotV * NdotL + 0.0001;
                float3 specular = numerator / denominator;

                float3 Lo = (kD * albedo / 3.14159265 + specular) * lightCol * NdotL;

                // Soft shadow
                float shadow = 1.0;
                if (_ShadowEnabled > 0)
                {
                    shadow = aliceSoftShadow_LOD(p + n * 0.02, L, 0.02, _ShadowMaxDist, _ShadowSoftness, tier);
                }
                Lo *= shadow;

                // Ambient occlusion
                float ao = 1.0;
                if (_AOEnabled > 0)
                {
                    int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                                  (tier == ALICE_LOD_TIER_MED) ? 3 : 2;
                    ao = sdfAO(p, n, aoSteps);
                }

                // Ambient + GI
                float3 ambient = albedo * _AmbientStrength * ao;

                if (_GIEnabled > 0)
                {
                    float3 gi = aliceGI_LOD(p, n, L, lightCol, tier);
                    ambient += gi * albedo * ao;
                }

                // Surface detail noise
                float detail = noise3d(p * 10.0) * 0.08 + noise3d(p * 50.0) * 0.03;
                Lo += detail * albedo * ao;

                return ambient + Lo;
            }

            // =============================================================
            // Reflection Ray
            // =============================================================
            float3 traceReflection(float3 ro, float3 rd, float eps, int maxSteps)
            {
                float t = 0.0;
                for (int i = 0; i < 32; i++)
                {
                    if (i >= maxSteps) break;
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < eps * 2.0)
                    {
                        float2 hit = map2(p);
                        float3 n = sdfNormal(p, eps);
                        float3 V = -rd;
                        float3 albedo; float roughness; float metallic;
                        getMaterial(hit.y, albedo, roughness, metallic);

                        float3 L = normalize(_LightDir.xyz);
                        float NdotL = max(dot(n, L), 0.0);
                        return albedo * (_AmbientStrength + NdotL * (1.0 - _AmbientStrength));
                    }
                    t += d;
                    if (t > 100.0) break;
                }
                // Sky color for reflection miss
                return lerp(_FogColor.rgb, _FogColor.rgb + float3(0.1, 0.2, 0.4), rd.y * 0.5 + 0.5);
            }

            // =============================================================
            // Volumetric Fog Integration
            // =============================================================
            float3 volumetricFog(float3 ro, float3 rd, float tHit, int steps)
            {
                float3 accum = float3(0, 0, 0);
                float dt = tHit / float(steps);
                float transmittance = 1.0;

                for (int i = 0; i < steps; i++)
                {
                    float t = dt * (float(i) + 0.5);
                    float3 p = ro + rd * t;

                    // Height-based density falloff
                    float density = _VolFogDensity * exp(-max(p.y, 0.0) / max(_VolFogHeight, 0.1));

                    // Light scattering (simplified)
                    float3 L = normalize(_LightDir.xyz);
                    float phase = 0.25 + 0.75 * pow(max(dot(rd, L), 0.0), 4.0);

                    float3 luminance = _VolFogColor.rgb * _LightColor.rgb * phase * density;
                    float extinction = density;

                    transmittance *= exp(-extinction * dt);
                    accum += luminance * transmittance * dt;
                }
                return accum;
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
            // Fragment Shader
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

                // LOD
                float camDist = length(i.objectCenter - _WorldSpaceCameraPos);
                int tier = aliceLodTier(camDist);
                int maxSteps = (_MaxSteps > 0) ? _MaxSteps : aliceLodSteps(tier);
                float eps = (_SurfaceEpsilon > 0.0) ? _SurfaceEpsilon : aliceLodEpsilon(tier);
                float stepScale = aliceLodStepScale(tier);

                float t = 0.0;
                FragOutput o;

                // === RAYMARCHING LOOP ===
                for (int k = 0; k < 128; k++)
                {
                    if (k >= maxSteps) break;

                    float3 p = ro + rd * t;
                    float d = map(p);

                    if (d < eps)
                    {
                        // Hit! Get material and normal
                        float2 hit = map2(p);
                        float3 n = sdfNormal(p, eps);
                        float3 V = normalize(_WorldSpaceCameraPos - p);

                        // Material lookup
                        float3 albedo; float roughness; float metallic;
                        getMaterial(hit.y, albedo, roughness, metallic);

                        // PBR shading
                        float3 color = shadePBR(p, n, V, albedo, roughness, metallic, tier);

                        // Reflection
                        if (_ReflectionEnabled > 0)
                        {
                            float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
                            float3 F = fresnelSchlickRoughness(max(dot(n, V), 0.0), F0, roughness);
                            float3 reflDir = reflect(-V, n);
                            float3 reflColor = traceReflection(p + n * eps * 4.0, reflDir, eps * 2.0, _ReflectionMaxSteps);
                            color = lerp(color, reflColor, F * _ReflectionStrength * (1.0 - roughness));
                        }

                        // Distance fog
                        float fog = exp(-t * _FogDensity);
                        color = lerp(_FogColor.rgb, color, fog);

                        // Volumetric fog
                        if (_VolFogEnabled > 0)
                        {
                            int volSteps = (tier == ALICE_LOD_TIER_HIGH) ? 16 :
                                           (tier == ALICE_LOD_TIER_MED) ? 8 : 4;
                            float3 volFog = volumetricFog(ro, rd, t, volSteps);
                            color += volFog;
                        }

                        // Depth output for avatar occlusion
                        float4 clipPos = UnityWorldToClipPos(p);
                        o.color = fixed4(color, 1.0);

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

                // Volumetric fog even on miss
                if (_VolFogEnabled > 0)
                {
                    float3 volFog = volumetricFog(ro, rd, _MaxDist * 0.5, 8);
                    sky += volFog;
                }

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
