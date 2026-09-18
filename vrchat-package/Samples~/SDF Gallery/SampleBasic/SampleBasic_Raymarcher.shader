// ALICE-SDF Sample: Basic (Ground + Sphere)
// The simplest possible SDF world: a floor and a sphere. The same law as
// SampleBasic_Collider.Evaluate (examples/vrchat_basic_golden.rs).
//
// Rendering notes (same fixes as the Mochi sample): Cull Off so the volume
// renders from inside; closest-approach acceptance so a ray grazing the
// sphere's silhouette that runs out of steps is a hit, not the far depth;
// soft contact shadow of the sphere on the ground.
Shader "AliceSDF/Samples/Basic"
{
    Properties
    {
        _Color ("Color", Color) = (0.3, 0.85, 1.0, 1.0)
        _Color2 ("Color 2", Color) = (0.15, 0.5, 0.3, 1.0)
        _MaxDist ("Max Distance", Float) = 100.0
        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1.0, 1.0, -0.5, 0.0)
        _ShadowEnabled ("Enable Soft Shadow", Int) = 1
        _ShadowSoftness ("Shadow Softness", Range(1, 128)) = 16.0
        _ShadowMaxDist ("Shadow Max Distance", Float) = 10.0
        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.01, 0.01, 0.02, 1.0)
        _FogDensity ("Fog Density", Float) = 0.01
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

            float4 _Color; float4 _Color2;
            float _MaxDist; float4 _FogColor; float _FogDensity;
            float4 _LightDir;
            int _ShadowEnabled;
            float _ShadowSoftness, _ShadowMaxDist;

            // Closest-approach acceptance, metres per metre of ray length
            #define NEAR_MISS_PER_M 0.002

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

            // === SDF: Ground plane + Sphere ===
            float mapLaw(float3 p)
            {
                float ground = p.y;
                float sphere = sdSphere(p - float3(0, 1.5, 0), 1.5);
                return min(ground, sphere);
            }

            // The law above is written in the object's frame (position and
            // rotation; the scale is the volume cube's size, not the law's):
            // move or turn the prefab and the SDF comes along. The collider
            // does the same (AliceSDF_Collider.EvaluateWorld).
            float3 toLaw(float3 p)
            {
                float3 s = float3(length(unity_ObjectToWorld._m00_m10_m20),
                                  length(unity_ObjectToWorld._m01_m11_m21),
                                  length(unity_ObjectToWorld._m02_m12_m22));
                return mul(unity_WorldToObject, float4(p, 1.0)).xyz * s;
            }
            float map(float3 p) { return mapLaw(toLaw(p)); }

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

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            float normalEps(int tier) {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.001;
                if (tier == ALICE_LOD_TIER_MED)  return 0.003;
                return 0.01;
            }

            float3 calcN(float3 p, float e) {
                return normalize(float3(
                    map(p+float3(e,0,0))-map(p-float3(e,0,0)),
                    map(p+float3(0,e,0))-map(p-float3(0,e,0)),
                    map(p+float3(0,0,e))-map(p-float3(0,0,e))
                ));
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
                float camDist = length(i.objectCenter - ro);
                int tier = aliceLodTier(camDist);
                int maxSteps = aliceLodSteps(tier);
                float eps = aliceLodEpsilon(tier);
                float ss = aliceLodStepScale(tier);
                float t = 0.0;
                float tExit = exitDistance(ro, rd);
                FragOutput o;

                float bestD = 1e10;
                float bestT = 0.0;
                bool hit = false;
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

                if (hit) {
                    float3 p = ro + rd * t;
                    float3 n = calcN(p, normalEps(tier));
                    float3 lightDir = normalize(_LightDir.xyz);
                    float shadow = 1.0;
                    if (_ShadowEnabled > 0)
                        shadow = aliceSoftShadow_LOD(p + n * 0.02, lightDir, 0.02, _ShadowMaxDist, _ShadowSoftness, tier);
                    float3 col = lerp(_Color.rgb, _Color2.rgb, n.y*0.5+0.5);
                    float diff = max(dot(n, lightDir), 0.0);
                    float3 fc = col * (0.2 + diff * 0.8 * shadow);
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
