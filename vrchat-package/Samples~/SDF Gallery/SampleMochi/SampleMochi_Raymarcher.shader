// =============================================================================
// ALICE-SDF Sample: Mochi (Dynamic Soft-Body Blobs)
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
        _BlendK ("Mochi Blend (higher = stickier)", Float) = 0.5
        _GroundK ("Ground Stickiness", Float) = 0.15

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.83, 0.80, 0.76, 1.0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include "UnityCG.cginc"
            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc"

            // Inspector properties
            float4 _MochiColor, _MochiColor2, _GroundColor, _GroundColor2, _FogColor;
            float _MaxDist;
            float _BlendK, _GroundK;

            // Dynamic mochi data (set from UdonSharp)
            // xyz = world position, w = radius
            float4 _MochiData[16];
            float _MochiCount;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
            };

            // =================================================================
            // SDF: Ground + Dynamic Mochi Blobs
            // =================================================================
            float map(float3 p)
            {
                // Ground plane at Y=0
                float ground = p.y;

                // Combine all mochis with SmoothUnion
                // Note: opSmoothUnion(1e10, d, k) = d (math works out)
                float mochi = 1e10;
                for (int i = 0; i < 16; i++)
                {
                    if (i >= (int)_MochiCount) break;
                    float3 mp = _MochiData[i].xyz;
                    float mr = _MochiData[i].w;
                    float d = sdSphere(p - mp, mr);
                    mochi = opSmoothUnion(mochi, d, _BlendK);
                }

                // SmoothUnion with ground: mochi "squishes" onto floor
                return opSmoothUnion(ground, mochi, _GroundK);
            }

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            // Normal via central differences
            float3 calcN(float3 p) {
                float e = 0.001;
                return normalize(float3(
                    map(p + float3(e,0,0)) - map(p - float3(e,0,0)),
                    map(p + float3(0,e,0)) - map(p - float3(0,e,0)),
                    map(p + float3(0,0,e)) - map(p - float3(0,0,e))
                ));
            }

            // AO
            float calcAO(float3 p, float3 n) {
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < 5; i++) {
                    float h = 0.01 + 0.08 * float(i);
                    occ += (h - map(p + h * n)) * sca;
                    sca *= 0.9;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // Find distance to nearest mochi sphere (raw, no SmoothUnion)
            float nearestMochiDist(float3 p) {
                float d = 1e10;
                for (int i = 0; i < 16; i++) {
                    if (i >= (int)_MochiCount) break;
                    d = min(d, sdSphere(p - _MochiData[i].xyz, _MochiData[i].w));
                }
                return d;
            }

            // Simple noise for ground texture
            float hash2d(float2 p) {
                p = frac(p * float2(0.3183, 0.3671));
                p *= 17.0;
                return frac(p.x * p.y * (p.x + p.y));
            }

            v2f vert(appdata v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                return o;
            }

            struct FragOutput { fixed4 color : SV_Target; float depth : SV_Depth; };

            FragOutput frag(v2f i) {
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);
                float t = 0.0;
                FragOutput o;

                for (int k = 0; k < 128; k++) {
                    float3 p = ro + rd * t;
                    float d = map(p);

                    if (d < 0.001) {
                        float3 n = calcN(p);
                        float3 lightDir = normalize(float3(1, 1, -0.5));
                        float ao = calcAO(p, n);

                        // Determine surface: ground vs mochi
                        float mochiRaw = nearestMochiDist(p);
                        float groundRaw = p.y;

                        float3 fc;
                        if (mochiRaw < groundRaw)
                        {
                            // === MOCHI SURFACE ===
                            // Warm wrap lighting (subsurface scattering approx)
                            float3 col = lerp(_MochiColor.rgb, _MochiColor2.rgb,
                                              n.y * 0.5 + 0.5);
                            float wrap = max(dot(n, lightDir) + 0.4, 0.0) / 1.4;

                            // Fresnel rim for soft translucent look
                            float3 viewDir = normalize(ro - p);
                            float fresnel = pow(1.0 - abs(dot(n, viewDir)), 3.0);
                            float3 sss = float3(1.0, 0.88, 0.72) * fresnel * 0.25;

                            fc = col * (0.3 + wrap * 0.7) * ao + sss;
                        }
                        else
                        {
                            // === GROUND SURFACE ===
                            float3 col = lerp(_GroundColor.rgb, _GroundColor2.rgb,
                                              hash2d(p.xz * 2.0) * 0.3 + 0.35);
                            float diff = max(dot(n, lightDir), 0.0);
                            fc = col * (0.2 + diff * 0.8) * ao;
                        }

                        // Fog
                        fc = lerp(_FogColor.rgb, fc, exp(-t * 0.005));

                        float4 cp = UnityWorldToClipPos(p);
                        o.color = fixed4(fc, 1.0);
                        #if defined(UNITY_REVERSED_Z)
                            o.depth = cp.z / cp.w;
                        #else
                            o.depth = (cp.z / cp.w) * 0.5 + 0.5;
                        #endif
                        return o;
                    }

                    t += d;
                    if (t > _MaxDist) break;
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
