// ALICE-SDF Sample: Basic (Ground + Sphere)
// The simplest possible SDF world: a floor to stand on and a sphere.
Shader "AliceSDF/Samples/Basic"
{
    Properties
    {
        _Color ("Color", Color) = (0.3, 0.85, 1.0, 1.0)
        _Color2 ("Color 2", Color) = (0.15, 0.5, 0.3, 1.0)
        _MaxDist ("Max Distance", Float) = 100.0
        _FogColor ("Fog Color", Color) = (0.01, 0.01, 0.02, 1.0)
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

            float4 _Color; float4 _Color2;
            float _MaxDist; float4 _FogColor;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
            };

            // === SDF: Ground plane + Sphere ===
            float map(float3 p)
            {
                float ground = p.y;
                float sphere = sdSphere(p - float3(0, 1.5, 0), 1.5);
                return min(ground, sphere);
            }

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            float3 calcN(float3 p) {
                float e = 0.001;
                return normalize(float3(
                    map(p+float3(e,0,0))-map(p-float3(e,0,0)),
                    map(p+float3(0,e,0))-map(p-float3(0,e,0)),
                    map(p+float3(0,0,e))-map(p-float3(0,0,e))
                ));
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
                for (int k = 0; k < 96; k++) {
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < 0.001) {
                        float3 n = calcN(p);
                        float3 col = lerp(_Color.rgb, _Color2.rgb, n.y*0.5+0.5);
                        float diff = max(dot(n, normalize(float3(1,1,-0.5))), 0.0);
                        float3 fc = col * (0.2 + diff * 0.8);
                        float fog = exp(-t * 0.01);
                        fc = lerp(_FogColor.rgb, fc, fog);
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
