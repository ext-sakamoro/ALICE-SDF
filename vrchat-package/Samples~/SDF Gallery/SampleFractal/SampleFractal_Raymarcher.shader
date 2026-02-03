// ALICE-SDF Sample: Fractal (Menger Sponge Labyrinth)
// ONE formula creates an infinite labyrinth: Subtract(Box, Repeat(Cross))
Shader "AliceSDF/Samples/Fractal"
{
    Properties
    {
        _Color ("Color 1", Color) = (0.2, 0.8, 1.0, 1.0)
        _Color2 ("Color 2", Color) = (0.8, 0.3, 0.5, 1.0)
        _BoxSize ("Box Size", Float) = 50.0
        _HoleSize ("Hole Size", Float) = 2.0
        _RepeatScale ("Repeat Scale", Float) = 15.0
        _TwistAmount ("Twist", Range(0, 0.2)) = 0.02
        _MaxDist ("Max Distance", Float) = 200.0
        _FogDensity ("Fog Density", Float) = 0.005
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
            float _BoxSize; float _HoleSize; float _RepeatScale; float _TwistAmount;
            float _MaxDist; float _FogDensity; float4 _FogColor;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
            };

            // Cross-shaped hole (3 infinite bars)
            float sdCross(float3 p, float size) {
                float inf = 1000.0;
                float da = sdBox(p, float3(inf, size, size));
                float db = sdBox(p, float3(size, inf, size));
                float dc = sdBox(p, float3(size, size, inf));
                return min(da, min(db, dc));
            }

            // === Menger Sponge: ONE formula, INFINITE complexity ===
            float map(float3 p)
            {
                // Optional twist for organic feel
                if (_TwistAmount > 0.001) {
                    float angle = p.y * _TwistAmount;
                    float c = cos(angle); float s = sin(angle);
                    p.xz = float2(c*p.x - s*p.z, s*p.x + c*p.z);
                }

                // Step 1: One big box
                float box = sdBox(p, float3(_BoxSize, _BoxSize, _BoxSize));

                // Step 2: Infinite cross via repetition (space folding)
                float3 rp = opRepeatInfinite(p, float3(_RepeatScale, _RepeatScale, _RepeatScale));
                float cross = sdCross(rp, _HoleSize);

                // Step 3: Subtract = Menger Sponge fractal
                return max(-cross, box);
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

            float hash3d(float3 p) { p=frac(p*0.3183099+0.1); p*=17.0; return frac(p.x*p.y*p.z*(p.x+p.y+p.z)); }
            float noise3d(float3 p) {
                float3 i=floor(p); float3 f=frac(p); f=f*f*(3.0-2.0*f);
                return lerp(lerp(lerp(hash3d(i),hash3d(i+float3(1,0,0)),f.x),
                    lerp(hash3d(i+float3(0,1,0)),hash3d(i+float3(1,1,0)),f.x),f.y),
                    lerp(lerp(hash3d(i+float3(0,0,1)),hash3d(i+float3(1,0,1)),f.x),
                    lerp(hash3d(i+float3(0,1,1)),hash3d(i+float3(1,1,1)),f.x),f.y),f.z);
            }

            float fbm(float3 p) {
                float v=0.0; float a=0.5; float f=1.0;
                for(int i=0;i<4;i++){v+=a*noise3d(p*f);a*=0.5;f*=2.0;}
                return v;
            }

            float3 getColor(float3 p, float3 n) {
                float3 bc = lerp(_Color.rgb, _Color2.rgb, n.y*0.5+0.5);
                float d1 = fbm(p*0.5); float d2 = fbm(p*5.0)*0.3; float d3 = fbm(p*50.0)*0.1;
                float3 c = bc + d1 + d2 + d3;
                float f = pow(1.0-abs(dot(n,normalize(_WorldSpaceCameraPos-p))),2.0);
                return saturate(c + f*float3(0.1,0.2,0.3));
            }

            v2f vert(appdata v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                o.objectCenter = mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz;
                return o;
            }

            struct FragOutput { fixed4 color : SV_Target; float depth : SV_Depth; };

            FragOutput frag(v2f i) {
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);
                float camDist = length(i.objectCenter - ro);
                int tier = aliceLodTier(camDist);
                int maxSteps = aliceLodSteps(tier);
                float eps = aliceLodEpsilon(tier);
                float ss = aliceLodStepScale(tier);
                float t = 0.0;
                FragOutput o;
                for (int k = 0; k < 128; k++) {
                    if (k >= maxSteps) break;
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < eps) {
                        float3 n = calcN(p);
                        float3 col = getColor(p, n);
                        float diff = max(dot(n, normalize(float3(1,1,-0.5))), 0.0);
                        float ao = 1.0;
                        { // Simple AO
                            float occ=0.0; float sc=1.0;
                            for(int j=0;j<4;j++){float h=0.01+0.12*float(j)/3.0;occ+=(h-map(p+h*n))*sc;sc*=0.95;}
                            ao=saturate(1.0-3.0*occ);
                        }
                        float3 fc = col * (0.15 + diff*0.85) * ao;
                        float fog = exp(-t * _FogDensity);
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
                    t += d * ss;
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
