// ALICE-SDF Sample: Mix (Fractal Planet + Torus Ring + Onion Shell)
// Fusion of Cosmic and Fractal concepts: planets made of Menger Sponge.
Shader "AliceSDF/Samples/Mix"
{
    Properties
    {
        _Color ("Color 1", Color) = (0.9, 0.4, 0.1, 1.0)
        _Color2 ("Color 2", Color) = (0.1, 0.6, 0.9, 1.0)
        _PlanetRadius ("Planet Radius", Float) = 6.0
        _HoleSize ("Fractal Hole Size", Float) = 0.8
        _RepeatScale ("Fractal Repeat", Float) = 5.0
        _RingMajor ("Ring Major Radius", Float) = 10.0
        _RingMinor ("Ring Minor Radius", Float) = 0.3
        _OnionRadius ("Onion Radius", Float) = 3.0
        _OnionLayers ("Onion Layers", Range(1, 5)) = 3
        _OnionThickness ("Onion Thickness", Float) = 0.15
        _Smoothness ("Smooth Blend", Range(0.1, 3.0)) = 0.8
        _MaxDist ("Max Distance", Float) = 150.0
        _FogDensity ("Fog Density", Float) = 0.004
        _FogColor ("Fog Color", Color) = (0.02, 0.01, 0.03, 1.0)
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
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"
            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc"

            float4 _Color; float4 _Color2;
            float _PlanetRadius; float _HoleSize; float _RepeatScale;
            float _RingMajor; float _RingMinor;
            float _OnionRadius; float _OnionLayers; float _OnionThickness;
            float _Smoothness;
            float _MaxDist; float _FogDensity; float4 _FogColor;

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

            // Cross-shaped hole (3 infinite bars)
            float sdCross(float3 p, float size) {
                float inf = 1000.0;
                float da = sdBox(p, float3(inf, size, size));
                float db = sdBox(p, float3(size, inf, size));
                float dc = sdBox(p, float3(size, size, inf));
                return min(da, min(db, dc));
            }

            // Onion shell: concentric hollow spheres
            float sdOnion(float3 p, float radius, int layers, float thickness) {
                float d = length(p) - radius;
                for (int i = 0; i < 5; i++) {
                    if (i >= layers) break;
                    d = abs(d) - thickness;
                }
                return d;
            }

            // === Mix SDF: Fractal Planet + Ring + Onion ===
            float map(float3 p)
            {
                float time = _Time.y;

                // --- Fractal Planet (Sphere ∩ Menger Sponge) ---
                // Slow rotation
                float rotAngle = time * 0.1;
                float cr = cos(rotAngle); float sr = sin(rotAngle);
                float3 pp = float3(cr*p.x - sr*p.z, p.y, sr*p.x + cr*p.z);

                float planet = sdSphere(pp, _PlanetRadius);
                float3 rp = opRepeatInfinite(pp, float3(_RepeatScale, _RepeatScale, _RepeatScale));
                float cross = sdCross(rp, _HoleSize);
                float fractalPlanet = max(-cross, planet); // Subtract holes from sphere

                // --- Torus Ring (tilted, orbiting) ---
                float ringTilt = time * 0.05;
                float ct = cos(ringTilt); float st = sin(ringTilt);
                float3 ringP = float3(p.x, ct*p.y - st*p.z, st*p.y + ct*p.z);
                float ring = sdTorus(ringP, _RingMajor, _RingMinor);

                // --- Onion Shell (offset, orbiting) ---
                float orbitAngle = time * 0.2;
                float3 onionPos = float3(cos(orbitAngle) * 16.0, sin(orbitAngle) * 2.0, sin(orbitAngle) * 16.0);
                float onion = sdOnion(p - onionPos, _OnionRadius, (int)_OnionLayers, _OnionThickness);

                // --- Combine with smooth union ---
                float d = fractalPlanet;
                d = opSmoothUnion(d, ring, _Smoothness);
                d = opSmoothUnion(d, onion, _Smoothness * 0.5);
                return d;
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

            float3 getColor(float3 p, float3 n) {
                float3 bc = lerp(_Color.rgb, _Color2.rgb, n.y*0.5+0.5);
                float d1 = noise3d(p*2.0)*0.3;
                float d2 = noise3d(p*10.0)*0.15;
                float3 c = bc + d1 + d2;
                // Warm glow near center
                float centerDist = length(p);
                c += _Color.rgb * saturate(1.0 - centerDist / (_PlanetRadius * 1.5)) * 0.5;
                float f = pow(1.0-abs(dot(n,normalize(_WorldSpaceCameraPos-p))),2.0);
                return saturate(c + f*float3(0.1,0.15,0.2));
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
                // Space background
                float stars = step(0.998, hash3d(rd * 500.0));
                float3 sky = _FogColor.rgb + stars * 0.7;
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
