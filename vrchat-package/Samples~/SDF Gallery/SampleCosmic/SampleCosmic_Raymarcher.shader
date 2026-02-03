// ALICE-SDF Sample: Cosmic (Sun + Planet + Ring + Moon)
// Ported from CosmicDemo GPU Compute shader to VRChat pure HLSL.
// Uses _Time for orbital animation.
Shader "AliceSDF/Samples/Cosmic"
{
    Properties
    {
        _Color ("Sun Color", Color) = (1.0, 0.6, 0.1, 1.0)
        _Color2 ("Planet Color", Color) = (0.2, 0.5, 1.0, 1.0)
        _SunRadius ("Sun Radius", Float) = 8.0
        _PlanetRadius ("Planet Radius", Float) = 2.5
        _PlanetDistance ("Planet Distance", Float) = 18.0
        _Smoothness ("Smooth Blend", Range(0.1, 5.0)) = 1.5
        _RingTwist ("Ring Twist", Range(0, 3)) = 0.5
        _MaxDist ("Max Distance", Float) = 200.0
        _FogDensity ("Fog Density", Float) = 0.003
        _FogColor ("Fog Color", Color) = (0.01, 0.005, 0.02, 1.0)
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
            float _SunRadius; float _PlanetRadius; float _PlanetDistance;
            float _Smoothness; float _RingTwist;
            float _MaxDist; float _FogDensity; float4 _FogColor;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
            };

            // === Cosmic SDF: Sun + Planet(orbit) + Ring + Moon ===
            float map(float3 p)
            {
                float time = _Time.y;

                // Sun
                float sun = sdSphere(p, _SunRadius);

                // Planet (orbiting)
                float orbitAngle = time * 0.15;
                float3 planetPos = float3(cos(orbitAngle) * _PlanetDistance, 0, sin(orbitAngle) * _PlanetDistance);
                float planet = sdSphere(p - planetPos, _PlanetRadius);

                // Ring (tilted torus around planet)
                float3 ringP = p - planetPos;
                // Tilt 15 degrees
                float c15 = 0.9659; float s15 = 0.2588;
                ringP = float3(ringP.x, c15*ringP.y - s15*ringP.z, s15*ringP.y + c15*ringP.z);
                // Optional twist
                if (_RingTwist > 0.01) {
                    float ta = _RingTwist * ringP.y;
                    float ct = cos(ta); float st = sin(ta);
                    ringP.xz = float2(ct*ringP.x - st*ringP.z, st*ringP.x + ct*ringP.z);
                }
                float ring = sdTorus(ringP, _PlanetRadius * 1.8, 0.12);

                // Moon (orbiting planet)
                float moonOrbit = time * 0.4;
                float3 moonPos = planetPos + float3(cos(moonOrbit)*4.0, sin(moonOrbit)*1.5, sin(moonOrbit)*4.0);
                float moon = sdSphere(p - moonPos, 0.6);

                // Asteroid belt
                float asteroids = 1e10;
                float beltR = _PlanetDistance * 0.75;
                for (int i = 0; i < 6; i++) {
                    float angle = float(i) * 1.0472 + time * (0.1 + float(i)*0.02);
                    float3 aPos = float3(cos(angle)*beltR, (i%2==0?0.5:-0.5), sin(angle)*beltR);
                    asteroids = min(asteroids, sdSphere(p - aPos, 0.3 + float(i)*0.1));
                }

                // Combine with smooth union
                float d = sun;
                d = opSmoothUnion(d, planet, _Smoothness);
                d = opSmoothUnion(d, ring, _Smoothness * 0.5);
                d = opSmoothUnion(d, moon, _Smoothness);
                d = opSmoothUnion(d, asteroids, _Smoothness * 0.3);
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
                float d = noise3d(p*3.0)*0.3 + noise3d(p*15.0)*0.1;
                float3 c = bc + d;
                // Emissive glow near sun
                float sunDist = length(p);
                c += _Color.rgb * saturate(1.0 - sunDist / (_SunRadius * 2.0)) * 2.0;
                float f = pow(1.0 - abs(dot(n, normalize(_WorldSpaceCameraPos-p))), 2.0);
                return saturate(c + f * float3(0.15, 0.1, 0.05));
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
                        float3 fc = col * (0.15 + diff * 0.85);
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
                // Space background with stars
                float stars = step(0.998, hash3d(rd * 500.0));
                float3 sky = _FogColor.rgb + stars * 0.8;
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
