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
        _OnionOrbit ("Onion Orbit Radius", Float) = 16.0
        _Smoothness ("Smooth Blend", Range(0.1, 3.0)) = 0.8
        _MaxDist ("Max Distance", Float) = 150.0
        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1.0, 1.0, -0.5, 0.0)
        _FogDensity ("Fog Density", Float) = 0.004
        _FogColor ("Fog Color", Color) = (0.02, 0.01, 0.03, 1.0)
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

            float4 _LightDir;
            // Closest-approach acceptance, metres per metre of ray length
            #define NEAR_MISS_PER_M 0.002

            float4 _Color; float4 _Color2;
            float _PlanetRadius; float _HoleSize; float _RepeatScale;
            float _RingMajor; float _RingMinor;
            float _OnionRadius; float _OnionLayers; float _OnionThickness; float _OnionOrbit;
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
            float mapLaw(float3 p)
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
                float3 onionPos = float3(cos(orbitAngle) * _OnionOrbit, sin(orbitAngle) * _OnionOrbit * 0.125, sin(orbitAngle) * _OnionOrbit);
                float onion = sdOnion(p - onionPos, _OnionRadius, (int)_OnionLayers, _OnionThickness);

                // --- Combine with smooth union ---
                float d = fractalPlanet;
                d = opSmoothUnion(d, ring, _Smoothness);
                d = opSmoothUnion(d, onion, _Smoothness * 0.5);
                return d;
            }

            // The same scene with hard unions: exact outside every part, so the
            // ambient occlusion sees only real geometry (the smooth blend zone
            // under-reports distance and read as a dark ring at every junction)
            float mapOccluderLaw(float3 p)
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
                float3 onionPos = float3(cos(orbitAngle) * _OnionOrbit, sin(orbitAngle) * _OnionOrbit * 0.125, sin(orbitAngle) * _OnionOrbit);
                float onion = sdOnion(p - onionPos, _OnionRadius, (int)_OnionLayers, _OnionThickness);

                // --- Combine with smooth union ---
                float d = fractalPlanet;
                d = min(d, ring);
                d = min(d, onion);
                return d;
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
            float mapOccluder(float3 p) { return mapOccluderLaw(toLaw(p)); }

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
                float tExit = exitDistance(ro, rd);
                FragOutput o;
                // Closest approach along the ray: a ray grazing a silhouette that
                // runs out of steps within a pixel of the surface is a hit, not
                // the far depth (a dark seam otherwise)
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
                    {
                        float3 n = calcN(p, normalEps(tier));
                        float3 col = getColor(toLaw(p), n);
                        float diff = max(dot(n, normalize(_LightDir.xyz)), 0.0);
                        float ao = 1.0;
                        { // Simple AO
                            float occ=0.0; float sc=1.0;
                            for(int j=0;j<4;j++){float h=0.01+0.12*float(j)/3.0;occ+=(h-mapOccluder(p+h*n))*sc;sc*=0.95;}
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
