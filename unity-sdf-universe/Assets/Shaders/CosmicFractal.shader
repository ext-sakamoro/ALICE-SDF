// =============================================================================
// Cosmic × Fractal Demo Shader
// 4 Demo Modes: Normal / Fusion / Destruction / Morphing
// =============================================================================
Shader "SdfUniverse/CosmicFractal"
{
    Properties
    {
        _Color ("Color 1", Color) = (0.9, 0.4, 0.1, 1.0)
        _Color2 ("Color 2", Color) = (0.1, 0.6, 0.9, 1.0)

        // Cosmic
        _SunRadius ("Sun Radius", Float) = 8.0
        _PlanetRadius ("Planet Radius", Float) = 6.0
        _PlanetDistance ("Planet Distance", Float) = 22.0
        _Smoothness ("Smooth Blend", Float) = 1.2

        // Fractal
        _HoleSize ("Hole Size", Float) = 0.8
        _RepeatScale ("Repeat Scale", Float) = 5.0
        _TwistAmount ("Twist", Float) = 0.01

        // Ring & Moon
        _RingMajor ("Ring Major", Float) = 10.0
        _RingMinor ("Ring Minor", Float) = 0.25

        // Demo Mode (0=Normal, 1=Fusion, 2=Destruction, 3=Morph)
        _DemoMode ("Demo Mode", Int) = 0

        // Fusion mode
        _Sphere1Pos ("Sphere 1 Pos", Vector) = (0, 0, 0, 0)
        _Sphere2Pos ("Sphere 2 Pos", Vector) = (10, 0, 0, 0)
        _Sphere1Radius ("Sphere 1 Radius", Float) = 4.0
        _Sphere2Radius ("Sphere 2 Radius", Float) = 3.0
        _DynamicK ("Dynamic Smoothness", Float) = 0.5

        // Destruction mode
        _HoleCount ("Hole Count", Int) = 0
        _HoleRadius ("Hole Radius", Float) = 1.5

        // Morph mode
        _MorphT ("Morph T", Range(0, 1)) = 0.0
        _MorphA ("Morph Shape A", Int) = 0
        _MorphB ("Morph Shape B", Int) = 1

        // Rendering
        _MaxDist ("Max Distance", Float) = 250.0
        _FogDensity ("Fog Density", Float) = 0.004
        _FogColor ("Fog Color", Color) = (0.02, 0.01, 0.03, 1.0)
        _SurfaceEpsilon ("Surface Epsilon", Float) = 0.001
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.0
            #include "UnityCG.cginc"

            float4 _Color; float4 _Color2;
            float _SunRadius; float _PlanetRadius; float _PlanetDistance; float _Smoothness;
            float _HoleSize; float _RepeatScale; float _TwistAmount;
            float _RingMajor; float _RingMinor;
            int _DemoMode;

            // Fusion
            float4 _Sphere1Pos; float4 _Sphere2Pos;
            float _Sphere1Radius; float _Sphere2Radius; float _DynamicK;

            // Destruction
            int _HoleCount; float _HoleRadius;
            float4 _HolePositions[16];

            // Morph
            float _MorphT; int _MorphA; int _MorphB;

            // Rendering
            float _MaxDist; float _FogDensity; float4 _FogColor; float _SurfaceEpsilon;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
            };

            // =====================================================================
            // SDF Primitives
            // =====================================================================

            float sdSphere(float3 p, float r) { return length(p) - r; }
            float sdBox(float3 p, float3 b) { float3 d = abs(p) - b; return length(max(d,0)) + min(max(d.x,max(d.y,d.z)),0); }
            float sdTorus(float3 p, float R, float r) { float2 q = float2(length(p.xz)-R, p.y); return length(q)-r; }
            float sdCylinder(float3 p, float r, float h) { float2 d = float2(length(p.xz)-r, abs(p.y)-h); return min(max(d.x,d.y),0) + length(max(d,0)); }

            float opSmoothUnion(float a, float b, float k) {
                float h = max(k - abs(a - b), 0.0) / k;
                return min(a, b) - h * h * k * 0.25;
            }

            float3 opRepeatInfinite(float3 p, float3 s) {
                return p - s * round(p / s);
            }

            float sdCross(float3 p, float size) {
                float inf = 1000.0;
                float da = sdBox(p, float3(inf, size, size));
                float db = sdBox(p, float3(size, inf, size));
                float dc = sdBox(p, float3(size, size, inf));
                return min(da, min(db, dc));
            }

            // =====================================================================
            // Mode 0: Normal (Cosmic × Fractal)
            // =====================================================================

            float mapNormal(float3 p)
            {
                float time = _Time.y;

                // Sun
                float sun = sdSphere(p, _SunRadius);

                // Fractal Planet
                float orbitAngle = time * 0.1;
                float3 planetPos = float3(cos(orbitAngle)*_PlanetDistance, 0, sin(orbitAngle)*_PlanetDistance);
                float3 pp = p - planetPos;

                // Slow self-rotation
                float rot = time * 0.15;
                float cr = cos(rot); float sr = sin(rot);
                pp = float3(cr*pp.x - sr*pp.z, pp.y, sr*pp.x + cr*pp.z);

                float planet = sdSphere(pp, _PlanetRadius);
                float3 rp = opRepeatInfinite(pp, float3(_RepeatScale, _RepeatScale, _RepeatScale));
                float cross = sdCross(rp, _HoleSize);
                float fractalPlanet = max(-cross, planet);

                // Ring
                float3 ringP = p - planetPos;
                float c15 = 0.9659; float s15 = 0.2588;
                ringP = float3(ringP.x, c15*ringP.y - s15*ringP.z, s15*ringP.y + c15*ringP.z);
                float ring = sdTorus(ringP, _RingMajor, _RingMinor);

                // Moon
                float moonOrbit = time * 0.3;
                float3 moonPos = planetPos + float3(cos(moonOrbit)*5.0, sin(moonOrbit)*1.5, sin(moonOrbit)*5.0);
                float moon = sdSphere(p - moonPos, 0.8);

                // Combine
                float d = sun;
                d = opSmoothUnion(d, fractalPlanet, _Smoothness);
                d = opSmoothUnion(d, ring, _Smoothness * 0.5);
                d = opSmoothUnion(d, moon, _Smoothness * 0.8);
                return d;
            }

            // =====================================================================
            // Mode 1: Dynamic Fusion (Metaball-like collision)
            // =====================================================================

            float mapFusion(float3 p)
            {
                float s1 = sdSphere(p - _Sphere1Pos.xyz, _Sphere1Radius);
                float s2 = sdSphere(p - _Sphere2Pos.xyz, _Sphere2Radius);

                // Dynamic smoothness based on distance between spheres
                float d = opSmoothUnion(s1, s2, _DynamicK);

                // Floor for reference
                float ground = p.y + 5.0;
                d = min(d, ground);

                return d;
            }

            // =====================================================================
            // Mode 2: Destruction (Dynamic hole punching)
            // =====================================================================

            float mapDestruction(float3 p)
            {
                // Base scene: big box + sphere
                float scene = sdBox(p, float3(15, 10, 15));

                // Fractal detail on the box
                float3 rp = opRepeatInfinite(p, float3(_RepeatScale, _RepeatScale, _RepeatScale));
                float cross = sdCross(rp, _HoleSize * 0.5);
                scene = max(-cross, scene);

                // Subtract holes from clicks
                for (int i = 0; i < 16; i++)
                {
                    if (i >= _HoleCount) break;
                    float hole = sdSphere(p - _HolePositions[i].xyz, _HoleRadius);
                    scene = max(-hole, scene);  // Subtraction
                }

                return scene;
            }

            // =====================================================================
            // Mode 3: Morphing (Shape interpolation)
            // =====================================================================

            float sdfShape(float3 p, int shapeId)
            {
                // 0: Sphere, 1: Box, 2: Torus, 3: Fractal
                if (shapeId == 0) return sdSphere(p, 5.0);
                if (shapeId == 1) return sdBox(p, float3(4.0, 4.0, 4.0));
                if (shapeId == 2) return sdTorus(p, 4.0, 1.2);

                // 3: Menger Sponge
                float box = sdBox(p, float3(5,5,5));
                float3 rp = opRepeatInfinite(p, float3(3.5, 3.5, 3.5));
                float cross = sdCross(rp, 0.6);
                return max(-cross, box);
            }

            float mapMorph(float3 p)
            {
                // Slow rotation
                float time = _Time.y;
                float a = time * 0.3;
                float ca = cos(a); float sa = sin(a);
                p = float3(ca*p.x - sa*p.z, p.y, sa*p.x + ca*p.z);

                float dA = sdfShape(p, _MorphA);
                float dB = sdfShape(p, _MorphB);

                // Smooth interpolation between shapes
                float d = lerp(dA, dB, _MorphT);

                // Floor
                float ground = p.y + 8.0;
                d = min(d, ground);

                return d;
            }

            // =====================================================================
            // Unified map()
            // =====================================================================

            float map(float3 p)
            {
                if (_DemoMode == 1) return mapFusion(p);
                if (_DemoMode == 2) return mapDestruction(p);
                if (_DemoMode == 3) return mapMorph(p);
                return mapNormal(p);
            }

            // =====================================================================
            // Rendering (shared)
            // =====================================================================

            float3 calcN(float3 p) {
                float e = max(0.0005, _SurfaceEpsilon * 0.5);
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
                float d1 = noise3d(p*2.0)*0.25;
                float d2 = noise3d(p*8.0)*0.1;
                float3 c = bc + d1 + d2;

                // Mode-specific coloring
                if (_DemoMode == 1) {
                    // Fusion: hot glow at blend region
                    float s1 = sdSphere(p - _Sphere1Pos.xyz, _Sphere1Radius);
                    float s2 = sdSphere(p - _Sphere2Pos.xyz, _Sphere2Radius);
                    float blend = saturate(1.0 - abs(s1 - s2) / (_DynamicK + 0.01));
                    c += float3(1.0, 0.3, 0.05) * blend * 2.0;
                }
                else if (_DemoMode == 2) {
                    // Destruction: red glow at hole edges
                    for (int i = 0; i < 16; i++) {
                        if (i >= _HoleCount) break;
                        float dist = length(p - _HolePositions[i].xyz);
                        float glow = saturate(1.0 - (dist - _HoleRadius) / 1.0);
                        c += float3(1.0, 0.2, 0.05) * glow * 1.5;
                    }
                }
                else if (_DemoMode == 3) {
                    // Morph: rainbow based on morph progress
                    float hue = _MorphT * 3.0 + _Time.y * 0.1;
                    float3 rainbow = saturate(float3(
                        abs(frac(hue) * 6.0 - 3.0) - 1.0,
                        abs(frac(hue + 0.333) * 6.0 - 3.0) - 1.0,
                        abs(frac(hue + 0.666) * 6.0 - 3.0) - 1.0
                    ));
                    c = lerp(c, rainbow, 0.4);
                }
                else {
                    // Normal: warm glow near sun
                    float sunDist = length(p);
                    c += _Color.rgb * saturate(1.0 - sunDist / (_SunRadius * 2.0)) * 1.5;
                }

                float f = pow(1.0-abs(dot(n,normalize(_WorldSpaceCameraPos-p))),2.0);
                return saturate(c + f*float3(0.1,0.12,0.15));
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

                float t = 0.0;
                int maxSteps = 128;
                float eps = _SurfaceEpsilon;
                FragOutput o;

                for (int k = 0; k < 128; k++) {
                    if (k >= maxSteps) break;
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < eps) {
                        float3 n = calcN(p);
                        float3 col = getColor(p, n);
                        float diff = max(dot(n, normalize(float3(1,1,-0.5))), 0.0);

                        // AO
                        float ao = 1.0;
                        float occ=0.0; float sc=1.0;
                        for(int j=0;j<4;j++){float h=0.01+0.12*float(j)/3.0;occ+=(h-map(p+h*n))*sc;sc*=0.95;}
                        ao=saturate(1.0-3.0*occ);

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
                    t += d;
                    if (t > _MaxDist) break;
                }

                // Sky
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
