// =============================================================================
// SDF Surface Raymarching Shader - TRUE Infinite Resolution
// =============================================================================
// This shader renders SDF as a SOLID SURFACE using raymarching.
// No particles, no gaps, no polygons - just pure mathematical perfection.
//
// Features:
//   - Raymarching: Finds exact SDF surface for each pixel
//   - Procedural Texturing: Colors from math, never pixelates
//   - Infinite Detail: More detail appears as you zoom in
//
// "Zoom x1,000,000 - Still perfectly sharp. That's not a bug, that's math."
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "SdfUniverse/InfiniteSurface"
{
    Properties
    {
        _Color ("Main Color", Color) = (0.2, 0.8, 1.0, 1.0)
        _Color2 ("Secondary Color", Color) = (0.8, 0.3, 0.5, 1.0)
        _DetailScale ("Detail Scale", Float) = 10.0
        _BoxSize ("Box Size", Float) = 50.0
        _HoleSize ("Hole Size", Float) = 2.0
        _RepeatScale ("Repeat Scale", Float) = 15.0
        _TwistAmount ("Twist Amount", Float) = 0.02
        _MaxSteps ("Max Raymarching Steps", Int) = 128
        _MaxDist ("Max Distance", Float) = 500.0
        _SurfaceEpsilon ("Surface Epsilon", Float) = 0.0001
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
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
            };

            // Properties
            float4 _Color;
            float4 _Color2;
            float _DetailScale;
            float _BoxSize;
            float _HoleSize;
            float _RepeatScale;
            float _TwistAmount;
            int _MaxSteps;
            float _MaxDist;
            float _SurfaceEpsilon;

            // =================================================================
            // SDF Primitives
            // =================================================================

            float sdBox(float3 p, float3 b)
            {
                float3 q = abs(p) - b;
                return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
            }

            float sdCross(float3 p, float size)
            {
                float inf = 1000.0;
                float da = sdBox(p, float3(inf, size, size));
                float db = sdBox(p, float3(size, inf, size));
                float dc = sdBox(p, float3(size, size, inf));
                return min(da, min(db, dc));
            }

            // Infinite repetition (space folding)
            float3 opRepeat(float3 p, float3 c)
            {
                return fmod(p + c * 0.5, c) - c * 0.5;
            }

            // Subtraction
            float opSubtraction(float d1, float d2)
            {
                return max(-d1, d2);
            }

            // =================================================================
            // Scene SDF: Subtract(Box, Repeat(Cross))
            // ONE formula creates INFINITE complexity
            // =================================================================

            float map(float3 p)
            {
                // Optional twist for organic feel
                if (_TwistAmount > 0.001)
                {
                    float angle = p.y * _TwistAmount;
                    float c = cos(angle);
                    float s = sin(angle);
                    float2 xz = float2(c * p.x - s * p.z, s * p.x + c * p.z);
                    p.x = xz.x;
                    p.z = xz.y;
                }

                // STEP 1: ONE BIG BOX
                float box = sdBox(p, float3(_BoxSize, _BoxSize, _BoxSize));

                // STEP 2: INFINITE CROSS via REPEAT
                float3 rp = opRepeat(p, float3(_RepeatScale, _RepeatScale, _RepeatScale));
                float cross = sdCross(rp, _HoleSize);

                // STEP 3: SUBTRACT = Infinite labyrinth from 1 formula
                return opSubtraction(cross, box);
            }

            // =================================================================
            // Normal Calculation (gradient of SDF)
            // =================================================================

            float3 calcNormal(float3 p)
            {
                const float eps = 0.0001;
                float3 n;
                n.x = map(p + float3(eps, 0, 0)) - map(p - float3(eps, 0, 0));
                n.y = map(p + float3(0, eps, 0)) - map(p - float3(0, eps, 0));
                n.z = map(p + float3(0, 0, eps)) - map(p - float3(0, 0, eps));
                return normalize(n);
            }

            // =================================================================
            // Procedural Texturing - INFINITE RESOLUTION COLOR
            // =================================================================
            // No textures! Colors are calculated from position.
            // Zoom x1,000,000? Still calculating new detail. Never blurs.
            // =================================================================

            float hash(float3 p)
            {
                p = frac(p * 0.3183099 + 0.1);
                p *= 17.0;
                return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
            }

            float noise(float3 p)
            {
                float3 i = floor(p);
                float3 f = frac(p);
                f = f * f * (3.0 - 2.0 * f);

                return lerp(
                    lerp(lerp(hash(i + float3(0,0,0)), hash(i + float3(1,0,0)), f.x),
                         lerp(hash(i + float3(0,1,0)), hash(i + float3(1,1,0)), f.x), f.y),
                    lerp(lerp(hash(i + float3(0,0,1)), hash(i + float3(1,0,1)), f.x),
                         lerp(hash(i + float3(0,1,1)), hash(i + float3(1,1,1)), f.x), f.y),
                    f.z
                );
            }

            // Fractal Brownian Motion - multi-octave noise
            float fbm(float3 p)
            {
                float value = 0.0;
                float amplitude = 0.5;
                float frequency = 1.0;

                for (int i = 0; i < 5; i++)
                {
                    value += amplitude * noise(p * frequency);
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
                return value;
            }

            float3 getColor(float3 p, float3 n)
            {
                // Base color from normal direction
                float3 baseColor = lerp(_Color.rgb, _Color2.rgb, n.y * 0.5 + 0.5);

                // Multi-scale procedural detail (THE KEY TO INFINITE RESOLUTION)
                // Each scale adds more detail - no matter how close you get
                float detail1 = fbm(p * _DetailScale);
                float detail2 = fbm(p * _DetailScale * 10.0) * 0.3;
                float detail3 = fbm(p * _DetailScale * 100.0) * 0.1;
                float detail4 = fbm(p * _DetailScale * 1000.0) * 0.03;

                // Combine all detail levels
                float detail = detail1 + detail2 + detail3 + detail4;

                // Apply detail to color
                float3 color = baseColor + detail * 0.3;

                // Add subtle iridescence based on view angle
                float fresnel = pow(1.0 - abs(dot(n, normalize(_WorldSpaceCameraPos - p))), 2.0);
                color += fresnel * float3(0.1, 0.2, 0.3);

                return saturate(color);
            }

            // =================================================================
            // Ambient Occlusion (soft shadows in crevices)
            // =================================================================

            float calcAO(float3 pos, float3 nor)
            {
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < 5; i++)
                {
                    float h = 0.01 + 0.12 * float(i) / 4.0;
                    float d = map(pos + h * nor);
                    occ += (h - d) * sca;
                    sca *= 0.95;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // =================================================================
            // Vertex Shader
            // =================================================================

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                return o;
            }

            // =================================================================
            // Fragment Shader - RAYMARCHING with DEPTH OUTPUT
            // =================================================================
            // SV_Depth enables proper occlusion with mesh objects (VRM avatars)
            // "背景は数式、役者はポリゴン" - SDF world + mesh characters
            // =================================================================

            struct FragOutput
            {
                fixed4 color : SV_Target;
                float depth : SV_Depth;  // Depth buffer integration for mesh occlusion
            };

            FragOutput frag(v2f i)
            {
                float3 rayOrigin = _WorldSpaceCameraPos;
                float3 rayDir = normalize(i.rayDir);

                float t = 0.0;
                float d = 0.0;

                FragOutput o;

                // === RAYMARCHING LOOP ===
                // March along ray until we hit the SDF surface
                for (int k = 0; k < _MaxSteps; k++)
                {
                    float3 p = rayOrigin + rayDir * t;
                    d = map(p);

                    // Surface found!
                    if (d < _SurfaceEpsilon)
                    {
                        // Calculate normal
                        float3 n = calcNormal(p);

                        // Calculate procedural color (INFINITE DETAIL)
                        float3 color = getColor(p, n);

                        // Simple lighting
                        float3 lightDir = normalize(float3(1.0, 1.0, -0.5));
                        float diff = max(dot(n, lightDir), 0.0);
                        float amb = 0.2;

                        // Ambient occlusion
                        float ao = calcAO(p, n);

                        // Final color
                        float3 finalColor = color * (amb + diff * 0.8) * ao;

                        // Fog based on distance (optional)
                        float fog = exp(-t * 0.005);
                        finalColor = lerp(float3(0.01, 0.01, 0.02), finalColor, fog);

                        // === DEPTH OUTPUT ===
                        // Convert world hit position to clip space depth
                        // This allows Unity to properly occlude mesh objects (VRM avatars)
                        float3 hitPos = rayOrigin + rayDir * t;
                        float4 clipPos = UnityWorldToClipPos(hitPos);

                        o.color = fixed4(finalColor, 1.0);
                        o.depth = clipPos.z / clipPos.w;

                        #if defined(UNITY_REVERSED_Z)
                            // DirectX, Metal, Vulkan: depth range [1, 0]
                            o.depth = clipPos.z / clipPos.w;
                        #else
                            // OpenGL: depth range [0, 1], need to remap from [-1, 1]
                            o.depth = (clipPos.z / clipPos.w) * 0.5 + 0.5;
                        #endif

                        return o;
                    }

                    t += d * 0.9; // Slight understep for stability

                    // Too far - no hit
                    if (t > _MaxDist) break;
                }

                // Background (sky gradient) - at far plane depth
                float3 sky = lerp(
                    float3(0.01, 0.01, 0.03),
                    float3(0.05, 0.1, 0.2),
                    rayDir.y * 0.5 + 0.5
                );

                o.color = fixed4(sky, 1.0);

                #if defined(UNITY_REVERSED_Z)
                    o.depth = 0.0; // Far plane (reversed Z)
                #else
                    o.depth = 1.0; // Far plane (normal Z)
                #endif

                return o;
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}
