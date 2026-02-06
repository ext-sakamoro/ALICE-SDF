// =============================================================================
// ALICE-SDF Sample: Terrain Sculpting (Dig & Build)
// =============================================================================
// A flat ground that players can sculpt in real-time with VR hands.
// Left hand adds terrain (SmoothUnion), right hand digs (SmoothSubtraction).
// Both rendering and collision use the exact same SDF — dig a hole and you
// actually fall in. Build a hill and you can climb it.
//
// This is impossible with traditional VRChat meshes because MeshColliders
// cannot be updated at runtime. ALICE-SDF evaluates the same formula for
// both pixels and physics, so visual = collision at all times.
//
// Sculpt data is sent from UdonSharp via Material.SetVectorArray.
// Up to 48 simultaneous sculpt operations (circular buffer).
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Samples/TerrainSculpt"
{
    Properties
    {
        [Header(Terrain Colors)]
        _GrassColor ("Grass", Color) = (0.35, 0.55, 0.25, 1.0)
        _DirtColor ("Dirt", Color) = (0.55, 0.40, 0.25, 1.0)
        _RockColor ("Rock (Underground)", Color) = (0.40, 0.38, 0.35, 1.0)

        [Header(Cursor)]
        _AddCursorColor ("Add Cursor (Left Hand)", Color) = (0.3, 0.6, 1.0, 1.0)
        _SubCursorColor ("Dig Cursor (Right Hand)", Color) = (1.0, 0.3, 0.2, 1.0)

        [Header(Raymarching)]
        _MaxDist ("Max Distance", Float) = 100.0

        [Header(Sculpting)]
        _AddSmooth ("Add Smoothness", Float) = 0.25
        _SubSmooth ("Dig Smoothness", Float) = 0.15
        _SculptRadius ("Sculpt Radius (for cursor)", Float) = 0.3

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.70, 0.80, 0.90, 1.0)
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
            float4 _GrassColor, _DirtColor, _RockColor, _FogColor;
            float4 _AddCursorColor, _SubCursorColor;
            float _MaxDist;
            float _AddSmooth, _SubSmooth, _SculptRadius;

            // Dynamic sculpt data (set from UdonSharp)
            // xyz = world position, w = radius (positive = add, negative = dig)
            float4 _SculptData[48];
            float _SculptCount;

            // Hand cursor positions (xyz = pos, w = 1 if near terrain, 0 if not)
            float4 _LeftHand;
            float4 _RightHand;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
            };

            // =================================================================
            // SDF: Ground Plane + Dynamic Sculpt Operations
            // =================================================================
            float map(float3 p)
            {
                // Base terrain: flat ground at Y=0
                float terrain = p.y;

                // Apply sculpt operations in order
                for (int i = 0; i < 48; i++)
                {
                    if (i >= (int)_SculptCount) break;

                    float3 sp = _SculptData[i].xyz;
                    float rw = _SculptData[i].w;

                    if (rw > 0.001)
                    {
                        // Add terrain (left hand): SmoothUnion
                        float hill = sdSphere(p - sp, rw);
                        terrain = opSmoothUnion(terrain, hill, _AddSmooth);
                    }
                    else if (rw < -0.001)
                    {
                        // Dig terrain (right hand): SmoothSubtraction
                        float hole = sdSphere(p - sp, -rw);
                        terrain = opSmoothSubtraction(terrain, hole, _SubSmooth);
                    }
                }

                return terrain;
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

            // Simple noise for texture variation
            float hash2d(float2 p) {
                p = frac(p * float2(0.3183, 0.3671));
                p *= 17.0;
                return frac(p.x * p.y * (p.x + p.y));
            }

            // Height + normal based terrain coloring
            float3 terrainColor(float3 p, float3 n) {
                float h = p.y;
                float top = saturate(n.y); // 1 = flat top, 0 = vertical/underside

                // Texture variation
                float noise = hash2d(p.xz * 3.0) * 0.12;

                // Surface color: flat = grass, steep = dirt
                float3 grass = _GrassColor.rgb * (0.94 + noise);
                float3 dirt = _DirtColor.rgb * (0.94 + noise);
                float3 rock = _RockColor.rgb * (0.92 + noise);

                float3 surfaceCol = lerp(dirt, grass, top);

                // Underground blend: deeper = more rock
                float underground = smoothstep(0.0, -0.5, h);

                return lerp(surfaceCol, rock, underground);
            }

            // Cursor overlay: colored glow near each hand
            float3 cursorOverlay(float3 p) {
                float3 overlay = float3(0, 0, 0);

                if (_LeftHand.w > 0.5) {
                    float d = length(p - _LeftHand.xyz);
                    float glow = smoothstep(_SculptRadius * 1.5, _SculptRadius * 0.2, d);
                    overlay += _AddCursorColor.rgb * glow * 0.45;
                }
                if (_RightHand.w > 0.5) {
                    float d = length(p - _RightHand.xyz);
                    float glow = smoothstep(_SculptRadius * 1.5, _SculptRadius * 0.2, d);
                    overlay += _SubCursorColor.rgb * glow * 0.45;
                }

                return overlay;
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
                        float3 lightDir = normalize(float3(1, 1.5, -0.5));
                        float ao = calcAO(p, n);

                        // Terrain color
                        float3 baseCol = terrainColor(p, n);

                        // Lighting
                        float diff = max(dot(n, lightDir), 0.0);
                        float3 fc = baseCol * (0.25 + diff * 0.75) * ao;

                        // Hand cursor overlay
                        fc += cursorOverlay(p);

                        // Fog
                        fc = lerp(_FogColor.rgb, fc, exp(-t * 0.004));

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
