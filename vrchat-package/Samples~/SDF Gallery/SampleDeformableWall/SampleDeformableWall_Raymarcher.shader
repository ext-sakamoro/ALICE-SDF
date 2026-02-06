// =============================================================================
// ALICE-SDF Sample: DeformableWall
// =============================================================================
// Interactive wall that dents when touched/hit. Dents recover over time.
// Impact data is sent from UdonSharp via Material.SetVectorArray.
// Up to 16 simultaneous dents supported.
//
// Setup:
//   1. Place a Cube scaled to cover the wall + ground area
//   2. Assign this shader's material to the Cube
//   3. Attach SampleDeformableWall_Collider.cs (UdonSharp)
//   4. Touch/hit the wall in VR - it deforms!
//
// SDF formula: Union(ground, SmoothSubtract(wall, dent_spheres...))
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Samples/DeformableWall"
{
    Properties
    {
        [Header(Colors)]
        _WallColor ("Wall Color", Color) = (0.82, 0.78, 0.72, 1.0)
        _GroundColor ("Ground Color", Color) = (0.35, 0.42, 0.3, 1.0)
        _DentColor ("Dent Glow Color", Color) = (1.0, 0.6, 0.3, 1.0)

        [Header(Raymarching)]
        _MaxDist ("Max Distance", Float) = 100.0

        [Header(Wall Dimensions)]
        _WallWidth ("Wall Half-Width", Float) = 5.0
        _WallHeight ("Wall Half-Height", Float) = 2.5
        _WallThick ("Wall Half-Thickness", Float) = 0.2

        [Header(Deformation)]
        _DentRadius ("Dent Radius", Float) = 0.35
        _DecaySpeed ("Recovery Speed", Float) = 0.5
        _DentSmooth ("Dent Smoothness", Float) = 0.08

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.65, 0.7, 0.78, 1.0)
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
            float4 _WallColor, _GroundColor, _DentColor, _FogColor;
            float _MaxDist;
            float _WallWidth, _WallHeight, _WallThick;
            float _DentRadius, _DecaySpeed, _DentSmooth;

            // Dynamic impact data (set from UdonSharp)
            // xyz = world position of impact, w = Time.time at impact
            float4 _ImpactPoints[16];
            float _ImpactCount;

            struct appdata { float4 vertex : POSITION; };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
            };

            // =================================================================
            // SDF: Ground + Wall with dynamic dents
            // =================================================================
            float map(float3 p)
            {
                // Ground plane at Y=0
                float ground = p.y;

                // Wall standing on the ground
                float3 wc = float3(0, _WallHeight, 0);
                float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                float wall = sdBox(p - wc, wh);

                // Carve dents via SmoothSubtraction of decaying spheres
                for (int i = 0; i < 16; i++)
                {
                    if (i >= (int)_ImpactCount) break;

                    float age = _Time.y - _ImpactPoints[i].w;
                    float decay = exp(-age * _DecaySpeed);
                    float r = _DentRadius * decay;

                    if (r < 0.005) continue;

                    float dent = sdSphere(p - _ImpactPoints[i].xyz, r);
                    wall = opSmoothSubtraction(wall, dent, _DentSmooth);
                }

                return min(ground, wall);
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

            // Ambient Occlusion
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

            // Dent freshness at hit point (for coloring, called once per pixel)
            float dentFreshness(float3 p) {
                float f = 0.0;
                for (int i = 0; i < 16; i++) {
                    if (i >= (int)_ImpactCount) break;
                    float age = _Time.y - _ImpactPoints[i].w;
                    float decay = exp(-age * _DecaySpeed);
                    float dist = length(p - _ImpactPoints[i].xyz);
                    f = max(f, decay * smoothstep(_DentRadius * 1.5, 0.0, dist));
                }
                return f;
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

                        // Determine surface: ground vs wall
                        float groundD = p.y;
                        float3 wc = float3(0, _WallHeight, 0);
                        float3 wh = float3(_WallWidth, _WallHeight, _WallThick);
                        float wallD = sdBox(p - wc, wh);

                        float3 baseColor;
                        if (groundD < wallD + 0.01)
                        {
                            // Ground
                            baseColor = _GroundColor.rgb;
                        }
                        else
                        {
                            // Wall - blend toward DentColor based on freshness
                            baseColor = _WallColor.rgb;
                            float fresh = dentFreshness(p);
                            baseColor = lerp(baseColor, _DentColor.rgb, saturate(fresh));
                        }

                        // Lighting
                        float3 lightDir = normalize(float3(1, 1, -0.5));
                        float diff = max(dot(n, lightDir), 0.0);
                        float ao = calcAO(p, n);
                        float3 fc = baseColor * (0.2 + diff * 0.8) * ao;

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
