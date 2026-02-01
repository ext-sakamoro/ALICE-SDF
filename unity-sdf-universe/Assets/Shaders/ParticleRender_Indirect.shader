// =============================================================================
// Ultimate Particle Shader
// =============================================================================
// GPU Instancing Indirect - Matrix computation on GPU
// Transfer: 24 bytes per particle (vs 64 bytes for Matrix4x4)
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "SdfUniverse/ParticleRender_Indirect"
{
    Properties
    {
        _Color ("Base Color", Color) = (0.3, 0.9, 1, 1)
        _Size ("Particle Size", Float) = 0.1
        _Brightness ("Brightness", Range(0, 5)) = 2.0
        _CoreGlow ("Core Glow", Range(0, 1)) = 0.3
    }

    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        LOD 100

        Blend SrcAlpha One  // Additive blending for glow
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma instancing_options procedural:setup

            #include "UnityCG.cginc"

            // =====================================================
            // Data Structures
            // =====================================================

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 color : TEXCOORD1;
                float speed : TEXCOORD2;
            };

            // Particle data from ComputeBuffer (matches C# struct)
            struct ParticleData
            {
                float3 position;
                float3 velocity;
            };

            // =====================================================
            // Uniforms & Buffers
            // =====================================================

            float4 _Color;
            float _Size;
            float _Brightness;
            float _CoreGlow;

            #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
                StructuredBuffer<ParticleData> _ParticlesBuffer;
            #endif

            // =====================================================
            // Instancing Setup (GPU Matrix Generation)
            // =====================================================

            void setup()
            {
                #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
                    ParticleData data = _ParticlesBuffer[unity_InstanceID];
                    float3 pos = data.position;

                    // Build transformation matrix on GPU
                    // This replaces CPU-side Matrix4x4.TRS computation
                    unity_ObjectToWorld._11_21_31_41 = float4(1, 0, 0, 0);
                    unity_ObjectToWorld._12_22_32_42 = float4(0, 1, 0, 0);
                    unity_ObjectToWorld._13_23_33_43 = float4(0, 0, 1, 0);
                    unity_ObjectToWorld._14_24_34_44 = float4(pos.x, pos.y, pos.z, 1);

                    // Inverse matrix (simplified for translation only)
                    unity_WorldToObject = unity_ObjectToWorld;
                    unity_WorldToObject._14_24_34 *= -1;
                #endif
            }

            // =====================================================
            // Vertex Shader (Billboard + Size)
            // =====================================================

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);

                float3 worldPos = float3(0, 0, 0);
                float3 velocity = float3(0, 0, 0);

                #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
                    ParticleData data = _ParticlesBuffer[unity_InstanceID];
                    worldPos = data.position;
                    velocity = data.velocity;
                #endif

                // Billboard: face camera
                float3 viewDir = normalize(_WorldSpaceCameraPos - worldPos);
                float3 right = normalize(cross(float3(0, 1, 0), viewDir));
                float3 up = cross(viewDir, right);

                // Handle edge case where viewDir is parallel to up
                if (length(right) < 0.001)
                {
                    right = float3(1, 0, 0);
                    up = float3(0, 0, 1);
                }

                // Apply size and build final position
                float3 offset = (right * v.vertex.x + up * v.vertex.y) * _Size;
                float3 finalPos = worldPos + offset;

                o.vertex = mul(UNITY_MATRIX_VP, float4(finalPos, 1));
                o.uv = v.uv;

                // Color based on velocity (faster = warmer color)
                float speed = length(velocity);
                o.speed = saturate(speed / 5.0);

                // Base color with velocity influence
                float3 slowColor = _Color.rgb;
                float3 fastColor = float3(1, 0.5, 0.2); // Orange for fast particles
                o.color = lerp(slowColor, fastColor, o.speed * 0.5) * _Brightness;

                return o;
            }

            // =====================================================
            // Fragment Shader (Glowing Circle)
            // =====================================================

            fixed4 frag(v2f i) : SV_Target
            {
                // Distance from center
                float2 centered = i.uv - 0.5;
                float dist = length(centered) * 2.0;

                // Soft circular falloff
                float alpha = saturate(1.0 - dist);
                alpha = pow(alpha, 1.5);

                // Core glow (bright center)
                float core = saturate(1.0 - dist * 3.0);
                core = pow(core, 3.0) * _CoreGlow;

                // Final color with core highlight
                float3 finalColor = i.color + core * 2.0;

                // Add slight pulse based on position (for visual interest)
                // float pulse = sin(_Time.y * 2.0 + i.speed * 10.0) * 0.1 + 1.0;
                // finalColor *= pulse;

                return fixed4(finalColor, alpha);
            }
            ENDCG
        }
    }

    FallBack "Particles/Standard Unlit"
}
