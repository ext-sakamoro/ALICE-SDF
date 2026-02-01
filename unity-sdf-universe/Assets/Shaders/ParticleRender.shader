// =============================================================================
// SDF Particle Render Shader
// =============================================================================
// GPU Instanced particle rendering with distance-based coloring
// Supports up to 1M particles at 60fps
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "SdfUniverse/ParticleRender"
{
    Properties
    {
        _Color ("Color", Color) = (0, 1, 1, 1)
        _EmissionColor ("Emission", Color) = (0.2, 0.5, 1, 1)
        _Size ("Size", Range(0.001, 0.1)) = 0.02
        _Brightness ("Brightness", Range(0, 5)) = 1.5
        _FresnelPower ("Fresnel Power", Range(0, 5)) = 2
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
            "IgnoreProjector" = "True"
        }

        LOD 100
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma target 4.5

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
                float3 viewDir : TEXCOORD2;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            float4 _Color;
            float4 _EmissionColor;
            float _Size;
            float _Brightness;
            float _FresnelPower;

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_TRANSFER_INSTANCE_ID(v, o);

                // Billboard: face camera
                float3 worldPos = mul(unity_ObjectToWorld, float4(0, 0, 0, 1)).xyz;
                float3 viewDir = normalize(_WorldSpaceCameraPos - worldPos);

                // Create billboard axes
                float3 right = normalize(cross(float3(0, 1, 0), viewDir));
                float3 up = cross(viewDir, right);

                // Scale and offset
                float3 billboardPos = worldPos
                    + right * v.vertex.x * _Size
                    + up * v.vertex.y * _Size;

                o.vertex = mul(UNITY_MATRIX_VP, float4(billboardPos, 1));
                o.uv = v.uv;
                o.worldPos = billboardPos;
                o.viewDir = viewDir;

                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);

                // Circular particle shape
                float2 centered = i.uv - 0.5;
                float dist = length(centered) * 2;

                // Soft circle with glow
                float alpha = saturate(1 - dist);
                alpha = pow(alpha, 0.5); // Soften edges

                // Core glow
                float core = saturate(1 - dist * 2);
                core = pow(core, 2);

                // Fresnel effect for depth
                float fresnel = pow(1 - saturate(dot(normalize(i.viewDir), float3(0, 0, 1))), _FresnelPower);

                // Final color
                float3 color = lerp(_Color.rgb, _EmissionColor.rgb, core);
                color *= _Brightness;
                color += fresnel * _EmissionColor.rgb * 0.5;

                // Clip fully transparent pixels
                clip(alpha - 0.01);

                return fixed4(color, alpha * _Color.a);
            }
            ENDCG
        }
    }

    FallBack "Sprites/Default"
}
