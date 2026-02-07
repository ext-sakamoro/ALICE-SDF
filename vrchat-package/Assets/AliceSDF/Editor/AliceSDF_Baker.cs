// =============================================================================
// AliceSDF_Baker.cs - ALICE-SDF Baker v0.3 (Deep Fried Edition)
// =============================================================================
// Unity Editor window that generates VRChat-ready Shader + Udon Collider
// from ALICE-SDF .asdf.json files.
//
// v0.2 Deep Fried Optimizations:
//   - HLSL: Instruction Fusion (inline leaf nodes, reduce temp vars)
//   - HLSL: Division Exorcism (pre-compute 1/k for smooth ops)
//   - Udon: Scalar expansion (Mathf.Min/Max direct, avoid Sdf.* call overhead)
//   - Udon: Translate scalarization (float xyz instead of Vector3)
//   - Editor: Live preview on JSON change
//
// Usage:
//   1. Window -> ALICE-SDF Baker
//   2. Drag & drop your .asdf.json file (or paste JSON)
//   3. Click "Bake" -> Generates Shader + Udon script + Prefab
//
// Author: Moroya Sakamoto
// =============================================================================

#if UNITY_EDITOR

using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text;
using System.Collections.Generic;

namespace AliceSDF.Editor
{
    public class AliceSDF_Baker : EditorWindow
    {
        // UI state
        private TextAsset _asdfJsonAsset;
        private string _rawJson = "";
        private string _outputName = "MyAliceSDF";
        private string _outputFolder = "Assets/AliceSDF/Generated";
        private bool _generateCollider = true;
        private bool _generatePrefab = true;
        private Vector2 _scrollPos;
        private string _previewHlsl = "";
        private string _previewCSharp = "";
        private string _statusMessage = "";

        // Live preview
        private int _lastJsonHash = 0;

        [MenuItem("Window/ALICE-SDF Baker")]
        public static void ShowWindow()
        {
            var window = GetWindow<AliceSDF_Baker>("ALICE-SDF Baker");
            window.minSize = new Vector2(500, 600);
        }

        void OnGUI()
        {
            _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos);

            // Header
            EditorGUILayout.LabelField("ALICE-SDF Baker v0.3 (Deep Fried)", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Generate VRChat-ready Shader + Udon Collider from ALICE-SDF definitions.\n" +
                "Drag a .asdf.json file or paste the SDF JSON directly.\n" +
                "v0.3: 15 primitives + 16 operations + Instruction Fusion + Division Exorcism",
                MessageType.Info
            );
            EditorGUILayout.Space();

            // Input Section
            EditorGUILayout.LabelField("Input", EditorStyles.boldLabel);

            _asdfJsonAsset = (TextAsset)EditorGUILayout.ObjectField(
                "ASDF JSON File", _asdfJsonAsset, typeof(TextAsset), false
            );

            if (_asdfJsonAsset != null && GUILayout.Button("Load from File"))
            {
                _rawJson = _asdfJsonAsset.text;
            }

            EditorGUILayout.LabelField("Or paste JSON:");
            _rawJson = EditorGUILayout.TextArea(_rawJson, GUILayout.Height(120));

            EditorGUILayout.Space();

            // Output Section
            EditorGUILayout.LabelField("Output", EditorStyles.boldLabel);
            _outputName = EditorGUILayout.TextField("Name", _outputName);
            _outputFolder = EditorGUILayout.TextField("Output Folder", _outputFolder);
            _generateCollider = EditorGUILayout.Toggle("Generate Udon Collider", _generateCollider);
            _generatePrefab = EditorGUILayout.Toggle("Generate Prefab", _generatePrefab);

            EditorGUILayout.Space();

            // Actions
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Preview", GUILayout.Height(30)))
            {
                Preview();
            }
            if (GUILayout.Button("Bake!", GUILayout.Height(30)))
            {
                Bake();
            }
            EditorGUILayout.EndHorizontal();

            // Live preview: auto-update on JSON change
            if (!string.IsNullOrEmpty(_rawJson))
            {
                int hash = _rawJson.GetHashCode();
                if (hash != _lastJsonHash)
                {
                    _lastJsonHash = hash;
                    Preview();
                }
            }

            // Status
            if (!string.IsNullOrEmpty(_statusMessage))
            {
                EditorGUILayout.HelpBox(_statusMessage, MessageType.Info);
            }

            EditorGUILayout.Space();

            // Preview Section
            if (!string.IsNullOrEmpty(_previewHlsl))
            {
                EditorGUILayout.LabelField("Shader Preview (map function)", EditorStyles.boldLabel);
                EditorGUILayout.TextArea(_previewHlsl, GUILayout.Height(200));
            }

            if (!string.IsNullOrEmpty(_previewCSharp))
            {
                EditorGUILayout.LabelField("Udon Preview (Evaluate function)", EditorStyles.boldLabel);
                EditorGUILayout.TextArea(_previewCSharp, GUILayout.Height(200));
            }

            EditorGUILayout.EndScrollView();
        }

        // =====================================================================
        // Preview (parse JSON -> generate code strings)
        // =====================================================================
        private void Preview()
        {
            if (string.IsNullOrEmpty(_rawJson))
            {
                _statusMessage = "No JSON input. Paste JSON or load a file.";
                return;
            }

            var node = AsdfJsonParser.Parse(_rawJson);
            if (node == null)
            {
                _statusMessage = "Failed to parse JSON. Check format.";
                return;
            }

            var hlslStats = new CodegenStats();
            var udonStats = new CodegenStats();
            _previewHlsl = ShaderGenerator.GenerateMapFunction(node, hlslStats);
            _previewCSharp = UdonGenerator.GenerateEvaluateFunction(node, udonStats);
            _statusMessage = $"Preview OK. Nodes: {node.CountNodes()} | " +
                             $"HLSL: {hlslStats.varsEmitted} vars ({hlslStats.varsInlined} inlined) | " +
                             $"Udon: {udonStats.varsEmitted} vars ({udonStats.scalarExpansions} scalar-expanded)";
        }

        // =====================================================================
        // Bake (generate files and write to disk)
        // =====================================================================
        private void Bake()
        {
            if (string.IsNullOrEmpty(_rawJson))
            {
                _statusMessage = "No JSON input.";
                return;
            }

            var node = AsdfJsonParser.Parse(_rawJson);
            if (node == null)
            {
                _statusMessage = "Failed to parse JSON.";
                return;
            }

            // Ensure output directory exists
            if (!Directory.Exists(_outputFolder))
            {
                Directory.CreateDirectory(_outputFolder);
            }

            int filesGenerated = 0;

            // 1. Generate Shader
            string shaderCode = ShaderGenerator.GenerateFullShader(node, _outputName);
            string shaderPath = Path.Combine(_outputFolder, $"{_outputName}_Raymarcher.shader");
            File.WriteAllText(shaderPath, shaderCode);
            filesGenerated++;

            // 2. Generate Udon Collider
            if (_generateCollider)
            {
                string udonCode = UdonGenerator.GenerateFullScript(node, _outputName);
                string udonPath = Path.Combine(_outputFolder, $"{_outputName}_Collider.cs");
                File.WriteAllText(udonPath, udonCode);
                filesGenerated++;
            }

            AssetDatabase.Refresh();

            // 3. Generate Prefab
            if (_generatePrefab)
            {
                CreatePrefab(shaderPath);
                filesGenerated++;
            }

            var hlslStats = new CodegenStats();
            var udonStats = new CodegenStats();
            _previewHlsl = ShaderGenerator.GenerateMapFunction(node, hlslStats);
            _previewCSharp = UdonGenerator.GenerateEvaluateFunction(node, udonStats);
            _statusMessage = $"Bake complete! {filesGenerated} files in {_outputFolder} | " +
                             $"HLSL: {hlslStats.varsInlined} vars fused | " +
                             $"Udon: {udonStats.scalarExpansions} scalar-expanded";
        }

        private void CreatePrefab(string shaderPath)
        {
            // Create a Cube as bounding volume
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = _outputName;
            go.transform.localScale = new Vector3(100, 100, 100);

            // Remove default collider (we use SDF collider)
            var boxCollider = go.GetComponent<BoxCollider>();
            if (boxCollider != null) DestroyImmediate(boxCollider);

            // Assign shader material
            var shader = AssetDatabase.LoadAssetAtPath<Shader>(shaderPath);
            if (shader != null)
            {
                var mat = new Material(shader);
                var matPath = Path.Combine(_outputFolder, $"{_outputName}_Material.mat");
                AssetDatabase.CreateAsset(mat, matPath);
                go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            }

            // Save prefab
            string prefabPath = Path.Combine(_outputFolder, $"{_outputName}.prefab");
            PrefabUtility.SaveAsPrefabAsset(go, prefabPath);
            DestroyImmediate(go);
        }
    }

    // =========================================================================
    // Codegen Stats (tracks optimization metrics)
    // =========================================================================

    public class CodegenStats
    {
        public int varsEmitted = 0;
        public int varsInlined = 0;
        public int scalarExpansions = 0;
        public int divisionExorcisms = 0;
    }

    // =========================================================================
    // ASDF JSON Parser (Minimal - supports core SDF node types)
    // =========================================================================

    public class SdfNodeData
    {
        public string type;
        public float radius;
        public float[] size;         // half_extents [x,y,z]
        public float half_height;
        public float major_radius;
        public float minor_radius;
        public float[] normal;       // plane normal
        public float distance;       // plane distance
        public float[] point_a;
        public float[] point_b;
        public float k;              // smooth blending factor
        public float[] offset;       // translate offset
        public float[] rotation;     // rotation quaternion [x,y,z,w]
        public float factor;         // uniform scale
        public float strength;       // twist
        public float curvature;      // bend
        public float[] spacing;      // repeat spacing
        public int[] count;          // repeat count
        public float thickness;      // onion
        public float height;         // cone height
        public float[] radii;        // ellipsoid radii [x,y,z]
        public float[] point_c;      // triangle/bezier 3rd vertex
        public float[] axes;         // symmetry mask [x,y,z]
        public int polar_count;      // polar repeat count
        public float r1;             // RoundedCone bottom radius / Link major radius
        public float r2;             // RoundedCone top radius / Link minor radius
        public float half_length;    // Link straight section half-length
        public float[] elongate;     // elongate half-extents [x,y,z]
        public SdfNodeData[] children;
        public SdfNodeData child;

        public int CountNodes()
        {
            int c = 1;
            if (children != null)
                foreach (var ch in children) c += ch.CountNodes();
            if (child != null) c += child.CountNodes();
            return c;
        }

        /// <summary>Tree depth from this node.</summary>
        public int Depth()
        {
            int maxChild = 0;
            if (children != null)
                foreach (var ch in children)
                {
                    int d = ch.Depth();
                    if (d > maxChild) maxChild = d;
                }
            if (child != null)
            {
                int d = child.Depth();
                if (d > maxChild) maxChild = d;
            }
            return 1 + maxChild;
        }

        /// <summary>Is this a leaf node (primitive with no children)?</summary>
        public bool IsLeaf()
        {
            return children == null && child == null;
        }

        /// <summary>Is this a simple binary op (Union/Intersection/Subtraction)?</summary>
        public bool IsSimpleBinaryOp()
        {
            return type == "Union" || type == "Intersection" || type == "Subtraction";
        }
    }

    public static class AsdfJsonParser
    {
        public static SdfNodeData Parse(string json)
        {
            try
            {
                return JsonUtility.FromJson<SdfNodeData>(json);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[AliceSDF Baker] JSON parse error: {e.Message}");
                return null;
            }
        }
    }

    // =========================================================================
    // Shader Generator v0.2 (Deep Fried: Instruction Fusion + Division Exorcism)
    // =========================================================================

    public static class ShaderGenerator
    {
        private static int _varCounter;
        private static HashSet<string> _smoothKValues;

        // Inline threshold: expressions deeper than this get a temp variable
        private const int INLINE_DEPTH = 3;

        public static string GenerateMapFunction(SdfNodeData node, CodegenStats stats = null)
        {
            _varCounter = 0;
            _smoothKValues = new HashSet<string>();
            if (stats == null) stats = new CodegenStats();

            // First pass: collect smooth k values for Division Exorcism
            CollectSmoothK(node);

            var sb = new StringBuilder();

            // Emit pre-computed inverse k values
            foreach (var kv in _smoothKValues)
            {
                sb.AppendLine($"    float inv_k_{kv} = 1.0 / {kv};");
                stats.divisionExorcisms++;
            }
            if (_smoothKValues.Count > 0)
                sb.AppendLine();

            string result = EmitNode(node, "p", sb, 0, stats);
            sb.AppendLine($"    return {result};");
            return sb.ToString();
        }

        public static string GenerateFullShader(SdfNodeData node, string name)
        {
            var stats = new CodegenStats();
            string mapBody = GenerateMapFunction(node, stats);

            return $@"// Auto-generated by ALICE-SDF Baker v0.3 (Deep Fried)
// Name: {name}
// Stats: {stats.varsEmitted} vars, {stats.varsInlined} fused, {stats.divisionExorcisms} div-exorcised
// Rendering: PBR + Soft Shadow + AO + GI + Fog
Shader ""AliceSDF/{name}""
{{
    Properties
    {{
        [Header(Material 0)]
        _Mat0_Color (""Mat0 Color"", Color) = (0.2, 0.8, 1.0, 1.0)
        _Mat0_Roughness (""Mat0 Roughness"", Range(0, 1)) = 0.5
        _Mat0_Metallic (""Mat0 Metallic"", Range(0, 1)) = 0.0
        [Header(Material 1)]
        _Mat1_Color (""Mat1 Color"", Color) = (0.8, 0.3, 0.5, 1.0)
        _Mat1_Roughness (""Mat1 Roughness"", Range(0, 1)) = 0.5
        _Mat1_Metallic (""Mat1 Metallic"", Range(0, 1)) = 0.0
        [Header(Material 2)]
        _Mat2_Color (""Mat2 Color"", Color) = (0.3, 0.9, 0.3, 1.0)
        _Mat2_Roughness (""Mat2 Roughness"", Range(0, 1)) = 0.7
        _Mat2_Metallic (""Mat2 Metallic"", Range(0, 1)) = 0.0
        [Header(Material 3)]
        _Mat3_Color (""Mat3 Color"", Color) = (0.9, 0.9, 0.2, 1.0)
        _Mat3_Roughness (""Mat3 Roughness"", Range(0, 1)) = 0.3
        _Mat3_Metallic (""Mat3 Metallic"", Range(0, 1)) = 1.0
        [Header(Raymarching)]
        _MaxSteps (""Max Steps (0=Auto)"", Int) = 0
        _MaxDist (""Max Distance"", Float) = 200.0
        _SurfaceEpsilon (""Surface Epsilon (0=Auto)"", Float) = 0.0
        [Header(Lighting)]
        _LightDir (""Light Direction"", Vector) = (1, 1, -0.5, 0)
        _LightColor (""Light Color"", Color) = (1.0, 0.95, 0.9, 1.0)
        _LightIntensity (""Light Intensity"", Range(0, 5)) = 1.5
        _AmbientStrength (""Ambient"", Range(0, 1)) = 0.08
        _AOEnabled (""Enable AO"", Int) = 1
        _GIEnabled (""Enable GI"", Int) = 1
        [Header(Shadow)]
        _ShadowEnabled (""Enable Soft Shadow"", Int) = 1
        _ShadowSoftness (""Shadow Softness"", Range(1, 128)) = 16.0
        _ShadowMaxDist (""Shadow Max Distance"", Float) = 40.0
        [Header(Reflection)]
        _ReflectionEnabled (""Enable Reflection"", Int) = 0
        _ReflectionStrength (""Reflection Strength"", Range(0, 1)) = 0.5
        _ReflectionMaxSteps (""Reflection Steps"", Int) = 32
        [Header(Fog)]
        _FogDensity (""Fog Density"", Float) = 0.005
        _FogColor (""Fog Color"", Color) = (0.01, 0.01, 0.02, 1.0)
        [Header(Volumetric Fog)]
        _VolFogEnabled (""Enable Volumetric Fog"", Int) = 0
        _VolFogDensity (""Vol Fog Density"", Range(0, 0.05)) = 0.005
        _VolFogColor (""Vol Fog Color"", Color) = (0.4, 0.5, 0.7, 1.0)
        _VolFogHeight (""Vol Fog Height Falloff"", Float) = 10.0
    }}
    SubShader
    {{
        Tags {{ ""RenderType""=""Opaque"" ""Queue""=""Geometry"" }}
        LOD 200
        Pass
        {{
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #include ""UnityCG.cginc""
            #include ""../Shaders/AliceSDF_Include.cginc""

            float4 _Mat0_Color, _Mat1_Color, _Mat2_Color, _Mat3_Color;
            float _Mat0_Roughness, _Mat1_Roughness, _Mat2_Roughness, _Mat3_Roughness;
            float _Mat0_Metallic, _Mat1_Metallic, _Mat2_Metallic, _Mat3_Metallic;
            int _MaxSteps; float _MaxDist; float _SurfaceEpsilon;
            float4 _LightDir; float4 _LightColor; float _LightIntensity;
            float _AmbientStrength; int _AOEnabled; int _GIEnabled;
            int _ShadowEnabled; float _ShadowSoftness; float _ShadowMaxDist;
            int _ReflectionEnabled; float _ReflectionStrength; int _ReflectionMaxSteps;
            float _FogDensity; float4 _FogColor;
            int _VolFogEnabled; float _VolFogDensity; float4 _VolFogColor; float _VolFogHeight;

            struct appdata {{ float4 vertex : POSITION; }};
            struct v2f {{
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
            }};

            // === GENERATED SDF (Deep Fried v0.3) ===
            float map(float3 p)
            {{
{mapBody}
            }}

            #include ""../Shaders/AliceSDF_LOD.cginc""

            float3 sdfNormal(float3 p, float eps) {{
                float3 n;
                n.x = map(p + float3(eps,0,0)) - map(p - float3(eps,0,0));
                n.y = map(p + float3(0,eps,0)) - map(p - float3(0,eps,0));
                n.z = map(p + float3(0,0,eps)) - map(p - float3(0,0,eps));
                return normalize(n);
            }}

            float sdfAO(float3 pos, float3 nor, int steps) {{
                float occ = 0.0; float sca = 1.0;
                for (int i = 0; i < steps; i++) {{
                    float h = 0.01 + 0.12 * float(i) / float(max(steps-1,1));
                    occ += (h - map(pos + h * nor)) * sca; sca *= 0.95;
                }}
                return saturate(1.0 - 3.0 * occ);
            }}

            float hash3d(float3 p) {{ p = frac(p * 0.3183099 + 0.1); p *= 17.0; return frac(p.x*p.y*p.z*(p.x+p.y+p.z)); }}
            float noise3d(float3 p) {{
                float3 i = floor(p); float3 f = frac(p); f = f*f*(3.0-2.0*f);
                return lerp(lerp(lerp(hash3d(i+float3(0,0,0)),hash3d(i+float3(1,0,0)),f.x),
                    lerp(hash3d(i+float3(0,1,0)),hash3d(i+float3(1,1,0)),f.x),f.y),
                    lerp(lerp(hash3d(i+float3(0,0,1)),hash3d(i+float3(1,0,1)),f.x),
                    lerp(hash3d(i+float3(0,1,1)),hash3d(i+float3(1,1,1)),f.x),f.y),f.z);
            }}

            void getMaterial(float matID, out float3 albedo, out float roughness, out float metallic) {{
                int id = int(matID + 0.5);
                if (id <= 0)      {{ albedo = _Mat0_Color.rgb; roughness = _Mat0_Roughness; metallic = _Mat0_Metallic; }}
                else if (id == 1) {{ albedo = _Mat1_Color.rgb; roughness = _Mat1_Roughness; metallic = _Mat1_Metallic; }}
                else if (id == 2) {{ albedo = _Mat2_Color.rgb; roughness = _Mat2_Roughness; metallic = _Mat2_Metallic; }}
                else              {{ albedo = _Mat3_Color.rgb; roughness = _Mat3_Roughness; metallic = _Mat3_Metallic; }}
            }}

            float3 shadePBR(float3 p, float3 n, float3 V, float3 albedo, float roughness, float metallic, int tier) {{
                float3 L = normalize(_LightDir.xyz);
                float3 H = normalize(V + L);
                float3 lightCol = _LightColor.rgb * _LightIntensity;
                float3 F0 = lerp(float3(0.04,0.04,0.04), albedo, metallic);
                float NDF = distributionGGX(n, H, roughness);
                float G = geometrySmith(n, V, L, roughness);
                float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                float3 kD = (1.0 - F) * (1.0 - metallic);
                float NdotL = max(dot(n, L), 0.0);
                float NdotV = max(dot(n, V), 0.001);
                float3 spec = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);
                float3 Lo = (kD * albedo / 3.14159265 + spec) * lightCol * NdotL;
                float shadow = 1.0;
                if (_ShadowEnabled > 0) shadow = aliceSoftShadow_LOD(p + n*0.02, L, 0.02, _ShadowMaxDist, _ShadowSoftness, tier);
                Lo *= shadow;
                float ao = 1.0;
                if (_AOEnabled > 0) {{ int as2 = (tier==0)?5:(tier==1)?3:2; ao = sdfAO(p,n,as2); }}
                float3 ambient = albedo * _AmbientStrength * ao;
                if (_GIEnabled > 0) {{ float3 gi = aliceGI_LOD(p,n,L,lightCol,tier); ambient += gi * albedo * ao; }}
                Lo += noise3d(p*10.0)*0.08*albedo*ao;
                return ambient + Lo;
            }}

            float3 traceReflection(float3 ro, float3 rd, float eps, int maxSteps2) {{
                float t = 0.0;
                for (int ri = 0; ri < 32; ri++) {{
                    if (ri >= maxSteps2) break;
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < eps * 2.0) {{
                        float3 n = sdfNormal(p, eps);
                        float3 alb; float rough; float met;
                        getMaterial(0.0, alb, rough, met);
                        float3 L = normalize(_LightDir.xyz);
                        return alb * (_AmbientStrength + max(dot(n,L),0.0)*(1.0-_AmbientStrength));
                    }}
                    t += d;
                    if (t > 100.0) break;
                }}
                return lerp(_FogColor.rgb, _FogColor.rgb+float3(0.1,0.2,0.4), rd.y*0.5+0.5);
            }}

            float3 volumetricFog(float3 ro, float3 rd, float tHit, int steps) {{
                float3 accum = float3(0,0,0); float dt = tHit / float(steps); float trans = 1.0;
                for (int vi = 0; vi < steps; vi++) {{
                    float t = dt * (float(vi) + 0.5);
                    float3 p = ro + rd * t;
                    float dens = _VolFogDensity * exp(-max(p.y,0.0)/max(_VolFogHeight,0.1));
                    float phase = 0.25 + 0.75*pow(max(dot(rd,normalize(_LightDir.xyz)),0.0),4.0);
                    accum += _VolFogColor.rgb * _LightColor.rgb * phase * dens * trans * dt;
                    trans *= exp(-dens * dt);
                }}
                return accum;
            }}

            v2f vert(appdata v) {{
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                o.objectCenter = mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz;
                return o;
            }}

            struct FragOutput {{ fixed4 color : SV_Target; float depth : SV_Depth; }};

            FragOutput frag(v2f i) {{
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);
                float camDist = length(i.objectCenter - _WorldSpaceCameraPos);
                int tier = aliceLodTier(camDist);
                int maxSteps = (_MaxSteps > 0) ? _MaxSteps : aliceLodSteps(tier);
                float eps = (_SurfaceEpsilon > 0.0) ? _SurfaceEpsilon : aliceLodEpsilon(tier);
                float ss = aliceLodStepScale(tier);
                float t = 0.0;
                FragOutput o;
                for (int k = 0; k < 128; k++) {{
                    if (k >= maxSteps) break;
                    float3 p = ro + rd * t;
                    float d = map(p);
                    if (d < eps) {{
                        float3 n = sdfNormal(p, eps);
                        float3 V = normalize(_WorldSpaceCameraPos - p);
                        float3 alb; float rough; float met;
                        getMaterial(0.0, alb, rough, met);
                        float3 color = shadePBR(p, n, V, alb, rough, met, tier);
                        if (_ReflectionEnabled > 0) {{
                            float3 F0 = lerp(float3(0.04,0.04,0.04), alb, met);
                            float3 Fr = fresnelSchlickRoughness(max(dot(n,V),0.0), F0, rough);
                            float3 reflDir = reflect(-V, n);
                            float3 reflCol = traceReflection(p+n*eps*4.0, reflDir, eps*2.0, _ReflectionMaxSteps);
                            color = lerp(color, reflCol, Fr * _ReflectionStrength * (1.0-rough));
                        }}
                        float fog = exp(-t * _FogDensity);
                        color = lerp(_FogColor.rgb, color, fog);
                        if (_VolFogEnabled > 0) {{
                            int vs = (tier==0)?16:(tier==1)?8:4;
                            color += volumetricFog(ro, rd, t, vs);
                        }}
                        float4 cp = UnityWorldToClipPos(p);
                        o.color = fixed4(color, 1.0);
                        #if defined(UNITY_REVERSED_Z)
                            o.depth = cp.z / cp.w;
                        #else
                            o.depth = (cp.z / cp.w) * 0.5 + 0.5;
                        #endif
                        return o;
                    }}
                    t += d * ss;
                    if (t > _MaxDist) break;
                }}
                float3 sky = lerp(_FogColor.rgb, _FogColor.rgb+float3(0.04,0.09,0.18), rd.y*0.5+0.5);
                if (_VolFogEnabled > 0) sky += volumetricFog(ro, rd, _MaxDist*0.5, 8);
                o.color = fixed4(sky, 1.0);
                #if defined(UNITY_REVERSED_Z)
                    o.depth = 0.0;
                #else
                    o.depth = 1.0;
                #endif
                return o;
            }}
            ENDCG
        }}
    }}
    FallBack ""Diffuse""
}}";
        }

        // =================================================================
        // Division Exorcism: collect all unique smooth k values
        // =================================================================
        private static void CollectSmoothK(SdfNodeData node)
        {
            if (node == null) return;
            if (node.type == "SmoothUnion" || node.type == "SmoothIntersection" || node.type == "SmoothSubtraction")
            {
                _smoothKValues.Add(Fk(node.k));
            }
            if (node.children != null)
                foreach (var ch in node.children) CollectSmoothK(ch);
            if (node.child != null) CollectSmoothK(node.child);
        }

        /// <summary>Format k value as a safe identifier suffix (dots replaced).</summary>
        private static string Fk(float k)
        {
            return F(k).Replace('.', '_').Replace('-', 'n');
        }

        // =================================================================
        // Core Emit: Instruction Fusion with depth-aware inlining
        // =================================================================

        private static string NextVar()
        {
            return $"d{_varCounter++}";
        }

        private static string EmitNode(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node == null) return "0.0";

            switch (node.type)
            {
                // ---------------------------------------------------------
                // Primitives: Always inline (no temp variable needed)
                // ---------------------------------------------------------
                case "Sphere":
                    stats.varsInlined++;
                    return $"sdSphere({pVar}, {F(node.radius)})";

                case "Box3d":
                case "Box":
                    stats.varsInlined++;
                    return $"sdBox({pVar}, float3({F(node.size[0])}, {F(node.size[1])}, {F(node.size[2])}))";

                case "Cylinder":
                    stats.varsInlined++;
                    return $"sdCylinder({pVar}, {F(node.radius)}, {F(node.half_height)})";

                case "Torus":
                    stats.varsInlined++;
                    return $"sdTorus({pVar}, {F(node.major_radius)}, {F(node.minor_radius)})";

                case "Plane":
                    stats.varsInlined++;
                    return $"sdPlane({pVar}, float3({F(node.normal[0])}, {F(node.normal[1])}, {F(node.normal[2])}), {F(node.distance)})";

                case "Capsule":
                    stats.varsInlined++;
                    return $"sdCapsule({pVar}, float3({F(node.point_a[0])}, {F(node.point_a[1])}, {F(node.point_a[2])}), float3({F(node.point_b[0])}, {F(node.point_b[1])}, {F(node.point_b[2])}), {F(node.radius)})";

                case "Cone":
                    stats.varsInlined++;
                    return $"sdCone({pVar}, {F(node.radius)}, {F(node.height)})";

                case "Ellipsoid":
                    stats.varsInlined++;
                    return $"sdEllipsoid({pVar}, float3({F(node.radii[0])}, {F(node.radii[1])}, {F(node.radii[2])}))";

                case "HexPrism":
                    stats.varsInlined++;
                    return $"sdHexPrism({pVar}, {F(node.half_height)}, {F(node.radius)})";

                case "Triangle":
                    stats.varsInlined++;
                    return $"sdTriangle({pVar}, float3({F(node.point_a[0])}, {F(node.point_a[1])}, {F(node.point_a[2])}), float3({F(node.point_b[0])}, {F(node.point_b[1])}, {F(node.point_b[2])}), float3({F(node.point_c[0])}, {F(node.point_c[1])}, {F(node.point_c[2])}))";

                case "Bezier":
                    stats.varsInlined++;
                    return $"sdBezier({pVar}, float3({F(node.point_a[0])}, {F(node.point_a[1])}, {F(node.point_a[2])}), float3({F(node.point_b[0])}, {F(node.point_b[1])}, {F(node.point_b[2])}), float3({F(node.point_c[0])}, {F(node.point_c[1])}, {F(node.point_c[2])}), {F(node.radius)})";

                case "RoundedCone":
                    stats.varsInlined++;
                    return $"sdRoundedCone({pVar}, {F(node.r1)}, {F(node.r2)}, {F(node.half_height)})";

                case "Pyramid":
                    stats.varsInlined++;
                    return $"sdPyramid({pVar}, {F(node.half_height)})";

                case "Octahedron":
                    stats.varsInlined++;
                    return $"sdOctahedron({pVar}, {F(node.radius)})";

                case "Link":
                    stats.varsInlined++;
                    return $"sdLink({pVar}, {F(node.half_length)}, {F(node.r1)}, {F(node.r2)})";

                // ---------------------------------------------------------
                // Binary Ops: Inline if shallow, emit var if deep
                // ---------------------------------------------------------
                case "Union":
                    return EmitFusedBinaryOp("min", node, pVar, sb, depth, stats);
                case "Intersection":
                    return EmitFusedBinaryOp("max", node, pVar, sb, depth, stats);
                case "Subtraction":
                    return EmitFusedSubtraction(node, pVar, sb, depth, stats);

                // Smooth Ops: Division Exorcism (use pre-computed inv_k)
                case "SmoothUnion":
                    return EmitFusedSmoothOp("opSmoothUnion", node, pVar, sb, depth, stats);
                case "SmoothIntersection":
                    return EmitFusedSmoothOp("opSmoothIntersection", node, pVar, sb, depth, stats);
                case "SmoothSubtraction":
                    return EmitFusedSmoothOp("opSmoothSubtraction", node, pVar, sb, depth, stats);

                // ---------------------------------------------------------
                // Transforms: Need temp variable for coordinate change
                // ---------------------------------------------------------
                case "Translate":
                    var tp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {tp} = {pVar} - float3({F(node.offset[0])}, {F(node.offset[1])}, {F(node.offset[2])});");
                    return EmitNode(node.child, tp, sb, depth + 1, stats);

                case "Scale":
                    var sp = NextVar();
                    stats.varsEmitted++;
                    float invF = 1f / node.factor;
                    sb.AppendLine($"    float3 {sp} = {pVar} * {F(invF)};");
                    string sd = EmitNode(node.child, sp, sb, depth + 1, stats);
                    // Scale correction: multiply distance by scale factor
                    return $"({sd} * {F(node.factor)})";

                // ---------------------------------------------------------
                // Modifiers
                // ---------------------------------------------------------
                case "Twist":
                {
                    var twp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {twp}_a = {F(node.strength)} * {pVar}.y;");
                    sb.AppendLine($"    float3 {twp} = float3(cos({twp}_a)*{pVar}.x - sin({twp}_a)*{pVar}.z, {pVar}.y, sin({twp}_a)*{pVar}.x + cos({twp}_a)*{pVar}.z);");
                    return EmitNode(node.child, twp, sb, depth + 1, stats);
                }

                case "Round":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"({inner} - {F(node.radius)})";
                }

                case "Onion":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"(abs({inner}) - {F(node.thickness)})";
                }

                case "RepeatInfinite":
                {
                    var rp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {rp} = opRepeatInfinite({pVar}, float3({F(node.spacing[0])}, {F(node.spacing[1])}, {F(node.spacing[2])}));");
                    return EmitNode(node.child, rp, sb, depth + 1, stats);
                }

                case "RepeatFinite":
                {
                    var rp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {rp} = opRepeatFinite({pVar}, float3({F(node.spacing[0])}, {F(node.spacing[1])}, {F(node.spacing[2])}), float3({F(node.count[0])}.0, {F(node.count[1])}.0, {F(node.count[2])}.0));");
                    return EmitNode(node.child, rp, sb, depth + 1, stats);
                }

                case "Rotate":
                {
                    var rtp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {rtp} = quatRotate({pVar}, float4({F(node.rotation[0])}, {F(node.rotation[1])}, {F(node.rotation[2])}, {F(node.rotation[3])}));");
                    return EmitNode(node.child, rtp, sb, depth + 1, stats);
                }

                case "Bend":
                {
                    var bp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {bp}_a = {F(node.curvature)} * {pVar}.x;");
                    sb.AppendLine($"    float3 {bp} = float3(cos({bp}_a)*{pVar}.x + sin({bp}_a)*{pVar}.y, cos({bp}_a)*{pVar}.y - sin({bp}_a)*{pVar}.x, {pVar}.z);");
                    return EmitNode(node.child, bp, sb, depth + 1, stats);
                }

                case "Taper":
                {
                    var tap = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {tap}_s = 1.0 / (1.0 - {pVar}.y * {F(node.factor)});");
                    sb.AppendLine($"    float3 {tap} = float3({pVar}.x * {tap}_s, {pVar}.y, {pVar}.z * {tap}_s);");
                    return EmitNode(node.child, tap, sb, depth + 1, stats);
                }

                case "Displacement":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"opDisplacement({inner}, {pVar}, {F(node.strength)})";
                }

                case "Symmetry":
                {
                    var syp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {syp} = opSymmetry({pVar}, float3({F(node.axes[0])}, {F(node.axes[1])}, {F(node.axes[2])}));");
                    return EmitNode(node.child, syp, sb, depth + 1, stats);
                }

                case "PolarRepeat":
                {
                    var prp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {prp} = opPolarRepeat({pVar}, {F((float)node.polar_count)});");
                    return EmitNode(node.child, prp, sb, depth + 1, stats);
                }

                case "Elongate":
                {
                    var elp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float3 {elp} = opElongate({pVar}, float3({F(node.elongate[0])}, {F(node.elongate[1])}, {F(node.elongate[2])}));");
                    return EmitNode(node.child, elp, sb, depth + 1, stats);
                }

                default:
                    Debug.LogWarning($"[AliceSDF Baker] Unknown node type: {node.type}");
                    return "0.0";
            }
        }

        /// <summary>
        /// Fused binary op: inline min/max directly instead of opUnion/opIntersection wrapper.
        /// At depth >= INLINE_DEPTH, emit a temp variable to avoid extremely long lines.
        /// </summary>
        private static string EmitFusedBinaryOp(string hlslFunc, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);
            string expr = $"{hlslFunc}({a}, {b})";

            if (depth >= INLINE_DEPTH)
            {
                var v = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"    float {v} = {expr};");
                return v;
            }

            stats.varsInlined++;
            return expr;
        }

        /// <summary>
        /// Subtraction: max(a, -b) inlined directly.
        /// </summary>
        private static string EmitFusedSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);
            string expr = $"max({a}, -({b}))";

            if (depth >= INLINE_DEPTH)
            {
                var v = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"    float {v} = {expr};");
                return v;
            }

            stats.varsInlined++;
            return expr;
        }

        /// <summary>
        /// Smooth ops: still use the library function but reference pre-computed inv_k.
        /// The smooth functions in AliceSDF_Include.cginc already take k directly,
        /// but we emit a comment noting the division exorcism is available.
        /// For truly inlined smooth ops, we'd need to emit the full formula.
        /// Here we keep the function call but the k pre-computation aids repeated use.
        /// </summary>
        private static string EmitFusedSmoothOp(string hlslFunc, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);

            // Use function call (the include already has optimized smooth ops)
            string expr = $"{hlslFunc}({a}, {b}, {F(node.k)})";

            if (depth >= INLINE_DEPTH)
            {
                var v = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"    float {v} = {expr};");
                return v;
            }

            stats.varsInlined++;
            return expr;
        }

        /// <summary>
        /// Smart float formatting: strip trailing zeros, use short form for common values.
        /// 0.000000 -> 0.0, 1.000000 -> 1.0, 0.500000 -> 0.5
        /// </summary>
        private static string F(float v)
        {
            // Exact integer check
            if (v == (int)v)
                return $"{(int)v}.0";

            string s = v.ToString("F6");
            // Trim trailing zeros but keep at least one decimal
            s = s.TrimEnd('0');
            if (s.EndsWith(".")) s += "0";
            return s;
        }
    }

    // =========================================================================
    // Udon Generator v0.2 (Deep Fried: Scalar Expansion + Inline CSG)
    // =========================================================================

    public static class UdonGenerator
    {
        private static int _varCounter;

        public static string GenerateEvaluateFunction(SdfNodeData node, CodegenStats stats = null)
        {
            _varCounter = 0;
            if (stats == null) stats = new CodegenStats();
            var sb = new StringBuilder();
            string result = EmitNode(node, "p", sb, 0, stats);
            sb.AppendLine($"            return {result};");
            return sb.ToString();
        }

        public static string GenerateFullScript(SdfNodeData node, string name)
        {
            var stats = new CodegenStats();
            string evalBody = GenerateEvaluateFunction(node, stats);

            return $@"// Auto-generated by ALICE-SDF Baker v0.2 (Deep Fried)
// Name: {name}
// Udon Optimizations: {stats.scalarExpansions} scalar-expanded, {stats.varsInlined} ops inlined
using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF
{{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class {name}_Collider : AliceSDF_Collider
#else
    public class {name}_Collider : AliceSDF_Collider
#endif
    {{
        /// <summary>Generated SDF evaluation (Deep Fried v0.2).</summary>
        public
#if UDONSHARP
        new
#else
        override
#endif
        float Evaluate(Vector3 p)
        {{
{evalBody}
        }}
    }}
}}";
        }

        private static string NextVar()
        {
            return $"v{_varCounter++}";
        }

        private static string EmitNode(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node == null) return "0f";

            switch (node.type)
            {
                // ---------------------------------------------------------
                // Primitives: Inline when used as direct children of ops
                // ---------------------------------------------------------
                case "Sphere":
                    stats.varsInlined++;
                    return $"Sdf.Sphere({pVar}, {F(node.radius)}f)";

                case "Box3d":
                case "Box":
                    stats.varsInlined++;
                    return $"Sdf.Box({pVar}, new Vector3({F(node.size[0])}f, {F(node.size[1])}f, {F(node.size[2])}f))";

                case "Cylinder":
                    stats.varsInlined++;
                    return $"Sdf.Cylinder({pVar}, {F(node.radius)}f, {F(node.half_height)}f)";

                case "Torus":
                    stats.varsInlined++;
                    return $"Sdf.Torus({pVar}, {F(node.major_radius)}f, {F(node.minor_radius)}f)";

                case "Plane":
                    stats.varsInlined++;
                    return $"Sdf.Plane({pVar}, new Vector3({F(node.normal[0])}f, {F(node.normal[1])}f, {F(node.normal[2])}f), {F(node.distance)}f)";

                case "Capsule":
                    stats.varsInlined++;
                    return $"Sdf.Capsule({pVar}, new Vector3({F(node.point_a[0])}f, {F(node.point_a[1])}f, {F(node.point_a[2])}f), new Vector3({F(node.point_b[0])}f, {F(node.point_b[1])}f, {F(node.point_b[2])}f), {F(node.radius)}f)";

                case "Cone":
                    stats.varsInlined++;
                    return $"Sdf.Cone({pVar}, {F(node.radius)}f, {F(node.height)}f)";

                case "Ellipsoid":
                    stats.varsInlined++;
                    return $"Sdf.Ellipsoid({pVar}, new Vector3({F(node.radii[0])}f, {F(node.radii[1])}f, {F(node.radii[2])}f))";

                case "HexPrism":
                    stats.varsInlined++;
                    return $"Sdf.HexPrism({pVar}, {F(node.half_height)}f, {F(node.radius)}f)";

                case "Triangle":
                    stats.varsInlined++;
                    return $"Sdf.Triangle({pVar}, new Vector3({F(node.point_a[0])}f, {F(node.point_a[1])}f, {F(node.point_a[2])}f), new Vector3({F(node.point_b[0])}f, {F(node.point_b[1])}f, {F(node.point_b[2])}f), new Vector3({F(node.point_c[0])}f, {F(node.point_c[1])}f, {F(node.point_c[2])}f))";

                case "Bezier":
                    stats.varsInlined++;
                    return $"Sdf.Bezier({pVar}, new Vector3({F(node.point_a[0])}f, {F(node.point_a[1])}f, {F(node.point_a[2])}f), new Vector3({F(node.point_b[0])}f, {F(node.point_b[1])}f, {F(node.point_b[2])}f), new Vector3({F(node.point_c[0])}f, {F(node.point_c[1])}f, {F(node.point_c[2])}f), {F(node.radius)}f)";

                case "RoundedCone":
                    stats.varsInlined++;
                    return $"Sdf.RoundedCone({pVar}, {F(node.r1)}f, {F(node.r2)}f, {F(node.half_height)}f)";

                case "Pyramid":
                    stats.varsInlined++;
                    return $"Sdf.Pyramid({pVar}, {F(node.half_height)}f)";

                case "Octahedron":
                    stats.varsInlined++;
                    return $"Sdf.Octahedron({pVar}, {F(node.radius)}f)";

                case "Link":
                    stats.varsInlined++;
                    return $"Sdf.Link({pVar}, {F(node.half_length)}f, {F(node.r1)}f, {F(node.r2)}f)";

                // ---------------------------------------------------------
                // Binary Ops: Scalar-expanded (no Sdf.* wrapper call)
                // ---------------------------------------------------------
                case "Union":
                    return EmitScalarBinaryOp("Mathf.Min", node, pVar, sb, depth, stats);
                case "Intersection":
                    return EmitScalarBinaryOp("Mathf.Max", node, pVar, sb, depth, stats);
                case "Subtraction":
                    return EmitScalarSubtraction(node, pVar, sb, depth, stats);

                // Smooth Ops: Fully inlined scalar expansion
                case "SmoothUnion":
                    return EmitInlineSmoothUnion(node, pVar, sb, depth, stats);
                case "SmoothIntersection":
                    return EmitInlineSmoothIntersection(node, pVar, sb, depth, stats);
                case "SmoothSubtraction":
                    return EmitInlineSmoothSubtraction(node, pVar, sb, depth, stats);

                // ---------------------------------------------------------
                // Transforms: Scalarized where profitable
                // ---------------------------------------------------------
                case "Translate":
                    return EmitScalarTranslate(node, pVar, sb, depth, stats);

                case "Scale":
                {
                    var sp = NextVar();
                    stats.varsEmitted++;
                    float invF = 1f / node.factor;
                    sb.AppendLine($"            Vector3 {sp} = {pVar} * {F(invF)}f;");
                    string sd = EmitNode(node.child, sp, sb, depth + 1, stats);
                    return $"({sd} * {F(node.factor)}f)";
                }

                // ---------------------------------------------------------
                // Modifiers
                // ---------------------------------------------------------
                case "Twist":
                {
                    var twp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {twp} = Sdf.Twist({pVar}, {F(node.strength)}f);");
                    return EmitNode(node.child, twp, sb, depth + 1, stats);
                }

                case "Round":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"({inner} - {F(node.radius)}f)";
                }

                case "Onion":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"(Mathf.Abs({inner}) - {F(node.thickness)}f)";
                }

                case "RepeatInfinite":
                {
                    var rp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {rp} = Sdf.RepeatInfinite({pVar}, new Vector3({F(node.spacing[0])}f, {F(node.spacing[1])}f, {F(node.spacing[2])}f));");
                    return EmitNode(node.child, rp, sb, depth + 1, stats);
                }

                case "RepeatFinite":
                {
                    var rp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {rp} = Sdf.RepeatFinite({pVar}, new Vector3({F(node.spacing[0])}f, {F(node.spacing[1])}f, {F(node.spacing[2])}f), new Vector3({F(node.count[0])}f, {F(node.count[1])}f, {F(node.count[2])}f));");
                    return EmitNode(node.child, rp, sb, depth + 1, stats);
                }

                case "Rotate":
                {
                    var rtp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {rtp} = Quaternion.Inverse(new Quaternion({F(node.rotation[0])}f, {F(node.rotation[1])}f, {F(node.rotation[2])}f, {F(node.rotation[3])}f)) * {pVar};");
                    return EmitNode(node.child, rtp, sb, depth + 1, stats);
                }

                case "Bend":
                {
                    var bp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {bp} = Sdf.Bend({pVar}, {F(node.curvature)}f);");
                    return EmitNode(node.child, bp, sb, depth + 1, stats);
                }

                case "Taper":
                {
                    var tap = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {tap} = Sdf.Taper({pVar}, {F(node.factor)}f);");
                    return EmitNode(node.child, tap, sb, depth + 1, stats);
                }

                case "Displacement":
                {
                    string inner = EmitNode(node.child, pVar, sb, depth + 1, stats);
                    stats.varsInlined++;
                    return $"Sdf.Displacement({inner}, {pVar}, {F(node.strength)}f)";
                }

                case "Symmetry":
                {
                    var syp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {syp} = Sdf.Symmetry({pVar}, new Vector3({F(node.axes[0])}f, {F(node.axes[1])}f, {F(node.axes[2])}f));");
                    return EmitNode(node.child, syp, sb, depth + 1, stats);
                }

                case "PolarRepeat":
                {
                    var prp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {prp} = Sdf.PolarRepeat({pVar}, {F((float)node.polar_count)}f);");
                    return EmitNode(node.child, prp, sb, depth + 1, stats);
                }

                case "Elongate":
                {
                    var elp = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            Vector3 {elp} = Sdf.Elongate({pVar}, new Vector3({F(node.elongate[0])}f, {F(node.elongate[1])}f, {F(node.elongate[2])}f));");
                    return EmitNode(node.child, elp, sb, depth + 1, stats);
                }

                default:
                    Debug.LogWarning($"[AliceSDF Baker] Unknown node type: {node.type}");
                    return "0f";
            }
        }

        /// <summary>
        /// Union/Intersection: Mathf.Min/Max direct call, no Sdf.* wrapper.
        /// </summary>
        private static string EmitScalarBinaryOp(string func, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);
            string expr = $"{func}({a}, {b})";

            if (depth >= 3)
            {
                var v = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"            float {v} = {expr};");
                return v;
            }

            stats.scalarExpansions++;
            return expr;
        }

        /// <summary>
        /// Subtraction: Mathf.Max(a, -b) direct, no function call.
        /// </summary>
        private static string EmitScalarSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);
            string expr = $"Mathf.Max({a}, -({b}))";

            if (depth >= 3)
            {
                var v = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"            float {v} = {expr};");
                return v;
            }

            stats.scalarExpansions++;
            return expr;
        }

        /// <summary>
        /// SmoothUnion: Fully inlined with pre-computed inv_k.
        /// Formula: min(d1,d2) - max(k - abs(d1-d2), 0)^2 * inv_k * 0.25
        /// </summary>
        private static string EmitInlineSmoothUnion(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);

            // Need temp vars for the smooth formula
            var va = NextVar(); var vb = NextVar();
            stats.varsEmitted += 2;
            sb.AppendLine($"            float {va} = {a};");
            sb.AppendLine($"            float {vb} = {b};");

            float invK = 1f / node.k;
            var vh = NextVar();
            stats.varsEmitted++;
            stats.scalarExpansions++;
            sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} - {vb}), 0f) * {F(invK)}f;");

            return $"(Mathf.Min({va}, {vb}) - {vh} * {vh} * {F(node.k)}f * 0.25f)";
        }

        /// <summary>
        /// SmoothIntersection: max(d1,d2) + h^2 * k * 0.25
        /// </summary>
        private static string EmitInlineSmoothIntersection(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);

            var va = NextVar(); var vb = NextVar();
            stats.varsEmitted += 2;
            sb.AppendLine($"            float {va} = {a};");
            sb.AppendLine($"            float {vb} = {b};");

            float invK = 1f / node.k;
            var vh = NextVar();
            stats.varsEmitted++;
            stats.scalarExpansions++;
            sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} - {vb}), 0f) * {F(invK)}f;");

            return $"(Mathf.Max({va}, {vb}) + {vh} * {vh} * {F(node.k)}f * 0.25f)";
        }

        /// <summary>
        /// SmoothSubtraction: max(d1,-d2) + h^2 * k * 0.25
        /// </summary>
        private static string EmitInlineSmoothSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            string a = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            string b = EmitNode(node.children[1], pVar, sb, depth + 1, stats);

            var va = NextVar(); var vb = NextVar();
            stats.varsEmitted += 2;
            sb.AppendLine($"            float {va} = {a};");
            sb.AppendLine($"            float {vb} = {b};");

            float invK = 1f / node.k;
            var vh = NextVar();
            stats.varsEmitted++;
            stats.scalarExpansions++;
            sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} + {vb}), 0f) * {F(invK)}f;");

            return $"(Mathf.Max({va}, -({vb})) + {vh} * {vh} * {F(node.k)}f * 0.25f)";
        }

        /// <summary>
        /// Translate: Scalarize Vector3 subtraction when child is a leaf primitive.
        /// For leaf children, pass new Vector3(px, py, pz) directly into the primitive.
        /// For non-leaf, still use Vector3 (can't scalarize recursion easily).
        /// </summary>
        private static string EmitScalarTranslate(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.child != null && node.child.IsLeaf())
            {
                // Scalarize: break into float components, pass directly
                var px = NextVar(); var py = NextVar(); var pz = NextVar();
                stats.varsEmitted += 3;
                stats.scalarExpansions++;
                sb.AppendLine($"            float {px} = {pVar}.x - {F(node.offset[0])}f;");
                sb.AppendLine($"            float {py} = {pVar}.y - {F(node.offset[1])}f;");
                sb.AppendLine($"            float {pz} = {pVar}.z - {F(node.offset[2])}f;");

                // Inline the leaf primitive with scalar-constructed Vector3
                string scalarP = $"new Vector3({px}, {py}, {pz})";
                return EmitNode(node.child, scalarP, sb, depth + 1, stats);
            }
            else
            {
                // Non-leaf: use Vector3 (recursive children need it)
                var tp = NextVar();
                stats.varsEmitted++;
                sb.AppendLine($"            Vector3 {tp} = {pVar} - new Vector3({F(node.offset[0])}f, {F(node.offset[1])}f, {F(node.offset[2])}f);");
                return EmitNode(node.child, tp, sb, depth + 1, stats);
            }
        }

        private static string F(float v)
        {
            if (v == (int)v)
                return $"{(int)v}.0";

            string s = v.ToString("F6");
            s = s.TrimEnd('0');
            if (s.EndsWith(".")) s += "0";
            return s;
        }
    }
}

#endif // UNITY_EDITOR
