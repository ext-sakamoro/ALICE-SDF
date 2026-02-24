// =============================================================================
// AliceSDF_Baker.cs - ALICE-SDF Baker v0.2 (Deep Fried Edition)
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
            EditorGUILayout.LabelField("ALICE-SDF Baker v0.2 (Deep Fried)", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Generate VRChat-ready Shader + Udon Collider from ALICE-SDF definitions.\n" +
                "Drag a .asdf.json file or paste the SDF JSON directly.\n" +
                "v0.2: Instruction Fusion + Division Exorcism + Scalar Expansion",
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

            try
            {
                var hlslStats = new CodegenStats();
                var udonStats = new CodegenStats();
                _previewHlsl = ShaderGenerator.GenerateMapFunction(node, hlslStats);
                _previewCSharp = UdonGenerator.GenerateEvaluateFunction(node, udonStats);
                _statusMessage = $"Preview OK. Nodes: {node.CountNodes()} | " +
                                 $"HLSL: {hlslStats.varsEmitted} vars ({hlslStats.varsInlined} inlined) | " +
                                 $"Udon: {udonStats.varsEmitted} vars ({udonStats.scalarExpansions} scalar-expanded)";
            }
            catch (System.Exception e)
            {
                _previewHlsl = "";
                _previewCSharp = "";
                _statusMessage = $"Codegen error: {e.Message}";
                Debug.LogError($"[AliceSDF Baker] Codegen failed: {e}");
            }
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

    [System.Serializable]
    public class SdfNodeData
    {
        // --- Internal field names (used by codegen) ---
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
        public SdfNodeData[] children;
        public SdfNodeData child;

        // --- JSON camelCase aliases (JsonUtility field-name compatibility) ---
        public float smoothness;     // alias for k
        public float majorRadius;    // alias for major_radius
        public float minorRadius;    // alias for minor_radius
        public float halfHeight;     // alias for half_height
        public float[] pointA;       // alias for point_a
        public float[] pointB;       // alias for point_b

        /// <summary>
        /// Merge camelCase JSON aliases into snake_case internal fields, recursively.
        /// Called after JsonUtility.FromJson to unify field naming.
        /// </summary>
        public void Normalize()
        {
            if (k == 0f && smoothness != 0f) k = smoothness;
            if (major_radius == 0f && majorRadius != 0f) major_radius = majorRadius;
            if (minor_radius == 0f && minorRadius != 0f) minor_radius = minorRadius;
            if (half_height == 0f && halfHeight != 0f) half_height = halfHeight;
            if (point_a == null && pointA != null) point_a = pointA;
            if (point_b == null && pointB != null) point_b = pointB;

            if (children != null)
                foreach (var ch in children) ch?.Normalize();
            if (child != null) child.Normalize();
        }

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
                var node = JsonUtility.FromJson<SdfNodeData>(json);
                node?.Normalize();
                return node;
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
        private static HashSet<float> _smoothKValues;

        // Inline threshold: expressions deeper than this get a temp variable
        private const int INLINE_DEPTH = 3;

        public static string GenerateMapFunction(SdfNodeData node, CodegenStats stats = null)
        {
            _varCounter = 0;
            _smoothKValues = new HashSet<float>();
            if (stats == null) stats = new CodegenStats();

            // First pass: collect smooth k values for Division Exorcism
            CollectSmoothK(node);

            var sb = new StringBuilder();

            // Emit pre-computed inverse k values
            foreach (var kv in _smoothKValues)
            {
                sb.AppendLine($"    float inv_k_{Fk(kv)} = 1.0 / {F(kv)};");
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

            return $@"// Auto-generated by ALICE-SDF Baker v0.2 (Deep Fried)
// Name: {name}
// Stats: {stats.varsEmitted} vars, {stats.varsInlined} fused, {stats.divisionExorcisms} div-exorcised
Shader ""AliceSDF/{name}""
{{
    Properties
    {{
        [Header(Colors)]
        _Color (""Primary Color"", Color) = (0.2, 0.8, 1.0, 1.0)
        _Color2 (""Secondary Color"", Color) = (0.8, 0.3, 0.5, 1.0)
        [Header(Raymarching)]
        _MaxSteps (""Max Steps (0=Auto)"", Int) = 0
        _MaxDist (""Max Distance"", Float) = 200.0
        _SurfaceEpsilon (""Surface Epsilon (0=Auto)"", Float) = 0.0
        [Header(Lighting)]
        _AmbientStrength (""Ambient"", Range(0, 1)) = 0.2
        _AOEnabled (""Enable AO"", Int) = 1
        [Header(Fog)]
        _FogDensity (""Fog Density"", Float) = 0.005
        _FogColor (""Fog Color"", Color) = (0.01, 0.01, 0.02, 1.0)
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
            #include ""Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc""

            float4 _Color; float4 _Color2;
            int _MaxSteps; float _MaxDist; float _SurfaceEpsilon;
            float _AmbientStrength; int _AOEnabled;
            float _FogDensity; float4 _FogColor;

            struct appdata {{ float4 vertex : POSITION; }};
            struct v2f {{
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
            }};

            // === GENERATED SDF (Deep Fried v0.2) ===
            float map(float3 p)
            {{
{mapBody}
            }}

            #include ""Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc""

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
            float3 getColor(float3 p, float3 n) {{
                float3 bc = lerp(_Color.rgb, _Color2.rgb, n.y*0.5+0.5);
                float d = noise3d(p*10.0)*0.3+noise3d(p*50.0)*0.1;
                float3 c = bc + d;
                float f = pow(1.0-abs(dot(n,normalize(_WorldSpaceCameraPos-p))),2.0);
                return saturate(c + f*float3(0.1,0.2,0.3));
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
                        float3 col = getColor(p, n);
                        float3 ld = normalize(float3(1,1,-0.5));
                        float diff = max(dot(n, ld), 0.0);
                        float ao = 1.0;
                        if (_AOEnabled > 0) {{ int as2 = (tier==0)?5:(tier==1)?3:2; ao = sdfAO(p,n,as2); }}
                        float3 fc = col * (_AmbientStrength + diff*(1.0-_AmbientStrength)) * ao;
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
                    }}
                    t += d * ss;
                    if (t > _MaxDist) break;
                }}
                float3 sky = lerp(_FogColor.rgb, _FogColor.rgb+float3(0.04,0.09,0.18), rd.y*0.5+0.5);
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
                _smoothKValues.Add(node.k);
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

                default:
                    Debug.LogWarning($"[AliceSDF Baker] Unknown node type: {node.type}");
                    return "0.0";
            }
        }

        /// <summary>
        /// Fused binary op: inline min/max directly instead of opUnion/opIntersection wrapper.
        /// Supports N-ary children via left-fold: min(min(min(a, b), c), d).
        /// At depth >= INLINE_DEPTH, emit a temp variable to avoid extremely long lines.
        /// </summary>
        private static string EmitFusedBinaryOp(string hlslFunc, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] {node.type} requires at least 1 child, got 0");
                return "0.0";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            // Left-fold: op(op(a, b), c) ...
            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);
                string expr = $"{hlslFunc}({result}, {next})";

                if (depth >= INLINE_DEPTH || i < node.children.Length - 1)
                {
                    var v = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {v} = {expr};");
                    result = v;
                }
                else
                {
                    stats.varsInlined++;
                    result = expr;
                }
            }
            return result;
        }

        /// <summary>
        /// Subtraction: max(a, -b) inlined directly.
        /// N-ary: A - B - C - D = max(max(max(A, -B), -C), -D).
        /// </summary>
        private static string EmitFusedSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] Subtraction requires at least 1 child, got 0");
                return "0.0";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);
                string expr = $"max({result}, -({next}))";

                if (depth >= INLINE_DEPTH || i < node.children.Length - 1)
                {
                    var v = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {v} = {expr};");
                    result = v;
                }
                else
                {
                    stats.varsInlined++;
                    result = expr;
                }
            }
            return result;
        }

        /// <summary>
        /// Smooth ops: use the library function with left-fold for N-ary children.
        /// opSmoothUnion(opSmoothUnion(a, b, k), c, k) ...
        /// </summary>
        private static string EmitFusedSmoothOp(string hlslFunc, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] {hlslFunc} requires at least 1 child, got 0");
                return "0.0";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);
                string expr = $"{hlslFunc}({result}, {next}, {F(node.k)})";

                if (depth >= INLINE_DEPTH || i < node.children.Length - 1)
                {
                    var v = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"    float {v} = {expr};");
                    result = v;
                }
                else
                {
                    stats.varsInlined++;
                    result = expr;
                }
            }
            return result;
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

                default:
                    Debug.LogWarning($"[AliceSDF Baker] Unknown node type: {node.type}");
                    return "0f";
            }
        }

        /// <summary>
        /// Union/Intersection: Mathf.Min/Max direct call, no Sdf.* wrapper.
        /// N-ary left-fold: Min(Min(Min(a, b), c), d).
        /// </summary>
        private static string EmitScalarBinaryOp(string func, SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] {node.type} requires at least 1 child, got 0");
                return "0f";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);
                string expr = $"{func}({result}, {next})";

                if (depth >= 3 || i < node.children.Length - 1)
                {
                    var v = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            float {v} = {expr};");
                    result = v;
                }
                else
                {
                    stats.scalarExpansions++;
                    result = expr;
                }
            }
            return result;
        }

        /// <summary>
        /// Subtraction: Mathf.Max(a, -b) direct, no function call.
        /// N-ary: A - B - C = Max(Max(A, -B), -C).
        /// </summary>
        private static string EmitScalarSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] Subtraction requires at least 1 child, got 0");
                return "0f";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);
            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);
                string expr = $"Mathf.Max({result}, -({next}))";

                if (depth >= 3 || i < node.children.Length - 1)
                {
                    var v = NextVar();
                    stats.varsEmitted++;
                    sb.AppendLine($"            float {v} = {expr};");
                    result = v;
                }
                else
                {
                    stats.scalarExpansions++;
                    result = expr;
                }
            }
            return result;
        }

        /// <summary>
        /// SmoothUnion: Fully inlined with pre-computed inv_k.
        /// Formula: min(d1,d2) - max(k - abs(d1-d2), 0)^2 * inv_k * 0.25
        /// N-ary left-fold: smoothUnion(smoothUnion(a, b), c) ...
        /// </summary>
        private static string EmitInlineSmoothUnion(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] SmoothUnion requires at least 1 child, got 0");
                return "0f";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            float invK = (node.k != 0f) ? 1f / node.k : 0f;
            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);

                var va = NextVar(); var vb = NextVar();
                stats.varsEmitted += 2;
                sb.AppendLine($"            float {va} = {result};");
                sb.AppendLine($"            float {vb} = {next};");

                var vh = NextVar();
                stats.varsEmitted++;
                stats.scalarExpansions++;
                sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} - {vb}), 0f) * {F(invK)}f;");

                result = $"(Mathf.Min({va}, {vb}) - {vh} * {vh} * {F(node.k)}f * 0.25f)";
            }
            return result;
        }

        /// <summary>
        /// SmoothIntersection: max(d1,d2) + h^2 * k * 0.25
        /// N-ary left-fold.
        /// </summary>
        private static string EmitInlineSmoothIntersection(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] SmoothIntersection requires at least 1 child, got 0");
                return "0f";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            float invK = (node.k != 0f) ? 1f / node.k : 0f;
            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);

                var va = NextVar(); var vb = NextVar();
                stats.varsEmitted += 2;
                sb.AppendLine($"            float {va} = {result};");
                sb.AppendLine($"            float {vb} = {next};");

                var vh = NextVar();
                stats.varsEmitted++;
                stats.scalarExpansions++;
                sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} - {vb}), 0f) * {F(invK)}f;");

                result = $"(Mathf.Max({va}, {vb}) + {vh} * {vh} * {F(node.k)}f * 0.25f)";
            }
            return result;
        }

        /// <summary>
        /// SmoothSubtraction: max(d1,-d2) + h^2 * k * 0.25
        /// N-ary left-fold: A - B - C = smoothSub(smoothSub(A, B), C).
        /// </summary>
        private static string EmitInlineSmoothSubtraction(SdfNodeData node, string pVar, StringBuilder sb, int depth, CodegenStats stats)
        {
            if (node.children == null || node.children.Length == 0)
            {
                Debug.LogWarning($"[AliceSDF Baker] SmoothSubtraction requires at least 1 child, got 0");
                return "0f";
            }
            if (node.children.Length == 1)
                return EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            float invK = (node.k != 0f) ? 1f / node.k : 0f;
            string result = EmitNode(node.children[0], pVar, sb, depth + 1, stats);

            for (int i = 1; i < node.children.Length; i++)
            {
                string next = EmitNode(node.children[i], pVar, sb, depth + 1, stats);

                var va = NextVar(); var vb = NextVar();
                stats.varsEmitted += 2;
                sb.AppendLine($"            float {va} = {result};");
                sb.AppendLine($"            float {vb} = {next};");

                var vh = NextVar();
                stats.varsEmitted++;
                stats.scalarExpansions++;
                sb.AppendLine($"            float {vh} = Mathf.Max({F(node.k)}f - Mathf.Abs({va} + {vb}), 0f) * {F(invK)}f;");

                result = $"(Mathf.Max({va}, -({vb})) + {vh} * {vh} * {F(node.k)}f * 0.25f)";
            }
            return result;
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
