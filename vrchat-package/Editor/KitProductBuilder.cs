// =============================================================================
// KitProductBuilder.cs - "ALICE-SDF/Build Kit Product"
// =============================================================================
// Generates the stand-alone AliceSDF Kit product (the BOOTH unitypackage)
// from the seven samples, so the samples stay the single source of the laws:
//
//   Assets/AliceSDFKit/
//     Scripts/AliceSDF_Collider.cs        base collider (namespace AliceSDFKit)
//     Scripts/Alice{Mochi,Wall,Terrain}.cs, AliceDecor{Basic,Cosmic,Fractal,Mix}.cs
//     Shaders/AliceSDF_Include.cginc, AliceSDF_LOD.cginc, <sample>.shader (renamed AliceSDFKit/<name>)
//     Materials/<prefab>.mat
//     Prefabs/AliceSDF <name>.prefab       root at the ground / law origin, child Volume cube + collider
//     Programs/<script>_UdonProgram.asset
//     README.md / README_JP.md / LICENSE.txt
//
// then exports it to <repo>/vrchat-package/Product~/AliceSDFKit_<version>.unitypackage.
// Re-run after any change to a sample; never edit the generated folder.
// Everything compiles into Assembly-CSharp (no asmdef), so the package needs
// nothing but the VRChat Worlds SDK.
//
// Author: Moroya Sakamoto
// =============================================================================

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AliceSDF.Editor
{
    public static class KitProductBuilder
    {
        private const string PendingKey = "AliceSDF.KitProduct.pending";
        private const string PackageRoot = "Packages/com.alice.sdf";
        private const string Out = "Assets/AliceSDFKit";
        private const string ProductName = "AliceSDFKit";

        // One row per generated script (sample file -> product class)
        private struct ScriptDef
        {
            public string sampleDir, sampleFile, productClass, sampleClass;
        }

        // One row per prefab (a sample may give several: the Mochi presets)
        private struct PrefabDef
        {
            public string name;            // prefab / material name
            public string productClass;    // AliceSDFKit.<class>
            public string shaderName;      // AliceSDFKit/<name>
            public Vector3 cubePos, cubeScale;
            public bool terrainSupport;
            public System.Action<Material, SerializedObject> preset;   // optional colours / knobs
        }

        private static readonly ScriptDef[] Scripts =
        {
            new ScriptDef { sampleDir = "SampleMochi",          sampleFile = "SampleMochi_Collider.cs",          sampleClass = "SampleMochi_Collider",          productClass = "AliceMochi" },
            new ScriptDef { sampleDir = "SampleDeformableWall", sampleFile = "SampleDeformableWall_Collider.cs", sampleClass = "SampleDeformableWall_Collider", productClass = "AliceWall" },
            new ScriptDef { sampleDir = "SampleTerrainSculpt",  sampleFile = "SampleTerrainSculpt_Collider.cs",  sampleClass = "SampleTerrainSculpt_Collider",  productClass = "AliceTerrain" },
            new ScriptDef { sampleDir = "SampleBasic",          sampleFile = "SampleBasic_Collider.cs",          sampleClass = "SampleBasic_Collider",          productClass = "AliceDecorBasic" },
            new ScriptDef { sampleDir = "SampleCosmic",         sampleFile = "SampleCosmic_Collider.cs",         sampleClass = "SampleCosmic_Collider",         productClass = "AliceDecorCosmic" },
            new ScriptDef { sampleDir = "SampleFractal",        sampleFile = "SampleFractal_Collider.cs",        sampleClass = "SampleFractal_Collider",        productClass = "AliceDecorFractal" },
            new ScriptDef { sampleDir = "SampleMix",            sampleFile = "SampleMix_Collider.cs",            sampleClass = "SampleMix_Collider",            productClass = "AliceDecorMix" },
        };

        // sample shader file -> product shader name
        private static readonly string[][] Shaders =
        {
            new[] { "SampleMochi/SampleMochi_Raymarcher.shader",                   "AliceSDF/Samples/Mochi",          "AliceSDFKit/Mochi" },
            new[] { "SampleDeformableWall/SampleDeformableWall_Raymarcher.shader", "AliceSDF/Samples/DeformableWall", "AliceSDFKit/Wall" },
            new[] { "SampleTerrainSculpt/SampleTerrainSculpt_Raymarcher.shader",   "AliceSDF/Samples/TerrainSculpt",  "AliceSDFKit/Terrain" },
            new[] { "SampleBasic/SampleBasic_Raymarcher.shader",                   "AliceSDF/Samples/Basic",          "AliceSDFKit/DecorBasic" },
            new[] { "SampleCosmic/SampleCosmic_Raymarcher.shader",                 "AliceSDF/Samples/Cosmic",         "AliceSDFKit/DecorCosmic" },
            new[] { "SampleFractal/SampleFractal_Raymarcher.shader",               "AliceSDF/Samples/Fractal",        "AliceSDFKit/DecorFractal" },
            new[] { "SampleMix/SampleMix_Raymarcher.shader",                       "AliceSDF/Samples/Mix",            "AliceSDFKit/DecorMix" },
        };

        private static PrefabDef Mochi(string preset, Color mochi, Color highlight, Color ground, Color detail,
                                       float blendK, float groundK, float density, float gravity)
        {
            return new PrefabDef
            {
                name = "AliceSDF Mochi (" + preset + ")", productClass = "AliceMochi", shaderName = "AliceSDFKit/Mochi",
                cubePos = new Vector3(0, 1, 0), cubeScale = new Vector3(4, 2, 4),
                preset = (mat, so) =>
                {
                    mat.SetColor("_MochiColor", mochi); mat.SetColor("_MochiColor2", highlight);
                    mat.SetColor("_GroundColor", ground); mat.SetColor("_GroundColor2", detail);
                    mat.SetFloat("_BlendK", blendK); mat.SetFloat("_GroundK", groundK);
                    so.FindProperty("applyColors").boolValue = true;
                    so.FindProperty("mochiColor").colorValue = mochi;
                    so.FindProperty("mochiHighlight").colorValue = highlight;
                    so.FindProperty("groundColor").colorValue = ground;
                    so.FindProperty("groundDetail").colorValue = detail;
                    so.FindProperty("useCustomLayout").boolValue = true;
                    so.FindProperty("initialCount").intValue = 5;
                    so.FindProperty("initialRadius").floatValue = 0.3f;
                    so.FindProperty("ringRadius").floatValue = 0.7f;
                    so.FindProperty("blendK").floatValue = blendK;
                    so.FindProperty("groundK").floatValue = groundK;
                    so.FindProperty("mochiDensity").floatValue = density;
                    so.FindProperty("gravity").floatValue = gravity;
                    so.FindProperty("groundOffset").vector3Value = new Vector3(0, -1, 0);
                }
            };
        }

        private static readonly PrefabDef[] Prefabs =
        {
            Mochi("Mochi", new Color(0.96f, 0.93f, 0.88f), new Color(0.99f, 0.96f, 0.92f), new Color(0.55f, 0.46f, 0.36f), new Color(0.50f, 0.42f, 0.33f), 0.5f, 0.15f, 1000f, 3f),
            Mochi("Slime", new Color(0.45f, 0.85f, 0.35f), new Color(0.70f, 1.0f, 0.55f), new Color(0.30f, 0.32f, 0.30f), new Color(0.26f, 0.28f, 0.26f), 0.9f, 0.3f, 600f, 2f),
            Mochi("Water", new Color(0.55f, 0.75f, 1.0f), new Color(0.85f, 0.95f, 1.0f), new Color(0.75f, 0.72f, 0.62f), new Color(0.70f, 0.66f, 0.56f), 0.3f, 0.1f, 1000f, 5f),
            new PrefabDef { name = "AliceSDF Wall", productClass = "AliceWall", shaderName = "AliceSDFKit/Wall",
                            cubePos = new Vector3(0, 4, 0), cubeScale = new Vector3(12, 8, 12),
                            preset = (mat, so) => { so.FindProperty("groundOffset").vector3Value = new Vector3(0, -4, 0); } },
            new PrefabDef { name = "AliceSDF Terrain", productClass = "AliceTerrain", shaderName = "AliceSDFKit/Terrain",
                            cubePos = new Vector3(0, 3, 0), cubeScale = new Vector3(20, 10, 20), terrainSupport = true,
                            preset = (mat, so) => { so.FindProperty("groundOffset").vector3Value = new Vector3(0, -3, 0); } },
            new PrefabDef { name = "AliceSDF Decor Basic",   productClass = "AliceDecorBasic",   shaderName = "AliceSDFKit/DecorBasic",   cubePos = Vector3.zero, cubeScale = Vector3.one * 100f },
            new PrefabDef { name = "AliceSDF Decor Cosmic",  productClass = "AliceDecorCosmic",  shaderName = "AliceSDFKit/DecorCosmic",  cubePos = Vector3.zero, cubeScale = Vector3.one * 200f },
            new PrefabDef { name = "AliceSDF Decor Fractal", productClass = "AliceDecorFractal", shaderName = "AliceSDFKit/DecorFractal", cubePos = Vector3.zero, cubeScale = Vector3.one * 200f },
            new PrefabDef { name = "AliceSDF Decor Mix",     productClass = "AliceDecorMix",     shaderName = "AliceSDFKit/DecorMix",     cubePos = Vector3.zero, cubeScale = Vector3.one * 200f },
        };

        [InitializeOnLoadMethod]
        private static void Resume()
        {
            if (!EditorPrefs.GetBool(PendingKey, false)) return;
            NextTick(() =>
            {
                if (EditorApplication.isCompiling) { Resume(); return; }
                EditorPrefs.SetBool(PendingKey, false);
                string path = BuildCore();
                if (!string.IsNullOrEmpty(path)) Debug.Log("[ALICE-SDF] Kit product done: " + path);
            });
        }

        private static void NextTick(System.Action action)
        {
            EditorApplication.CallbackFunction once = null;
            once = () => { EditorApplication.update -= once; action(); };
            EditorApplication.update += once;
        }

        [MenuItem("ALICE-SDF/Build Kit Product")]
        public static void Build()
        {
            string path = BuildCore();
            if (path != null) EditorUtility.DisplayDialog("ALICE-SDF Kit Product", "Exported:\n" + path, "OK");
        }

        // Headless: -executeMethod AliceSDF.Editor.KitProductBuilder.BuildBatch
        public static void BuildBatch()
        {
            if (BuildCore() == null) EditorApplication.Exit(2);
        }

        public static string BuildCore()
        {
            string packageFull = Path.GetFullPath(PackageRoot);
            string galleryFull = Path.Combine(packageFull, "Samples~", "SDF Gallery");
            if (!Directory.Exists(galleryFull))
                throw new DirectoryNotFoundException("SDF Gallery samples not found: " + galleryFull);

            foreach (var d in new[] { Out, Out + "/Scripts", Out + "/Shaders", Out + "/Materials", Out + "/Prefabs", Out + "/Programs" })
                if (!AssetDatabase.IsValidFolder(d)) Directory.CreateDirectory(d);

            // --- Scripts: base collider + the seven samples, renamed into one namespace ---
            bool scriptChanged = false;
            string baseCs = File.ReadAllText(Path.Combine(packageFull, "Runtime", "Udon", "AliceSDF_Collider.cs"));
            baseCs = baseCs.Replace("namespace AliceSDF\r\n", "namespace AliceSDFKit\r\n").Replace("namespace AliceSDF\n", "namespace AliceSDFKit\n");
            baseCs = "// AliceSDF Kit - generated from ALICE-SDF Runtime/Udon/AliceSDF_Collider.cs by KitProductBuilder; do not edit here\n" + baseCs;
            scriptChanged |= WriteIfChanged(Out + "/Scripts/AliceSDF_Collider.cs", baseCs);
            foreach (var s in Scripts)
            {
                string cs = File.ReadAllText(Path.Combine(galleryFull, s.sampleDir, s.sampleFile));
                cs = cs.Replace("namespace AliceSDF.Samples", "namespace AliceSDFKit")
                       .Replace(s.sampleClass, s.productClass);
                cs = "// AliceSDF Kit - generated from the ALICE-SDF " + s.sampleDir + " sample by KitProductBuilder; do not edit here\n" + cs;
                scriptChanged |= WriteIfChanged(Out + "/Scripts/" + s.productClass + ".cs", cs);
            }

            // --- Shaders: the two includes next to them, includes rewritten, names renamed ---
            foreach (var inc in new[] { "AliceSDF_Include.cginc", "AliceSDF_LOD.cginc" })
                WriteIfChanged(Out + "/Shaders/" + inc, File.ReadAllText(Path.Combine(packageFull, "Runtime", "Shaders", inc)));
            foreach (var sh in Shaders)
            {
                string src = File.ReadAllText(Path.Combine(galleryFull, sh[0]));
                src = src.Replace("Shader \"" + sh[1] + "\"", "Shader \"" + sh[2] + "\"")
                         .Replace("#include \"Packages/com.alice.sdf/Runtime/Shaders/", "#include \"");
                if (src.Contains("Packages/com.alice.sdf/"))
                    throw new System.InvalidOperationException(sh[0] + " still references the package: the product must be stand-alone");
                WriteIfChanged(Out + "/Shaders/" + Path.GetFileName(sh[0]).Replace("_Raymarcher", "").Replace("Sample", "Kit"), src);
            }

            // --- Docs / license ---
            File.WriteAllText(Out + "/README.md", ReadmeEn());
            File.WriteAllText(Out + "/README_JP.md", ReadmeJp());
            string lic = Path.Combine(packageFull, "LICENSE.md");
            if (File.Exists(lic)) File.Copy(lic, Out + "/LICENSE.txt", true);

            bool anyTypeMissing = FindType("AliceSDFKit.AliceSDF_Collider") == null;
            foreach (var s in Scripts) anyTypeMissing |= FindType("AliceSDFKit." + s.productClass) == null;
            if (anyTypeMissing || scriptChanged)
            {
                EditorPrefs.SetBool(PendingKey, true);
                AssetDatabase.Refresh();
                Debug.Log("[ALICE-SDF] Kit product phase 1: files written, compiling (phase 2 follows the reload)");
                return null;
            }
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            foreach (var sh in Shaders)
                if (Shader.Find(sh[2]) == null) throw new System.InvalidOperationException(sh[2] + " did not compile");

#if UDONSHARP
            // Program assets: one per script (the base class needs one too, or
            // UdonSharp reports the subclasses as outside a U# assembly)
            bool createdProgram = false;
            var programScripts = new List<string> { "AliceSDF_Collider" };
            foreach (var s in Scripts) programScripts.Add(s.productClass);
            foreach (var name in programScripts)
            {
                string programPath = Out + "/Programs/" + name + "_UdonProgram.asset";
                if (AssetDatabase.LoadAssetAtPath<UdonSharp.UdonSharpProgramAsset>(programPath) != null) continue;
                var script = AssetDatabase.LoadAssetAtPath<MonoScript>(Out + "/Scripts/" + name + ".cs");
                var asset = ScriptableObject.CreateInstance<UdonSharp.UdonSharpProgramAsset>();
                asset.sourceCsScript = script;
                AssetDatabase.CreateAsset(asset, programPath);
                createdProgram = true;
            }
            if (createdProgram)
            {
                AssetDatabase.SaveAssets();
                UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
                Debug.Log("[ALICE-SDF] Kit product phase 2: program assets compiled, prefabs follow");
                NextTick(() =>
                {
                    string p = BuildCore();
                    if (p != null) Debug.Log("[ALICE-SDF] Kit product phase 3 done: " + p);
                });
                return null;
            }
#endif

            // --- Materials + prefabs ---
            var pending = new List<GameObject>();
            foreach (var p in Prefabs)
            {
                var shader = Shader.Find(p.shaderName);
                string matPath = Out + "/Materials/" + p.name + ".mat";
                var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
                if (mat == null) { mat = new Material(shader); AssetDatabase.CreateAsset(mat, matPath); }
                mat.shader = shader;

                // Root at the ground / law origin (what the user places), the
                // volume cube a child so its scale stays out of the root
                var root = new GameObject(p.name);
                var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.name = "Volume";
                cube.transform.SetParent(root.transform, false);
                cube.transform.localPosition = p.cubePos;
                cube.transform.localScale = p.cubeScale;
                Object.DestroyImmediate(cube.GetComponent<BoxCollider>());
                cube.GetComponent<MeshRenderer>().sharedMaterial = mat;
                var type = FindType("AliceSDFKit." + p.productClass);
                var comp = cube.AddComponent(type);
                var so = new SerializedObject(comp);
                if (p.terrainSupport)
                {
                    // Level box the player stands on, re-placed by the script every
                    // frame; a sibling of the cube so its scale is not inherited
                    var support = new GameObject("TerrainSupport");
                    support.transform.SetParent(root.transform, false);
                    support.transform.localScale = new Vector3(1.5f, 0.2f, 1.5f);
                    support.transform.localPosition = new Vector3(0f, -0.1f, 0f);
                    support.AddComponent<BoxCollider>();
                    var rb = support.AddComponent<Rigidbody>();
                    rb.isKinematic = true;
                    so.FindProperty("support").objectReferenceValue = support.transform;
                }
                if (p.preset != null) p.preset(mat, so);
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(mat);
                pending.Add(root);
            }
            AssetDatabase.SaveAssets();

#if UDONSHARP
            UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
            foreach (var root in pending)
            {
                foreach (var proxy in root.GetComponentsInChildren<UdonSharp.UdonSharpBehaviour>())
                {
                    if (UdonSharpEditor.UdonSharpEditorUtility.GetBackingUdonBehaviour(proxy) == null)
                        UdonSharpEditor.UdonSharpEditorUtility.CreateBehaviourForProxy(proxy);
                    UdonSharpEditor.UdonSharpEditorUtility.CopyProxyToUdon(proxy);
                }
                PrefabUtility.SaveAsPrefabAsset(root, Out + "/Prefabs/" + root.name + ".prefab");
                Object.DestroyImmediate(root);
            }
#else
            foreach (var root in pending)
            {
                PrefabUtility.SaveAsPrefabAsset(root, Out + "/Prefabs/" + root.name + ".prefab");
                Object.DestroyImmediate(root);
            }
#endif
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return Export(packageFull);
        }

        private static bool WriteIfChanged(string path, string text)
        {
            if (File.Exists(path) && File.ReadAllText(path) == text) return false;
            File.WriteAllText(path, text);
            return true;
        }

        private static string Export(string packageFull)
        {
            string version = PackageVersion(packageFull);
            string outDir = Path.Combine(packageFull, "Product~");
            Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, ProductName + "_" + version + ".unitypackage");
            AssetDatabase.ExportPackage(Out, outPath, ExportPackageOptions.Recurse);
            Debug.Log("[ALICE-SDF] Kit product exported: " + outPath);
            return outPath;
        }

        private static string PackageVersion(string packageFull)
        {
            string json = File.ReadAllText(Path.Combine(packageFull, "package.json"));
            var m = System.Text.RegularExpressions.Regex.Match(json, "\"version\"\\s*:\\s*\"([^\"]+)\"");
            return m.Success ? m.Groups[1].Value : "0.0.0";
        }

        private static System.Type FindType(string fullName)
        {
            foreach (var asm in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                var t = asm.GetType(fullName);
                if (t != null) return t;
            }
            return null;
        }

        private static string ReadmeEn()
        {
            return
@"# AliceSDF Kit — math-made interactive things for your VRChat world

Seven prefabs, no polygons: everything is a raymarched signed distance field, and the same
formula that draws it is what your body collides with. Drop a prefab in, place it, build.
PC and Quest (on Quest every shader picks a lighter march / shadow / AO budget by itself).

## Install (3 steps)
1. Import `AliceSDFKit_<version>.unitypackage` into a VRChat **Worlds** project (SDK 3.7 or newer, UdonSharp is part of the SDK). No other package.
2. Drag a prefab from `Assets/AliceSDFKit/Prefabs/` into your scene and place its root where you want it.
   The root is the ground level (Mochi, Wall, Terrain) or the centre of the shape (Decor). Turn it as you like.
   The child `Volume` cube is the space the effect can use; scale it to the area you want.
3. Build & Test.

## What is in the box
| Prefab | It is | Play (VR / desktop) |
|---|---|---|
| AliceSDF Mochi (Mochi / Slime / Water) | soft blobs on a ground patch, three presets | grab: hand in / hold left click; split: grip / right click; merge: push together; walk in: your body dents them |
| AliceSDF Wall | a wall that dents where it is hit and heals | punch: hand / hold left click; walk into it |
| AliceSDF Terrain | ground you can build up and dig, and stand on | left hand / left click builds, right hand / right click digs; walk on what you made |
| AliceSDF Decor Basic / Cosmic / Fractal / Mix | walk-in showpieces (sphere on a plane, an orbiting planet system, a twisted fractal, a mixed scene) | walk around and into them; they push you out of solid parts |

Interactive prefabs are synced: everyone sees the same mochis, dents and terrain (owner-authoritative).
Keep a floor collider under Mochi and Wall (their ground is drawn, not walked on). Terrain carries you by itself.

## Tune (Inspector of the component on `Volume`, no material editing needed)
Every prefab has a **Look** section (colours, and your own textures: drop an image in, it is projected without UVs)
and a **Placement** section (`Ground Offset`: where the ground is relative to the root; the prefabs are set up so the root is the ground).
Then per prefab: Mochi — count / size / stickiness / weight / VR grab rules; Wall — size / dent radius / recovery;
Terrain — brush radius / smoothness / capacity; Decor — the shape parameters. `Log Events` prints one line per event to the client log.

## Support
Message the shop page on BOOTH with: your SDK version, what you did, what you expected, and the `[Mochi]` / `[Wall]` / `[Terrain]` / `[SDF]`
log lines (turn on Log Events). Reproductions in a fresh Worlds project get fixed first.

Built with ALICE-SDF (https://github.com/ext-sakamoro/ALICE-SDF). See LICENSE.txt.
";
        }

        private static string ReadmeJp()
        {
            return
@"# AliceSDF Kit — 数式でできた、触れる VRChat ワールドギミック集

prefab 7 個、ポリゴン 0: 全部レイマーチングの距離関数で、描いている数式そのものに体がぶつかります
置いて、位置を決めて、ビルドするだけ PC と Quest 両対応 (Quest では shader が自動で march / 影 / AO の予算を落とします)

## 導入 (3 step)
1. VRChat **Worlds** project (SDK 3.7 以降、UdonSharp は SDK 同梱) に `AliceSDFKit_<version>.unitypackage` を import 他の package は不要
2. `Assets/AliceSDFKit/Prefabs/` の prefab を scene に drag して、root を置きたい場所へ
   root が地面の高さ (Mochi / Wall / Terrain) または形の中心 (Decor) です 向きは自由
   子の `Volume` cube が効果の使える空間なので、欲しい広さに scale してください
3. Build & Test

## 同梱物
| Prefab | 内容 | 遊び方 (VR / デスクトップ) |
|---|---|---|
| AliceSDF Mochi (Mochi / Slime / Water) | 地面パッチの上のやわらかい塊、preset 3 種 | 掴む: 手を入れる / 左クリック押しっぱなし、分裂: グリップ / 右クリック、合体: 押し合わせる、体当たり: 体の形に凹む |
| AliceSDF Wall | 殴ると凹んで戻る壁 | 殴る: 手 / 左クリック押しっぱなし、歩いて押す |
| AliceSDF Terrain | 盛って掘れて、その上に立てる地面 | 左手 / 左クリックで盛る、右手 / 右クリックで掘る、作った地形の上を歩く |
| AliceSDF Decor Basic / Cosmic / Fractal / Mix | 中に入れる飾り (球と床、公転する惑星系、ねじれたフラクタル、混合 scene) | 歩き回る、固い所からは押し出される |

インタラクティブな prefab は同期します (全員が同じ餅・凹み・地形を見る、owner 権威)
Mochi と Wall の地面は「描いているだけ」なので床 collider をその高さに置いてください Terrain は自分で足場を出します

## 調整 (`Volume` のコンポーネントの Inspector だけ、material は触らなくてよい)
全 prefab に **Look** (色と、自分のテクスチャ: 画像を drop すると UV なしで投影) と **Placement** (`Ground Offset`: root から見た地面の位置、prefab は root = 地面に設定済) があります
あとは prefab ごとに: Mochi — 個数 / 大きさ / 粘り / 重さ / VR の掴みルール、Wall — 大きさ / 凹み半径 / 回復速度、
Terrain — ブラシ半径 / 滑らかさ / 容量、Decor — 形のパラメータ `Log Events` を on にすると client log にイベントごと 1 行出ます

## サポート
BOOTH のショップページからメッセージで: SDK のバージョン、やったこと、期待した動き、`[Mochi]` / `[Wall]` / `[Terrain]` / `[SDF]` の log 行 (Log Events を on に)
新規 Worlds project で再現できるものから優先して直します

ALICE-SDF (https://github.com/ext-sakamoro/ALICE-SDF) 製 LICENSE.txt 参照
";
        }
    }
}
