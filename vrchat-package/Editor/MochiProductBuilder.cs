// =============================================================================
// MochiProductBuilder.cs - "ALICE-SDF/Build Mochi Product"
// =============================================================================
// Generates the stand-alone AliceMochi product (the BOOTH unitypackage) from
// the Mochi sample, so the sample stays the single source of the law:
//
//   Assets/AliceMochi/
//     Scripts/AliceMochi.cs          SampleMochi_Collider renamed (no AliceSDF dependency)
//     Shaders/AliceMochi.shader      SampleMochi_Raymarcher renamed (already stand-alone)
//     Materials/{Mochi,Slime,Water}.mat
//     Prefabs/AliceMochi {Mochi,Slime,Water}.prefab   volume cube + collider, drop in and build
//     AliceMochi_UdonProgram.asset
//     README.md / README_JP.md / LICENSE.txt
//
// then exports it to <repo>/vrchat-package/Product~/AliceMochi_<version>.unitypackage.
// Re-run after any change to the sample; never edit the generated folder.
//
// Author: Moroya Sakamoto
// =============================================================================

using System.IO;
using UnityEditor;
using UnityEngine;

namespace AliceSDF.Editor
{
    public static class MochiProductBuilder
    {
        // A freshly written script compiles only after a domain reload, so the
        // build is two phases: write the files, reload, then materials /
        // prefabs / export (Resume runs phase 2 when this flag is set)
        private const string PendingKey = "AliceSDF.MochiProduct.pending";

        [InitializeOnLoadMethod]
        private static void Resume()
        {
            if (!EditorPrefs.GetBool(PendingKey, false)) return;
            NextTick(() =>
            {
                if (EditorApplication.isCompiling) { Resume(); return; }
                EditorPrefs.SetBool(PendingKey, false);
                string path = BuildCore();
                if (!string.IsNullOrEmpty(path)) Debug.Log("[ALICE-SDF] Mochi product phase 2 done: " + path);
            });
        }

        private const string PackageRoot = "Packages/com.alice.sdf";
        private const string SampleDir = PackageRoot + "/Samples~/SDF Gallery/SampleMochi";
        private const string Out = "Assets/AliceMochi";

        private struct Preset
        {
            public string name;
            public Color mochi, highlight, ground, detail;
            public float blendK, groundK, density, gravity;
        }

        private static readonly Preset[] Presets =
        {
            new Preset { name = "Mochi", mochi = new Color(0.96f, 0.93f, 0.88f), highlight = new Color(0.99f, 0.96f, 0.92f),
                         ground = new Color(0.55f, 0.46f, 0.36f), detail = new Color(0.50f, 0.42f, 0.33f),
                         blendK = 0.5f, groundK = 0.15f, density = 1000f, gravity = 3f },
            new Preset { name = "Slime", mochi = new Color(0.45f, 0.85f, 0.35f), highlight = new Color(0.70f, 1.0f, 0.55f),
                         ground = new Color(0.30f, 0.32f, 0.30f), detail = new Color(0.26f, 0.28f, 0.26f),
                         blendK = 0.9f, groundK = 0.3f, density = 600f, gravity = 2f },
            new Preset { name = "Water", mochi = new Color(0.55f, 0.75f, 1.0f), highlight = new Color(0.85f, 0.95f, 1.0f),
                         ground = new Color(0.75f, 0.72f, 0.62f), detail = new Color(0.70f, 0.66f, 0.56f),
                         blendK = 0.3f, groundK = 0.1f, density = 1000f, gravity = 5f },
        };

        // Run once on the next editor update (delayCall proved unreliable across
        // a UdonSharp compile)
        private static void NextTick(System.Action action)
        {
            EditorApplication.CallbackFunction once = null;
            once = () => { EditorApplication.update -= once; action(); };
            EditorApplication.update += once;
        }

        [MenuItem("ALICE-SDF/Build Mochi Product")]
        public static void Build()
        {
            string path = BuildCore();
            EditorUtility.DisplayDialog("ALICE-SDF Mochi Product", "Exported:\n" + path, "OK");
        }

        // Headless: -executeMethod AliceSDF.Editor.MochiProductBuilder.BuildBatch
        // (run it twice on a project that has never built the product: the first
        // run writes and compiles the script, the second exports)
        public static void BuildBatch()
        {
            if (BuildCore() == null) EditorApplication.Exit(2);
        }

        public static string BuildCore()
        {
            string packageFull = Path.GetFullPath(PackageRoot);
            string sampleFull = Path.Combine(packageFull, "Samples~", "SDF Gallery", "SampleMochi");
            if (!Directory.Exists(sampleFull))
                throw new DirectoryNotFoundException("Mochi sample not found: " + sampleFull);

            foreach (var d in new[] { Out, Out + "/Scripts", Out + "/Shaders", Out + "/Materials", Out + "/Prefabs" })
                if (!AssetDatabase.IsValidFolder(d)) Directory.CreateDirectory(d);

            // --- Script: rename namespace / class, drop the sample header link ---
            string cs = File.ReadAllText(Path.Combine(sampleFull, "SampleMochi_Collider.cs"));
            cs = cs.Replace("namespace AliceSDF.Samples", "namespace AliceMochi")
                   .Replace("SampleMochi_Collider", "AliceMochi")
                   .Replace("// ALICE-SDF Sample: Mochi Collider & Interaction (UdonSharp)",
                            "// AliceMochi - generated from the ALICE-SDF Mochi sample by MochiProductBuilder; do not edit here");
            string csPath = Out + "/Scripts/AliceMochi.cs";
            bool scriptChanged = !File.Exists(csPath) || File.ReadAllText(csPath) != cs;
            File.WriteAllText(csPath, cs);

            // --- Shader: rename; it is stand-alone (no package include) ---
            string sh = File.ReadAllText(Path.Combine(sampleFull, "SampleMochi_Raymarcher.shader"));
            if (sh.Contains("Packages/com.alice.sdf"))
                throw new System.InvalidOperationException("Mochi shader includes the package: the product must be stand-alone");
            sh = sh.Replace("Shader \"AliceSDF/Samples/Mochi\"", "Shader \"AliceMochi/Mochi\"");
            File.WriteAllText(Out + "/Shaders/AliceMochi.shader", sh);

            // --- Docs / license ---
            File.WriteAllText(Out + "/README.md", ReadmeEn());
            File.WriteAllText(Out + "/README_JP.md", ReadmeJp());
            string lic = Path.Combine(packageFull, "LICENSE.md");
            if (File.Exists(lic)) File.Copy(lic, Out + "/LICENSE.txt", true);

            var type = FindType("AliceMochi.AliceMochi");
            if (type == null || scriptChanged)
            {
                // Phase 1: the script is new, let Unity compile it; Resume continues
                EditorPrefs.SetBool(PendingKey, true);
                AssetDatabase.Refresh();
                Debug.Log("[ALICE-SDF] Mochi product phase 1: files written, compiling (phase 2 follows the reload)");
                return null;
            }
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var shader = Shader.Find("AliceMochi/Mochi");
            if (shader == null) throw new System.InvalidOperationException("AliceMochi/Mochi shader did not compile");

#if UDONSHARP
            // Program asset for the product script (the prefab needs it)
            string programPath = Out + "/AliceMochi_UdonProgram.asset";
            if (AssetDatabase.LoadAssetAtPath<UdonSharp.UdonSharpProgramAsset>(programPath) == null)
            {
                var script = AssetDatabase.LoadAssetAtPath<MonoScript>(Out + "/Scripts/AliceMochi.cs");
                var asset = ScriptableObject.CreateInstance<UdonSharp.UdonSharpProgramAsset>();
                asset.sourceCsScript = script;
                AssetDatabase.CreateAsset(asset, programPath);
                AssetDatabase.SaveAssets();
                UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
                // The proxy's script version is registered on a later editor tick
                // (CopyProxyToUdon throws "outdated script version" in this one):
                // phase 2 ends here, phase 3 (materials / prefabs / export) follows
                Debug.Log("[ALICE-SDF] Mochi product phase 2: program asset compiled, prefabs follow");
                NextTick(() =>
                {
                    string p = BuildCore();
                    if (p != null) Debug.Log("[ALICE-SDF] Mochi product phase 3 done: " + p);
                });
                return null;
            }
#endif

            // --- Materials + prefabs per preset ---
            var pending = new System.Collections.Generic.List<GameObject>();
            foreach (var p in Presets)
            {
                string matPath = Out + "/Materials/" + p.name + ".mat";
                var mat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
                if (mat == null)
                {
                    mat = new Material(shader);
                    AssetDatabase.CreateAsset(mat, matPath);
                }
                mat.shader = shader;
                mat.SetColor("_MochiColor", p.mochi);
                mat.SetColor("_MochiColor2", p.highlight);
                mat.SetColor("_GroundColor", p.ground);
                mat.SetColor("_GroundColor2", p.detail);
                mat.SetFloat("_BlendK", p.blendK);
                mat.SetFloat("_GroundK", p.groundK);
                EditorUtility.SetDirty(mat);

                var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.name = "AliceMochi " + p.name;
                go.transform.position = new Vector3(0, 1, 0);
                go.transform.localScale = new Vector3(4, 2, 4);
                Object.DestroyImmediate(go.GetComponent<BoxCollider>());
                go.GetComponent<MeshRenderer>().sharedMaterial = mat;
                var comp = go.AddComponent(type);
                var so = new SerializedObject(comp);
                so.FindProperty("applyColors").boolValue = true;
                so.FindProperty("mochiColor").colorValue = p.mochi;
                so.FindProperty("mochiHighlight").colorValue = p.highlight;
                so.FindProperty("groundColor").colorValue = p.ground;
                so.FindProperty("groundDetail").colorValue = p.detail;
                so.FindProperty("useCustomLayout").boolValue = true;
                so.FindProperty("initialCount").intValue = 5;
                so.FindProperty("initialRadius").floatValue = 0.3f;
                so.FindProperty("ringRadius").floatValue = 0.7f;
                so.FindProperty("blendK").floatValue = p.blendK;
                so.FindProperty("groundK").floatValue = p.groundK;
                so.FindProperty("mochiDensity").floatValue = p.density;
                so.FindProperty("gravity").floatValue = p.gravity;
                so.ApplyModifiedPropertiesWithoutUndo();
                pending.Add(go);
            }
            AssetDatabase.SaveAssets();

#if UDONSHARP
            // A UdonSharp compile right before asking for the backing behaviours:
            // in the call that created or rewrote the program asset,
            // CreateBehaviourForProxy throws (outdated script version / null
            // key) until a compile has registered the script version
            UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
            foreach (var go in pending)
            {
                var proxy = go.GetComponent<UdonSharp.UdonSharpBehaviour>();
                if (proxy != null)
                {
                    if (UdonSharpEditor.UdonSharpEditorUtility.GetBackingUdonBehaviour(proxy) == null)
                        UdonSharpEditor.UdonSharpEditorUtility.CreateBehaviourForProxy(proxy);
                    UdonSharpEditor.UdonSharpEditorUtility.CopyProxyToUdon(proxy);
                }
                PrefabUtility.SaveAsPrefabAsset(go, Out + "/Prefabs/" + go.name + ".prefab");
                Object.DestroyImmediate(go);
            }
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return Export(packageFull);
#else
            foreach (var go in pending)
            {
                PrefabUtility.SaveAsPrefabAsset(go, Out + "/Prefabs/" + go.name + ".prefab");
                Object.DestroyImmediate(go);
            }
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            return Export(packageFull);
#endif
        }

        private static string Export(string packageFull)
        {

            string version = PackageVersion(packageFull);
            string outDir = Path.Combine(packageFull, "Product~");
            Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, "AliceMochi_" + version + ".unitypackage");
            AssetDatabase.ExportPackage(Out, outPath, ExportPackageOptions.Recurse);
            Debug.Log("[ALICE-SDF] Mochi product exported: " + outPath);
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
@"# AliceMochi — squishy mochi blobs for your VRChat world

Soft blobs you can grab, pull apart, push together and walk into, in VR and on desktop.
One prefab, no other packages: drop it in and build.

## Install (3 steps)
1. Import `AliceMochi_<version>.unitypackage` into a VRChat **Worlds** project (SDK 3.7 or newer, UdonSharp is part of the SDK).
2. Drag `Assets/AliceMochi/Prefabs/AliceMochi Mochi.prefab` (or Slime / Water) into your scene.
   Put it where you want the mochis: the bottom face of the cube is their ground (the prefab can sit at any position or height).
   The cube is the volume they can move in; scale it to the area you want. Keep a floor collider at that ground level.
3. Build & Test. PC only (the raymarching shader is not built for Quest).

## Play
| Action | VR | Desktop |
|---|---|---|
| Grab / move | hand inside a mochi | hold left click on it, move the view |
| Split | grip while holding | right click while holding |
| Merge | push two mochis together | same |
| Walk in | your body dents it and pushes it away | same |

## Tune (Inspector of the AliceMochi component, no material editing needed)
| Section | Knob | What it does |
|---|---|---|
| Look | Mochi Color / Highlight / Ground Color / Ground Detail | colours (Apply Colors on) |
| Mochis | Initial Count / Initial Radius / Ring Radius | how many, how big, how they start |
| Mochi Settings | Blend K (stickiness) / Ground K / Min Radius / Gravity | feel of the blobs |
| Player Body | Player Radius / Player Mass / Mochi Density / Dent K | how your body presses in and how much a mochi gives way |
| Interaction | Grab Threshold / Grab Dwell / Split On Pull / Release Distance / Merge Threshold | VR hand rules |
| Debug | Log Events | one `[Mochi] ...` line per event in the client log |

Everyone in the instance sees the same mochis (synced, owner-authoritative). Up to 16 mochis.

## Support
Message the shop page on BOOTH with: your SDK version, what you did, what you expected, and the Console / client log lines
starting with `[Mochi]` (turn on Log Events). Reproductions in a fresh Worlds project get fixed first.

Built with ALICE-SDF (https://github.com/ext-sakamoro/ALICE-SDF). See LICENSE.txt.
";
        }

        private static string ReadmeJp()
        {
            return
@"# AliceMochi — VRChat ワールド用のやわらかい餅

掴む・千切る・くっつける・体で押す、VR でもデスクトップでも動く餅の塊です
prefab 1 個、他の package 不要: 置いてビルドするだけ

## 導入 (3 step)
1. VRChat **Worlds** project (SDK 3.7 以降、UdonSharp は SDK 同梱) に `AliceMochi_<version>.unitypackage` を import
2. `Assets/AliceMochi/Prefabs/AliceMochi Mochi.prefab` (または Slime / Water) を scene に drag
   餅を出したい場所に置いてください: Cube の底面が餅の地面です (prefab はどの位置・高さでも可)
   Cube は餅が動ける範囲 (volume) なので置きたい広さに scale、その地面の高さに床 collider を置いてください
3. Build & Test PC 専用 (レイマーチングシェーダーは Quest 向けにビルドしていません)

## 遊び方
| 操作 | VR | デスクトップ |
|---|---|---|
| 掴む / 動かす | 手を餅の中に入れる | 餅の上で左クリック押しっぱなし、視点を動かす |
| 分裂 | 持ったままグリップ | 持ったまま右クリック |
| 合体 | 餅同士を押し合わせる | 同じ |
| 体当たり | 体の形に凹んで餅が逃げる | 同じ |

## 調整 (AliceMochi コンポーネントの Inspector だけ、material は触らなくてよい)
| 区分 | つまみ | 効果 |
|---|---|---|
| Look | Mochi Color / Highlight / Ground Color / Ground Detail | 色 (Apply Colors on) |
| Mochis | Initial Count / Initial Radius / Ring Radius | 最初の個数・大きさ・並び |
| Mochi Settings | Blend K (粘り) / Ground K / Min Radius / Gravity | 餅の手触り |
| Player Body | Player Radius / Player Mass / Mochi Density / Dent K | 体の食い込みと餅の逃げ方 |
| Interaction | Grab Threshold / Grab Dwell / Split On Pull / Release Distance / Merge Threshold | VR の手のルール |
| Debug | Log Events | イベントごとに client log へ `[Mochi] ...` 1 行 |

インスタンス内の全員が同じ餅を見ます (同期、owner 権威) 餅は最大 16 個

## サポート
BOOTH のショップページからメッセージで: SDK のバージョン、やったこと、期待した動き、Console / client log の `[Mochi]` で始まる行 (Log Events を on に)
新規 Worlds project で再現できるものから優先して直します

ALICE-SDF (https://github.com/ext-sakamoro/ALICE-SDF) 製 LICENSE.txt 参照
";
        }
    }
}
