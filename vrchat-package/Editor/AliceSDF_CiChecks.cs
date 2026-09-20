// =============================================================================
// AliceSDF_CiChecks.cs - headless checks for the Unity side of the package
// =============================================================================
// What the Rust / dotnet CI cannot see: do the shaders compile (for the
// editor platform and, when Unity runs with -buildTarget Android, for
// GLES3), do the scripts and the UdonSharp programs compile, do the sample
// scenes generate with a descriptor / spawn / PipelineManager, does the Kit
// product build and export with every prefab, program, shader and texture.
//
// Batch entry (one Unity invocation per stage, the stage from the
// ALICE_CI_STAGE environment variable; a script loops the "kit" stage until
// it reports done because the product build crosses domain reloads):
//
//   unity -batchmode -nographics -projectPath <p> -quit
//         -executeMethod AliceSDF.Editor.AliceSDF_CiChecks.RunBatch
//
//   stages: setup | compile | samples | scenes | kit | verify | android
//   (setup first in a fresh project: it writes the scripting defines the
//   SDK and UdonSharp add on editor ticks, which never run headless)
//   result: exit code 0 = pass (or "run the kit stage again", see the
//           marker file Library/alice_ci_kit.txt), 1 = fail; every finding is
//           logged as [ALICE-CI] ... so the log is the report
//
// Menu: ALICE-SDF/Run CI Checks runs compile + scenes + verify in the editor.
//
// Author: Moroya Sakamoto
// =============================================================================

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace AliceSDF.Editor
{
    public static class AliceSDF_CiChecks
    {
        private const string Tag = "[ALICE-CI]";
        // Under the project (Library/, which survives -quit; Temp/ is wiped on exit),
        // not the process working directory (Unity is started
        // from its own Editor folder so the shader compiler finds its includes)
        private static string KitMarker { get { return Path.Combine(Application.dataPath, "..", "Library", "alice_ci_kit.txt"); } }

        private static readonly string[] SampleShaders =
        {
            "AliceSDF/Samples/Basic", "AliceSDF/Samples/Cosmic", "AliceSDF/Samples/Fractal", "AliceSDF/Samples/Mix",
            "AliceSDF/Samples/DeformableWall", "AliceSDF/Samples/Mochi", "AliceSDF/Samples/TerrainSculpt",
        };
        private static readonly string[] SampleNames = { "Basic", "Cosmic", "Fractal", "Mix", "DeformableWall", "Mochi", "TerrainSculpt" };

        [MenuItem("ALICE-SDF/Run CI Checks")]
        public static void RunMenu()
        {
            int fails = CheckCompile() + CheckShaders("editor") + CheckScenes() + CheckKit();
            Debug.Log(Tag + (fails == 0 ? " all checks passed" : " " + fails + " check(s) FAILED"));
        }

        public static void RunBatch()
        {
            string stage = System.Environment.GetEnvironmentVariable("ALICE_CI_STAGE") ?? "compile";
            int fails;
            switch (stage)
            {
                case "setup": fails = SetupDefines(); break;
                case "compile": fails = CheckCompile() + CheckShaders("editor"); break;
                case "samples": fails = ImportSamples(); break;
                case "scenes": fails = GenerateScenes() + CheckScenes(); break;
                case "kit": fails = BuildKitStage(); break;
                case "verify": fails = CheckCompile() + CheckShaders("editor") + CheckScenes() + CheckKit(); break;
                case "android": fails = CheckCompile() + CheckShaders("android"); break;
                default: Debug.LogError(Tag + " unknown stage " + stage); fails = 1; break;
            }
            Debug.Log(Tag + " stage " + stage + (fails == 0 ? " passed" : " FAILED (" + fails + ")"));
            EditorApplication.Exit(fails == 0 ? 0 : 1);
        }

        // --- setup: the project defines the SDK and UdonSharp normally add -----------
        // UDON / VRC_SDK_VRCSDK3 come from EnvConfig.SetActiveSDKDefines and
        // UDONSHARP from UdonSharpEditorManager, both on EditorApplication.update
        // in an interactive editor. A fresh CI project never runs those ticks,
        // the samples' `#if UDONSHARP` then picks the MonoBehaviour branch and
        // no scene gets an UdonBehaviour. Written for Standalone and Android
        // (the android stage switches target); the next invocation compiles
        // with them.
        private static readonly string[] RequiredDefines = { "UDON", "VRC_SDK_VRCSDK3", "UDONSHARP" };

        private static int SetupDefines()
        {
#if UDONSHARP
            var env = System.Type.GetType("VRC.Editor.EnvConfig, VRC.SDKBase.Editor");
            var m = env != null ? env.GetMethod("SetActiveSDKDefines", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static) : null;
            if (m != null) m.Invoke(null, null); else Debug.LogWarning(Tag + " EnvConfig.SetActiveSDKDefines not found; defines written directly");
            foreach (var group in new[] { BuildTargetGroup.Standalone, BuildTargetGroup.Android })
            {
                var defines = new List<string>(PlayerSettings.GetScriptingDefineSymbolsForGroup(group).Split(';'));
                defines.RemoveAll(string.IsNullOrEmpty);
                bool changed = false;
                foreach (var d in RequiredDefines)
                    if (!defines.Contains(d)) { defines.Add(d); changed = true; }
                if (changed) PlayerSettings.SetScriptingDefineSymbolsForGroup(group, string.Join(";", defines));
                Debug.Log(Tag + " defines " + group + (changed ? " updated: " : " ok: ") + string.Join(";", defines));
            }
            AssetDatabase.SaveAssets();
            return 0;
#else
            Debug.LogError(Tag + " setup: com.vrchat.worlds is not in the project (UDONSHARP undefined)");
            return 1;
#endif
        }

        private static int CheckDefines()
        {
            int fails = 0;
            var group = BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget);
            var defines = new List<string>(PlayerSettings.GetScriptingDefineSymbolsForGroup(group).Split(';'));
            foreach (var d in RequiredDefines)
                if (!defines.Contains(d)) { Debug.LogError(Tag + " scripting define " + d + " missing for " + group + " (run the setup stage first)"); fails++; }
            return fails;
        }

        // --- compile: no C# errors, U# compiles ---------------------------------
        private static int CheckCompile()
        {
            int fails = CheckDefines();
            if (EditorUtility.scriptCompilationFailed) { Debug.LogError(Tag + " C# compilation failed (see the errors above)"); fails++; }
#if UDONSHARP
            // CompileSync reports errors only through the log, so count them
            // there; every program with an error stays uncompiled otherwise
            int usErrors = 0;
            Application.LogCallback onLog = (msg, stack, type) =>
            {
                if ((type == LogType.Error || type == LogType.Exception) && (msg.Contains("UdonSharp") || msg.Contains("error CS"))) usErrors++;
            };
            Application.logMessageReceived += onLog;
            try
            {
                PrimeUdonSharpCaches();
                UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
            }
            catch (System.Exception e) { Debug.LogError(Tag + " UdonSharp compile threw: " + e.Message); fails++; }
            finally { Application.logMessageReceived -= onLog; }
            if (usErrors > 0) { Debug.LogError(Tag + " UdonSharp compile: " + usErrors + " error(s)"); fails++; }
            else Debug.Log(Tag + " UdonSharp CompileSync done, no errors");
#else
            Debug.LogWarning(Tag + " UDONSHARP not defined (VRChat SDK absent): UdonSharp compile skipped");
#endif
            return fails;
        }

#if UDONSHARP
        // Headless (-batchmode -quit) the U# compiler's lazy assembly cache is
        // first touched from its worker thread, where AssetDatabase.FindAssets
        // throws ("GetAllRegisteredPackages can only be called from the main
        // thread") and every script then fails to resolve UnityEngine. The
        // editor primes the cache on the main thread as a side effect of its
        // UI; do the same here through reflection (the methods are internal).
        private static void PrimeUdonSharpCaches()
        {
            var t = typeof(UdonSharp.Compiler.UdonSharpCompilerV1).Assembly.GetType("UdonSharp.Compiler.Udon.CompilerUdonInterface");
            if (t == null) { Debug.LogWarning(Tag + " CompilerUdonInterface not found; U# cache not primed"); return; }
            const System.Reflection.BindingFlags F = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static;
            foreach (var name in new[] { "AssemblyCacheInit", "CacheInit" })
            {
                var m = t.GetMethod(name, F);
                if (m == null) Debug.LogWarning(Tag + " CompilerUdonInterface." + name + " not found");
                else m.Invoke(null, null);
            }
        }
#endif

        // --- shaders: every package / sample / kit shader without errors -----------
        // Shaders are compiled for the active build target when imported, so
        // the "android" label only names what the caller started Unity with
        // (-buildTarget Android); the check itself is the same.
        private static int CheckShaders(string label)
        {
            int fails = 0, checkedCount = 0;
            foreach (var guid in AssetDatabase.FindAssets("t:Shader", new[] { "Assets", "Packages/com.alice.sdf" }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                var sh = AssetDatabase.LoadAssetAtPath<Shader>(path);
                if (sh == null) continue;
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
                sh = AssetDatabase.LoadAssetAtPath<Shader>(path);
                checkedCount++;
                if (ShaderUtil.ShaderHasError(sh))
                {
                    fails++;
                    foreach (var m in ShaderUtil.GetShaderMessages(sh))
                        if (m.severity == UnityEditor.Rendering.ShaderCompilerMessageSeverity.Error)
                            Debug.LogError(Tag + " shader error " + sh.name + " (" + path + ") L" + m.line + ": " + m.message + " [" + m.platform + "]");
                }
            }
            foreach (var n in SampleShaders)
                if (Shader.Find(n) == null) Debug.LogWarning(Tag + " sample shader not present (samples not imported?): " + n);
            Debug.Log(Tag + " shaders (" + label + ", target " + EditorUserBuildSettings.activeBuildTarget + "): " + checkedCount + " checked, " + fails + " with errors");
            return fails;
        }

        // --- samples / scenes -----------------------------------------------------
        private static int ImportSamples()
        {
            try { SampleSceneGenerator.ImportAllSamplesBatchNoExit(); return 0; }
            catch (System.Exception e) { Debug.LogError(Tag + " import samples: " + e.Message); return 1; }
        }

        private static int GenerateScenes()
        {
            try { SampleSceneGenerator.GenerateAllBatchNoExit(); return 0; }
            catch (System.Exception e) { Debug.LogError(Tag + " generate scenes: " + e.Message); return 1; }
        }

        private static int CheckScenes()
        {
            int fails = 0;
            foreach (var n in SampleNames)
            {
                string path = "Assets/AliceSDF_SampleScenes/SDF_" + n + ".unity";
                if (!File.Exists(path)) { Debug.LogError(Tag + " scene missing: " + path); fails++; continue; }
                UnityEditor.SceneManagement.EditorSceneManager.OpenScene(path);
                var vol = GameObject.Find("SDF_" + n);
                var rend = vol != null ? vol.GetComponent<MeshRenderer>() : null;
                string why = "";
#if UDONSHARP
                var desc = Object.FindObjectOfType<VRC.SDK3.Components.VRCSceneDescriptor>();
                var pm = Object.FindObjectOfType<VRC.Core.PipelineManager>();
                if (desc == null) why += " no VRCSceneDescriptor;";
                else if (desc.spawns == null || desc.spawns.Length == 0 || desc.spawns[0] == null) why += " no spawn;";
                if (pm == null) why += " no PipelineManager;";
#endif
                if (vol == null) why += " no SDF_" + n + " volume;";
                else if (rend == null || rend.sharedMaterial == null || rend.sharedMaterial.shader == null || ShaderUtil.ShaderHasError(rend.sharedMaterial.shader)) why += " volume material / shader broken;";
#if UDONSHARP
                if (vol != null && vol.GetComponent<VRC.Udon.UdonBehaviour>() == null) why += " no UdonBehaviour on the volume;";
                else if (vol != null)
                {
                    var ub = vol.GetComponent<VRC.Udon.UdonBehaviour>();
                    if (ub.programSource == null) why += " UdonBehaviour has no program source;";
                    else
                    {
                        var ps = ub.programSource as UdonSharp.UdonSharpProgramAsset;
                        if (ps != null && ps.GetSerializedUdonProgramAsset() == null) why += " program not compiled;";
                    }
                }
#endif
                if (why.Length > 0) { Debug.LogError(Tag + " scene " + n + ":" + why); fails++; }
            }
            Debug.Log(Tag + " scenes: " + SampleNames.Length + " checked, " + fails + " bad");
            return fails;
        }

        // --- kit: build across domain reloads, then verify ---------------------------
        // Returns 0 and writes "again" to the marker when another run is needed,
        // 0 + "done" when the package was exported, 1 on failure.
        private static int BuildKitStage()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(KitMarker));
            KitProductBuilder.ClearPending();
            try
            {
                string path = KitProductBuilder.BuildCore();
                KitProductBuilder.ClearPending();
                File.WriteAllText(KitMarker, path == null ? "again" : "done " + path);
                Debug.Log(Tag + " kit stage: " + (path == null ? "more runs needed (domain reload)" : "exported " + path));
                return 0;
            }
            catch (System.Exception e)
            {
                KitProductBuilder.ClearPending();
                File.WriteAllText(KitMarker, "failed");
                Debug.LogError(Tag + " kit build threw: " + e);
                return 1;
            }
        }

        private static int CheckKit()
        {
            int fails = 0;
            const string Out = "Assets/AliceSDFKit";
            if (!AssetDatabase.IsValidFolder(Out)) { Debug.LogError(Tag + " kit folder missing: " + Out + " (run the kit stage)"); return 1; }
            foreach (var name in new[] { "AliceSDF_Collider", "AliceMochi", "AliceWall", "AliceTerrain", "AliceDecorBasic", "AliceDecorCosmic", "AliceDecorFractal", "AliceDecorMix" })
            {
                if (!File.Exists(Out + "/Scripts/" + name + ".cs")) { Debug.LogError(Tag + " kit script missing: " + name); fails++; }
#if UDONSHARP
                var asset = AssetDatabase.LoadAssetAtPath<UdonSharp.UdonSharpProgramAsset>(Out + "/Programs/" + name + "_UdonProgram.asset");
                if (asset == null || asset.SerializedProgramAsset == null) { Debug.LogError(Tag + " kit program missing / not compiled: " + name); fails++; }
#endif
            }
            foreach (var name in new[] { "AliceSDF Mochi (Mochi)", "AliceSDF Mochi (Slime)", "AliceSDF Mochi (Water)", "AliceSDF Mochi (Sesame)", "AliceSDF Wall", "AliceSDF Terrain", "AliceSDF Decor Basic", "AliceSDF Decor Cosmic", "AliceSDF Decor Fractal", "AliceSDF Decor Mix" })
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(Out + "/Prefabs/" + name + ".prefab");
                if (prefab == null) { Debug.LogError(Tag + " kit prefab missing: " + name); fails++; continue; }
                var vol = prefab.transform.Find("Volume");
                if (vol == null) { Debug.LogError(Tag + " kit prefab has no Volume: " + name); fails++; continue; }
                var r = vol.GetComponent<MeshRenderer>();
                if (r == null || r.sharedMaterial == null || r.sharedMaterial.shader == null || ShaderUtil.ShaderHasError(r.sharedMaterial.shader)) { Debug.LogError(Tag + " kit prefab material / shader broken: " + name); fails++; }
                if (r != null && r.shadowCastingMode != UnityEngine.Rendering.ShadowCastingMode.Off) { Debug.LogError(Tag + " kit Volume casts shadows: " + name); fails++; }
#if UDONSHARP
                var ub = prefab.GetComponentInChildren<VRC.Udon.UdonBehaviour>();
                if (ub == null || ub.programSource == null) { Debug.LogError(Tag + " kit prefab has no backing UdonBehaviour / program: " + name); fails++; }
#endif
                if (name == "AliceSDF Terrain" && prefab.transform.Find("TerrainSupport") == null) { Debug.LogError(Tag + " kit Terrain prefab has no TerrainSupport"); fails++; }
            }
            foreach (var name in new[] { "Mochi", "Wall", "Terrain", "DecorBasic", "DecorCosmic", "DecorFractal", "DecorMix" })
            {
                var sh = Shader.Find("AliceSDFKit/" + name);
                if (sh == null || ShaderUtil.ShaderHasError(sh)) { Debug.LogError(Tag + " kit shader missing / broken: AliceSDFKit/" + name); fails++; }
            }
            foreach (var tex in new[] { "MochiSesame.png", "GroundWeave.png" })
                if (AssetDatabase.LoadAssetAtPath<Texture2D>(Out + "/Textures/" + tex) == null) { Debug.LogError(Tag + " kit texture missing: " + tex); fails++; }
            foreach (var inc in new[] { "AliceSDF_Include.cginc", "AliceSDF_LOD.cginc" })
                if (!File.Exists(Out + "/Shaders/" + inc)) { Debug.LogError(Tag + " kit include missing: " + inc); fails++; }
            string readme = Out + "/README.md";
            if (!File.Exists(readme) || !File.Exists(Out + "/README_JP.md")) { Debug.LogError(Tag + " kit README missing"); fails++; }
            Debug.Log(Tag + " kit: " + fails + " problem(s)");
            return fails;
        }
    }
}
