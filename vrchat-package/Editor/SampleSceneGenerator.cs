// ALICE-SDF: Sample Scene Generator
// Menu: ALICE-SDF > Generate Sample Scenes
// Creates ready-to-play scenes for each imported sample.
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

namespace AliceSDF.Editor
{
    public static class SampleSceneGenerator
    {
        private static readonly string OutputFolder = "Assets/AliceSDF_SampleScenes";

        private struct SampleDef
        {
            public string name;
            public string shaderName;
            public Vector3 cubePos;
            public Vector3 cubeScale;
            public Vector3 camPos;
            public Vector3 camLookAt;
            public Color bgColor;
            // Interactive samples: the UdonSharp behaviour that drives the
            // shader every frame, added to the cube (full type name; the
            // samples compile into Assembly-CSharp, which this editor
            // assembly does not reference, so it is resolved by name)
            public string colliderType;
        }

        private static readonly SampleDef[] Samples = new SampleDef[]
        {
            new SampleDef {
                name       = "Basic",
                shaderName = "AliceSDF/Samples/Basic",
                cubeScale  = Vector3.one * 100f,
                camPos     = new Vector3(5, 4, 8),
                camLookAt  = new Vector3(0, 1, 0),
                bgColor    = new Color(0.01f, 0.01f, 0.02f),
            },
            new SampleDef {
                name       = "Cosmic",
                shaderName = "AliceSDF/Samples/Cosmic",
                cubeScale  = Vector3.one * 200f,
                camPos     = new Vector3(40, 20, 40),
                camLookAt  = Vector3.zero,
                bgColor    = new Color(0.01f, 0.005f, 0.02f),
            },
            new SampleDef {
                name       = "Fractal",
                shaderName = "AliceSDF/Samples/Fractal",
                cubeScale  = Vector3.one * 200f,
                camPos     = new Vector3(30, 15, 30),
                camLookAt  = Vector3.zero,
                bgColor    = new Color(0.005f, 0.005f, 0.015f),
            },
            new SampleDef {
                name       = "Mix",
                shaderName = "AliceSDF/Samples/Mix",
                cubeScale  = Vector3.one * 200f,
                camPos     = new Vector3(20, 10, 25),
                camLookAt  = Vector3.zero,
                bgColor    = new Color(0.02f, 0.01f, 0.03f),
            },
            // Interactive samples (README "Setup (All Interactive Samples)"):
            // the cube is the raymarching bounding volume, its bottom face
            // sits on or under the SDF ground plane at y = 0
            new SampleDef {
                name         = "DeformableWall",
                shaderName   = "AliceSDF/Samples/DeformableWall",
                cubePos      = new Vector3(0, 4, 0),
                cubeScale    = new Vector3(12, 8, 12),
                camPos       = new Vector3(0, 1.6f, 6),
                camLookAt    = new Vector3(0, 1.5f, 0),
                bgColor      = new Color(0.65f, 0.7f, 0.78f),
                colliderType = "AliceSDF.Samples.SampleDeformableWall_Collider",
            },
            new SampleDef {
                name         = "Mochi",
                shaderName   = "AliceSDF/Samples/Mochi",
                cubePos      = new Vector3(0, 1, 0),
                cubeScale    = new Vector3(4, 2, 4),
                camPos       = new Vector3(0, 1.6f, 2.5f),
                camLookAt    = new Vector3(0, 0.3f, 0),
                bgColor      = new Color(0.83f, 0.80f, 0.76f),
                colliderType = "AliceSDF.Samples.SampleMochi_Collider",
            },
            new SampleDef {
                name         = "TerrainSculpt",
                shaderName   = "AliceSDF/Samples/TerrainSculpt",
                cubePos      = new Vector3(0, 3, 0),
                cubeScale    = new Vector3(20, 10, 20),
                camPos       = new Vector3(0, 1.6f, 6),
                camLookAt    = new Vector3(0, 0.5f, 0),
                bgColor      = new Color(0.6f, 0.75f, 0.9f),
                colliderType = "AliceSDF.Samples.SampleTerrainSculpt_Collider",
            },
        };

        // Copies every sample of this package into Assets/Samples/<displayName>/<version>/
        // (what the Package Manager "Import" button does), so the shaders and
        // *_Collider.cs scripts exist before Generate Sample Scenes runs.
        // Already-imported samples are left alone.
        [MenuItem("ALICE-SDF/Import All Samples")]
        public static void ImportAllSamples()
        {
            ImportAllSamplesCore();
        }

        [MenuItem("ALICE-SDF/Import All Samples", true)]
        public static bool ImportAllSamplesValidation()
        {
            return !EditorApplication.isPlaying;
        }

        // Headless: Unity -batchmode -quit -projectPath <p>
        //   -executeMethod AliceSDF.Editor.SampleSceneGenerator.ImportAllSamplesBatch
        // Exit code 1 when the package cannot be found or nothing could be imported.
        // Run this and GenerateAllBatch as two Unity invocations: the imported
        // scripts compile (domain reload) between them.
        public static void ImportAllSamplesBatch()
        {
            int imported = ImportAllSamplesCore();
            if (imported < 0)
                Fail("[ALICE-SDF] Import All Samples failed; see the log above.");
        }

        // Returns the number of samples imported by this call, or -1 on failure.
        private static int ImportAllSamplesCore()
        {
            var package = UnityEditor.PackageManager.PackageInfo.FindForAssembly(
                typeof(SampleSceneGenerator).Assembly);
            if (package == null)
            {
                Debug.LogError("[ALICE-SDF] PackageInfo not found for the AliceSDF.Editor assembly; is com.alice.sdf installed as a package (not copied into Assets/)?");
                return -1;
            }

            int imported = 0;
            int present = 0;
            int failed = 0;
            foreach (var sample in UnityEditor.PackageManager.UI.Sample.FindByPackage(package.name, package.version))
            {
                if (sample.isImported)
                {
                    present++;
                    continue;
                }
                if (sample.Import())
                {
                    Debug.Log($"[ALICE-SDF] Imported sample: {sample.displayName}");
                    imported++;
                }
                else
                {
                    Debug.LogError($"[ALICE-SDF] Failed to import sample: {sample.displayName}");
                    failed++;
                }
            }

            AssetDatabase.Refresh();
            Debug.Log($"[ALICE-SDF] Import All Samples: {imported} imported, {present} already present, {failed} failed ({package.name} {package.version}).");
            return failed == 0 && (imported + present) > 0 ? imported : -1;
        }

        [MenuItem("ALICE-SDF/Generate Sample Scenes")]
        public static void GenerateAll()
        {
            int created = GenerateAllCore(out int skipped);

            string msg = $"[ALICE-SDF] Scene generation complete: {created} created, {skipped} skipped.";
            EditorUtility.DisplayDialog("ALICE-SDF Sample Scenes", msg, "OK");

            // Open the first created scene
            if (created > 0)
            {
                foreach (var sample in Samples)
                {
                    string path = $"{OutputFolder}/SDF_{sample.name}.unity";
                    if (System.IO.File.Exists(path))
                    {
                        EditorSceneManager.OpenScene(path);
                        break;
                    }
                }
            }
        }

        [MenuItem("ALICE-SDF/Generate Sample Scenes", true)]
        public static bool GenerateAllValidation()
        {
            return !EditorApplication.isPlaying;
        }

        // Headless: Unity -batchmode -quit -projectPath <p>
        //   -executeMethod AliceSDF.Editor.SampleSceneGenerator.GenerateAllBatch
        // No dialog, no scene is opened. Exit code 1 when no scene was created
        // (no sample imported) so a script or an agent can tell it went wrong.
        // Samples that are not imported are skipped with a warning, as in the menu.
        public static void GenerateAllBatch()
        {
            int created = GenerateAllCore(out int skipped);
            if (created == 0)
                Fail($"[ALICE-SDF] No sample scene created ({skipped} skipped). Run ImportAllSamplesBatch (or Package Manager > Samples > Import) first.");
        }

        // Builds one scene per imported sample. Returns the number created.
        private static int GenerateAllCore(out int skipped)
        {
            if (!AssetDatabase.IsValidFolder(OutputFolder))
            {
                AssetDatabase.CreateFolder("Assets", "AliceSDF_SampleScenes");
            }

            int created = 0;
            skipped = 0;
            var interactive = new System.Collections.Generic.List<string>();

            foreach (var sample in Samples)
            {
                var shader = Shader.Find(sample.shaderName);
                if (shader == null)
                {
                    Debug.LogWarning(
                        $"[ALICE-SDF] Shader '{sample.shaderName}' not found. " +
                        $"Import the '{sample.name}' sample from Package Manager first.");
                    skipped++;
                    continue;
                }

                string scenePath = $"{OutputFolder}/SDF_{sample.name}.unity";
                BuildScene(sample, shader, scenePath);
                created++;
                if (!string.IsNullOrEmpty(sample.colliderType))
                    interactive.Add(scenePath);
            }

            AssetDatabase.Refresh();
            FinalizeUdonScenes(interactive);
            Debug.Log($"[ALICE-SDF] Scene generation complete: {created} created, {skipped} skipped.");
            return created;
        }

        // UdonSharp gives a UdonSharpBehaviour its backing UdonBehaviour only
        // when a UdonSharpProgramAsset for the script exists and the proxy's
        // script version is current, and it refreshes that on its own editor
        // ticks and when a scene is opened ("has not been fully setup, running
        // setup"). Inside one synchronous menu call a freshly imported sample
        // therefore ended up with the proxy component and no program
        // (2022.3.22f1 + SDK 3.10.1: Mochi had its asset from an earlier scene
        // and worked, DeformableWall / TerrainSculpt did not). So the program
        // asset is created and compiled while the scene is built
        // (EnsureUdonProgramAsset), and here, after every scene is saved, each
        // interactive scene is opened again, which runs UdonSharp's setup, and
        // saved with its backing behaviour.
        private static void FinalizeUdonScenes(System.Collections.Generic.List<string> scenePaths)
        {
#if UDONSHARP
            foreach (var path in scenePaths)
            {
                var scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                foreach (var proxy in Object.FindObjectsOfType<UdonSharp.UdonSharpBehaviour>())
                {
                    if (UdonSharpEditor.UdonSharpEditorUtility.GetBackingUdonBehaviour(proxy) != null) continue;
                    try { UdonSharpEditor.UdonSharpEditorUtility.CreateBehaviourForProxy(proxy); }
                    catch (System.Exception e) { Debug.LogWarning($"[ALICE-SDF] {path}: backing UdonBehaviour not created yet ({e.Message}); UdonSharp will create it when the scene is opened in the Editor."); }
                }
                EditorSceneManager.SaveScene(scene);
                int backed = 0, total = 0;
                foreach (var proxy in Object.FindObjectsOfType<UdonSharp.UdonSharpBehaviour>())
                {
                    total++;
                    if (UdonSharpEditor.UdonSharpEditorUtility.GetBackingUdonBehaviour(proxy) != null) backed++;
                }
                Debug.Log($"[ALICE-SDF] {path}: {backed}/{total} UdonSharp behaviours have a backing UdonBehaviour.");
            }
#endif
        }

        // In batch mode the process exit code is the only thing the caller sees.
        private static void Fail(string message)
        {
            Debug.LogError(message);
            if (Application.isBatchMode)
                EditorApplication.Exit(1);
        }

        private static void BuildScene(SampleDef sample, Shader shader, string scenePath)
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // --- Camera ---
            var camObj = new GameObject("Main Camera");
            var cam = camObj.AddComponent<Camera>();
            camObj.AddComponent<AudioListener>();
            camObj.tag = "MainCamera";
            cam.backgroundColor = sample.bgColor;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.farClipPlane = 500f;
            cam.nearClipPlane = 0.01f;
            cam.fieldOfView = 60f;
            camObj.transform.position = sample.camPos;
            camObj.transform.LookAt(sample.camLookAt);

            // --- Directional Light ---
            var lightObj = new GameObject("Directional Light");
            var light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(1f, 0.95f, 0.9f);
            light.intensity = 1.2f;
            lightObj.transform.rotation = Quaternion.Euler(45, -30, 0);

            // --- SDF Raymarching Surface (Cube) ---
            var cubeObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cubeObj.name = $"SDF_{sample.name}";
            cubeObj.transform.position = sample.cubePos;
            cubeObj.transform.localScale = sample.cubeScale;

            // Remove default collider (SDF handles collision)
            var boxCollider = cubeObj.GetComponent<BoxCollider>();
            if (boxCollider != null)
                Object.DestroyImmediate(boxCollider);

            // Apply SDF shader
            var mat = new Material(shader);
            mat.name = $"SDF_{sample.name}_Mat";

            // Save material as asset
            string matPath = $"{OutputFolder}/SDF_{sample.name}_Mat.mat";
            AssetDatabase.CreateAsset(mat, matPath);
            cubeObj.GetComponent<MeshRenderer>().sharedMaterial = mat;

            // --- Interactive samples: the UdonSharp collider on the cube ---
            // (UdonSharp's editor hooks create the backing UdonBehaviour)
            if (!string.IsNullOrEmpty(sample.colliderType))
            {
                var type = FindType(sample.colliderType);
                if (type == null)
                {
                    Debug.LogWarning($"[ALICE-SDF] {sample.colliderType} not found; add the *_Collider.cs component to SDF_{sample.name} by hand.");
                }
                else
                {
                    var component = cubeObj.AddComponent(type);
                    EnsureUdonProgramAsset(component, sample.name);
                }
            }

            // --- Info label (world-space canvas) ---
            CreateInfoCanvas(sample.name);

            // Save scene
            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"[ALICE-SDF] Created scene: {scenePath}");
        }

        // The UdonSharpProgramAsset UdonSharp needs for this script, created
        // next to the scenes and compiled if the type has none (see
        // FinalizeUdonScenes for why the backing behaviour comes later).
        private static void EnsureUdonProgramAsset(Component component, string sampleName)
        {
#if UDONSHARP
            var proxy = component as UdonSharp.UdonSharpBehaviour;
            if (proxy == null) return;
            if (UdonSharpEditor.UdonSharpEditorUtility.GetUdonSharpProgramAsset(proxy.GetType()) != null) return;
            var asset = ScriptableObject.CreateInstance<UdonSharp.UdonSharpProgramAsset>();
            asset.sourceCsScript = MonoScript.FromMonoBehaviour(proxy);
            string assetPath = $"{OutputFolder}/SDF_{sampleName}_UdonProgram.asset";
            AssetDatabase.CreateAsset(asset, assetPath);
            AssetDatabase.SaveAssets();
            UdonSharp.Compiler.UdonSharpCompilerV1.CompileSync(new UdonSharp.Compiler.UdonSharpCompileOptions());
            Debug.Log($"[ALICE-SDF] Created Udon program asset: {assetPath}");
#endif
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

        private static void CreateInfoCanvas(string sampleName)
        {
            var canvasObj = new GameObject("InfoCanvas");
            var canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 10;
            canvasObj.AddComponent<UnityEngine.UI.CanvasScaler>();

            var textObj = new GameObject("InfoText");
            textObj.transform.SetParent(canvasObj.transform, false);

            var rect = textObj.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 0);
            rect.anchorMax = new Vector2(1, 0.08f);
            rect.offsetMin = new Vector2(10, 5);
            rect.offsetMax = new Vector2(-10, -5);

            var text = textObj.AddComponent<UnityEngine.UI.Text>();
            text.text = $"ALICE-SDF Sample: {sampleName}  |  Shader: AliceSDF/Samples/{sampleName}  |  Polygons: 0  |  Resolution: INFINITE";
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = 16;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = new Color(0.6f, 0.9f, 1f, 0.8f);
        }
    }
}
