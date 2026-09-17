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

        [MenuItem("ALICE-SDF/Generate Sample Scenes")]
        public static void GenerateAll()
        {
            if (!AssetDatabase.IsValidFolder(OutputFolder))
            {
                AssetDatabase.CreateFolder("Assets", "AliceSDF_SampleScenes");
            }

            int created = 0;
            int skipped = 0;

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
            }

            AssetDatabase.Refresh();

            string msg = $"[ALICE-SDF] Scene generation complete: {created} created, {skipped} skipped.";
            Debug.Log(msg);
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
                    Debug.LogWarning($"[ALICE-SDF] {sample.colliderType} not found; add the *_Collider.cs component to SDF_{sample.name} by hand.");
                else
                    cubeObj.AddComponent(type);
            }

            // --- Info label (world-space canvas) ---
            CreateInfoCanvas(sample.name);

            // Save scene
            EditorSceneManager.SaveScene(scene, scenePath);
            Debug.Log($"[ALICE-SDF] Created scene: {scenePath}");
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
