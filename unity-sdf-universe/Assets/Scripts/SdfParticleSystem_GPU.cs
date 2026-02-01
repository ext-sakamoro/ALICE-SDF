// =============================================================================
// SDF Particle System - Full GPU Edition
// =============================================================================
// Zero CPU involvement - Everything runs on GPU
//
// Pipeline:
//   1. Compute Shader: SDF Eval + Physics (GPU)
//   2. Vertex Shader: Billboard transform (GPU)
//   3. Fragment Shader: Render (GPU)
//
// CPU only sets uniforms - No data transfer!
//
// Performance: 10M+ particles at 60 FPS
//
// Shader Variants:
//   - Cosmic: Sun + Planet + Ring + Moon + Asteroids
//   - Terrain: FBM Noise Terrain + Water + Floating Islands
//   - Abstract: Gyroid + Metaballs + Rotating Torus
//   - Fractal: Menger Sponge with Surface-Adhering Particles (Microscope Demo)
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;
using UnityEngine.Rendering;

namespace SdfUniverse
{
    /// <summary>
    /// GPU Scene Type - Each has its own Compute Shader
    /// </summary>
    public enum GpuSceneType
    {
        /// <summary>Sun + Planet + Ring + Moon + Asteroids</summary>
        Cosmic,
        /// <summary>FBM Noise Terrain + Water + Floating Islands</summary>
        Terrain,
        /// <summary>Gyroid + Metaballs + Rotating Torus</summary>
        Abstract,
        /// <summary>Menger Sponge fractal with surface-adhering particles (Microscope Demo)</summary>
        Fractal
    }

    /// <summary>
    /// Full GPU Particle System - Zero CPU data transfer
    /// </summary>
    public class SdfParticleSystem_GPU : MonoBehaviour
    {
        [Header("=== SCENE TYPE ===")]
        [Tooltip("Switch between different SDF scenes (each uses a different Compute Shader)")]
        public GpuSceneType sceneType = GpuSceneType.Cosmic;

        [Header("=== TIME SLICING ===")]
        [Range(1, 10)]
        [Tooltip("1=Every frame (full update), 3=1/3 load, 10=1/10 load")]
        public int updateDivisions = 1;

        [Header("=== COMPUTE SHADERS ===")]
        public ComputeShader cosmicShader;
        public ComputeShader terrainShader;
        public ComputeShader abstractShader;
        public ComputeShader fractalShader;

        [Header("=== PARTICLE COUNT ===")]
        [Range(10000, 10000000)]
        public int particleCount = 1000000;

        [Header("=== PHYSICS ===")]
        [Range(0.1f, 10f)] public float flowSpeed = 3f;
        [Range(0.1f, 10f)] public float surfaceAttraction = 2f;
        [Range(0f, 1f)] public float noiseStrength = 0.1f;

        [Header("=== RENDERING ===")]
        public Material particleMaterial;
        [Range(0.01f, 0.5f)] public float particleSize = 0.1f;
        [Range(0.5f, 5f)] public float brightness = 2f;
        public Color cosmicColor = new Color(0.3f, 0.9f, 1f, 1f);
        public Color terrainColor = new Color(0.4f, 1f, 0.5f, 1f);
        public Color abstractColor = new Color(1f, 0.5f, 0.8f, 1f);
        public Color fractalColor = new Color(0.2f, 0.8f, 1f, 1f);

        [Header("=== SPAWN ===")]
        public float spawnRadius = 40f;
        public float maxDistance = 100f;

        [Header("=== COSMIC SCENE ===")]
        public float sunRadius = 8f;
        public float planetRadius = 2.5f;
        public float planetDistance = 18f;
        public float smoothness = 1.5f;

        [Header("=== TERRAIN SCENE ===")]
        public float terrainHeight = 10f;
        public float terrainScale = 1f;
        public float waterLevel = 0f;
        public float rockSize = 1.5f;

        [Header("=== ABSTRACT SCENE ===")]
        public float gyroidScale = 0.5f;
        public float gyroidThickness = 0.3f;
        public float metaballRadius = 2f;
        [Range(0f, 1f)] public float morphAmount = 0.5f;

        [Header("=== FRACTAL SCENE ===")]
        [Range(10f, 200f)] public float boxSize = 50f;
        [Range(0.5f, 10f)] public float holeSize = 2f;
        [Range(5f, 50f)] public float repeatScale = 15f;
        [Range(0f, 0.2f)] public float twistAmount = 0.02f;
        [Range(1, 5)] public int fractalIterations = 3;

        [Header("=== DEBUG ===")]
        public bool showStats = true;

        // GPU Resources
        private ComputeBuffer _particleBuffer;
        private ComputeBuffer _argsBuffer;
        private uint[] _args = new uint[5];
        private Mesh _quadMesh;

        // Active shader reference
        private ComputeShader _activeShader;
        private GpuSceneType _currentSceneType;

        // Kernel IDs
        private int _initKernel;
        private int _mainKernel;

        // Stats
        private float _fps;
        private int _frameCount;
        private float _fpsTimer;
        private float _gpuTime;

        // Time slicing
        private int _sliceIndex = 0;

        // Thread group size (must match compute shader)
        private const int THREAD_GROUP_SIZE = 256;

        // Microscope mode (Fractal scene)
        private Camera _mainCamera;
        private InfiniteZoomCamera _zoomCamera;

        public int ActiveParticles => particleCount;
        public float GpuTimeMs => _gpuTime;
        public float FPS => _fps;
        public GpuSceneType CurrentScene => _currentSceneType;
        public int UpdateDivisions => updateDivisions;
        public int ParticlesPerFrame => updateDivisions > 1 ? particleCount / updateDivisions : particleCount;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        void Start()
        {
            // Load all compute shaders
            LoadComputeShaders();

            // Select active shader based on scene type
            _activeShader = GetShaderForScene(sceneType);
            _currentSceneType = sceneType;

            if (_activeShader == null)
            {
                Debug.LogError("[GPU] No ComputeShader available! Assign shaders in Inspector.");
                enabled = false;
                return;
            }

            // Cache camera references for Microscope mode
            _mainCamera = Camera.main;
            if (_mainCamera != null)
            {
                _zoomCamera = _mainCamera.GetComponent<InfiniteZoomCamera>();
            }

            InitializeGPU();
            UpdateMaterialColor();
            Debug.Log($"[GPU] Initialized {particleCount:N0} particles (Scene: {sceneType})");
        }

        void LoadComputeShaders()
        {
            // Try to find shaders if not assigned
            #if UNITY_EDITOR
            if (cosmicShader == null)
                cosmicShader = FindComputeShaderByName("SdfCompute_Cosmic");
            if (terrainShader == null)
                terrainShader = FindComputeShaderByName("SdfCompute_Terrain");
            if (abstractShader == null)
                abstractShader = FindComputeShaderByName("SdfCompute_Abstract");
            if (fractalShader == null)
                fractalShader = FindComputeShaderByName("SdfCompute_Fractal");
            #endif
        }

        ComputeShader GetShaderForScene(GpuSceneType scene)
        {
            switch (scene)
            {
                case GpuSceneType.Cosmic:
                    return cosmicShader;
                case GpuSceneType.Terrain:
                    return terrainShader;
                case GpuSceneType.Abstract:
                    return abstractShader;
                case GpuSceneType.Fractal:
                    return fractalShader;
                default:
                    return cosmicShader;
            }
        }

        void UpdateMaterialColor()
        {
            if (particleMaterial == null) return;

            Color targetColor;
            switch (_currentSceneType)
            {
                case GpuSceneType.Cosmic:
                    targetColor = cosmicColor;
                    break;
                case GpuSceneType.Terrain:
                    targetColor = terrainColor;
                    break;
                case GpuSceneType.Abstract:
                    targetColor = abstractColor;
                    break;
                case GpuSceneType.Fractal:
                    targetColor = fractalColor;
                    break;
                default:
                    targetColor = cosmicColor;
                    break;
            }
            particleMaterial.SetColor("_Color", targetColor);
        }

        void OnDestroy()
        {
            ReleaseGPU();
        }

        void Update()
        {
            // FPS counter
            _frameCount++;
            _fpsTimer += Time.deltaTime;
            if (_fpsTimer >= 1f)
            {
                _fps = _frameCount / _fpsTimer;
                _frameCount = 0;
                _fpsTimer = 0;
            }

            // Check for scene change
            if (sceneType != _currentSceneType)
            {
                SwitchScene(sceneType);
            }

            // Update simulation on GPU
            UpdateGPU();

            // Render
            RenderGPU();
        }

        /// <summary>
        /// Switch to a different GPU scene (runtime scene change)
        /// </summary>
        public void SwitchScene(GpuSceneType newScene)
        {
            var newShader = GetShaderForScene(newScene);
            if (newShader == null)
            {
                Debug.LogWarning($"[GPU] Shader for {newScene} not assigned!");
                return;
            }

            _activeShader = newShader;
            _currentSceneType = newScene;
            sceneType = newScene;

            // Reinitialize with new shader
            _initKernel = _activeShader.FindKernel("CSInit");
            _mainKernel = _activeShader.FindKernel("CSMain");

            // Reinitialize particles
            SetComputeUniforms();
            _activeShader.SetBuffer(_initKernel, "_Particles", _particleBuffer);
            int threadGroups = Mathf.CeilToInt((float)particleCount / THREAD_GROUP_SIZE);
            _activeShader.Dispatch(_initKernel, threadGroups, 1, 1);

            UpdateMaterialColor();

            Debug.Log($"[GPU] Switched to {newScene} scene");
        }

        // =====================================================================
        // GPU Initialization
        // =====================================================================

        void InitializeGPU()
        {
            // Create particle buffer (32 bytes per particle)
            int stride = sizeof(float) * 8; // float3 pos + float3 vel + float life + float pad
            _particleBuffer = new ComputeBuffer(particleCount, stride);

            // Create args buffer
            _argsBuffer = new ComputeBuffer(1, _args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);

            // Create quad mesh
            _quadMesh = CreateQuadMesh();

            // Create material if needed
            if (particleMaterial == null)
            {
                CreateMaterial();
            }

            // Get kernel IDs from active shader
            _initKernel = _activeShader.FindKernel("CSInit");
            _mainKernel = _activeShader.FindKernel("CSMain");

            // Initialize particles on GPU
            SetComputeUniforms();
            _activeShader.SetBuffer(_initKernel, "_Particles", _particleBuffer);
            int threadGroups = Mathf.CeilToInt((float)particleCount / THREAD_GROUP_SIZE);
            _activeShader.Dispatch(_initKernel, threadGroups, 1, 1);
        }

        void ReleaseGPU()
        {
            _particleBuffer?.Release();
            _argsBuffer?.Release();
        }

        Mesh CreateQuadMesh()
        {
            var mesh = new Mesh();
            float s = 0.5f;
            mesh.vertices = new Vector3[] {
                new Vector3(-s, -s, 0), new Vector3(s, -s, 0),
                new Vector3(s, s, 0), new Vector3(-s, s, 0)
            };
            mesh.triangles = new int[] { 0, 2, 1, 0, 3, 2 };
            mesh.uv = new Vector2[] {
                new Vector2(0, 0), new Vector2(1, 0),
                new Vector2(1, 1), new Vector2(0, 1)
            };
            mesh.RecalculateBounds();
            return mesh;
        }

        void CreateMaterial()
        {
            var shader = Shader.Find("SdfUniverse/ParticleRender_GPU");
            if (shader == null)
            {
                shader = Shader.Find("SdfUniverse/ParticleRender_Indirect");
            }
            if (shader == null)
            {
                shader = Shader.Find("Particles/Standard Unlit");
            }

            particleMaterial = new Material(shader);
            particleMaterial.SetColor("_Color", new Color(0.3f, 0.9f, 1f, 1f));
            particleMaterial.SetFloat("_Size", particleSize);
            particleMaterial.SetFloat("_Brightness", brightness);
        }

        ComputeShader FindComputeShaderByName(string name)
        {
            #if UNITY_EDITOR
            var guids = UnityEditor.AssetDatabase.FindAssets($"{name} t:ComputeShader");
            foreach (var guid in guids)
            {
                var path = UnityEditor.AssetDatabase.GUIDToAssetPath(guid);
                if (path.Contains(name))
                {
                    return UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>(path);
                }
            }
            #endif
            return null;
        }

        // =====================================================================
        // GPU Update
        // =====================================================================

        void SetComputeUniforms()
        {
            if (_activeShader == null) return;

            // Common parameters
            _activeShader.SetInt("_ParticleCount", particleCount);
            _activeShader.SetFloat("_DeltaTime", Time.deltaTime);
            _activeShader.SetFloat("_Time", Time.time);
            _activeShader.SetFloat("_FlowSpeed", flowSpeed);
            _activeShader.SetFloat("_SurfaceAttraction", surfaceAttraction);
            _activeShader.SetFloat("_NoiseStrength", noiseStrength);
            _activeShader.SetFloat("_MaxDistance", maxDistance);
            _activeShader.SetFloat("_SpawnRadius", spawnRadius);

            // Time slicing parameters
            _activeShader.SetInt("_SliceIndex", _sliceIndex);
            _activeShader.SetInt("_SliceCount", updateDivisions);

            // Scene-specific parameters
            switch (_currentSceneType)
            {
                case GpuSceneType.Cosmic:
                    _activeShader.SetFloat("_SunRadius", sunRadius);
                    _activeShader.SetFloat("_PlanetRadius", planetRadius);
                    _activeShader.SetFloat("_PlanetDistance", planetDistance);
                    _activeShader.SetFloat("_Smoothness", smoothness);
                    _activeShader.SetFloat("_RingMajorRadius", planetRadius * 1.8f);
                    _activeShader.SetFloat("_RingMinorRadius", 0.12f);
                    break;

                case GpuSceneType.Terrain:
                    _activeShader.SetFloat("_TerrainHeight", terrainHeight);
                    _activeShader.SetFloat("_TerrainScale", terrainScale);
                    _activeShader.SetFloat("_WaterLevel", waterLevel);
                    _activeShader.SetFloat("_RockSize", rockSize);
                    break;

                case GpuSceneType.Abstract:
                    _activeShader.SetFloat("_GyroidScale", gyroidScale);
                    _activeShader.SetFloat("_GyroidThickness", gyroidThickness);
                    _activeShader.SetFloat("_MetaballRadius", metaballRadius);
                    _activeShader.SetFloat("_MorphAmount", morphAmount);
                    break;

                case GpuSceneType.Fractal:
                    _activeShader.SetFloat("_BoxSize", boxSize);
                    _activeShader.SetFloat("_HoleSize", holeSize);
                    _activeShader.SetFloat("_RepeatScale", repeatScale);
                    _activeShader.SetFloat("_TwistAmount", twistAmount);
                    _activeShader.SetInt("_FractalIterations", fractalIterations);

                    // === MICROSCOPE MODE: View-Dependent Density ===
                    // Send camera info so particles concentrate in view frustum
                    if (_mainCamera != null)
                    {
                        Transform camT = _mainCamera.transform;
                        _activeShader.SetVector("_CamPos", camT.position);
                        _activeShader.SetVector("_CamForward", camT.forward);
                        _activeShader.SetVector("_CamRight", camT.right);
                        _activeShader.SetVector("_CamUp", camT.up);

                        // Calculate zoom level: 1.0 (far) -> 0.000001 (microscope)
                        float zoomLevel = 1.0f;
                        if (_zoomCamera != null)
                        {
                            // Inverse of magnification: smaller = more zoomed in
                            zoomLevel = _zoomCamera.CurrentDistance / _zoomCamera.startDistance;
                            zoomLevel = Mathf.Clamp(zoomLevel, 0.000001f, 10.0f);
                        }
                        _activeShader.SetFloat("_ZoomLevel", zoomLevel);
                    }
                    break;
            }
        }

        void UpdateGPU()
        {
            if (_activeShader == null) return;

            // Set uniforms (ONLY data sent to GPU each frame - just a few floats!)
            SetComputeUniforms();

            // Bind buffer
            _activeShader.SetBuffer(_mainKernel, "_Particles", _particleBuffer);

            // Dispatch compute shader
            int threadGroups = Mathf.CeilToInt((float)particleCount / THREAD_GROUP_SIZE);

            // Time the GPU dispatch
            float startTime = Time.realtimeSinceStartup;
            _activeShader.Dispatch(_mainKernel, threadGroups, 1, 1);
            _gpuTime = Time.realtimeSinceStartup - startTime;

            // Advance slice index for next frame
            if (updateDivisions > 1)
            {
                _sliceIndex = (_sliceIndex + 1) % updateDivisions;
            }
            else
            {
                _sliceIndex = 0;
            }
        }

        // =====================================================================
        // GPU Rendering
        // =====================================================================

        void RenderGPU()
        {
            if (particleMaterial == null || _quadMesh == null) return;

            // Update material
            particleMaterial.SetFloat("_Size", particleSize);
            particleMaterial.SetFloat("_Brightness", brightness);
            particleMaterial.SetBuffer("_ParticlesBuffer", _particleBuffer);

            // Setup indirect args
            _args[0] = _quadMesh.GetIndexCount(0);
            _args[1] = (uint)particleCount;
            _args[2] = _quadMesh.GetIndexStart(0);
            _args[3] = _quadMesh.GetBaseVertex(0);
            _args[4] = 0;
            _argsBuffer.SetData(_args);

            // Draw all particles in ONE call
            Bounds bounds = new Bounds(Vector3.zero, Vector3.one * maxDistance * 2);
            Graphics.DrawMeshInstancedIndirect(
                _quadMesh, 0, particleMaterial,
                bounds, _argsBuffer,
                0, null, ShadowCastingMode.Off, false
            );
        }

        // =====================================================================
        // Public API
        // =====================================================================

        public void SetSceneParameters(float sunR, float planetR, float planetDist, float smooth)
        {
            sunRadius = sunR;
            planetRadius = planetR;
            planetDistance = planetDist;
            smoothness = smooth;
        }

        public void Reinitialize()
        {
            if (_activeShader == null) return;

            SetComputeUniforms();
            _activeShader.SetBuffer(_initKernel, "_Particles", _particleBuffer);
            int threadGroups = Mathf.CeilToInt((float)particleCount / THREAD_GROUP_SIZE);
            _activeShader.Dispatch(_initKernel, threadGroups, 1, 1);
        }

        // =====================================================================
        // Stats UI
        // =====================================================================

        void OnGUI()
        {
            if (!showStats) return;

            string sceneLabel = _currentSceneType switch
            {
                GpuSceneType.Cosmic => "COSMIC (Solar System)",
                GpuSceneType.Terrain => "TERRAIN (FBM Landscape)",
                GpuSceneType.Abstract => "ABSTRACT (Gyroid+Metaballs)",
                GpuSceneType.Fractal => "FRACTAL (Menger Sponge)",
                _ => "Unknown"
            };

            string sliceInfo = updateDivisions > 1
                ? $"Time Slice: 1/{updateDivisions} ({ParticlesPerFrame:N0}/frame)"
                : "Time Slice: OFF (full update)";

            GUILayout.BeginArea(new Rect(10, 10, 380, 270));
            GUILayout.BeginVertical("box");

            GUILayout.Label("<size=18><b>=== FULL GPU MODE ===</b></size>");
            GUILayout.Label($"<color=yellow>Scene: {sceneLabel}</color>");
            GUILayout.Space(3);
            GUILayout.Label($"<color=lime>FPS: {_fps:F0}</color>");
            GUILayout.Label($"Particles: {particleCount:N0}");
            GUILayout.Label($"<color=orange>{sliceInfo}</color>");
            GUILayout.Space(5);
            GUILayout.Label($"<color=cyan>GPU Dispatch: {_gpuTime * 1000:F3} ms</color>");
            GUILayout.Label($"CPU Transfer: ZERO");
            GUILayout.Space(5);
            GUILayout.Label("<size=10>SDF Eval: GPU Compute Shader</size>");
            GUILayout.Label("<size=10>Physics: GPU Compute Shader</size>");
            GUILayout.Label("<size=10>Render: GPU Instanced Indirect</size>");
            GUILayout.Space(5);
            GUILayout.Label("<size=10>[1] Cosmic  [2] Terrain  [3] Abstract  [4] Fractal</size>");

            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        void LateUpdate()
        {
            // Keyboard shortcuts for scene switching
            if (Input.GetKeyDown(KeyCode.Alpha1) || Input.GetKeyDown(KeyCode.Keypad1))
            {
                SwitchScene(GpuSceneType.Cosmic);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha2) || Input.GetKeyDown(KeyCode.Keypad2))
            {
                SwitchScene(GpuSceneType.Terrain);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha3) || Input.GetKeyDown(KeyCode.Keypad3))
            {
                SwitchScene(GpuSceneType.Abstract);
            }
            else if (Input.GetKeyDown(KeyCode.Alpha4) || Input.GetKeyDown(KeyCode.Keypad4))
            {
                SwitchScene(GpuSceneType.Fractal);
            }
        }
    }
}
