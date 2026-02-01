// =============================================================================
// Fractal Demo - "The Fractal Dive" (Microscope Demo)
// =============================================================================
// Demonstrates INFINITE RESOLUTION by zooming into a fractal structure.
//
// Two rendering modes:
//   1. Particles: Surface-adhering point cloud (view-dependent density)
//   2. Raymarching: TRUE solid surface rendering (recommended for "wow" effect)
//
// Key Message: "Zoom x1,000,000 - Still perfectly sharp. Polygons: 0"
//
// Controls:
//   - Mouse Wheel: Logarithmic zoom (microscope level possible)
//   - RMB + Mouse: Rotate view
//   - WASD: Move target point
//   - [R]: Toggle Raymarching/Particles mode
//   - [Space]: Reset zoom
//
// Author: Moroya Sakamoto
// =============================================================================

using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using AliceSdf;

namespace SdfUniverse
{
    /// <summary>
    /// Rendering mode for the fractal
    /// </summary>
    public enum FractalRenderMode
    {
        /// <summary>Surface-adhering particles (GPU Compute)</summary>
        Particles,
        /// <summary>TRUE solid surface via raymarching (recommended)</summary>
        Raymarching
    }

    public class FractalDemo : MonoBehaviour
    {
        [Header("=== RENDERING MODE ===")]
        [Tooltip("Particles: Point cloud visualization\nRaymarching: TRUE solid surface (recommended)")]
        public FractalRenderMode renderMode = FractalRenderMode.Raymarching;

        [Header("=== PARTICLE MODE (if Particles selected) ===")]
        [Tooltip("CPU_Standard: Parallel.For\nCPU_RustSIMD_Burst: Burst + Indirect\nGPU_ComputeShader: Full GPU")]
        public ParticleMode particleMode = ParticleMode.GPU_ComputeShader;

        [Header("=== FRACTAL PARAMETERS ===")]
        [Range(10f, 200f)]
        public float boxSize = 50f;

        [Range(0.5f, 10f)]
        public float holeSize = 2f;

        [Range(5f, 50f)]
        public float repeatScale = 15f;

        [Range(0f, 0.2f)]
        public float twistAmount = 0.02f;

        [Range(1, 5)]
        public int fractalIterations = 3;

        [Header("=== PARTICLES ===")]
        [Range(100000, 10000000)]
        public int particleCount = 5000000;

        [Range(0.001f, 0.1f)]
        public float particleSize = 0.02f;

        [Header("=== RENDERING ===")]
        public Color particleColor = new Color(0.2f, 0.8f, 1f, 1f);
        [Range(0.5f, 5f)]
        public float brightness = 2f;

        [Header("=== RAYMARCHING SETTINGS ===")]
        public Color sdfColor1 = new Color(0.2f, 0.8f, 1f, 1f);
        public Color sdfColor2 = new Color(0.8f, 0.3f, 0.5f, 1f);
        [Range(1f, 100f)]
        public float detailScale = 10f;

        [Header("=== DEMO SETTINGS ===")]
        public bool autoZoomDemo = false;
        public float autoZoomDuration = 30f;

        // Components
        private SdfWorld _sdfWorld;
        private SdfParticleSystem _particleSystemCPU;
        private SdfParticleSystem_Ultimate _particleSystemUltimate;
        private SdfParticleSystem_GPU _particleSystemGPU;
        private InfiniteZoomCamera _zoomCamera;
        private Camera _mainCamera;

        // Raymarching
        private GameObject _raymarchObject;
        private Material _raymarchMaterial;
        private FractalRenderMode _currentRenderMode;

        // SDF (1つの数式で無限の構造を生成)
        private SdfNode _sdfNode;
        private CompiledSdf _compiledSdf;

        // UI
        private Canvas _canvas;
        private Text _titleText;
        private Text _statsText;
        private Text _controlsText;

        // State
        private bool _demoRunning = false;

        void Awake()
        {
            _currentRenderMode = renderMode;

            SetupCamera();
            SetupSdfWorld();

            if (renderMode == FractalRenderMode.Raymarching)
            {
                SetupRaymarching();
                Debug.Log("[FractalDemo] Initialized - Raymarching Mode (TRUE Infinite Resolution)");
            }
            else
            {
                SetupParticleSystem();
                if (particleMode != ParticleMode.GPU_ComputeShader)
                {
                    BuildFractalWorld();
                }
                Debug.Log("[FractalDemo] Initialized - Particle Mode");
            }

            CreateUI();
            CreateLighting();

            Debug.Log("[FractalDemo] The Fractal Dive - Press [R] to toggle rendering mode");
        }

        void Start()
        {
            if (autoZoomDemo)
            {
                StartCoroutine(RunAutoZoomDemo());
            }
        }

        void Update()
        {
            HandleInput();
            UpdateFractalParameters();
            UpdateStats();
        }

        // =========================================================================
        // Setup
        // =========================================================================

        void SetupCamera()
        {
            _mainCamera = Camera.main;
            if (_mainCamera == null)
            {
                var camObj = new GameObject("Main Camera");
                _mainCamera = camObj.AddComponent<Camera>();
                camObj.AddComponent<AudioListener>();
                camObj.tag = "MainCamera";
            }

            _mainCamera.backgroundColor = new Color(0.005f, 0.005f, 0.015f);
            _mainCamera.clearFlags = CameraClearFlags.SolidColor;
            _mainCamera.farClipPlane = 1000f;
            _mainCamera.nearClipPlane = 0.0001f; // Very close for microscope zoom
            _mainCamera.fieldOfView = 60f;

            // Add infinite zoom camera controller
            _zoomCamera = _mainCamera.gameObject.GetComponent<InfiniteZoomCamera>();
            if (_zoomCamera == null)
            {
                _zoomCamera = _mainCamera.gameObject.AddComponent<InfiniteZoomCamera>();
            }

            _zoomCamera.minDistance = 0.0001f;
            _zoomCamera.maxDistance = 500f;
            _zoomCamera.startDistance = 150f;  // Reference for magnification calculation
            _zoomCamera.zoomSpeed = 20f;
            _zoomCamera.showZoomUI = true;

            // Position camera - start at x0.5 zoom (distance = startDistance * 2)
            float initialDistance = 300f;  // 150 / 300 = x0.5 magnification
            _mainCamera.transform.position = new Vector3(initialDistance, initialDistance * 0.5f, initialDistance);
            _mainCamera.transform.LookAt(Vector3.zero);

            // Set initial zoom distance after Start() runs
            StartCoroutine(SetInitialZoom(initialDistance));
        }

        System.Collections.IEnumerator SetInitialZoom(float distance)
        {
            yield return null; // Wait one frame for InfiniteZoomCamera.Start() to run
            if (_zoomCamera != null)
            {
                _zoomCamera.SetDistance(distance);
            }
        }

        void SetupSdfWorld()
        {
            _sdfWorld = GetComponent<SdfWorld>();
            if (_sdfWorld == null)
            {
                _sdfWorld = gameObject.AddComponent<SdfWorld>();
            }
            _sdfWorld.skyEnabled = false;
            _sdfWorld.groundEnabled = false;
        }

        void BuildFractalWorld()
        {
            // =================================================================
            // 無限に続く複雑な構造体を作る - たった1つの数式で！
            // =================================================================

            // 1. ベースとなる巨大な箱
            var baseBox = SdfNode.Box(boxSize, boxSize, boxSize);

            // 2. くり抜くための小さな十字架構造を作る
            float thickness = holeSize;
            float length = boxSize * 1.5f;
            var barX = SdfNode.Box(length, thickness, thickness);
            var barY = SdfNode.Box(thickness, length, thickness);
            var barZ = SdfNode.Box(thickness, thickness, length);
            var cross = barX.Union(barY.Union(barZ));

            // 3. 十字架を無限リピートさせる
            // これがフラクタル的な複雑さを生む（空間の折りたたみ）
            var repeatedCross = cross.Repeat(new Vector3(repeatScale, repeatScale, repeatScale));

            // 4. ベースからリピート十字架をくり抜く（メンガーのスポンジ風）
            var fractal = baseBox.Subtract(repeatedCross);

            // 5. 全体を少し捻って有機的にする
            _sdfNode = fractal.Twist(twistAmount);

            // JITコンパイル
            _compiledSdf = _sdfNode.Compile();

            // SdfWorldに設定（CPUモード用）
            if (_sdfWorld != null)
            {
                _sdfWorld.SetCompiledSdf(_compiledSdf);
            }

            Debug.Log($"[FractalDemo] Built fractal: 1 SDF formula, {_compiledSdf?.InstructionCount ?? 0} instructions");
        }

        void SetupParticleSystem()
        {
            switch (particleMode)
            {
                case ParticleMode.GPU_ComputeShader:
                    _particleSystemGPU = GetComponent<SdfParticleSystem_GPU>();
                    if (_particleSystemGPU == null)
                    {
                        _particleSystemGPU = gameObject.AddComponent<SdfParticleSystem_GPU>();
                    }

                    _particleSystemGPU.sceneType = GpuSceneType.Fractal;
                    _particleSystemGPU.particleCount = particleCount;
                    _particleSystemGPU.particleSize = particleSize;
                    _particleSystemGPU.brightness = brightness;
                    _particleSystemGPU.spawnRadius = boxSize * 1.5f;
                    _particleSystemGPU.maxDistance = boxSize * 3f;
                    _particleSystemGPU.flowSpeed = 0.1f;
                    _particleSystemGPU.surfaceAttraction = 10f;
                    _particleSystemGPU.noiseStrength = 0.3f;
                    _particleSystemGPU.boxSize = boxSize;
                    _particleSystemGPU.holeSize = holeSize;
                    _particleSystemGPU.repeatScale = repeatScale;
                    _particleSystemGPU.twistAmount = twistAmount;
                    _particleSystemGPU.fractalIterations = fractalIterations;
                    _particleSystemGPU.fractalColor = particleColor;
                    _particleSystemGPU.showStats = false;

                    Debug.Log($"[FractalDemo] GPU mode: {particleCount:N0} particles");
                    break;

                case ParticleMode.CPU_RustSIMD_Burst:
                    _particleSystemUltimate = GetComponent<SdfParticleSystem_Ultimate>();
                    if (_particleSystemUltimate == null)
                    {
                        _particleSystemUltimate = gameObject.AddComponent<SdfParticleSystem_Ultimate>();
                    }

                    _particleSystemUltimate.particleCount = particleCount;
                    _particleSystemUltimate.spawnRadius = boxSize * 1.5f;
                    _particleSystemUltimate.maxDistance = boxSize * 3f;
                    _particleSystemUltimate.particleSize = particleSize;
                    _particleSystemUltimate.flowSpeed = 0.1f;
                    _particleSystemUltimate.surfaceAttraction = 10f;
                    _particleSystemUltimate.noiseStrength = 0.3f;
                    _particleSystemUltimate.brightness = brightness;
                    _particleSystemUltimate.showStats = false;

                    Debug.Log($"[FractalDemo] CPU Burst mode: {particleCount:N0} particles");
                    break;

                case ParticleMode.CPU_Standard:
                default:
                    _particleSystemCPU = GetComponent<SdfParticleSystem>();
                    if (_particleSystemCPU == null)
                    {
                        _particleSystemCPU = gameObject.AddComponent<SdfParticleSystem>();
                    }

                    _particleSystemCPU.particleCount = particleCount;
                    _particleSystemCPU.spawnRadius = boxSize * 1.5f;
                    _particleSystemCPU.maxDistance = boxSize * 3f;
                    _particleSystemCPU.particleSize = particleSize;
                    _particleSystemCPU.flowSpeed = 0.1f;
                    _particleSystemCPU.surfaceAttraction = 10f;
                    _particleSystemCPU.noiseStrength = 0.3f;
                    _particleSystemCPU.flowMode = ParticleFlowMode.SurfaceFlow;

                    Debug.Log($"[FractalDemo] CPU Standard mode: {particleCount:N0} particles");
                    break;
            }
        }

        void SetupRaymarching()
        {
            // Create a large cube that acts as a "screen" for raymarching
            // The shader will render the SDF inside it
            _raymarchObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _raymarchObject.name = "SDF_Raymarching_Surface";
            _raymarchObject.transform.position = Vector3.zero;
            _raymarchObject.transform.localScale = Vector3.one * boxSize * 4f;

            // Find and apply raymarching shader
            var shader = Shader.Find("SdfUniverse/InfiniteSurface");
            if (shader == null)
            {
                Debug.LogError("[FractalDemo] SdfUniverse/InfiniteSurface shader not found!");
                return;
            }

            _raymarchMaterial = new Material(shader);
            _raymarchMaterial.SetColor("_Color", sdfColor1);
            _raymarchMaterial.SetColor("_Color2", sdfColor2);
            _raymarchMaterial.SetFloat("_DetailScale", detailScale);
            _raymarchMaterial.SetFloat("_BoxSize", boxSize);
            _raymarchMaterial.SetFloat("_HoleSize", holeSize);
            _raymarchMaterial.SetFloat("_RepeatScale", repeatScale);
            _raymarchMaterial.SetFloat("_TwistAmount", twistAmount);
            _raymarchMaterial.SetInt("_MaxSteps", 128);
            _raymarchMaterial.SetFloat("_MaxDist", boxSize * 10f);
            _raymarchMaterial.SetFloat("_SurfaceEpsilon", 0.0001f);

            _raymarchObject.GetComponent<Renderer>().material = _raymarchMaterial;

            Debug.Log("[FractalDemo] Raymarching setup complete - TRUE solid surface rendering");
        }

        void ToggleRenderMode()
        {
            if (_currentRenderMode == FractalRenderMode.Raymarching)
            {
                // Switch to Particles
                _currentRenderMode = FractalRenderMode.Particles;
                renderMode = FractalRenderMode.Particles;

                // Hide raymarching object
                if (_raymarchObject != null)
                    _raymarchObject.SetActive(false);

                // Setup particles if not already
                if (_particleSystemGPU == null && _particleSystemCPU == null && _particleSystemUltimate == null)
                {
                    SetupParticleSystem();
                }
                else
                {
                    // Reactivate existing particle system
                    if (_particleSystemGPU != null) _particleSystemGPU.enabled = true;
                    if (_particleSystemCPU != null) _particleSystemCPU.enabled = true;
                    if (_particleSystemUltimate != null) _particleSystemUltimate.enabled = true;
                }

                Debug.Log("[FractalDemo] Switched to Particle mode");
            }
            else
            {
                // Switch to Raymarching
                _currentRenderMode = FractalRenderMode.Raymarching;
                renderMode = FractalRenderMode.Raymarching;

                // Disable particle systems
                if (_particleSystemGPU != null) _particleSystemGPU.enabled = false;
                if (_particleSystemCPU != null) _particleSystemCPU.enabled = false;
                if (_particleSystemUltimate != null) _particleSystemUltimate.enabled = false;

                // Setup raymarching if not already
                if (_raymarchObject == null)
                {
                    SetupRaymarching();
                }
                else
                {
                    _raymarchObject.SetActive(true);
                }

                Debug.Log("[FractalDemo] Switched to Raymarching mode (TRUE Infinite Resolution)");
            }
        }

        void CreateLighting()
        {
            if (FindObjectOfType<Light>() == null)
            {
                var lightObj = new GameObject("Fractal Light");
                var light = lightObj.AddComponent<Light>();
                light.type = LightType.Directional;
                light.color = new Color(0.8f, 0.9f, 1f);
                light.intensity = 1.2f;
                lightObj.transform.rotation = Quaternion.Euler(45, -30, 0);
            }
        }

        // =========================================================================
        // UI
        // =========================================================================

        void CreateUI()
        {
            var canvasObj = new GameObject("FractalCanvas");
            _canvas = canvasObj.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 100;
            canvasObj.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvasObj.AddComponent<GraphicRaycaster>();

            // Title (top center)
            var titleObj = CreateText(_canvas.transform, "Title", "THE FRACTAL DIVE", 48,
                new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                new Color(0.3f, 0.9f, 1f));
            _titleText = titleObj.GetComponent<Text>();
            var titleRect = titleObj.GetComponent<RectTransform>();
            titleRect.anchoredPosition = new Vector2(0, -30);

            // Subtitle
            CreateText(_canvas.transform, "Subtitle", "Infinite Resolution SDF Visualization", 20,
                new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                new Color(0.6f, 0.6f, 0.6f)).GetComponent<RectTransform>().anchoredPosition = new Vector2(0, -80);

            // Stats panel (top left)
            var statsPanel = new GameObject("StatsPanel");
            statsPanel.transform.SetParent(_canvas.transform, false);
            var statsPanelRect = statsPanel.AddComponent<RectTransform>();
            statsPanelRect.anchorMin = new Vector2(0, 0.7f);
            statsPanelRect.anchorMax = new Vector2(0.25f, 1f);
            statsPanelRect.offsetMin = new Vector2(10, 10);
            statsPanelRect.offsetMax = new Vector2(-10, -100);

            var statsBg = statsPanel.AddComponent<Image>();
            statsBg.color = new Color(0, 0, 0, 0.7f);

            var statsObj = CreateText(statsPanel.transform, "Stats", "", 14,
                Vector2.zero, Vector2.one, new Color(0.4f, 1f, 0.6f));
            _statsText = statsObj.GetComponent<Text>();
            _statsText.alignment = TextAnchor.UpperLeft;
            var statsRect = statsObj.GetComponent<RectTransform>();
            statsRect.offsetMin = new Vector2(10, 10);
            statsRect.offsetMax = new Vector2(-10, -10);

            // Controls panel (bottom left)
            var controlsPanel = new GameObject("ControlsPanel");
            controlsPanel.transform.SetParent(_canvas.transform, false);
            var controlsPanelRect = controlsPanel.AddComponent<RectTransform>();
            controlsPanelRect.anchorMin = new Vector2(0, 0);
            controlsPanelRect.anchorMax = new Vector2(0.22f, 0.25f);
            controlsPanelRect.offsetMin = new Vector2(10, 10);
            controlsPanelRect.offsetMax = new Vector2(-10, -10);

            var controlsBg = controlsPanel.AddComponent<Image>();
            controlsBg.color = new Color(0, 0, 0, 0.6f);

            var controlsObj = CreateText(controlsPanel.transform, "Controls",
                "=== CONTROLS ===\n" +
                "[Scroll] Zoom\n" +
                "[RMB] Rotate\n" +
                "[WASD] Move\n" +
                "[R] Toggle Mode\n" +
                "[Space] Reset",
                12, Vector2.zero, Vector2.one, new Color(0.7f, 0.7f, 0.7f));
            _controlsText = controlsObj.GetComponent<Text>();
            _controlsText.alignment = TextAnchor.UpperLeft;
            var controlsRect = controlsObj.GetComponent<RectTransform>();
            controlsRect.offsetMin = new Vector2(8, 8);
            controlsRect.offsetMax = new Vector2(-8, -8);
        }

        GameObject CreateText(Transform parent, string name, string content, int fontSize,
            Vector2 anchorMin, Vector2 anchorMax, Color color)
        {
            var obj = new GameObject(name);
            obj.transform.SetParent(parent, false);

            var text = obj.AddComponent<Text>();
            text.text = content;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = color;

            var rect = obj.GetComponent<RectTransform>();
            rect.anchorMin = anchorMin;
            rect.anchorMax = anchorMax;
            rect.sizeDelta = Vector2.zero;
            rect.offsetMin = Vector2.zero;
            rect.offsetMax = Vector2.zero;

            return obj;
        }

        void UpdateStats()
        {
            if (_statsText == null) return;

            float fps = 1f / Time.deltaTime;
            string mode = "";
            string renderingInfo = "";

            // Determine mode and rendering info based on current render mode
            if (_currentRenderMode == FractalRenderMode.Raymarching)
            {
                mode = "<color=lime>RAYMARCHING</color>";
                renderingInfo = "Solid Surface (Per-Pixel)";
            }
            else
            {
                switch (particleMode)
                {
                    case ParticleMode.GPU_ComputeShader:
                        mode = "GPU Particles";
                        if (_particleSystemGPU != null)
                            renderingInfo = $"{_particleSystemGPU.ActiveParticles:N0} points";
                        break;
                    case ParticleMode.CPU_RustSIMD_Burst:
                        mode = "CPU Burst";
                        if (_particleSystemUltimate != null)
                            renderingInfo = $"{_particleSystemUltimate.ActiveParticles:N0} points";
                        break;
                    default:
                        mode = "CPU Standard";
                        if (_particleSystemCPU != null)
                            renderingInfo = $"{_particleSystemCPU.ActiveParticles:N0} points";
                        break;
                }
            }

            string zoomText = "";
            string statusText = "Ready";

            if (_zoomCamera != null)
            {
                float mag = _zoomCamera.ZoomMagnification;

                if (mag >= 1000000)
                    zoomText = $"x{mag / 1000000:F1}M";
                else if (mag >= 1000)
                    zoomText = $"x{mag / 1000:F1}K";
                else
                    zoomText = $"x{mag:F1}";

                // Dynamic status based on zoom level
                if (mag >= 100000)
                    statusText = "RESOLVING INFINITY...";
                else if (mag >= 10000)
                    statusText = "Atomic Scale";
                else if (mag >= 1000)
                    statusText = "Microscope Mode";
                else if (mag >= 100)
                    statusText = "Macro View";
                else
                    statusText = "Overview";
            }

            string miracleText = _currentRenderMode == FractalRenderMode.Raymarching
                ? "TRUE SOLID SURFACE"
                : "Point Cloud Visualization";

            _statsText.text = $@"=== THE FRACTAL DIVE ===
Mode: {mode}
Rendering: {renderingInfo}
Status: <color=yellow>{statusText}</color>

FPS: <color=lime>{fps:F0}</color>

=== THE 5MB MIRACLE ===
Zoom: <color=lime>{zoomText}</color>
Polygons: <color=cyan>0</color>
Textures: <color=cyan>0</color>
SDF Formula: <color=orange>1</color>
Resolution: <color=yellow>INFINITE</color>

{miracleText}
[R] Toggle Mode | [Space] Reset";
        }

        // =========================================================================
        // Input
        // =========================================================================

        void HandleInput()
        {
            // [R] Toggle render mode (Raymarching vs Particles)
            if (Input.GetKeyDown(KeyCode.R))
            {
                ToggleRenderMode();
            }

            // Reset zoom
            if (Input.GetKeyDown(KeyCode.Space) && _zoomCamera != null)
            {
                _zoomCamera.ResetZoom();
            }

            // Adjust parameters with arrow keys
            if (Input.GetKey(KeyCode.UpArrow))
            {
                boxSize = Mathf.Clamp(boxSize + Time.deltaTime * 10, 10, 200);
            }
            if (Input.GetKey(KeyCode.DownArrow))
            {
                boxSize = Mathf.Clamp(boxSize - Time.deltaTime * 10, 10, 200);
            }
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                twistAmount = Mathf.Clamp(twistAmount - Time.deltaTime * 0.05f, 0, 0.2f);
            }
            if (Input.GetKey(KeyCode.RightArrow))
            {
                twistAmount = Mathf.Clamp(twistAmount + Time.deltaTime * 0.05f, 0, 0.2f);
            }
        }

        void UpdateFractalParameters()
        {
            if (particleMode == ParticleMode.GPU_ComputeShader && _particleSystemGPU != null)
            {
                _particleSystemGPU.boxSize = boxSize;
                _particleSystemGPU.holeSize = holeSize;
                _particleSystemGPU.repeatScale = repeatScale;
                _particleSystemGPU.twistAmount = twistAmount;
                _particleSystemGPU.fractalIterations = fractalIterations;
                _particleSystemGPU.particleSize = particleSize;
                _particleSystemGPU.brightness = brightness;
                _particleSystemGPU.fractalColor = particleColor;
            }

            // Update raymarching material parameters
            if (_raymarchMaterial != null)
            {
                _raymarchMaterial.SetFloat("_BoxSize", boxSize);
                _raymarchMaterial.SetFloat("_HoleSize", holeSize);
                _raymarchMaterial.SetFloat("_RepeatScale", repeatScale);
                _raymarchMaterial.SetFloat("_TwistAmount", twistAmount);
                _raymarchMaterial.SetColor("_Color", sdfColor1);
                _raymarchMaterial.SetColor("_Color2", sdfColor2);

                // Adaptive detail scale based on zoom level
                if (_zoomCamera != null)
                {
                    float mag = _zoomCamera.ZoomMagnification;
                    float adaptiveDetail = detailScale * (1f + Mathf.Log10(mag + 1f));
                    _raymarchMaterial.SetFloat("_DetailScale", adaptiveDetail);

                    // Adjust epsilon for precision at close range
                    float epsilon = Mathf.Max(0.00001f, _zoomCamera.CurrentDistance * 0.00001f);
                    _raymarchMaterial.SetFloat("_SurfaceEpsilon", epsilon);
                }
            }
        }

        // =========================================================================
        // Auto Zoom Demo
        // =========================================================================

        IEnumerator RunAutoZoomDemo()
        {
            _demoRunning = true;

            yield return new WaitForSeconds(2f); // Initial pause

            // Phase 1: Zoom in dramatically
            if (_zoomCamera != null)
            {
                _zoomCamera.ZoomTo(0.001f, autoZoomDuration * 0.6f);
            }

            yield return new WaitForSeconds(autoZoomDuration * 0.6f);

            // Phase 2: Hold at microscope level
            yield return new WaitForSeconds(3f);

            // Phase 3: Zoom back out
            if (_zoomCamera != null)
            {
                _zoomCamera.ZoomTo(150f, autoZoomDuration * 0.3f);
            }

            yield return new WaitForSeconds(autoZoomDuration * 0.3f);

            _demoRunning = false;
        }

        // =========================================================================
        // Gizmos
        // =========================================================================

        void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one * boxSize * 2);

            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(Vector3.zero, boxSize * 1.5f);
        }
    }
}
