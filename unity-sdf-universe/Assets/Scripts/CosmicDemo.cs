// =============================================================================
// Cosmic Demo - "5MB Miracle" Demonstration
// =============================================================================
// JUST PRESS PLAY - The universe loads in 5MB
//
// Demo Flow:
// 1. Loading: "Loading Universe... 5.2MB" with progress bar
// 2. Reveal: Dramatic fade-in of the solar system
// 3. Camera Tour: Sun → Planet → Ring → Orbit view
// 4. Free Roam: WASD + Mouse controls
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
    /// Particle processing mode
    /// </summary>
    public enum ParticleMode
    {
        /// <summary>CPU Standard - Parallel.For + EvalGradientSoA</summary>
        CPU_Standard,
        /// <summary>CPU Ultimate - Burst Jobs + DrawMeshInstancedIndirect</summary>
        CPU_RustSIMD_Burst,
        /// <summary>Full GPU - Compute Shader (Zero CPU transfer)</summary>
        GPU_ComputeShader
    }

    public class CosmicDemo : MonoBehaviour
    {
        [Header("=== MODE ===")]
        [Tooltip("CPU_Standard: Parallel.For\nCPU_RustSIMD_Burst: Burst + Indirect\nGPU_ComputeShader: Full GPU")]
        public ParticleMode particleMode = ParticleMode.GPU_ComputeShader;

        [Header("=== GPU SCENE (GPU Mode Only) ===")]
        [Tooltip("Cosmic: Solar System\nTerrain: FBM Landscape\nAbstract: Gyroid+Metaballs")]
        public GpuSceneType gpuSceneType = GpuSceneType.Cosmic;

        [Header("=== TIME SLICING ===")]
        [Range(1, 10)]
        [Tooltip("1=Every frame (full), 3=1/3 load, 10=1/10 load")]
        public int updateDivisions = 1;

        [Header("=== COSMIC PARAMETERS ===")]
        [Range(1f, 50f)] public float sunRadius = 8f;
        [Range(0.5f, 10f)] public float planetRadius = 2.5f;
        [Range(0f, 5f)] public float ringTwist = 0.5f;
        [Range(5f, 30f)] public float planetDistance = 18f;
        [Range(0f, 10f)] public float smoothness = 1.5f;

        [Header("=== PARTICLES ===")]
        [Range(10000, 10000000)] public int particleCount = 1000000;

        [Header("=== DEMO TIMING ===")]
        public float tourDuration = 15f;

        // Auto-created components
        private SdfWorld _sdfWorld;
        private SdfParticleSystem _particleSystem;
        private SdfParticleSystem_Ultimate _particleSystemUltimate;
        private SdfParticleSystem_GPU _particleSystemGPU;
        private Camera _mainCamera;

        // UI
        private Canvas _canvas;
        private Text _titleText;
        private Text _sizeText;
        private Text _statsText;
        private Image _loadingBarBg;
        private Image _loadingBarFill;
        private CanvasGroup _loadingPanel;
        private CanvasGroup _statsPanel;

        // Control indicator
        private CanvasGroup _controlsPanel;
        private Image _keyW, _keyA, _keyS, _keyD;
        private Image _keyQ, _keyE, _keyShift;
        private Image _arrowUp, _arrowDown, _arrowLeft, _arrowRight;
        private Color _keyNormal = new Color(0.2f, 0.2f, 0.2f, 0.8f);
        private Color _keyPressed = new Color(0.3f, 0.9f, 1f, 1f);

        // State
        private DemoPhase _phase = DemoPhase.Tour;

        // Camera tour
        private Vector3[] _tourPoints;
        private Vector3[] _tourLookAts;
        private float _tourProgress;

        public enum DemoPhase { Tour, FreeRoam }

        // =========================================================================
        // Bootstrap
        // =========================================================================

        void Awake()
        {
            SetupCamera();
            SetupSdfWorld();
            SetupParticleSystem();
            CreateUI();
            CreateLighting();
            BuildCosmicWorld();
            SetupCameraTour();

            Debug.Log("[CosmicDemo] Universe initialized");
        }

        void Start()
        {
            StartCoroutine(RunDemo());
        }

        void Update()
        {
            UpdateStats();
            UpdateControlIndicators();

            switch (_phase)
            {
                case DemoPhase.Tour:
                    // Camera moves automatically
                    break;
                case DemoPhase.FreeRoam:
                    UpdateFreeRoam();
                    break;
            }
        }

        void UpdateControlIndicators()
        {
            if (_controlsPanel == null || _controlsPanel.alpha < 0.1f) return;

            // WASD
            UpdateKeyHighlight(_keyW, Input.GetKey(KeyCode.W));
            UpdateKeyHighlight(_keyA, Input.GetKey(KeyCode.A));
            UpdateKeyHighlight(_keyS, Input.GetKey(KeyCode.S));
            UpdateKeyHighlight(_keyD, Input.GetKey(KeyCode.D));
            UpdateKeyHighlight(_keyQ, Input.GetKey(KeyCode.Q));
            UpdateKeyHighlight(_keyE, Input.GetKey(KeyCode.E));
            UpdateKeyHighlight(_keyShift, Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift));

            // Arrows
            UpdateKeyHighlight(_arrowUp, Input.GetKey(KeyCode.UpArrow));
            UpdateKeyHighlight(_arrowDown, Input.GetKey(KeyCode.DownArrow));
            UpdateKeyHighlight(_arrowLeft, Input.GetKey(KeyCode.LeftArrow));
            UpdateKeyHighlight(_arrowRight, Input.GetKey(KeyCode.RightArrow));
        }

        void UpdateKeyHighlight(Image key, bool pressed)
        {
            if (key == null) return;
            key.color = Color.Lerp(key.color, pressed ? _keyPressed : _keyNormal, Time.deltaTime * 15f);
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

            _mainCamera.backgroundColor = new Color(0.01f, 0.005f, 0.02f);
            _mainCamera.clearFlags = CameraClearFlags.SolidColor;
            _mainCamera.farClipPlane = 500f;
            _mainCamera.nearClipPlane = 0.1f;
            _mainCamera.fieldOfView = 60f;

            // Start far away
            _mainCamera.transform.position = new Vector3(60, 30, 60);
            _mainCamera.transform.LookAt(Vector3.zero);
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

        void SetupParticleSystem()
        {
            switch (particleMode)
            {
                case ParticleMode.GPU_ComputeShader:
                    // FULL GPU MODE - Zero CPU transfer
                    _particleSystemGPU = GetComponent<SdfParticleSystem_GPU>();
                    if (_particleSystemGPU == null)
                    {
                        _particleSystemGPU = gameObject.AddComponent<SdfParticleSystem_GPU>();
                    }

                    // Set scene type and time slicing FIRST (before Start() runs)
                    _particleSystemGPU.sceneType = gpuSceneType;
                    _particleSystemGPU.updateDivisions = updateDivisions;

                    _particleSystemGPU.particleCount = particleCount;
                    _particleSystemGPU.spawnRadius = 35f;
                    _particleSystemGPU.maxDistance = 80f;
                    _particleSystemGPU.particleSize = 0.12f;
                    _particleSystemGPU.flowSpeed = 3f;
                    _particleSystemGPU.surfaceAttraction = 2f;
                    _particleSystemGPU.noiseStrength = 0.1f;
                    _particleSystemGPU.brightness = 2f;
                    _particleSystemGPU.showStats = false;

                    // Set SDF scene parameters (Cosmic)
                    _particleSystemGPU.sunRadius = sunRadius;
                    _particleSystemGPU.planetRadius = planetRadius;
                    _particleSystemGPU.planetDistance = planetDistance;
                    _particleSystemGPU.smoothness = smoothness;

                    Debug.Log($"[CosmicDemo] Using GPU Compute Shader mode (Scene: {gpuSceneType})");
                    break;

                case ParticleMode.CPU_RustSIMD_Burst:
                    // ULTIMATE DEEP FRIED MODE - Burst + Indirect
                    _particleSystemUltimate = GetComponent<SdfParticleSystem_Ultimate>();
                    if (_particleSystemUltimate == null)
                    {
                        _particleSystemUltimate = gameObject.AddComponent<SdfParticleSystem_Ultimate>();
                    }

                    _particleSystemUltimate.particleCount = particleCount;
                    _particleSystemUltimate.spawnRadius = 35f;
                    _particleSystemUltimate.maxDistance = 80f;
                    _particleSystemUltimate.particleSize = 0.12f;
                    _particleSystemUltimate.flowSpeed = 3f;
                    _particleSystemUltimate.surfaceAttraction = 2f;
                    _particleSystemUltimate.noiseStrength = 0.1f;
                    _particleSystemUltimate.brightness = 2f;
                    _particleSystemUltimate.showStats = false;

                    Debug.Log("[CosmicDemo] Using CPU Rust SIMD + Burst mode");
                    break;

                case ParticleMode.CPU_Standard:
                default:
                    // Standard mode - Parallel.For
                    _particleSystem = GetComponent<SdfParticleSystem>();
                    if (_particleSystem == null)
                    {
                        _particleSystem = gameObject.AddComponent<SdfParticleSystem>();
                    }

                    _particleSystem.particleCount = particleCount;
                    _particleSystem.spawnRadius = 35f;
                    _particleSystem.maxDistance = 80f;
                    _particleSystem.particleSize = 0.15f;
                    _particleSystem.flowSpeed = 3f;
                    _particleSystem.surfaceAttraction = 0.3f;
                    _particleSystem.noiseStrength = 0.1f;
                    _particleSystem.flowMode = ParticleFlowMode.SurfaceFlow;

                    Debug.Log("[CosmicDemo] Using CPU Standard mode");
                    break;
            }
        }

        void SetupCameraTour()
        {
            // Camera tour waypoints
            _tourPoints = new Vector3[] {
                new Vector3(50, 25, 50),                          // Wide view
                new Vector3(30, 15, 30),                          // Closer
                new Vector3(sunRadius + 5, 3, 0),                 // Sun close-up
                new Vector3(planetDistance + 8, 5, 8),            // Planet view
                new Vector3(planetDistance, 8, -10),              // Ring view
                new Vector3(35, 20, 35),                          // Pull back
            };

            _tourLookAts = new Vector3[] {
                Vector3.zero,
                Vector3.zero,
                Vector3.zero,
                new Vector3(planetDistance, 0, 0),
                new Vector3(planetDistance, 0, 0),
                Vector3.zero,
            };
        }

        void BuildCosmicWorld()
        {
            _sdfWorld.objects.Clear();

            // === SUN (Center) ===
            _sdfWorld.objects.Add(new SdfObjectDefinition {
                name = "Sun",
                shapeType = SdfShapeType.Sphere,
                radius = sunRadius,
                position = Vector3.zero
            });

            // === PLANET 1 (Main) ===
            _sdfWorld.objects.Add(new SdfObjectDefinition {
                name = "Planet",
                shapeType = SdfShapeType.Sphere,
                radius = planetRadius,
                position = new Vector3(planetDistance, 0, 0),
                animate = true,
                animationSpeed = 0.15f,
                animationAxis = Vector3.up
            });

            // === RING (Around Planet) ===
            _sdfWorld.objects.Add(new SdfObjectDefinition {
                name = "Ring",
                shapeType = SdfShapeType.Torus,
                majorRadius = planetRadius * 1.8f,
                minorRadius = 0.12f,
                position = new Vector3(planetDistance, 0, 0),
                rotation = new Vector3(15, 0, 10),
                twistStrength = ringTwist,
                animate = true,
                animationSpeed = 0.15f
            });

            // === MOON ===
            _sdfWorld.objects.Add(new SdfObjectDefinition {
                name = "Moon",
                shapeType = SdfShapeType.Sphere,
                radius = 0.6f,
                position = new Vector3(planetDistance + 4, 1.5f, 0),
                animate = true,
                animationSpeed = 0.4f
            });

            // === PLANET 2 (Distant) ===
            _sdfWorld.objects.Add(new SdfObjectDefinition {
                name = "Planet2",
                shapeType = SdfShapeType.Sphere,
                radius = planetRadius * 0.5f,
                position = new Vector3(-planetDistance * 0.6f, 3, planetDistance * 0.4f),
                animate = true,
                animationSpeed = 0.25f
            });

            // === ASTEROID BELT (Multiple small spheres) ===
            float beltRadius = planetDistance * 0.75f;
            for (int i = 0; i < 6; i++)
            {
                float angle = i * Mathf.PI * 2f / 6f;
                Vector3 pos = new Vector3(
                    Mathf.Cos(angle) * beltRadius,
                    (i % 2 == 0 ? 0.5f : -0.5f),
                    Mathf.Sin(angle) * beltRadius
                );
                _sdfWorld.objects.Add(new SdfObjectDefinition {
                    name = $"Asteroid{i}",
                    shapeType = SdfShapeType.Sphere,
                    radius = 0.3f + (i * 0.1f),
                    position = pos,
                    animate = true,
                    animationSpeed = 0.1f + (i * 0.02f)
                });
            }

            _sdfWorld.globalSmoothness = smoothness;
            _sdfWorld.RebuildWorld();

            Debug.Log($"[CosmicDemo] Built universe: {_sdfWorld.objects.Count} objects");
        }

        // =========================================================================
        // UI Creation
        // =========================================================================

        void CreateUI()
        {
            var canvasObj = new GameObject("DemoCanvas");
            _canvas = canvasObj.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 100;
            canvasObj.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvasObj.AddComponent<GraphicRaycaster>();

            CreateLoadingPanel();
            CreateStatsPanel();
            CreateControlsPanel();
        }

        void CreateLoadingPanel()
        {
            // Panel
            var panelObj = new GameObject("LoadingPanel");
            panelObj.transform.SetParent(_canvas.transform, false);
            _loadingPanel = panelObj.AddComponent<CanvasGroup>();

            var panelRect = panelObj.AddComponent<RectTransform>();
            panelRect.anchorMin = Vector2.zero;
            panelRect.anchorMax = Vector2.one;
            panelRect.sizeDelta = Vector2.zero;

            var panelBg = panelObj.AddComponent<Image>();
            panelBg.color = new Color(0.02f, 0.01f, 0.03f, 1f);

            // Title
            var titleObj = CreateText(panelObj, "Title", "ALICE-SDF", 72,
                new Vector2(0.5f, 0.7f), new Vector2(0.5f, 0.85f), Color.cyan);
            _titleText = titleObj.GetComponent<Text>();

            // Subtitle
            CreateText(panelObj, "Subtitle", "The 5MB Miracle", 36,
                new Vector2(0.5f, 0.6f), new Vector2(0.5f, 0.7f), new Color(0.7f, 0.7f, 0.7f));

            // Loading text
            var sizeObj = CreateText(panelObj, "Size", "Loading Universe... (0.0 MB)", 28,
                new Vector2(0.5f, 0.4f), new Vector2(0.5f, 0.5f), Color.white);
            _sizeText = sizeObj.GetComponent<Text>();

            // Loading bar background
            var barBgObj = new GameObject("LoadingBarBg");
            barBgObj.transform.SetParent(panelObj.transform, false);
            _loadingBarBg = barBgObj.AddComponent<Image>();
            _loadingBarBg.color = new Color(0.15f, 0.15f, 0.15f, 1f);
            var barBgRect = barBgObj.GetComponent<RectTransform>();
            barBgRect.anchorMin = new Vector2(0.25f, 0.32f);
            barBgRect.anchorMax = new Vector2(0.75f, 0.35f);
            barBgRect.sizeDelta = Vector2.zero;

            // Loading bar fill
            var barFillObj = new GameObject("LoadingBarFill");
            barFillObj.transform.SetParent(barBgObj.transform, false);
            _loadingBarFill = barFillObj.AddComponent<Image>();
            _loadingBarFill.color = new Color(0.3f, 0.9f, 1f, 1f);
            var barFillRect = barFillObj.GetComponent<RectTransform>();
            barFillRect.anchorMin = Vector2.zero;
            barFillRect.anchorMax = new Vector2(0f, 1f);
            barFillRect.sizeDelta = Vector2.zero;
            barFillRect.pivot = new Vector2(0, 0.5f);
        }

        void CreateStatsPanel()
        {
            var panelObj = new GameObject("StatsPanel");
            panelObj.transform.SetParent(_canvas.transform, false);
            _statsPanel = panelObj.AddComponent<CanvasGroup>();
            _statsPanel.alpha = 0;

            var panelRect = panelObj.AddComponent<RectTransform>();
            panelRect.anchorMin = new Vector2(0, 0.65f);
            panelRect.anchorMax = new Vector2(0.28f, 1f);
            panelRect.offsetMin = new Vector2(10, 10);
            panelRect.offsetMax = new Vector2(-10, -10);

            var panelBg = panelObj.AddComponent<Image>();
            panelBg.color = new Color(0, 0, 0, 0.75f);

            var statsObj = CreateText(panelObj, "Stats", "", 16,
                Vector2.zero, Vector2.one, new Color(0.4f, 1f, 0.6f));
            _statsText = statsObj.GetComponent<Text>();
            _statsText.alignment = TextAnchor.UpperLeft;
            var statsRect = statsObj.GetComponent<RectTransform>();
            statsRect.offsetMin = new Vector2(12, 12);
            statsRect.offsetMax = new Vector2(-12, -12);
        }

        void CreateControlsPanel()
        {
            // Main panel (bottom-left)
            var panelObj = new GameObject("ControlsPanel");
            panelObj.transform.SetParent(_canvas.transform, false);
            _controlsPanel = panelObj.AddComponent<CanvasGroup>();
            _controlsPanel.alpha = 0; // Hidden until FreeRoam

            var panelRect = panelObj.AddComponent<RectTransform>();
            panelRect.anchorMin = new Vector2(0, 0);
            panelRect.anchorMax = new Vector2(0.25f, 0.35f);
            panelRect.offsetMin = new Vector2(15, 15);
            panelRect.offsetMax = new Vector2(-15, -15);

            var panelBg = panelObj.AddComponent<Image>();
            panelBg.color = new Color(0, 0, 0, 0.6f);

            // Key size
            float keySize = 40f;
            float spacing = 5f;

            // === WASD Keys (left side) ===
            float wasdStartX = 20f;
            float wasdStartY = -60f;

            // W
            _keyW = CreateKeyImage(panelObj, "W", "W",
                wasdStartX + keySize + spacing, wasdStartY, keySize);

            // A S D
            _keyA = CreateKeyImage(panelObj, "A", "A",
                wasdStartX, wasdStartY - keySize - spacing, keySize);
            _keyS = CreateKeyImage(panelObj, "S", "S",
                wasdStartX + keySize + spacing, wasdStartY - keySize - spacing, keySize);
            _keyD = CreateKeyImage(panelObj, "D", "D",
                wasdStartX + (keySize + spacing) * 2, wasdStartY - keySize - spacing, keySize);

            // Q E (above A and D)
            _keyQ = CreateKeyImage(panelObj, "Q", "Q",
                wasdStartX, wasdStartY, keySize);
            _keyE = CreateKeyImage(panelObj, "E", "E",
                wasdStartX + (keySize + spacing) * 2, wasdStartY, keySize);

            // Shift (below A)
            _keyShift = CreateKeyImage(panelObj, "Shift", "SHIFT",
                wasdStartX, wasdStartY - (keySize + spacing) * 2, keySize * 2 + spacing);

            // === Arrow Keys (right side) ===
            float arrowStartX = 160f;
            float arrowStartY = -60f;

            // Up
            _arrowUp = CreateKeyImage(panelObj, "Up", "\u25B2",
                arrowStartX + keySize + spacing, arrowStartY, keySize);

            // Left Down Right
            _arrowLeft = CreateKeyImage(panelObj, "Left", "\u25C0",
                arrowStartX, arrowStartY - keySize - spacing, keySize);
            _arrowDown = CreateKeyImage(panelObj, "Down", "\u25BC",
                arrowStartX + keySize + spacing, arrowStartY - keySize - spacing, keySize);
            _arrowRight = CreateKeyImage(panelObj, "Right", "\u25B6",
                arrowStartX + (keySize + spacing) * 2, arrowStartY - keySize - spacing, keySize);

            // Labels
            CreateKeyLabel(panelObj, "Move", wasdStartX + keySize + spacing, wasdStartY + 30f);
            CreateKeyLabel(panelObj, "Boost", wasdStartX + keySize * 0.5f + spacing, wasdStartY - (keySize + spacing) * 2 - 15f);
            CreateKeyLabel(panelObj, "Sun", arrowStartX + keySize + spacing, arrowStartY + 30f);
            CreateKeyLabel(panelObj, "Blend", arrowStartX + keySize + spacing, arrowStartY - (keySize + spacing) * 2 - 15f);

            // RMB indicator
            CreateKeyLabel(panelObj, "[RMB] Look", wasdStartX + 60f, wasdStartY - (keySize + spacing) * 3 - 10f);
        }

        Image CreateKeyImage(GameObject parent, string name, string label, float x, float y, float width)
        {
            var keyObj = new GameObject($"Key_{name}");
            keyObj.transform.SetParent(parent.transform, false);

            var rect = keyObj.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 1);
            rect.anchorMax = new Vector2(0, 1);
            rect.pivot = new Vector2(0, 1);
            rect.anchoredPosition = new Vector2(x, y);
            rect.sizeDelta = new Vector2(width, 40f);

            var img = keyObj.AddComponent<Image>();
            img.color = _keyNormal;

            // Key label
            var textObj = new GameObject("Label");
            textObj.transform.SetParent(keyObj.transform, false);

            var textRect = textObj.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;

            var text = textObj.AddComponent<Text>();
            text.text = label;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = label.Length > 2 ? 12 : 18;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = Color.white;

            return img;
        }

        void CreateKeyLabel(GameObject parent, string labelText, float x, float y)
        {
            var labelObj = new GameObject($"Label_{labelText}");
            labelObj.transform.SetParent(parent.transform, false);

            var rect = labelObj.AddComponent<RectTransform>();
            rect.anchorMin = new Vector2(0, 1);
            rect.anchorMax = new Vector2(0, 1);
            rect.pivot = new Vector2(0.5f, 1);
            rect.anchoredPosition = new Vector2(x, y);
            rect.sizeDelta = new Vector2(80, 20);

            var text = labelObj.AddComponent<Text>();
            text.text = labelText;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = 12;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = new Color(0.7f, 0.7f, 0.7f);
        }

        GameObject CreateText(GameObject parent, string name, string content, int fontSize,
            Vector2 anchorMin, Vector2 anchorMax, Color color)
        {
            var obj = new GameObject(name);
            obj.transform.SetParent(parent.transform, false);

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

        void CreateLighting()
        {
            if (FindObjectOfType<Light>() == null)
            {
                var lightObj = new GameObject("Sun Light");
                var light = lightObj.AddComponent<Light>();
                light.type = LightType.Directional;
                light.color = new Color(1f, 0.95f, 0.85f);
                light.intensity = 1.5f;
                lightObj.transform.rotation = Quaternion.Euler(45, -30, 0);
            }
        }

        // =========================================================================
        // Demo Sequence
        // =========================================================================

        IEnumerator RunDemo()
        {
            // Immediate start - no loading animation
            _loadingPanel.gameObject.SetActive(false);
            _statsPanel.alpha = 1f;
            if (_controlsPanel != null)
                _controlsPanel.alpha = 1f;

            // Phase 1: Camera Tour
            _phase = DemoPhase.Tour;
            yield return StartCoroutine(TourPhase());

            // Phase 2: Free Roam
            _phase = DemoPhase.FreeRoam;
            Debug.Log("[CosmicDemo] Demo complete! WASD to move, RMB to look.");
        }

        // Legacy loading methods (kept for reference, not used)
        /*
        IEnumerator LoadingPhase()
        {
            _loadingPanel.alpha = 1f;
            float elapsed = 0f;

            // Animate title
            _titleText.color = new Color(0, 0, 0, 0);

            while (elapsed < loadingDuration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / loadingDuration;

                // Smooth progress
                float progress = Mathf.SmoothStep(0, 1, t);

                // Update bar
                var barRect = _loadingBarFill.GetComponent<RectTransform>();
                barRect.anchorMax = new Vector2(progress, 1);

                // Update size text with dramatic effect
                float size = Mathf.Lerp(0, 5.2f, progress);
                _sizeText.text = $"Loading Universe... ({size:F1} MB)";

                // Fade in title
                float titleAlpha = Mathf.Clamp01(t * 2);
                _titleText.color = new Color(0, titleAlpha, titleAlpha, titleAlpha);

                yield return null;
            }

            // Complete
            _sizeText.text = "Universe Ready! (5.2 MB)";
            _sizeText.color = new Color(0.4f, 1f, 0.6f);
            yield return new WaitForSeconds(0.5f);
        }

        IEnumerator RevealPhase()
        {
            float elapsed = 0f;

            while (elapsed < revealDuration)
            {
                elapsed += Time.deltaTime;
                float t = elapsed / revealDuration;

                // Fade out loading panel
                _loadingPanel.alpha = 1f - Mathf.SmoothStep(0, 1, t);

                // Fade in stats panel and controls
                float fadeIn = Mathf.SmoothStep(0, 1, t);
                _statsPanel.alpha = fadeIn;
                if (_controlsPanel != null)
                    _controlsPanel.alpha = fadeIn;

                yield return null;
            }

            _loadingPanel.gameObject.SetActive(false);
        }
        */

        IEnumerator TourPhase()
        {
            float elapsed = 0f;
            int segments = _tourPoints.Length - 1;
            float segmentDuration = tourDuration / segments;

            while (elapsed < tourDuration)
            {
                elapsed += Time.deltaTime;
                _tourProgress = elapsed / tourDuration;

                // Determine current segment
                float segmentProgress = (elapsed % segmentDuration) / segmentDuration;
                int segment = Mathf.Min((int)(elapsed / segmentDuration), segments - 1);

                // Smooth interpolation
                float t = Mathf.SmoothStep(0, 1, segmentProgress);

                // Interpolate position and look-at
                Vector3 pos = Vector3.Lerp(_tourPoints[segment], _tourPoints[segment + 1], t);
                Vector3 lookAt = Vector3.Lerp(_tourLookAts[segment], _tourLookAts[segment + 1], t);

                _mainCamera.transform.position = pos;
                _mainCamera.transform.LookAt(lookAt);

                yield return null;
            }
        }

        // =========================================================================
        // Free Roam
        // =========================================================================

        void UpdateFreeRoam()
        {
            float moveSpeed = 15f * Time.deltaTime;
            float rotSpeed = 80f * Time.deltaTime;

            if (Input.GetKey(KeyCode.LeftShift)) moveSpeed *= 3f;

            if (Input.GetKey(KeyCode.W)) _mainCamera.transform.Translate(Vector3.forward * moveSpeed);
            if (Input.GetKey(KeyCode.S)) _mainCamera.transform.Translate(Vector3.back * moveSpeed);
            if (Input.GetKey(KeyCode.A)) _mainCamera.transform.Translate(Vector3.left * moveSpeed);
            if (Input.GetKey(KeyCode.D)) _mainCamera.transform.Translate(Vector3.right * moveSpeed);
            if (Input.GetKey(KeyCode.Q)) _mainCamera.transform.Translate(Vector3.down * moveSpeed);
            if (Input.GetKey(KeyCode.E)) _mainCamera.transform.Translate(Vector3.up * moveSpeed);

            if (Input.GetMouseButton(1))
            {
                float mx = Input.GetAxis("Mouse X") * rotSpeed;
                float my = Input.GetAxis("Mouse Y") * rotSpeed;
                _mainCamera.transform.Rotate(Vector3.up, mx, Space.World);
                _mainCamera.transform.Rotate(Vector3.right, -my, Space.Self);
            }

            // Real-time morphing with arrow keys
            // Up/Down: Sun size
            if (Input.GetKey(KeyCode.UpArrow))
            {
                sunRadius = Mathf.Clamp(sunRadius + Time.deltaTime * 3, 1, 50);
                UpdateSdfParameters();
            }
            if (Input.GetKey(KeyCode.DownArrow))
            {
                sunRadius = Mathf.Clamp(sunRadius - Time.deltaTime * 3, 1, 50);
                UpdateSdfParameters();
            }

            // Left/Right: Smoothness (blend between shapes)
            if (Input.GetKey(KeyCode.LeftArrow))
            {
                smoothness = Mathf.Clamp(smoothness - Time.deltaTime * 2, 0.1f, 10);
                UpdateSdfParameters();
            }
            if (Input.GetKey(KeyCode.RightArrow))
            {
                smoothness = Mathf.Clamp(smoothness + Time.deltaTime * 2, 0.1f, 10);
                UpdateSdfParameters();
            }
        }

        void UpdateSdfParameters()
        {
            if (particleMode == ParticleMode.GPU_ComputeShader)
            {
                // GPU mode: Update compute shader uniforms
                if (_particleSystemGPU != null)
                {
                    _particleSystemGPU.SetSceneParameters(sunRadius, planetRadius, planetDistance, smoothness);
                }
            }
            else
            {
                // CPU modes: Update SdfWorld and rebuild
                if (_sdfWorld != null && _sdfWorld.objects.Count > 0)
                {
                    _sdfWorld.objects[0].radius = sunRadius;
                    _sdfWorld.globalSmoothness = smoothness;
                    _sdfWorld.RebuildWorld();
                }
            }
        }

        // =========================================================================
        // Stats
        // =========================================================================

        void UpdateStats()
        {
            if (_statsText == null || _statsPanel.alpha < 0.1f) return;

            float fps = 1f / Time.deltaTime;
            int particles = 0;
            float evalMs = 0;
            float physicsMs = 0;
            float gpuMs = 0;
            string mode = "Standard";
            string pipeline = "Parallel.For";

            switch (particleMode)
            {
                case ParticleMode.GPU_ComputeShader:
                    if (_particleSystemGPU != null)
                    {
                        particles = _particleSystemGPU.ActiveParticles;
                        gpuMs = _particleSystemGPU.GpuTimeMs * 1000f;
                        string sceneLabel = _particleSystemGPU.CurrentScene switch
                        {
                            GpuSceneType.Cosmic => "Cosmic",
                            GpuSceneType.Terrain => "Terrain",
                            GpuSceneType.Abstract => "Abstract",
                            _ => "Unknown"
                        };
                        mode = $"FULL GPU ({sceneLabel})";
                        pipeline = "Compute Shader";
                    }
                    break;

                case ParticleMode.CPU_RustSIMD_Burst:
                    if (_particleSystemUltimate != null)
                    {
                        particles = _particleSystemUltimate.ActiveParticles;
                        evalMs = _particleSystemUltimate.EvalTimeMs;
                        physicsMs = _particleSystemUltimate.PhysicsTimeMs;
                        mode = "ULTIMATE";
                        pipeline = "Burst + Indirect";
                    }
                    break;

                case ParticleMode.CPU_Standard:
                default:
                    if (_particleSystem != null)
                    {
                        particles = _particleSystem.ActiveParticles;
                        evalMs = _particleSystem.EvalTimeMs;
                        physicsMs = _particleSystem.UpdateTimeMs - evalMs;
                        mode = "Standard";
                        pipeline = "Parallel.For";
                    }
                    break;
            }

            float compileMs = _sdfWorld?.LastCompileTimeMs ?? 0;
            float totalMs = particleMode == ParticleMode.GPU_ComputeShader ? gpuMs : (evalMs + physicsMs);
            float throughput = particles > 0 && totalMs > 0.001f ? (particles / totalMs / 1000f) : 0;

            string timingInfo;
            if (particleMode == ParticleMode.GPU_ComputeShader)
            {
                string sliceInfo = "";
                if (_particleSystemGPU != null && _particleSystemGPU.UpdateDivisions > 1)
                {
                    sliceInfo = $"\nTime Slice: 1/{_particleSystemGPU.UpdateDivisions} ({_particleSystemGPU.ParticlesPerFrame:N0}/f)";
                }
                timingInfo = $@"GPU Dispatch: {gpuMs:F3} ms
CPU Transfer: ZERO{sliceInfo}
Total Frame: {gpuMs:F3} ms";
            }
            else
            {
                timingInfo = $@"Rust SIMD Eval: {evalMs:F2} ms
Burst Physics: {physicsMs:F2} ms
Total Frame: {totalMs:F2} ms";
            }

            string sceneHint = particleMode == ParticleMode.GPU_ComputeShader
                ? "\n[1]Cosmic [2]Terrain [3]Abstract"
                : "";

            _statsText.text = $@"=== {mode} DEEP FRIED ===
Pipeline: {pipeline}

FPS: {fps:F0}
Particles: {particles:N0}

{timingInfo}

Throughput: {throughput:F1}M pts/s
SDF Compile: {compileMs:F2} ms

=== CONTROLS ===
[WASD] Move  [QE] Up/Down
[Shift] Boost  [RMB] Look{sceneHint}

=== THE 5MB MIRACLE ===
Engine: 5.2 MB | Res: INF";
        }
    }
}
