// =============================================================================
// Cosmic × Fractal Demo - "The Fractal Universe"
// =============================================================================
// 4 Demo Modes showcasing SDF capabilities impossible with meshes:
//
//   [1] NORMAL    - Cosmic × Fractal solar system (fractal planet)
//   [2] FUSION    - Two SDF blobs merge/separate like liquid metal
//   [3] DESTROY   - Click to punch holes in fractal geometry
//   [4] MORPH     - Sphere → Box → Torus → Fractal smooth interpolation
//
// Controls:
//   [1][2][3][4]  Switch demo mode
//   [WASD]        Move            [QE]    Up/Down
//   [Shift]       Boost           [RMB]   Look
//   [Scroll]      Zoom            [Space] Reset zoom
//   [R]           Toggle Raymarching/Particles
//
//   Mode-specific:
//   [2] Fusion:  Arrow keys move Sphere 2, [F/G] adjust smoothness
//   [3] Destroy: Left-click to punch holes, [C] clear all holes
//   [4] Morph:   Auto-cycles, [M] pause/resume
//
// Author: Moroya Sakamoto
// =============================================================================

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using AliceSdf;

namespace SdfUniverse
{
    public class CosmicFractalDemo : MonoBehaviour
    {
        [Header("=== RENDERING MODE ===")]
        public FractalRenderMode renderMode = FractalRenderMode.Raymarching;

        [Header("=== PARTICLE MODE (if Particles selected) ===")]
        public ParticleMode particleMode = ParticleMode.GPU_ComputeShader;

        [Header("=== COSMIC PARAMETERS ===")]
        [Range(1f, 30f)] public float sunRadius = 8f;
        [Range(5f, 40f)] public float planetDistance = 22f;
        [Range(0f, 5f)] public float smoothness = 1.2f;

        [Header("=== FRACTAL PLANET ===")]
        [Range(2f, 20f)] public float planetRadius = 6f;
        [Range(0.2f, 5f)] public float holeSize = 0.8f;
        [Range(1f, 20f)] public float repeatScale = 5f;
        [Range(0f, 0.1f)] public float twistAmount = 0.01f;

        [Header("=== RING & MOON ===")]
        [Range(5f, 20f)] public float ringMajor = 10f;
        [Range(0.05f, 1f)] public float ringMinor = 0.25f;
        [Range(0.3f, 3f)] public float moonRadius = 0.8f;
        [Range(2f, 10f)] public float moonDistance = 5f;

        [Header("=== PARTICLES ===")]
        [Range(100000, 10000000)] public int particleCount = 2000000;
        [Range(0.001f, 0.1f)] public float particleSize = 0.03f;
        [Range(0.5f, 5f)] public float brightness = 2f;

        [Header("=== RAYMARCHING COLORS ===")]
        public Color sdfColor1 = new Color(0.9f, 0.4f, 0.1f, 1f);
        public Color sdfColor2 = new Color(0.1f, 0.6f, 0.9f, 1f);

        [Header("=== DEMO TIMING ===")]
        public float tourDuration = 20f;
        public float diveDuration = 12f;

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

        // SDF (Rust FFI)
        private SdfNode _sdfNode;
        private CompiledSdf _compiledSdf;

        // UI
        private Canvas _canvas;
        private Text _titleText;
        private Text _subtitleText;
        private Text _statsText;
        private Text _controlsText;

        // Camera tour
        private Vector3[] _tourPoints;
        private Vector3[] _tourLookAts;

        // State
        private DemoPhase _phase = DemoPhase.Tour;
        public enum DemoPhase { Tour, Dive, FreeRoam }

        // === Interactive Demo Mode ===
        private int _demoMode = 0; // 0=Normal, 1=Fusion, 2=Destruction, 3=Morph

        // Fusion state
        private Vector3 _sphere1Pos = new Vector3(-5, 0, 0);
        private Vector3 _sphere2Pos = new Vector3(5, 0, 0);
        private float _fusionK = 0.5f;

        // Destruction state
        private List<Vector4> _holePositions = new List<Vector4>();
        private float _holeRadius = 1.5f;
        private const int MaxHoles = 16;

        // Morph state
        private float _morphT = 0f;
        private int _morphA = 0; // 0=Sphere, 1=Box, 2=Torus, 3=Fractal
        private int _morphB = 1;
        private bool _morphPaused = false;
        private float _morphCycleTime = 0f;
        private float _morphCycleDuration = 4f; // seconds per shape transition

        private static readonly string[] ShapeNames = { "Sphere", "Box", "Torus", "Fractal" };

        // =========================================================================
        // Bootstrap
        // =========================================================================

        void Awake()
        {
            _currentRenderMode = renderMode;

            SetupCamera();
            SetupSdfWorld();

            if (renderMode == FractalRenderMode.Raymarching)
            {
                SetupRaymarching();
            }
            else
            {
                SetupParticleSystem();
                if (particleMode != ParticleMode.GPU_ComputeShader)
                {
                    BuildFractalUniverse();
                }
            }

            CreateUI();
            CreateLighting();
            SetupCameraTour();

            Debug.Log("[CosmicFractalDemo] The Fractal Universe initialized — [1]Normal [2]Fusion [3]Destroy [4]Morph");
        }

        void Start()
        {
            StartCoroutine(RunDemo());
        }

        void Update()
        {
            HandleInput();
            UpdateModeLogic();
            UpdateParameters();
            UpdateStats();

            if (_phase == DemoPhase.FreeRoam)
            {
                UpdateFreeRoam();
            }
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
            _mainCamera.farClipPlane = 1000f;
            _mainCamera.nearClipPlane = 0.01f;
            _mainCamera.fieldOfView = 60f;

            _zoomCamera = _mainCamera.gameObject.GetComponent<InfiniteZoomCamera>();
            if (_zoomCamera == null)
            {
                _zoomCamera = _mainCamera.gameObject.AddComponent<InfiniteZoomCamera>();
            }
            _zoomCamera.minDistance = 0.01f;
            _zoomCamera.maxDistance = 500f;
            _zoomCamera.startDistance = 100f;
            _zoomCamera.zoomSpeed = 20f;
            _zoomCamera.showZoomUI = true;

            _mainCamera.transform.position = new Vector3(80, 40, 80);
            _mainCamera.transform.LookAt(Vector3.zero);

            StartCoroutine(SetInitialZoom(120f));
        }

        IEnumerator SetInitialZoom(float distance)
        {
            yield return null;
            if (_zoomCamera != null)
                _zoomCamera.SetDistance(distance);
        }

        void SetupSdfWorld()
        {
            _sdfWorld = GetComponent<SdfWorld>();
            if (_sdfWorld == null)
                _sdfWorld = gameObject.AddComponent<SdfWorld>();
            _sdfWorld.skyEnabled = false;
            _sdfWorld.groundEnabled = false;
        }

        void BuildFractalUniverse()
        {
            var sun = SdfNode.Sphere(sunRadius);

            var planetSphere = SdfNode.Sphere(planetRadius);
            float length = planetRadius * 3f;
            var barX = SdfNode.Box(length, holeSize, holeSize);
            var barY = SdfNode.Box(holeSize, length, holeSize);
            var barZ = SdfNode.Box(holeSize, holeSize, length);
            var cross = barX.Union(barY.Union(barZ));
            var repeatedCross = cross.Repeat(new Vector3(repeatScale, repeatScale, repeatScale));
            var fractalPlanet = planetSphere.Subtract(repeatedCross);

            if (twistAmount > 0.001f)
                fractalPlanet = fractalPlanet.Twist(twistAmount);

            fractalPlanet = fractalPlanet.Translate(planetDistance, 0, 0);

            var ring = SdfNode.Torus(ringMajor, ringMinor);
            ring = ring.RotateEuler(new Vector3(15, 0, 10));
            ring = ring.Translate(planetDistance, 0, 0);

            var moon = SdfNode.Sphere(moonRadius);
            moon = moon.Translate(planetDistance + moonDistance, 2f, 0);

            var scene = sun.SmoothUnion(fractalPlanet, smoothness);
            scene = scene.SmoothUnion(ring, smoothness * 0.5f);
            scene = scene.SmoothUnion(moon, smoothness * 0.8f);

            _sdfNode = scene;
            _compiledSdf = _sdfNode.Compile();

            if (_sdfWorld != null)
                _sdfWorld.SetCompiledSdf(_compiledSdf);

            Debug.Log($"[CosmicFractalDemo] Built fractal universe: {_compiledSdf?.InstructionCount ?? 0} instructions");
        }

        // =========================================================================
        // Raymarching
        // =========================================================================

        void SetupRaymarching()
        {
            _raymarchObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _raymarchObject.name = "SDF_CosmicFractal_Surface";
            _raymarchObject.transform.position = Vector3.zero;
            _raymarchObject.transform.localScale = Vector3.one * 200f;

            var shader = Shader.Find("SdfUniverse/CosmicFractal");
            if (shader == null)
                shader = Shader.Find("SdfUniverse/InfiniteSurface");

            if (shader == null)
            {
                Debug.LogError("[CosmicFractalDemo] No suitable raymarching shader found!");
                return;
            }

            _raymarchMaterial = new Material(shader);
            UpdateRaymarchMaterialParams();
            _raymarchObject.GetComponent<Renderer>().material = _raymarchMaterial;

            Debug.Log("[CosmicFractalDemo] Raymarching setup complete");
        }

        void UpdateRaymarchMaterialParams()
        {
            if (_raymarchMaterial == null) return;

            // Base params
            _raymarchMaterial.SetColor("_Color", sdfColor1);
            _raymarchMaterial.SetColor("_Color2", sdfColor2);
            _raymarchMaterial.SetFloat("_SunRadius", sunRadius);
            _raymarchMaterial.SetFloat("_PlanetRadius", planetRadius);
            _raymarchMaterial.SetFloat("_PlanetDistance", planetDistance);
            _raymarchMaterial.SetFloat("_Smoothness", smoothness);
            _raymarchMaterial.SetFloat("_HoleSize", holeSize);
            _raymarchMaterial.SetFloat("_RepeatScale", repeatScale);
            _raymarchMaterial.SetFloat("_TwistAmount", twistAmount);
            _raymarchMaterial.SetFloat("_RingMajor", ringMajor);
            _raymarchMaterial.SetFloat("_RingMinor", ringMinor);
            _raymarchMaterial.SetFloat("_MaxDist", 300f);

            // Demo mode
            _raymarchMaterial.SetInt("_DemoMode", _demoMode);

            // Fusion params
            _raymarchMaterial.SetVector("_Sphere1Pos", new Vector4(_sphere1Pos.x, _sphere1Pos.y, _sphere1Pos.z, 0));
            _raymarchMaterial.SetVector("_Sphere2Pos", new Vector4(_sphere2Pos.x, _sphere2Pos.y, _sphere2Pos.z, 0));
            _raymarchMaterial.SetFloat("_DynamicK", _fusionK);

            // Destruction params
            _raymarchMaterial.SetInt("_HoleCount", _holePositions.Count);
            _raymarchMaterial.SetFloat("_HoleRadius", _holeRadius);
            if (_holePositions.Count > 0)
            {
                // Pad to 16
                Vector4[] holes = new Vector4[16];
                for (int i = 0; i < 16; i++)
                    holes[i] = i < _holePositions.Count ? _holePositions[i] : Vector4.zero;
                _raymarchMaterial.SetVectorArray("_HolePositions", holes);
            }

            // Morph params
            _raymarchMaterial.SetFloat("_MorphT", _morphT);
            _raymarchMaterial.SetInt("_MorphA", _morphA);
            _raymarchMaterial.SetInt("_MorphB", _morphB);

            // Adaptive epsilon
            if (_zoomCamera != null)
            {
                float epsilon = Mathf.Max(0.0005f, _zoomCamera.CurrentDistance * 0.0001f);
                _raymarchMaterial.SetFloat("_SurfaceEpsilon", epsilon);
            }
        }

        // =========================================================================
        // Particle System
        // =========================================================================

        void SetupParticleSystem()
        {
            switch (particleMode)
            {
                case ParticleMode.GPU_ComputeShader:
                    _particleSystemGPU = GetComponent<SdfParticleSystem_GPU>();
                    if (_particleSystemGPU == null)
                        _particleSystemGPU = gameObject.AddComponent<SdfParticleSystem_GPU>();

                    _particleSystemGPU.sceneType = GpuSceneType.Fractal;
                    _particleSystemGPU.particleCount = particleCount;
                    _particleSystemGPU.particleSize = particleSize;
                    _particleSystemGPU.brightness = brightness;
                    _particleSystemGPU.spawnRadius = planetDistance * 2f;
                    _particleSystemGPU.maxDistance = planetDistance * 4f;
                    _particleSystemGPU.flowSpeed = 1f;
                    _particleSystemGPU.surfaceAttraction = 5f;
                    _particleSystemGPU.noiseStrength = 0.2f;
                    _particleSystemGPU.sunRadius = sunRadius;
                    _particleSystemGPU.planetRadius = planetRadius;
                    _particleSystemGPU.planetDistance = planetDistance;
                    _particleSystemGPU.smoothness = smoothness;
                    _particleSystemGPU.boxSize = planetRadius;
                    _particleSystemGPU.holeSize = holeSize;
                    _particleSystemGPU.repeatScale = repeatScale;
                    _particleSystemGPU.twistAmount = twistAmount;
                    _particleSystemGPU.showStats = false;
                    break;

                case ParticleMode.CPU_RustSIMD_Burst:
                    _particleSystemUltimate = GetComponent<SdfParticleSystem_Ultimate>();
                    if (_particleSystemUltimate == null)
                        _particleSystemUltimate = gameObject.AddComponent<SdfParticleSystem_Ultimate>();

                    _particleSystemUltimate.particleCount = particleCount;
                    _particleSystemUltimate.spawnRadius = planetDistance * 2f;
                    _particleSystemUltimate.maxDistance = planetDistance * 4f;
                    _particleSystemUltimate.particleSize = particleSize;
                    _particleSystemUltimate.flowSpeed = 1f;
                    _particleSystemUltimate.surfaceAttraction = 5f;
                    _particleSystemUltimate.noiseStrength = 0.2f;
                    _particleSystemUltimate.brightness = brightness;
                    _particleSystemUltimate.showStats = false;
                    break;

                case ParticleMode.CPU_Standard:
                default:
                    _particleSystemCPU = GetComponent<SdfParticleSystem>();
                    if (_particleSystemCPU == null)
                        _particleSystemCPU = gameObject.AddComponent<SdfParticleSystem>();

                    _particleSystemCPU.particleCount = particleCount;
                    _particleSystemCPU.spawnRadius = planetDistance * 2f;
                    _particleSystemCPU.maxDistance = planetDistance * 4f;
                    _particleSystemCPU.particleSize = particleSize;
                    _particleSystemCPU.flowSpeed = 1f;
                    _particleSystemCPU.surfaceAttraction = 5f;
                    _particleSystemCPU.noiseStrength = 0.2f;
                    _particleSystemCPU.flowMode = ParticleFlowMode.SurfaceFlow;
                    break;
            }
        }

        void ToggleRenderMode()
        {
            if (_currentRenderMode == FractalRenderMode.Raymarching)
            {
                _currentRenderMode = FractalRenderMode.Particles;
                if (_raymarchObject != null) _raymarchObject.SetActive(false);
                if (_particleSystemGPU == null && _particleSystemCPU == null && _particleSystemUltimate == null)
                    SetupParticleSystem();
                else
                {
                    if (_particleSystemGPU != null) _particleSystemGPU.enabled = true;
                    if (_particleSystemCPU != null) _particleSystemCPU.enabled = true;
                    if (_particleSystemUltimate != null) _particleSystemUltimate.enabled = true;
                }
            }
            else
            {
                _currentRenderMode = FractalRenderMode.Raymarching;
                if (_particleSystemGPU != null) _particleSystemGPU.enabled = false;
                if (_particleSystemCPU != null) _particleSystemCPU.enabled = false;
                if (_particleSystemUltimate != null) _particleSystemUltimate.enabled = false;
                if (_raymarchObject == null) SetupRaymarching();
                else _raymarchObject.SetActive(true);
            }
        }

        // =========================================================================
        // Camera Tour
        // =========================================================================

        void SetupCameraTour()
        {
            Vector3 planetPos = new Vector3(planetDistance, 0, 0);
            _tourPoints = new Vector3[] {
                new Vector3(80, 40, 80),
                new Vector3(40, 15, 40),
                new Vector3(sunRadius + 3, 2, 0),
                new Vector3(planetDistance - 2, 8, 15),
                new Vector3(planetDistance + planetRadius + 2, 3, 3),
                new Vector3(planetDistance, 0, 0),
                new Vector3(50, 25, 50),
            };
            _tourLookAts = new Vector3[] {
                Vector3.zero, Vector3.zero, Vector3.zero,
                planetPos, planetPos,
                planetPos + new Vector3(1, 0, 0),
                Vector3.zero,
            };
        }

        void CreateLighting()
        {
            if (FindObjectOfType<Light>() == null)
            {
                var lightObj = new GameObject("Cosmic Light");
                var light = lightObj.AddComponent<Light>();
                light.type = LightType.Directional;
                light.color = new Color(1f, 0.9f, 0.8f);
                light.intensity = 1.5f;
                lightObj.transform.rotation = Quaternion.Euler(35, -45, 0);
            }
        }

        // =========================================================================
        // Demo Sequence
        // =========================================================================

        IEnumerator RunDemo()
        {
            _phase = DemoPhase.Tour;
            yield return StartCoroutine(TourPhase());

            _phase = DemoPhase.Dive;
            yield return StartCoroutine(DivePhase());

            _phase = DemoPhase.FreeRoam;
            Debug.Log("[CosmicFractalDemo] Free Roam! [1]Normal [2]Fusion [3]Destroy [4]Morph");
        }

        IEnumerator TourPhase()
        {
            float elapsed = 0f;
            int segments = _tourPoints.Length - 1;
            float segmentDuration = tourDuration / segments;

            while (elapsed < tourDuration)
            {
                elapsed += Time.deltaTime;
                float segmentProgress = (elapsed % segmentDuration) / segmentDuration;
                int segment = Mathf.Min((int)(elapsed / segmentDuration), segments - 1);
                float t = Mathf.SmoothStep(0, 1, segmentProgress);

                _mainCamera.transform.position = Vector3.Lerp(_tourPoints[segment], _tourPoints[segment + 1], t);
                _mainCamera.transform.LookAt(Vector3.Lerp(_tourLookAts[segment], _tourLookAts[segment + 1], t));
                yield return null;
            }
        }

        IEnumerator DivePhase()
        {
            Vector3 planetPos = new Vector3(planetDistance, 0, 0);
            Vector3 startPos = planetPos + new Vector3(planetRadius + 1, 1, 1);
            float stopDist = planetRadius * 0.8f;
            Vector3 endPos = planetPos + new Vector3(stopDist, stopDist * 0.4f, stopDist * 0.3f);
            float elapsed = 0f;

            while (elapsed < diveDuration)
            {
                elapsed += Time.deltaTime;
                float t = Mathf.SmoothStep(0, 1, elapsed / diveDuration);
                float logT = 1f - Mathf.Pow(1f - t, 3f);

                Vector3 pos = Vector3.Lerp(startPos, endPos, logT);
                _mainCamera.transform.position = pos;
                _mainCamera.transform.LookAt(planetPos);

                if (_zoomCamera != null)
                    _zoomCamera.SetDistance(Vector3.Distance(pos, planetPos));

                yield return null;
            }
            _mainCamera.transform.position = endPos;
        }

        // =========================================================================
        // Input & Mode Logic
        // =========================================================================

        void HandleInput()
        {
            // Mode switching
            if (Input.GetKeyDown(KeyCode.Alpha1)) SwitchMode(0);
            if (Input.GetKeyDown(KeyCode.Alpha2)) SwitchMode(1);
            if (Input.GetKeyDown(KeyCode.Alpha3)) SwitchMode(2);
            if (Input.GetKeyDown(KeyCode.Alpha4)) SwitchMode(3);

            if (Input.GetKeyDown(KeyCode.R)) ToggleRenderMode();
            if (Input.GetKeyDown(KeyCode.Space) && _zoomCamera != null) _zoomCamera.ResetZoom();

            // Mode-specific input
            switch (_demoMode)
            {
                case 1: HandleFusionInput(); break;
                case 2: HandleDestructionInput(); break;
                case 3: HandleMorphInput(); break;
            }
        }

        void SwitchMode(int mode)
        {
            _demoMode = mode;

            string[] modeNames = { "NORMAL", "FUSION", "DESTRUCTION", "MORPHING" };
            Debug.Log($"[CosmicFractalDemo] Mode: {modeNames[mode]}");

            // Reset camera for non-normal modes
            if (mode == 1)
            {
                // Fusion: side view
                _mainCamera.transform.position = new Vector3(0, 8, 25);
                _mainCamera.transform.LookAt(Vector3.zero);
                _sphere1Pos = new Vector3(-5, 0, 0);
                _sphere2Pos = new Vector3(5, 0, 0);
                _fusionK = 0.5f;
            }
            else if (mode == 2)
            {
                // Destruction: front view of fractal box
                _mainCamera.transform.position = new Vector3(0, 5, 30);
                _mainCamera.transform.LookAt(Vector3.zero);
                _holePositions.Clear();
            }
            else if (mode == 3)
            {
                // Morph: front view
                _mainCamera.transform.position = new Vector3(0, 3, 20);
                _mainCamera.transform.LookAt(Vector3.zero);
                _morphT = 0f;
                _morphA = 0;
                _morphB = 1;
                _morphCycleTime = 0f;
                _morphPaused = false;
            }
            else
            {
                // Normal: overview
                _mainCamera.transform.position = new Vector3(50, 25, 50);
                _mainCamera.transform.LookAt(Vector3.zero);
            }

            // Update title
            if (_titleText != null)
            {
                string[] titles = { "THE FRACTAL UNIVERSE", "LIQUID METAL FUSION", "SDF DESTRUCTION", "INFINITE MORPHING" };
                _titleText.text = titles[mode];
            }
            if (_subtitleText != null)
            {
                string[] subtitles = {
                    "Cosmic x Fractal Fusion",
                    "Two SDFs merge and separate — impossible with meshes",
                    "Click to punch holes in fractal geometry",
                    "Sphere → Box → Torus → Fractal — infinite resolution morph"
                };
                _subtitleText.text = subtitles[mode];
            }
        }

        // --- Fusion Input ---
        void HandleFusionInput()
        {
            float speed = 8f * Time.deltaTime;
            if (Input.GetKey(KeyCode.UpArrow))    _sphere2Pos.z -= speed;
            if (Input.GetKey(KeyCode.DownArrow))   _sphere2Pos.z += speed;
            if (Input.GetKey(KeyCode.LeftArrow))   _sphere2Pos.x -= speed;
            if (Input.GetKey(KeyCode.RightArrow))  _sphere2Pos.x += speed;
            if (Input.GetKey(KeyCode.F)) _fusionK = Mathf.Clamp(_fusionK - Time.deltaTime * 2, 0.1f, 8f);
            if (Input.GetKey(KeyCode.G)) _fusionK = Mathf.Clamp(_fusionK + Time.deltaTime * 2, 0.1f, 8f);

            // Auto-compute dynamic K based on distance
            float dist = Vector3.Distance(_sphere1Pos, _sphere2Pos);
            float autoK = Mathf.Lerp(4f, 0.1f, Mathf.Clamp01(dist / 15f));
            _fusionK = Mathf.Lerp(_fusionK, autoK, Time.deltaTime * 3f);
        }

        // --- Destruction Input ---
        void HandleDestructionInput()
        {
            if (Input.GetKeyDown(KeyCode.C))
            {
                _holePositions.Clear();
                Debug.Log("[CosmicFractalDemo] Holes cleared");
            }

            if (Input.GetMouseButtonDown(0) && _holePositions.Count < MaxHoles)
            {
                // Raycast from camera to find hit point on SDF surface
                Ray ray = _mainCamera.ScreenPointToRay(Input.mousePosition);
                Vector3 hitPoint = RaycastSdf(ray.origin, ray.direction, 100f);
                if (hitPoint != Vector3.zero)
                {
                    _holePositions.Add(new Vector4(hitPoint.x, hitPoint.y, hitPoint.z, 0));
                    Debug.Log($"[CosmicFractalDemo] Hole #{_holePositions.Count} at {hitPoint:F1}");
                }
            }
        }

        // Simple CPU-side SDF raycast for destruction mode
        Vector3 RaycastSdf(Vector3 ro, Vector3 rd, float maxDist)
        {
            float t = 0f;
            for (int i = 0; i < 64; i++)
            {
                Vector3 p = ro + rd * t;
                // Approximate the destruction scene SDF on CPU
                float d = EvalDestructionSdf(p);
                if (d < 0.05f) return p;
                t += Mathf.Max(d, 0.1f);
                if (t > maxDist) break;
            }
            return Vector3.zero;
        }

        float EvalDestructionSdf(Vector3 p)
        {
            // Box
            Vector3 d = new Vector3(Mathf.Abs(p.x) - 15, Mathf.Abs(p.y) - 10, Mathf.Abs(p.z) - 15);
            float box = new Vector3(Mathf.Max(d.x, 0), Mathf.Max(d.y, 0), Mathf.Max(d.z, 0)).magnitude
                        + Mathf.Min(Mathf.Max(d.x, Mathf.Max(d.y, d.z)), 0);

            // Fractal cross (approximate)
            Vector3 rp = p - Vector3.Scale(
                new Vector3(repeatScale, repeatScale, repeatScale),
                new Vector3(Mathf.Round(p.x / repeatScale), Mathf.Round(p.y / repeatScale), Mathf.Round(p.z / repeatScale)));
            float hs = holeSize * 0.5f;
            float barX = Mathf.Max(Mathf.Abs(rp.y) - hs, Mathf.Abs(rp.z) - hs);
            float barY = Mathf.Max(Mathf.Abs(rp.x) - hs, Mathf.Abs(rp.z) - hs);
            float barZ = Mathf.Max(Mathf.Abs(rp.x) - hs, Mathf.Abs(rp.y) - hs);
            float cross = Mathf.Min(barX, Mathf.Min(barY, barZ));

            float scene = Mathf.Max(-cross, box);

            // Existing holes
            for (int i = 0; i < _holePositions.Count; i++)
            {
                Vector3 hp = new Vector3(_holePositions[i].x, _holePositions[i].y, _holePositions[i].z);
                float hole = (p - hp).magnitude - _holeRadius;
                scene = Mathf.Max(-hole, scene);
            }

            return scene;
        }

        // --- Morph Input ---
        void HandleMorphInput()
        {
            if (Input.GetKeyDown(KeyCode.M))
            {
                _morphPaused = !_morphPaused;
                Debug.Log($"[CosmicFractalDemo] Morph {(_morphPaused ? "PAUSED" : "RESUMED")}");
            }
        }

        // =========================================================================
        // Mode Logic (per-frame)
        // =========================================================================

        void UpdateModeLogic()
        {
            if (_demoMode == 3 && !_morphPaused)
            {
                // Auto-cycle morphing
                _morphCycleTime += Time.deltaTime;
                float totalPhase = _morphCycleTime / _morphCycleDuration;

                int shapeCount = 4;
                int currentPair = (int)totalPhase % shapeCount;
                _morphA = currentPair;
                _morphB = (currentPair + 1) % shapeCount;
                _morphT = totalPhase - Mathf.Floor(totalPhase);

                // Smooth ease
                _morphT = _morphT * _morphT * (3f - 2f * _morphT);
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

            // Normal mode: arrow keys adjust scene params
            if (_demoMode == 0)
            {
                if (Input.GetKey(KeyCode.UpArrow))
                {
                    sunRadius = Mathf.Clamp(sunRadius + Time.deltaTime * 3, 1, 30);
                    UpdateSdfParameters();
                }
                if (Input.GetKey(KeyCode.DownArrow))
                {
                    sunRadius = Mathf.Clamp(sunRadius - Time.deltaTime * 3, 1, 30);
                    UpdateSdfParameters();
                }
                if (Input.GetKey(KeyCode.LeftArrow))
                {
                    smoothness = Mathf.Clamp(smoothness - Time.deltaTime * 2, 0.1f, 5);
                    UpdateSdfParameters();
                }
                if (Input.GetKey(KeyCode.RightArrow))
                {
                    smoothness = Mathf.Clamp(smoothness + Time.deltaTime * 2, 0.1f, 5);
                    UpdateSdfParameters();
                }
            }
        }

        void UpdateSdfParameters()
        {
            if (particleMode == ParticleMode.GPU_ComputeShader && _particleSystemGPU != null)
                _particleSystemGPU.SetSceneParameters(sunRadius, planetRadius, planetDistance, smoothness);
            else if (_sdfWorld != null)
                BuildFractalUniverse();
        }

        void UpdateParameters()
        {
            UpdateRaymarchMaterialParams();

            if (particleMode == ParticleMode.GPU_ComputeShader && _particleSystemGPU != null)
            {
                _particleSystemGPU.boxSize = planetRadius;
                _particleSystemGPU.holeSize = holeSize;
                _particleSystemGPU.repeatScale = repeatScale;
                _particleSystemGPU.twistAmount = twistAmount;
                _particleSystemGPU.particleSize = particleSize;
                _particleSystemGPU.brightness = brightness;
            }
        }

        // =========================================================================
        // UI
        // =========================================================================

        void CreateUI()
        {
            var canvasObj = new GameObject("CosmicFractalCanvas");
            _canvas = canvasObj.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 100;
            canvasObj.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvasObj.AddComponent<GraphicRaycaster>();

            // Title
            var titleObj = CreateText(_canvas.transform, "Title", "THE FRACTAL UNIVERSE", 48,
                new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Color(1f, 0.6f, 0.2f));
            _titleText = titleObj.GetComponent<Text>();
            titleObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, -30);

            // Subtitle
            var subObj = CreateText(_canvas.transform, "Subtitle", "Cosmic x Fractal Fusion", 20,
                new Vector2(0.5f, 1f), new Vector2(0.5f, 1f), new Color(0.6f, 0.6f, 0.6f));
            _subtitleText = subObj.GetComponent<Text>();
            subObj.GetComponent<RectTransform>().anchoredPosition = new Vector2(0, -80);

            // Stats panel (top left)
            var statsPanel = new GameObject("StatsPanel");
            statsPanel.transform.SetParent(_canvas.transform, false);
            var statsPanelRect = statsPanel.AddComponent<RectTransform>();
            statsPanelRect.anchorMin = new Vector2(0, 0.6f);
            statsPanelRect.anchorMax = new Vector2(0.3f, 1f);
            statsPanelRect.offsetMin = new Vector2(10, 10);
            statsPanelRect.offsetMax = new Vector2(-10, -100);

            statsPanel.AddComponent<Image>().color = new Color(0, 0, 0, 0.75f);

            var statsObj = CreateText(statsPanel.transform, "Stats", "", 13,
                Vector2.zero, Vector2.one, new Color(0.4f, 1f, 0.6f));
            _statsText = statsObj.GetComponent<Text>();
            _statsText.alignment = TextAnchor.UpperLeft;
            statsObj.GetComponent<RectTransform>().offsetMin = new Vector2(10, 10);
            statsObj.GetComponent<RectTransform>().offsetMax = new Vector2(-10, -10);

            // Controls panel (bottom left)
            var controlsPanel = new GameObject("ControlsPanel");
            controlsPanel.transform.SetParent(_canvas.transform, false);
            var controlsPanelRect = controlsPanel.AddComponent<RectTransform>();
            controlsPanelRect.anchorMin = new Vector2(0, 0);
            controlsPanelRect.anchorMax = new Vector2(0.25f, 0.28f);
            controlsPanelRect.offsetMin = new Vector2(10, 10);
            controlsPanelRect.offsetMax = new Vector2(-10, -10);

            controlsPanel.AddComponent<Image>().color = new Color(0, 0, 0, 0.6f);

            var controlsObj = CreateText(controlsPanel.transform, "Controls",
                "=== DEMO MODES ===\n" +
                "[1] Normal  [2] Fusion\n" +
                "[3] Destroy [4] Morph\n\n" +
                "[WASD] Move  [QE] Up/Down\n" +
                "[Shift] Boost  [RMB] Look\n" +
                "[R] Toggle Render Mode",
                11, Vector2.zero, Vector2.one, new Color(0.7f, 0.7f, 0.7f));
            _controlsText = controlsObj.GetComponent<Text>();
            _controlsText.alignment = TextAnchor.UpperLeft;
            controlsObj.GetComponent<RectTransform>().offsetMin = new Vector2(8, 8);
            controlsObj.GetComponent<RectTransform>().offsetMax = new Vector2(-8, -8);
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

        // =========================================================================
        // Stats
        // =========================================================================

        void UpdateStats()
        {
            if (_statsText == null) return;

            float fps = 1f / Time.deltaTime;

            string[] modeNames = {
                "<color=lime>NORMAL</color> — Cosmic x Fractal",
                "<color=orange>FUSION</color> — Liquid Metal",
                "<color=red>DESTRUCTION</color> — Punch Holes",
                "<color=cyan>MORPHING</color> — Shape Shift"
            };

            string modeInfo = "";
            switch (_demoMode)
            {
                case 0:
                    modeInfo = $"Sun: {sunRadius:F1}  Blend: {smoothness:F1}\nPlanet: Menger r={planetRadius:F1}";
                    break;
                case 1:
                    float dist = Vector3.Distance(_sphere1Pos, _sphere2Pos);
                    modeInfo = $"Distance: {dist:F1}\nSmooth K: {_fusionK:F2}\n<color=yellow>[Arrows] Move  [F/G] K</color>";
                    break;
                case 2:
                    modeInfo = $"Holes: {_holePositions.Count}/{MaxHoles}\nRadius: {_holeRadius:F1}\n<color=yellow>[LMB] Punch  [C] Clear</color>";
                    break;
                case 3:
                    modeInfo = $"{ShapeNames[_morphA]} → {ShapeNames[_morphB]}\nProgress: {_morphT:P0}\n<color=yellow>[M] {(_morphPaused ? "Resume" : "Pause")}</color>";
                    break;
            }

            string renderMode = _currentRenderMode == FractalRenderMode.Raymarching
                ? "<color=lime>RAYMARCHING</color>" : "Particles";

            string phaseText = _phase switch
            {
                DemoPhase.Tour => "<color=yellow>TOUR</color>",
                DemoPhase.Dive => "<color=orange>DIVE</color>",
                DemoPhase.FreeRoam => "<color=lime>FREE</color>",
                _ => ""
            };

            _statsText.text = $@"=== FRACTAL UNIVERSE ===
Demo: {modeNames[_demoMode]}
Render: {renderMode}  Phase: {phaseText}
FPS: <color=lime>{fps:F0}</color>

{modeInfo}

=== SDF IMPOSSIBLE ===
Polygons: <color=cyan>0</color>
Formulas: <color=orange>{(_demoMode == 0 ? "4" : "dynamic")}</color>
Resolution: <color=yellow>INFINITE</color>
Mesh can't do this: <color=lime>TRUE</color>";
        }

        // =========================================================================
        // Gizmos
        // =========================================================================

        void OnDrawGizmosSelected()
        {
            if (_demoMode == 1)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawWireSphere(_sphere1Pos, 4f);
                Gizmos.color = Color.blue;
                Gizmos.DrawWireSphere(_sphere2Pos, 3f);
            }
            else if (_demoMode == 2)
            {
                Gizmos.color = Color.red;
                foreach (var h in _holePositions)
                    Gizmos.DrawWireSphere(new Vector3(h.x, h.y, h.z), _holeRadius);
            }
            else
            {
                Gizmos.color = new Color(1f, 0.6f, 0.1f, 0.5f);
                Gizmos.DrawWireSphere(Vector3.zero, sunRadius);
                Vector3 pp = new Vector3(planetDistance, 0, 0);
                Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.5f);
                Gizmos.DrawWireSphere(pp, planetRadius);
            }
        }
    }
}
