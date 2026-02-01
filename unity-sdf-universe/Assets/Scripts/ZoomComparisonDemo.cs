// =============================================================================
// Zoom Comparison Demo - Traditional vs ALICE-SDF
// =============================================================================
// Side-by-side comparison showing:
//   LEFT:  Traditional Mesh + Texture (pixelates when zoomed)
//   RIGHT: Raymarching + Procedural Texture (infinite resolution)
//
// "One side blurs. The other side reveals new detail. Forever."
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;
using UnityEngine.UI;

namespace SdfUniverse
{
    public class ZoomComparisonDemo : MonoBehaviour
    {
        [Header("=== ZOOM SETTINGS ===")]
        [Range(1f, 50f)]
        public float zoomSpeed = 15f;
        public float minDistance = 0.0001f;
        public float maxDistance = 100f;
        public float startDistance = 50f;

        [Header("=== TRADITIONAL SIDE (LEFT) ===")]
        public GameObject traditionalObject;
        public Texture2D lowResTexture;

        [Header("=== SDF SIDE (RIGHT) ===")]
        public Material sdfMaterial;
        public float detailScale = 10f;

        [Header("=== UI ===")]
        public bool showComparisonUI = true;

        // Private
        private Camera _leftCamera;
        private Camera _rightCamera;
        private GameObject _leftSphere;
        private GameObject _rightCube;
        private float _currentDistance;
        private float _yaw = 0f;
        private float _pitch = 30f;
        private Canvas _canvas;
        private Text _leftLabel;
        private Text _rightLabel;
        private Text _zoomLabel;
        private Text _instructionLabel;

        void Start()
        {
            _currentDistance = startDistance;
            SetupCameras();
            SetupObjects();
            CreateUI();

            Debug.Log("[ZoomComparison] Demo initialized - Scroll to zoom, see the difference!");
        }

        void SetupCameras()
        {
            // Destroy existing main camera
            if (Camera.main != null)
            {
                DestroyImmediate(Camera.main.gameObject);
            }

            // LEFT CAMERA (Traditional)
            var leftCamObj = new GameObject("LeftCamera");
            _leftCamera = leftCamObj.AddComponent<Camera>();
            _leftCamera.rect = new Rect(0, 0, 0.5f, 1f);
            _leftCamera.backgroundColor = new Color(0.1f, 0.05f, 0.05f);
            _leftCamera.clearFlags = CameraClearFlags.SolidColor;
            _leftCamera.nearClipPlane = 0.0001f;
            _leftCamera.farClipPlane = 1000f;
            _leftCamera.depth = 0;

            // RIGHT CAMERA (SDF Raymarching)
            var rightCamObj = new GameObject("RightCamera");
            _rightCamera = rightCamObj.AddComponent<Camera>();
            _rightCamera.rect = new Rect(0.5f, 0, 0.5f, 1f);
            _rightCamera.backgroundColor = new Color(0.02f, 0.02f, 0.05f);
            _rightCamera.clearFlags = CameraClearFlags.SolidColor;
            _rightCamera.nearClipPlane = 0.0001f;
            _rightCamera.farClipPlane = 1000f;
            _rightCamera.depth = 0;

            // Add audio listener to left camera
            leftCamObj.AddComponent<AudioListener>();
        }

        void SetupObjects()
        {
            // === LEFT SIDE: Traditional Mesh + Texture ===
            _leftSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _leftSphere.name = "Traditional_Sphere";
            _leftSphere.transform.position = new Vector3(-30, 0, 0);
            _leftSphere.transform.localScale = Vector3.one * 10f;

            // Apply low-res texture material
            var traditionalMat = new Material(Shader.Find("Standard"));
            if (lowResTexture != null)
            {
                traditionalMat.mainTexture = lowResTexture;
            }
            else
            {
                // Create a procedural low-res texture
                traditionalMat.mainTexture = CreateLowResTexture(64);
            }
            traditionalMat.SetFloat("_Glossiness", 0.3f);
            _leftSphere.GetComponent<Renderer>().material = traditionalMat;

            // === RIGHT SIDE: SDF Raymarching ===
            // Create a large cube that acts as a "screen" for raymarching
            _rightCube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            _rightCube.name = "SDF_Raymarching";
            _rightCube.transform.position = new Vector3(30, 0, 0);
            _rightCube.transform.localScale = Vector3.one * 100f;

            // Apply SDF raymarching material
            if (sdfMaterial == null)
            {
                var shader = Shader.Find("SdfUniverse/InfiniteSurface");
                if (shader != null)
                {
                    sdfMaterial = new Material(shader);
                    sdfMaterial.SetColor("_Color", new Color(0.2f, 0.8f, 1f));
                    sdfMaterial.SetColor("_Color2", new Color(0.8f, 0.3f, 0.5f));
                    sdfMaterial.SetFloat("_DetailScale", detailScale);
                    sdfMaterial.SetFloat("_BoxSize", 50f);
                    sdfMaterial.SetFloat("_HoleSize", 2f);
                    sdfMaterial.SetFloat("_RepeatScale", 15f);
                }
                else
                {
                    Debug.LogError("[ZoomComparison] SdfUniverse/InfiniteSurface shader not found!");
                    sdfMaterial = new Material(Shader.Find("Standard"));
                }
            }
            _rightCube.GetComponent<Renderer>().material = sdfMaterial;

            // Create lighting
            var lightObj = new GameObject("Directional Light");
            var light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = Color.white;
            light.intensity = 1f;
            lightObj.transform.rotation = Quaternion.Euler(45, -30, 0);
        }

        Texture2D CreateLowResTexture(int size)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGB24, false);
            tex.filterMode = FilterMode.Point; // Make pixelation obvious

            // Create a simple checkerboard/brick pattern
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    bool checker = ((x / 8) + (y / 8)) % 2 == 0;
                    float noise = Random.Range(0.8f, 1f);
                    Color c = checker ?
                        new Color(0.6f * noise, 0.3f * noise, 0.2f * noise) :
                        new Color(0.4f * noise, 0.2f * noise, 0.15f * noise);
                    tex.SetPixel(x, y, c);
                }
            }
            tex.Apply();
            return tex;
        }

        void CreateUI()
        {
            var canvasObj = new GameObject("ComparisonCanvas");
            _canvas = canvasObj.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 100;
            canvasObj.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvasObj.AddComponent<GraphicRaycaster>();

            // Left label
            _leftLabel = CreateLabel("LeftLabel", "TRADITIONAL\nMesh + Texture",
                new Vector2(0.25f, 0.95f), new Color(1f, 0.5f, 0.5f));

            // Right label
            _rightLabel = CreateLabel("RightLabel", "ALICE-SDF\nRaymarching + Math",
                new Vector2(0.75f, 0.95f), new Color(0.5f, 1f, 0.8f));

            // Zoom label (center bottom)
            _zoomLabel = CreateLabel("ZoomLabel", "Zoom: x1.0",
                new Vector2(0.5f, 0.08f), Color.white, 32);

            // Instruction label
            _instructionLabel = CreateLabel("Instructions",
                "[Scroll] Zoom  |  [RMB+Mouse] Rotate  |  [Space] Reset",
                new Vector2(0.5f, 0.02f), new Color(0.6f, 0.6f, 0.6f), 14);

            // Divider line
            var dividerObj = new GameObject("Divider");
            dividerObj.transform.SetParent(_canvas.transform, false);
            var dividerImage = dividerObj.AddComponent<Image>();
            dividerImage.color = new Color(1, 1, 1, 0.3f);
            var dividerRect = dividerObj.GetComponent<RectTransform>();
            dividerRect.anchorMin = new Vector2(0.5f, 0);
            dividerRect.anchorMax = new Vector2(0.5f, 1);
            dividerRect.sizeDelta = new Vector2(2, 0);
        }

        Text CreateLabel(string name, string content, Vector2 anchor, Color color, int fontSize = 24)
        {
            var obj = new GameObject(name);
            obj.transform.SetParent(_canvas.transform, false);

            var text = obj.AddComponent<Text>();
            text.text = content;
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.alignment = TextAnchor.MiddleCenter;
            text.color = color;

            var rect = obj.GetComponent<RectTransform>();
            rect.anchorMin = anchor;
            rect.anchorMax = anchor;
            rect.anchoredPosition = Vector2.zero;
            rect.sizeDelta = new Vector2(400, 100);

            return text;
        }

        void Update()
        {
            HandleZoom();
            HandleRotation();
            HandleReset();
            UpdateCameras();
            UpdateUI();
            UpdateSdfMaterial();
        }

        void HandleZoom()
        {
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) < 0.001f) return;

            // Logarithmic zoom
            float logDist = Mathf.Log(_currentDistance);
            logDist -= scroll * zoomSpeed;
            _currentDistance = Mathf.Exp(logDist);
            _currentDistance = Mathf.Clamp(_currentDistance, minDistance, maxDistance);
        }

        void HandleRotation()
        {
            if (Input.GetMouseButton(1))
            {
                _yaw += Input.GetAxis("Mouse X") * 3f;
                _pitch -= Input.GetAxis("Mouse Y") * 3f;
                _pitch = Mathf.Clamp(_pitch, -89f, 89f);
            }
        }

        void HandleReset()
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                _currentDistance = startDistance;
                _yaw = 0;
                _pitch = 30;
            }
        }

        void UpdateCameras()
        {
            Quaternion rotation = Quaternion.Euler(_pitch, _yaw, 0);

            // Left camera looks at left object
            Vector3 leftTarget = _leftSphere.transform.position;
            Vector3 leftDir = rotation * Vector3.back;
            _leftCamera.transform.position = leftTarget + leftDir * _currentDistance;
            _leftCamera.transform.LookAt(leftTarget);

            // Right camera looks at right object (center of SDF world)
            Vector3 rightTarget = Vector3.zero; // SDF is centered at origin
            Vector3 rightDir = rotation * Vector3.back;
            _rightCamera.transform.position = rightTarget + rightDir * _currentDistance;
            _rightCamera.transform.LookAt(rightTarget);
        }

        void UpdateUI()
        {
            float magnification = startDistance / _currentDistance;

            string magText;
            if (magnification >= 1000000)
                magText = $"x{magnification / 1000000:F1}M";
            else if (magnification >= 1000)
                magText = $"x{magnification / 1000:F1}K";
            else
                magText = $"x{magnification:F1}";

            _zoomLabel.text = $"Zoom: {magText}\nDistance: {_currentDistance:E2}";

            // Update labels based on zoom level
            if (magnification >= 100)
            {
                _leftLabel.text = "TRADITIONAL\n<color=red>PIXELATED!</color>";
                _rightLabel.text = "ALICE-SDF\n<color=lime>STILL SHARP!</color>";
            }
            else if (magnification >= 10)
            {
                _leftLabel.text = "TRADITIONAL\n<color=yellow>Getting blurry...</color>";
                _rightLabel.text = "ALICE-SDF\n<color=cyan>Infinite detail</color>";
            }
            else
            {
                _leftLabel.text = "TRADITIONAL\nMesh + Texture";
                _rightLabel.text = "ALICE-SDF\nRaymarching + Math";
            }
        }

        void UpdateSdfMaterial()
        {
            if (sdfMaterial == null) return;

            // Increase detail scale as we zoom in
            // This reveals more micro-detail at higher zoom levels
            float magnification = startDistance / _currentDistance;
            float adaptiveDetail = detailScale * Mathf.Log10(magnification + 1) + detailScale;
            sdfMaterial.SetFloat("_DetailScale", adaptiveDetail);

            // Adjust surface epsilon for precision at close range
            float epsilon = Mathf.Max(0.00001f, _currentDistance * 0.00001f);
            sdfMaterial.SetFloat("_SurfaceEpsilon", epsilon);
        }

        void OnDestroy()
        {
            if (_leftSphere != null) Destroy(_leftSphere);
            if (_rightCube != null) Destroy(_rightCube);
            if (_leftCamera != null) Destroy(_leftCamera.gameObject);
            if (_rightCamera != null) Destroy(_rightCamera.gameObject);
        }
    }
}
