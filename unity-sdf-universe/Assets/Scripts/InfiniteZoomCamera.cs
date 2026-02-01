// =============================================================================
// Infinite Zoom Camera - Logarithmic Scale Zoom
// =============================================================================
// Enables microscope-level zooming without precision issues.
// Uses logarithmic scale for smooth zooming at any distance.
//
// Controls:
//   - Mouse Wheel: Zoom in/out
//   - RMB + Mouse: Look around
//   - WASD: Move (optional)
//
// Author: Moroya Sakamoto
// =============================================================================

using UnityEngine;

namespace SdfUniverse
{
    public class InfiniteZoomCamera : MonoBehaviour
    {
        [Header("=== TARGET ===")]
        [Tooltip("The center point to zoom towards")]
        public Transform target;
        public Vector3 targetOffset = Vector3.zero;

        [Header("=== ZOOM ===")]
        [Range(1f, 50f)]
        public float zoomSpeed = 15.0f;
        [Tooltip("Minimum distance (microscope level)")]
        public float minDistance = 0.0001f;
        [Tooltip("Maximum distance (overview level)")]
        public float maxDistance = 500.0f;
        [Tooltip("Starting distance")]
        public float startDistance = 100.0f;

        [Header("=== ROTATION ===")]
        [Range(1f, 10f)]
        public float rotationSpeed = 3.0f;
        public bool invertY = false;

        [Header("=== MOVEMENT ===")]
        public bool enableWASD = true;
        [Range(0.1f, 10f)]
        public float moveSpeedMultiplier = 1.0f;

        [Header("=== UI ===")]
        public bool showZoomUI = true;

        // State
        private float _currentDistance;
        private float _yaw;
        private float _pitch;
        private Vector3 _targetPosition;

        // Stats
        public float CurrentDistance => _currentDistance;
        public float ZoomScale => 1.0f / Mathf.Max(_currentDistance, 0.0001f);
        public float ZoomMagnification => startDistance / Mathf.Max(_currentDistance, 0.0001f);

        void Start()
        {
            // Initialize distance
            if (target != null)
            {
                _currentDistance = Vector3.Distance(transform.position, target.position + targetOffset);
            }
            else
            {
                _currentDistance = startDistance;
            }

            // Initialize rotation from current orientation
            Vector3 euler = transform.eulerAngles;
            _yaw = euler.y;
            _pitch = euler.x;

            // Clamp initial pitch
            if (_pitch > 180f) _pitch -= 360f;
            _pitch = Mathf.Clamp(_pitch, -89f, 89f);

            _targetPosition = target != null ? target.position + targetOffset : Vector3.zero;

            Debug.Log($"[InfiniteZoom] Initialized at distance: {_currentDistance:F4}");
        }

        void Update()
        {
            UpdateTargetPosition();
            HandleZoom();
            HandleRotation();
            HandleMovement();
            UpdateCameraPosition();
        }

        void UpdateTargetPosition()
        {
            if (target != null)
            {
                _targetPosition = target.position + targetOffset;
            }
        }

        void HandleZoom()
        {
            float scroll = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(scroll) < 0.001f) return;

            // Logarithmic zoom calculation
            // The closer we are, the smaller the step (finer control)
            float logDist = Mathf.Log(_currentDistance);
            logDist -= scroll * zoomSpeed;
            _currentDistance = Mathf.Exp(logDist);

            // Clamp to valid range
            _currentDistance = Mathf.Clamp(_currentDistance, minDistance, maxDistance);
        }

        void HandleRotation()
        {
            // RMB to rotate
            if (Input.GetMouseButton(1))
            {
                float mouseX = Input.GetAxis("Mouse X") * rotationSpeed;
                float mouseY = Input.GetAxis("Mouse Y") * rotationSpeed * (invertY ? 1f : -1f);

                _yaw += mouseX;
                _pitch += mouseY;
                _pitch = Mathf.Clamp(_pitch, -89f, 89f);
            }
        }

        void HandleMovement()
        {
            if (!enableWASD) return;

            // Movement speed scales with distance (logarithmic)
            float moveSpeed = _currentDistance * moveSpeedMultiplier * Time.deltaTime;

            Vector3 move = Vector3.zero;

            if (Input.GetKey(KeyCode.W)) move += transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= transform.forward;
            if (Input.GetKey(KeyCode.A)) move -= transform.right;
            if (Input.GetKey(KeyCode.D)) move += transform.right;
            if (Input.GetKey(KeyCode.E)) move += transform.up;
            if (Input.GetKey(KeyCode.Q)) move -= transform.up;

            if (Input.GetKey(KeyCode.LeftShift))
            {
                moveSpeed *= 3f;
            }

            if (move.sqrMagnitude > 0.01f)
            {
                _targetPosition += move.normalized * moveSpeed;
            }
        }

        void UpdateCameraPosition()
        {
            // Calculate position on sphere around target
            Quaternion rotation = Quaternion.Euler(_pitch, _yaw, 0);
            Vector3 direction = rotation * Vector3.back;

            transform.position = _targetPosition + direction * _currentDistance;
            transform.LookAt(_targetPosition);
        }

        // =====================================================================
        // Public API
        // =====================================================================

        /// <summary>
        /// Set zoom to specific distance
        /// </summary>
        public void SetDistance(float distance)
        {
            _currentDistance = Mathf.Clamp(distance, minDistance, maxDistance);
        }

        /// <summary>
        /// Set zoom to specific magnification (relative to start)
        /// </summary>
        public void SetMagnification(float magnification)
        {
            _currentDistance = startDistance / Mathf.Max(magnification, 0.0001f);
            _currentDistance = Mathf.Clamp(_currentDistance, minDistance, maxDistance);
        }

        /// <summary>
        /// Reset to starting position
        /// </summary>
        public void ResetZoom()
        {
            _currentDistance = startDistance;
        }

        /// <summary>
        /// Animate zoom to target distance
        /// </summary>
        public void ZoomTo(float targetDistance, float duration = 1f)
        {
            StartCoroutine(ZoomAnimation(targetDistance, duration));
        }

        private System.Collections.IEnumerator ZoomAnimation(float targetDist, float duration)
        {
            float startDist = _currentDistance;
            float elapsed = 0f;

            // Use logarithmic interpolation for smooth zoom
            float logStart = Mathf.Log(startDist);
            float logEnd = Mathf.Log(targetDist);

            while (elapsed < duration)
            {
                elapsed += Time.deltaTime;
                float t = Mathf.SmoothStep(0, 1, elapsed / duration);
                float logCurrent = Mathf.Lerp(logStart, logEnd, t);
                _currentDistance = Mathf.Exp(logCurrent);
                yield return null;
            }

            _currentDistance = targetDist;
        }

        // =====================================================================
        // UI
        // =====================================================================

        void OnGUI()
        {
            if (!showZoomUI) return;

            // Zoom indicator in bottom-right
            float boxWidth = 220;
            float boxHeight = 80;
            Rect rect = new Rect(Screen.width - boxWidth - 20, Screen.height - boxHeight - 20, boxWidth, boxHeight);

            GUILayout.BeginArea(rect);
            GUILayout.BeginVertical("box");

            // Format magnification nicely
            float mag = ZoomMagnification;
            string magText;
            if (mag >= 1000000)
                magText = $"x{mag / 1000000:F1}M";
            else if (mag >= 1000)
                magText = $"x{mag / 1000:F1}K";
            else if (mag >= 1)
                magText = $"x{mag:F1}";
            else
                magText = $"x{mag:F4}";

            GUILayout.Label($"<size=24><b>Zoom: {magText}</b></size>");
            GUILayout.Label($"<size=12>Distance: {_currentDistance:E2}</size>");
            GUILayout.Label($"<size=10><color=cyan>Polygons: 0</color></size>");

            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.yellow;
            Vector3 pos = target != null ? target.position + targetOffset : _targetPosition;
            Gizmos.DrawWireSphere(pos, 0.5f);

            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(transform.position, pos);
        }
    }
}
