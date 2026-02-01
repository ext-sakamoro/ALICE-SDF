// =============================================================================
// SDF Particle System
// =============================================================================
// 100K-1M Particles flowing on SDF surfaces at 60+ FPS
// Uses SIMD-accelerated batch gradient evaluation
//
// Performance:
//   - Old: 6x Eval() per particle = 600K calls for 100K particles = SLOW
//   - New: 1x EvalGradientSoA() = 100K calls batched = FAST
//
// Author: Moroya Sakamoto
// =============================================================================

using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;
using AliceSdf;

namespace SdfUniverse
{
    public enum ParticleFlowMode
    {
        SurfaceFlow,
        Orbit,
        Attract,
        Repel
    }

    /// <summary>
    /// High-performance SDF particle system
    /// </summary>
    [RequireComponent(typeof(SdfWorld))]
    public class SdfParticleSystem : MonoBehaviour
    {
        [Header("=== PARTICLE COUNT ===")]
        [Range(10000, 1000000)]
        public int particleCount = 100000;

        [Header("=== BEHAVIOR ===")]
        public ParticleFlowMode flowMode = ParticleFlowMode.SurfaceFlow;
        [Range(0.01f, 10f)] public float flowSpeed = 1f;
        [Range(0.001f, 1f)] public float surfaceAttraction = 0.1f;
        [Range(0f, 1f)] public float noiseStrength = 0.2f;
        [Range(0.01f, 2f)] public float noiseScale = 0.5f;

        [Header("=== APPEARANCE ===")]
        [Range(0.01f, 1f)] public float particleSize = 0.1f;
        public Gradient colorGradient;
        public Material particleMaterial;

        [Header("=== SPAWN ===")]
        public Vector3 spawnCenter = Vector3.zero;
        public float spawnRadius = 5f;
        public bool respawnOutOfBounds = true;
        public float maxDistance = 50f;

        [Header("=== PERFORMANCE ===")]
        public bool useGpuInstancing = true;
        public bool parallelUpdate = true;
        [Range(1, 120)] public int updateRate = 60;

        [Header("=== DEBUG ===")]
        public bool showStats = false;
        public bool logPerformance = false;

        // Core data - SoA layout for SIMD
        private SdfWorld _sdfWorld;
        private float[] _posX, _posY, _posZ;
        private float[] _velX, _velY, _velZ;
        private float[] _normX, _normY, _normZ;  // Gradients from SIMD
        private float[] _distances;
        private float[] _lifetimes;
        private Color[] _colors;
        private int _activeCount;

        // GPU Instancing
        private Matrix4x4[] _matrices;
        private MaterialPropertyBlock _propertyBlock;
        private Mesh _particleMesh;

        // Stats
        private float _lastUpdateTime;
        private float _lastEvalTime;
        private float _fps;
        private int _frameCount;
        private float _fpsTimer;

        public int ActiveParticles => _activeCount;
        public float EvalTimeMs => _lastEvalTime * 1000f;
        public float UpdateTimeMs => _lastUpdateTime * 1000f;
        public float FPS => _fps;

        void Awake()
        {
            _sdfWorld = GetComponent<SdfWorld>();

            if (colorGradient == null || colorGradient.colorKeys.Length == 0)
            {
                colorGradient = new Gradient();
                colorGradient.SetKeys(
                    new GradientColorKey[] {
                        new GradientColorKey(new Color(0.2f, 0.8f, 1f), 0f),
                        new GradientColorKey(new Color(1f, 0.3f, 0.8f), 0.5f),
                        new GradientColorKey(new Color(1f, 0.9f, 0.3f), 1f)
                    },
                    new GradientAlphaKey[] {
                        new GradientAlphaKey(1f, 0f),
                        new GradientAlphaKey(1f, 1f)
                    }
                );
            }
        }

        void Start()
        {
            InitializeParticles();
            InitializeRendering();
        }

        void Update()
        {
            if (!_sdfWorld.IsReady) return;

            // FPS counter
            _frameCount++;
            _fpsTimer += Time.deltaTime;
            if (_fpsTimer >= 1f)
            {
                _fps = _frameCount / _fpsTimer;
                _frameCount = 0;
                _fpsTimer = 0;
            }

            // Update particles - DEEP FRIED PATH
            float startTime = Time.realtimeSinceStartup;
            UpdateParticlesDeepFried();
            _lastUpdateTime = Time.realtimeSinceStartup - startTime;

            // Render
            RenderParticles();
        }

        void OnDestroy()
        {
            // Cleanup
        }

        // =====================================================================
        // Initialization
        // =====================================================================

        private void InitializeParticles()
        {
            // Allocate SoA arrays
            _posX = new float[particleCount];
            _posY = new float[particleCount];
            _posZ = new float[particleCount];
            _velX = new float[particleCount];
            _velY = new float[particleCount];
            _velZ = new float[particleCount];
            _normX = new float[particleCount];
            _normY = new float[particleCount];
            _normZ = new float[particleCount];
            _distances = new float[particleCount];
            _lifetimes = new float[particleCount];
            _colors = new Color[particleCount];

            // Spawn initial particles
            for (int i = 0; i < particleCount; i++)
            {
                SpawnParticle(i);
            }

            _activeCount = particleCount;
            Debug.Log($"[SdfParticleSystem] Initialized {particleCount:N0} particles");
        }

        private void SpawnParticle(int i)
        {
            Vector3 pos = spawnCenter + UnityEngine.Random.insideUnitSphere * spawnRadius;
            _posX[i] = pos.x;
            _posY[i] = pos.y;
            _posZ[i] = pos.z;

            Vector3 vel = UnityEngine.Random.onUnitSphere * flowSpeed * 0.5f;
            _velX[i] = vel.x;
            _velY[i] = vel.y;
            _velZ[i] = vel.z;

            _lifetimes[i] = UnityEngine.Random.value;
            _colors[i] = colorGradient.Evaluate(_lifetimes[i]);
        }

        private void InitializeRendering()
        {
            _particleMesh = CreateQuadMesh();
            _propertyBlock = new MaterialPropertyBlock();
            _matrices = new Matrix4x4[particleCount];

            if (particleMaterial == null)
            {
                particleMaterial = CreateDefaultMaterial();
            }
        }

        private Mesh CreateQuadMesh()
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
            mesh.RecalculateNormals();
            return mesh;
        }

        private Material CreateDefaultMaterial()
        {
            Shader shader = Shader.Find("Particles/Standard Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            if (shader == null) shader = Shader.Find("Standard");

            var mat = new Material(shader);
            mat.color = new Color(0.4f, 0.9f, 1f, 1f);
            mat.enableInstancing = true;

            if (mat.HasProperty("_EmissionColor"))
            {
                mat.SetColor("_EmissionColor", new Color(0.4f, 0.9f, 1f));
                mat.EnableKeyword("_EMISSION");
            }

            Debug.Log($"[SdfParticleSystem] Material: {shader.name}");
            return mat;
        }

        // =====================================================================
        // DEEP FRIED Update - SIMD Batch Gradient Evaluation
        // =====================================================================

        private void UpdateParticlesDeepFried()
        {
            var sdf = _sdfWorld.CompiledWorld;
            if (sdf == null || !sdf.IsValid) return;

            float dt = Time.deltaTime;
            float time = Time.time;
            int count = _activeCount;

            // ================================================================
            // PHASE 1: Batch evaluate gradients (THE DEEP FRIED PATH)
            // One call evaluates ALL particles: distance + normal
            // ================================================================
            float evalStart = Time.realtimeSinceStartup;
            var result = sdf.EvalGradientSoA(
                _posX, _posY, _posZ,
                _normX, _normY, _normZ,
                _distances
            );
            _lastEvalTime = Time.realtimeSinceStartup - evalStart;

            if (!result.IsOk)
            {
                Debug.LogWarning($"[SdfParticleSystem] Gradient eval failed: {result.result}");
                return;
            }

            // ================================================================
            // PHASE 2: Update physics (Parallel for multi-core)
            // ================================================================
            if (parallelUpdate)
            {
                Parallel.For(0, count, i => UpdateSingleParticle(i, dt, time));
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    UpdateSingleParticle(i, dt, time);
                }
            }
        }

        private void UpdateSingleParticle(int i, float dt, float time)
        {
            float x = _posX[i];
            float y = _posY[i];
            float z = _posZ[i];
            float dist = _distances[i];
            float nx = _normX[i];
            float ny = _normY[i];
            float nz = _normZ[i];

            // Normalize gradient (it should already be normalized, but safety check)
            float nLen = Mathf.Sqrt(nx * nx + ny * ny + nz * nz);
            if (nLen > 0.0001f)
            {
                nx /= nLen;
                ny /= nLen;
                nz /= nLen;
            }

            float vx = _velX[i];
            float vy = _velY[i];
            float vz = _velZ[i];

            // Apply flow mode
            switch (flowMode)
            {
                case ParticleFlowMode.SurfaceFlow:
                    // Project velocity onto surface tangent
                    float dot = vx * nx + vy * ny + vz * nz;
                    vx -= dot * nx;
                    vy -= dot * ny;
                    vz -= dot * nz;

                    // Add rotational flow (cross product with normal)
                    float tx = vy * nz - vz * ny;
                    float ty = vz * nx - vx * nz;
                    float tz = vx * ny - vy * nx;
                    vx += tx * 0.3f;
                    vy += ty * 0.3f;
                    vz += tz * 0.3f;

                    // Attract to surface
                    vx -= nx * dist * surfaceAttraction * 10f;
                    vy -= ny * dist * surfaceAttraction * 10f;
                    vz -= nz * dist * surfaceAttraction * 10f;
                    break;

                case ParticleFlowMode.Orbit:
                    // Orbit around objects
                    float ox = ny;
                    float oy = -nx;
                    float oz = 0;
                    vx = ox * flowSpeed + vx * 0.95f;
                    vy = oy * flowSpeed + vy * 0.95f;
                    vz = oz * flowSpeed + vz * 0.95f;
                    vx -= nx * dist * surfaceAttraction * 5f;
                    vy -= ny * dist * surfaceAttraction * 5f;
                    vz -= nz * dist * surfaceAttraction * 5f;
                    break;

                case ParticleFlowMode.Attract:
                    vx -= nx * dist * surfaceAttraction * 10f;
                    vy -= ny * dist * surfaceAttraction * 10f;
                    vz -= nz * dist * surfaceAttraction * 10f;
                    break;

                case ParticleFlowMode.Repel:
                    float repelForce = surfaceAttraction / (dist * dist + 0.5f);
                    vx += nx * repelForce;
                    vy += ny * repelForce;
                    vz += nz * repelForce;
                    break;
            }

            // Add noise for organic movement
            if (noiseStrength > 0)
            {
                float px = Mathf.PerlinNoise(x * noiseScale + time * 0.5f, y * noiseScale) - 0.5f;
                float py = Mathf.PerlinNoise(y * noiseScale, z * noiseScale + time * 0.5f) - 0.5f;
                float pz = Mathf.PerlinNoise(z * noiseScale + time * 0.5f, x * noiseScale) - 0.5f;
                vx += px * noiseStrength;
                vy += py * noiseStrength;
                vz += pz * noiseStrength;
            }

            // Clamp velocity
            float velLen = Mathf.Sqrt(vx * vx + vy * vy + vz * vz);
            if (velLen > flowSpeed * 2f)
            {
                float scale = flowSpeed * 2f / velLen;
                vx *= scale;
                vy *= scale;
                vz *= scale;
            }

            // Integrate position
            x += vx * dt;
            y += vy * dt;
            z += vz * dt;

            // Store
            _posX[i] = x;
            _posY[i] = y;
            _posZ[i] = z;
            _velX[i] = vx;
            _velY[i] = vy;
            _velZ[i] = vz;

            // Update lifetime and color
            _lifetimes[i] += dt * 0.05f;
            if (_lifetimes[i] > 1f) _lifetimes[i] -= 1f;
            _colors[i] = colorGradient.Evaluate(_lifetimes[i]);

            // Respawn if out of bounds or NaN
            if (respawnOutOfBounds)
            {
                float distFromCenter = Mathf.Sqrt(x * x + y * y + z * z);
                if (distFromCenter > maxDistance || float.IsNaN(x))
                {
                    SpawnParticle(i);
                }
            }
        }

        // =====================================================================
        // Rendering
        // =====================================================================

        private void RenderParticles()
        {
            if (particleMaterial == null || _particleMesh == null) return;

            Camera cam = Camera.main;
            if (cam == null) return;

            int count = _activeCount;
            Quaternion camRot = cam.transform.rotation;
            Vector3 scale = Vector3.one * particleSize;

            // Build transformation matrices
            for (int i = 0; i < count; i++)
            {
                Vector3 pos = new Vector3(_posX[i], _posY[i], _posZ[i]);
                _matrices[i] = Matrix4x4.TRS(pos, camRot, scale);
            }

            // Draw in batches (Unity limit: 1023 per call)
            int batchSize = 1023;
            for (int i = 0; i < count; i += batchSize)
            {
                int batchCount = Mathf.Min(batchSize, count - i);
                var batch = new Matrix4x4[batchCount];
                Array.Copy(_matrices, i, batch, 0, batchCount);

                Graphics.DrawMeshInstanced(
                    _particleMesh, 0, particleMaterial,
                    batch, batchCount, _propertyBlock,
                    ShadowCastingMode.Off, false
                );
            }
        }

        public void SetParticleCount(int count)
        {
            if (count == particleCount) return;
            particleCount = Mathf.Clamp(count, 1000, 1000000);
            InitializeParticles();
            InitializeRendering();
        }

        void OnDrawGizmos()
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(spawnCenter, spawnRadius);
        }

        void OnGUI()
        {
            if (!showStats) return;

            GUILayout.BeginArea(new Rect(10, 10, 300, 200));
            GUILayout.BeginVertical("box");
            GUILayout.Label($"FPS: {_fps:F1}");
            GUILayout.Label($"Particles: {_activeCount:N0}");
            GUILayout.Label($"SDF Gradient: {EvalTimeMs:F2} ms");
            GUILayout.Label($"Physics: {(UpdateTimeMs - EvalTimeMs):F2} ms");
            GUILayout.Label($"Throughput: {(_activeCount / (EvalTimeMs + 0.001f) / 1000f):F1}M pts/s");
            GUILayout.EndVertical();
            GUILayout.EndArea();
        }
    }
}
