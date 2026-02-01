// =============================================================================
// SDF Particle System - Ultimate Edition
// =============================================================================
// The final form: 500万パーティクル at 60 FPS
//
// Optimization Stack:
// 1. Rust SIMD: EvalGradientSoA (8-wide AVX2)
// 2. Unity Burst: Physics computation (LLVM vectorization)
// 3. GPU Indirect: DrawMeshInstancedIndirect (no CPU matrix)
// 4. ComputeBuffer: 24 bytes/particle (vs 64 bytes Matrix4x4)
//
// Memory Transfer per Frame:
//   Old: 100万 × 64 bytes = 64 MB
//   New: 100万 × 24 bytes = 24 MB (62.5% reduction)
//
// CPU Matrix Computation:
//   Old: 100万 × Matrix4x4.TRS = Heavy
//   New: Zero (computed on GPU)
//
// Author: Moroya Sakamoto
// =============================================================================

using System;
using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using AliceSdf;

namespace SdfUniverse
{
    /// <summary>
    /// Particle data structure for GPU (matches shader)
    /// </summary>
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    public struct ParticleGpuData
    {
        public float3 position;
        public float3 velocity;
    }

    /// <summary>
    /// Ultimate Particle System
    /// </summary>
    [RequireComponent(typeof(SdfWorld))]
    public class SdfParticleSystem_Ultimate : MonoBehaviour
    {
        [Header("=== PARTICLE COUNT ===")]
        [Range(10000, 5000000)]
        public int particleCount = 1000000;

        [Header("=== PHYSICS ===")]
        [Range(0.1f, 10f)] public float flowSpeed = 3f;
        [Range(0.1f, 10f)] public float surfaceAttraction = 2f;
        [Range(0f, 1f)] public float noiseStrength = 0.1f;

        [Header("=== RENDERING ===")]
        public Material particleMaterial;
        [Range(0.01f, 0.5f)] public float particleSize = 0.1f;
        [Range(0.5f, 5f)] public float brightness = 2f;

        [Header("=== SPAWN ===")]
        public float spawnRadius = 40f;
        public float maxDistance = 100f;

        [Header("=== DEBUG ===")]
        public bool showStats = true;

        // Native Arrays (SoA for Rust + Burst)
        private NativeArray<float> _posX, _posY, _posZ;
        private NativeArray<float> _velX, _velY, _velZ;
        private NativeArray<float> _normX, _normY, _normZ;
        private NativeArray<float> _dist;

        // GPU Data
        private NativeArray<ParticleGpuData> _gpuData;
        private ComputeBuffer _particleBuffer;
        private ComputeBuffer _argsBuffer;
        private uint[] _args = new uint[5];
        private Mesh _quadMesh;

        // References
        private SdfWorld _sdfWorld;

        // Stats
        private float _evalTime;
        private float _physicsTime;
        private float _fps;
        private int _frameCount;
        private float _fpsTimer;

        public int ActiveParticles => particleCount;
        public float EvalTimeMs => _evalTime * 1000f;
        public float PhysicsTimeMs => _physicsTime * 1000f;
        public float FPS => _fps;

        // =====================================================================
        // Lifecycle
        // =====================================================================

        void Awake()
        {
            _sdfWorld = GetComponent<SdfWorld>();
        }

        void Start()
        {
            AllocateMemory();
            CreateMesh();
            CreateMaterial();
            InitializeParticles();

            Debug.Log($"[Ultimate] Initialized {particleCount:N0} particles");
        }

        void OnDestroy()
        {
            DisposeMemory();
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

            // Update simulation
            UpdateSimulation();

            // Render
            RenderIndirect();
        }

        // =====================================================================
        // Memory Management
        // =====================================================================

        void AllocateMemory()
        {
            var alloc = Allocator.Persistent;

            // SoA arrays for Rust + Burst
            _posX = new NativeArray<float>(particleCount, alloc);
            _posY = new NativeArray<float>(particleCount, alloc);
            _posZ = new NativeArray<float>(particleCount, alloc);
            _velX = new NativeArray<float>(particleCount, alloc);
            _velY = new NativeArray<float>(particleCount, alloc);
            _velZ = new NativeArray<float>(particleCount, alloc);
            _normX = new NativeArray<float>(particleCount, alloc);
            _normY = new NativeArray<float>(particleCount, alloc);
            _normZ = new NativeArray<float>(particleCount, alloc);
            _dist = new NativeArray<float>(particleCount, alloc);

            // GPU data (AoS for rendering)
            _gpuData = new NativeArray<ParticleGpuData>(particleCount, alloc);

            // Compute buffers
            int stride = UnsafeUtility.SizeOf<ParticleGpuData>(); // 24 bytes
            _particleBuffer = new ComputeBuffer(particleCount, stride);
            _argsBuffer = new ComputeBuffer(1, _args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        }

        void DisposeMemory()
        {
            if (_posX.IsCreated) _posX.Dispose();
            if (_posY.IsCreated) _posY.Dispose();
            if (_posZ.IsCreated) _posZ.Dispose();
            if (_velX.IsCreated) _velX.Dispose();
            if (_velY.IsCreated) _velY.Dispose();
            if (_velZ.IsCreated) _velZ.Dispose();
            if (_normX.IsCreated) _normX.Dispose();
            if (_normY.IsCreated) _normY.Dispose();
            if (_normZ.IsCreated) _normZ.Dispose();
            if (_dist.IsCreated) _dist.Dispose();
            if (_gpuData.IsCreated) _gpuData.Dispose();

            _particleBuffer?.Release();
            _argsBuffer?.Release();
        }

        void CreateMesh()
        {
            _quadMesh = new Mesh();
            float s = 0.5f;
            _quadMesh.vertices = new Vector3[] {
                new Vector3(-s, -s, 0), new Vector3(s, -s, 0),
                new Vector3(s, s, 0), new Vector3(-s, s, 0)
            };
            _quadMesh.triangles = new int[] { 0, 2, 1, 0, 3, 2 };
            _quadMesh.uv = new Vector2[] {
                new Vector2(0, 0), new Vector2(1, 0),
                new Vector2(1, 1), new Vector2(0, 1)
            };
            _quadMesh.RecalculateBounds();
        }

        void CreateMaterial()
        {
            if (particleMaterial == null)
            {
                var shader = Shader.Find("SdfUniverse/ParticleRender_Indirect");
                if (shader == null)
                {
                    Debug.LogError("[Ultimate] Shader 'SdfUniverse/ParticleRender_Indirect' not found!");
                    shader = Shader.Find("Particles/Standard Unlit");
                }

                particleMaterial = new Material(shader);
                particleMaterial.SetColor("_Color", new Color(0.3f, 0.9f, 1f, 1f));
                particleMaterial.SetFloat("_Size", particleSize);
                particleMaterial.SetFloat("_Brightness", brightness);
            }
        }

        void InitializeParticles()
        {
            // Initialize positions using Burst job
            var initJob = new InitializeParticlesJob
            {
                posX = _posX,
                posY = _posY,
                posZ = _posZ,
                velX = _velX,
                velY = _velY,
                velZ = _velZ,
                spawnRadius = spawnRadius,
                seed = (uint)UnityEngine.Random.Range(1, 100000)
            };
            initJob.Schedule(particleCount, 256).Complete();
        }

        // =====================================================================
        // Simulation (Rust + Burst)
        // =====================================================================

        void UpdateSimulation()
        {
            var compiled = _sdfWorld.CompiledWorld;
            if (compiled == null || !compiled.IsValid) return;

            float dt = Time.deltaTime;
            float time = Time.time;

            // =================================================================
            // PHASE 1: Rust SIMD - Evaluate SDF gradients
            // One call for ALL particles - DEEP FRIED PATH
            // =================================================================
            float evalStart = Time.realtimeSinceStartup;

            var result = compiled.EvalGradientSoA(
                _posX, _posY, _posZ,
                _normX, _normY, _normZ,
                _dist
            );

            _evalTime = Time.realtimeSinceStartup - evalStart;

            if (!result.IsOk)
            {
                Debug.LogWarning($"[Ultimate] SDF eval failed: {result.result}");
                return;
            }

            // =================================================================
            // PHASE 2: Burst - Physics + GPU data packing
            // =================================================================
            float physicsStart = Time.realtimeSinceStartup;

            var updateJob = new UpdatePhysicsAndPackJob
            {
                posX = _posX, posY = _posY, posZ = _posZ,
                velX = _velX, velY = _velY, velZ = _velZ,
                normX = _normX, normY = _normY, normZ = _normZ,
                dist = _dist,
                gpuData = _gpuData,
                deltaTime = dt,
                time = time,
                flowSpeed = flowSpeed,
                surfaceAttraction = surfaceAttraction,
                noiseStrength = noiseStrength,
                maxDistance = maxDistance,
                spawnRadius = spawnRadius,
                seed = (uint)(time * 1000)
            };

            updateJob.Schedule(particleCount, 256).Complete();

            _physicsTime = Time.realtimeSinceStartup - physicsStart;

            // =================================================================
            // PHASE 3: Upload to GPU (24 bytes per particle)
            // =================================================================
            _particleBuffer.SetData(_gpuData);
        }

        // =====================================================================
        // Rendering (GPU Indirect)
        // =====================================================================

        void RenderIndirect()
        {
            if (particleMaterial == null || _quadMesh == null) return;

            // Update material properties
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
        // Stats UI
        // =====================================================================

        void OnGUI()
        {
            if (!showStats) return;

            float throughput = particleCount / (EvalTimeMs + 0.001f) / 1000f;

            GUILayout.BeginArea(new Rect(10, 10, 350, 250));
            GUILayout.BeginVertical("box");

            GUILayout.Label("<size=18><b>=== ULTIMATE DEEP FRIED ===</b></size>");
            GUILayout.Label($"<color=cyan>FPS: {_fps:F0}</color>");
            GUILayout.Label($"Particles: {particleCount:N0}");
            GUILayout.Space(5);
            GUILayout.Label($"<color=yellow>Rust SDF Eval: {EvalTimeMs:F2} ms</color>");
            GUILayout.Label($"<color=yellow>Burst Physics: {PhysicsTimeMs:F2} ms</color>");
            GUILayout.Label($"GPU Transfer: 24 bytes/particle");
            GUILayout.Space(5);
            GUILayout.Label($"<color=lime>Throughput: {throughput:F1}M pts/s</color>");
            GUILayout.Space(5);
            GUILayout.Label("<size=10>DrawMeshInstancedIndirect</size>");
            GUILayout.Label("<size=10>Zero CPU matrix computation</size>");

            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        // =====================================================================
        // Burst Jobs
        // =====================================================================

        [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
        struct InitializeParticlesJob : IJobParallelFor
        {
            [WriteOnly] public NativeArray<float> posX, posY, posZ;
            [WriteOnly] public NativeArray<float> velX, velY, velZ;
            public float spawnRadius;
            public uint seed;

            public void Execute(int i)
            {
                var rng = new Unity.Mathematics.Random(seed + (uint)i * 1337);

                // Random position in sphere
                float3 dir = rng.NextFloat3Direction();
                float r = rng.NextFloat() * spawnRadius;
                float3 pos = dir * r;

                posX[i] = pos.x;
                posY[i] = pos.y;
                posZ[i] = pos.z;

                // Random initial velocity
                float3 vel = rng.NextFloat3Direction() * 0.5f;
                velX[i] = vel.x;
                velY[i] = vel.y;
                velZ[i] = vel.z;
            }
        }

        [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
        struct UpdatePhysicsAndPackJob : IJobParallelFor
        {
            public NativeArray<float> posX, posY, posZ;
            public NativeArray<float> velX, velY, velZ;

            [ReadOnly] public NativeArray<float> normX, normY, normZ;
            [ReadOnly] public NativeArray<float> dist;

            [WriteOnly] public NativeArray<ParticleGpuData> gpuData;

            public float deltaTime;
            public float time;
            public float flowSpeed;
            public float surfaceAttraction;
            public float noiseStrength;
            public float maxDistance;
            public float spawnRadius;
            public uint seed;

            public void Execute(int i)
            {
                // Read current state
                float3 p = new float3(posX[i], posY[i], posZ[i]);
                float3 v = new float3(velX[i], velY[i], velZ[i]);
                float3 n = new float3(normX[i], normY[i], normZ[i]);
                float d = dist[i];

                // Normalize gradient
                float nLen = math.length(n);
                if (nLen > 0.0001f)
                {
                    n /= nLen;
                }

                // === Surface flow physics ===

                // 1. Project velocity onto surface tangent
                float vDotN = math.dot(v, n);
                v -= vDotN * n;

                // 2. Add rotational flow (cross product for tangent direction)
                float3 tangent = math.cross(n, new float3(0, 1, 0));
                if (math.lengthsq(tangent) < 0.01f)
                {
                    tangent = math.cross(n, new float3(1, 0, 0));
                }
                tangent = math.normalize(tangent);

                v += tangent * flowSpeed * 0.5f;

                // 3. Surface attraction (pull towards SDF surface)
                v -= n * d * surfaceAttraction;

                // 4. Add noise for organic movement
                if (noiseStrength > 0)
                {
                    float3 noiseOffset = new float3(
                        noise.snoise(new float2(p.x * 0.1f + time * 0.3f, p.y * 0.1f)),
                        noise.snoise(new float2(p.y * 0.1f, p.z * 0.1f + time * 0.3f)),
                        noise.snoise(new float2(p.z * 0.1f + time * 0.3f, p.x * 0.1f))
                    );
                    v += noiseOffset * noiseStrength;
                }

                // 5. Clamp velocity
                float velLen = math.length(v);
                if (velLen > flowSpeed * 2f)
                {
                    v = v / velLen * flowSpeed * 2f;
                }

                // 6. Integrate position
                p += v * deltaTime;

                // 7. Respawn if out of bounds
                float distFromOrigin = math.length(p);
                if (distFromOrigin > maxDistance || math.any(math.isnan(p)))
                {
                    var rng = new Unity.Mathematics.Random(seed + (uint)i);
                    p = rng.NextFloat3Direction() * spawnRadius * rng.NextFloat();
                    v = rng.NextFloat3Direction() * 0.5f;
                }

                // Write back position/velocity
                posX[i] = p.x;
                posY[i] = p.y;
                posZ[i] = p.z;
                velX[i] = v.x;
                velY[i] = v.y;
                velZ[i] = v.z;

                // Pack for GPU (SoA -> AoS)
                gpuData[i] = new ParticleGpuData
                {
                    position = p,
                    velocity = v
                };
            }
        }
    }
}
