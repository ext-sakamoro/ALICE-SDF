// =============================================================================
// SDF World - Procedural Universe Definition
// =============================================================================
// Defines the entire world using SDFs: Sky, Ground, Objects
// No meshes required - everything is mathematical
//
// Author: Moroya Sakamoto
// =============================================================================

using System;
using System.Collections.Generic;
using UnityEngine;
using AliceSdf;

namespace SdfUniverse
{
    /// <summary>
    /// Shape type for world objects
    /// </summary>
    public enum SdfShapeType
    {
        Sphere,
        Box,
        Cylinder,
        Torus,
        Capsule,
        Metaball,     // Smooth union of spheres
        Custom
    }

    /// <summary>
    /// Definition of an SDF object in the world
    /// </summary>
    [Serializable]
    public class SdfObjectDefinition
    {
        public string name = "Object";
        public SdfShapeType shapeType = SdfShapeType.Sphere;
        public Vector3 position = Vector3.zero;
        public Vector3 rotation = Vector3.zero;
        public Vector3 scale = Vector3.one;
        public float smoothness = 0.3f;

        [Header("Shape Parameters")]
        public float radius = 1f;
        public Vector3 halfExtents = Vector3.one * 0.5f;
        public float majorRadius = 1f;
        public float minorRadius = 0.3f;

        [Header("Modifiers")]
        public float roundRadius = 0f;
        public float onionThickness = 0f;
        public float twistStrength = 0f;

        [Header("Animation")]
        public bool animate = false;
        public float animationSpeed = 1f;
        public Vector3 animationAxis = Vector3.up;
    }

    /// <summary>
    /// Manages the entire SDF-defined world
    /// </summary>
    public class SdfWorld : MonoBehaviour
    {
        [Header("=== SKY ===")]
        public bool skyEnabled = true;
        public float skyRadius = 100f;
        [Range(0, 1)] public float skyParticleRatio = 0.1f;

        [Header("=== GROUND ===")]
        public bool groundEnabled = true;
        public float groundHeight = -2f;
        public float groundNoiseAmplitude = 0.5f;
        public float groundNoiseFrequency = 0.1f;
        [Range(0, 1)] public float groundParticleRatio = 0.2f;

        [Header("=== OBJECTS ===")]
        public List<SdfObjectDefinition> objects = new List<SdfObjectDefinition>();
        [Range(0, 1)] public float objectParticleRatio = 0.7f;

        [Header("=== GLOBAL ===")]
        public float globalSmoothness = 0.2f;
        public bool infiniteRepeat = false;
        public Vector3 repeatSpacing = new Vector3(10, 10, 10);

        [Header("=== DEBUG ===")]
        public bool showBounds = false;
        public bool logPerformance = false;

        // Compiled SDF (reused every frame)
        private CompiledSdf _compiledWorld;
        private SdfNode _worldNode;
        private bool _needsRecompile = true;
        private float _lastCompileTime;

        // Object handles for cleanup
        private List<SdfNode> _tempNodes = new List<SdfNode>();

        public CompiledSdf CompiledWorld => _compiledWorld;
        public bool IsReady => _compiledWorld != null && _compiledWorld.IsValid;
        public float LastCompileTimeMs => _lastCompileTime * 1000f;

        /// <summary>
        /// Set an externally compiled SDF (e.g., from FractalDemo)
        /// This allows external code to define the world SDF directly.
        /// </summary>
        /// <param name="compiledSdf">Pre-compiled SDF to use</param>
        /// <param name="sourceNode">Optional: source SdfNode for reference (not disposed by SdfWorld)</param>
        public void SetCompiledSdf(CompiledSdf compiledSdf, SdfNode sourceNode = null)
        {
            // Cleanup existing world
            CleanupNodes();
            _compiledWorld?.Dispose();

            // Set new compiled SDF
            _compiledWorld = compiledSdf;
            _needsRecompile = false;

            // Disable auto-rebuild for external SDF
            groundEnabled = false;
            skyEnabled = false;
            objects.Clear();

            if (logPerformance && _compiledWorld != null)
            {
                Debug.Log($"[SdfWorld] External SDF set: {_compiledWorld.InstructionCount} instructions");
            }
        }

        void Start()
        {
            if (objects.Count == 0)
            {
                // Default: metaball demo
                objects.Add(new SdfObjectDefinition
                {
                    name = "Metaball Center",
                    shapeType = SdfShapeType.Sphere,
                    radius = 2f,
                    position = Vector3.zero
                });
                objects.Add(new SdfObjectDefinition
                {
                    name = "Metaball Orbit",
                    shapeType = SdfShapeType.Sphere,
                    radius = 1f,
                    position = new Vector3(2.5f, 0, 0),
                    animate = true,
                    animationSpeed = 0.5f
                });
            }

            RebuildWorld();
        }

        void Update()
        {
            // Animate objects
            bool needsRebuild = false;
            foreach (var obj in objects)
            {
                if (obj.animate)
                {
                    needsRebuild = true;
                    break;
                }
            }

            if (needsRebuild || _needsRecompile)
            {
                RebuildWorld();
            }
        }

        void OnDestroy()
        {
            CleanupNodes();
        }

        void OnValidate()
        {
            _needsRecompile = true;
        }

        /// <summary>
        /// Rebuild the entire world SDF
        /// </summary>
        public void RebuildWorld()
        {
            var startTime = Time.realtimeSinceStartup;

            CleanupNodes();

            try
            {
                _worldNode = BuildWorldSdf();

                if (_worldNode != null && _worldNode.IsValid)
                {
                    _compiledWorld?.Dispose();
                    _compiledWorld = _worldNode.Compile();
                    _needsRecompile = false;

                    if (logPerformance)
                    {
                        Debug.Log($"[SdfWorld] Compiled: {_compiledWorld.InstructionCount} instructions");
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[SdfWorld] Build failed: {e.Message}");
            }

            _lastCompileTime = Time.realtimeSinceStartup - startTime;

            if (logPerformance)
            {
                Debug.Log($"[SdfWorld] Rebuild time: {_lastCompileTime * 1000:F2}ms");
            }
        }

        /// <summary>
        /// Build the complete world SDF
        /// </summary>
        private SdfNode BuildWorldSdf()
        {
            SdfNode result = null;

            // 1. Build objects
            SdfNode objectsSdf = BuildObjectsSdf();
            if (objectsSdf != null)
            {
                result = objectsSdf;
            }

            // 2. Add ground
            if (groundEnabled)
            {
                var ground = SdfNode.Plane(Vector3.up, -groundHeight);
                _tempNodes.Add(ground);

                if (result != null)
                {
                    result = result.SmoothUnion(ground, globalSmoothness);
                    _tempNodes.Add(result);
                }
                else
                {
                    result = ground;
                }
            }

            // 3. Add sky dome (inverted sphere)
            if (skyEnabled)
            {
                // Sky is an inverted sphere - inside is positive distance
                var sky = SdfNode.Sphere(skyRadius);
                _tempNodes.Add(sky);
                // Note: For sky dome, we typically use intersection or special handling
                // Here we just add it for particle spawning purposes
            }

            // 4. Apply infinite repeat if enabled
            if (infiniteRepeat && result != null)
            {
                result = result.Repeat(repeatSpacing);
                _tempNodes.Add(result);
            }

            return result;
        }

        /// <summary>
        /// Build SDF for all objects
        /// </summary>
        private SdfNode BuildObjectsSdf()
        {
            if (objects.Count == 0) return null;

            SdfNode result = null;
            float time = Time.time;

            foreach (var obj in objects)
            {
                // Build base shape
                SdfNode shape = BuildShape(obj);
                if (shape == null) continue;
                _tempNodes.Add(shape);

                // Apply modifiers
                if (obj.roundRadius > 0)
                {
                    shape = shape.Round(obj.roundRadius);
                    _tempNodes.Add(shape);
                }
                if (obj.onionThickness > 0)
                {
                    shape = shape.Onion(obj.onionThickness);
                    _tempNodes.Add(shape);
                }
                if (Mathf.Abs(obj.twistStrength) > 0.001f)
                {
                    shape = shape.Twist(obj.twistStrength);
                    _tempNodes.Add(shape);
                }

                // Apply animation
                Vector3 pos = obj.position;
                if (obj.animate)
                {
                    float angle = time * obj.animationSpeed;
                    Quaternion rot = Quaternion.AngleAxis(angle * 180f, obj.animationAxis);
                    pos = rot * pos;
                }

                // Apply transforms
                if (obj.rotation != Vector3.zero)
                {
                    shape = shape.RotateEuler(obj.rotation * Mathf.Deg2Rad);
                    _tempNodes.Add(shape);
                }
                if (obj.scale != Vector3.one)
                {
                    shape = shape.Scale(obj.scale);
                    _tempNodes.Add(shape);
                }
                if (pos != Vector3.zero)
                {
                    shape = shape.Translate(pos);
                    _tempNodes.Add(shape);
                }

                // Combine with result
                if (result == null)
                {
                    result = shape;
                }
                else
                {
                    result = result.SmoothUnion(shape, obj.smoothness);
                    _tempNodes.Add(result);
                }
            }

            return result;
        }

        /// <summary>
        /// Build a single shape from definition
        /// </summary>
        private SdfNode BuildShape(SdfObjectDefinition def)
        {
            switch (def.shapeType)
            {
                case SdfShapeType.Sphere:
                    return SdfNode.Sphere(def.radius);

                case SdfShapeType.Box:
                    return SdfNode.Box(def.halfExtents);

                case SdfShapeType.Cylinder:
                    return SdfNode.Cylinder(def.radius, def.halfExtents.y);

                case SdfShapeType.Torus:
                    return SdfNode.Torus(def.majorRadius, def.minorRadius);

                case SdfShapeType.Capsule:
                    return SdfNode.Capsule(
                        Vector3.down * def.halfExtents.y,
                        Vector3.up * def.halfExtents.y,
                        def.radius);

                case SdfShapeType.Metaball:
                    return BuildMetaball(def);

                default:
                    return SdfNode.Sphere(def.radius);
            }
        }

        /// <summary>
        /// Build a metaball (smooth union of multiple spheres)
        /// </summary>
        private SdfNode BuildMetaball(SdfObjectDefinition def)
        {
            // Create a metaball from 3 spheres
            var s1 = SdfNode.Sphere(def.radius);
            _tempNodes.Add(s1);

            var s2 = SdfNode.Sphere(def.radius * 0.7f).Translate(def.radius * 0.8f, 0, 0);
            _tempNodes.Add(s2);

            var s3 = SdfNode.Sphere(def.radius * 0.5f).Translate(0, def.radius * 0.6f, 0);
            _tempNodes.Add(s3);

            var result = s1.SmoothUnion(s2, def.smoothness);
            _tempNodes.Add(result);

            result = result.SmoothUnion(s3, def.smoothness);
            _tempNodes.Add(result);

            return result;
        }

        /// <summary>
        /// Cleanup all temporary nodes
        /// </summary>
        private void CleanupNodes()
        {
            foreach (var node in _tempNodes)
            {
                node?.Dispose();
            }
            _tempNodes.Clear();

            _worldNode?.Dispose();
            _worldNode = null;
        }

        /// <summary>
        /// Evaluate distance at a point
        /// </summary>
        public float EvalDistance(Vector3 point)
        {
            if (_compiledWorld == null || !_compiledWorld.IsValid)
                return float.MaxValue;

            return _compiledWorld.Eval(point);
        }

        /// <summary>
        /// Get surface normal at a point using central differences
        /// </summary>
        public Vector3 GetNormal(Vector3 point, float epsilon = 0.001f)
        {
            if (_compiledWorld == null) return Vector3.up;

            float dx = _compiledWorld.Eval(point + Vector3.right * epsilon)
                     - _compiledWorld.Eval(point - Vector3.right * epsilon);
            float dy = _compiledWorld.Eval(point + Vector3.up * epsilon)
                     - _compiledWorld.Eval(point - Vector3.up * epsilon);
            float dz = _compiledWorld.Eval(point + Vector3.forward * epsilon)
                     - _compiledWorld.Eval(point - Vector3.forward * epsilon);

            return new Vector3(dx, dy, dz).normalized;
        }

        void OnDrawGizmosSelected()
        {
            if (!showBounds) return;

            Gizmos.color = Color.cyan;

            // Sky dome
            if (skyEnabled)
            {
                Gizmos.DrawWireSphere(Vector3.zero, skyRadius);
            }

            // Ground plane
            if (groundEnabled)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawWireCube(
                    new Vector3(0, groundHeight, 0),
                    new Vector3(20, 0.1f, 20));
            }

            // Objects
            Gizmos.color = Color.yellow;
            foreach (var obj in objects)
            {
                Gizmos.DrawWireSphere(obj.position, obj.radius);
            }
        }
    }
}
