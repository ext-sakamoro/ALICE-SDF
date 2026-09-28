// CI-only runtime smoke test for unity-sdf-universe.
//
// This project has no committed .unity scene (see README/SETUP_GUIDE — it is
// source you assemble yourself), so a plain batchmode compile check cannot
// prove the demo actually runs: SdfWorld/SdfParticleSystem's Awake/Start/Update
// are ordinary MonoBehaviour methods that Unity never calls outside play mode
// or a real scene. This drives them directly via reflection instead, so the
// native FFI path (EvalGradientSoA -> alice_sdf_eval_gradient_soa) and the
// physics integrator actually execute once, without needing a scene or a
// domain-reload-fragile batchmode play-mode session.
//
// Invoked by .github/workflows/unity-sdf-universe.yml via -executeMethod.
using System;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using SdfUniverse;

public static class CiSmokeTest
{
    const string P = "[CiSmokeTest]";
    static bool _hadLogError = false;

    static void OnLog(string condition, string stackTrace, LogType type)
    {
        if (type == LogType.Exception || type == LogType.Error)
        {
            Debug.Log($"{P} captured {type}: {condition}");
            _hadLogError = true;
        }
    }

    static void Invoke(object target, string method, object[] args = null)
    {
        var mi = target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        if (mi == null) throw new Exception($"method not found: {target.GetType().Name}.{method}");
        mi.Invoke(target, args);
    }

    public static void Run()
    {
        int exit = 0;
        GameObject go = null;
        Application.logMessageReceived += OnLog;
        try
        {
            go = new GameObject("CiSmokeTest");
            var world = go.AddComponent<SdfWorld>();
            var particles = go.AddComponent<SdfParticleSystem>();
            particles.particleCount = 2000;

            // Manual lifecycle: Unity does not call these outside play mode / a scene.
            Invoke(world, "Start");
            Debug.Log($"{P} world.IsReady={world.IsReady}");
            if (!world.IsReady)
            {
                Debug.LogError($"{P} FAIL: world did not become ready after Start()");
                exit = 1;
            }

            Invoke(particles, "Awake");
            Invoke(particles, "Start");
            Debug.Log($"{P} particles.ActiveParticles={particles.ActiveParticles}");

            var posXField = typeof(SdfParticleSystem).GetField("_posX", BindingFlags.Instance | BindingFlags.NonPublic);
            float posXBefore = ((Unity.Collections.NativeArray<float>)posXField.GetValue(particles))[0];

            // Time.deltaTime is 0 outside play mode: run the native gradient eval
            // once, then step the physics integrator directly with an explicit
            // non-zero dt so positions actually advance and can be checked.
            Invoke(particles, "UpdateParticlesDeepFried");
            var updateSingle = typeof(SdfParticleSystem).GetMethod("UpdateSingleParticle", BindingFlags.Instance | BindingFlags.NonPublic);
            for (int frame = 0; frame < 30; frame++)
            {
                updateSingle.Invoke(particles, new object[] { 0, 0.016f, (float)frame * 0.016f });
            }

            var posXArr = (Unity.Collections.NativeArray<float>)posXField.GetValue(particles);
            float posXAfter = posXArr[0];
            bool anyNaN = false;
            for (int i = 0; i < posXArr.Length; i++)
            {
                if (float.IsNaN(posXArr[i])) { anyNaN = true; break; }
            }

            Debug.Log($"{P} EvalTimeMs={particles.EvalTimeMs:F3} posX[0] before={posXBefore:F4} after={posXAfter:F4} anyNaN={anyNaN}");

            if (anyNaN)
            {
                Debug.LogError($"{P} FAIL: NaN detected in particle positions after 30 update steps");
                exit = 1;
            }
            if (Mathf.Approximately(posXBefore, posXAfter))
            {
                Debug.LogError($"{P} FAIL: posX[0] did not change across 30 explicit-dt steps (physics not advancing)");
                exit = 1;
            }

            if (_hadLogError) exit = 1;

            Debug.Log($"{P} done exit={exit}");
        }
        catch (Exception e)
        {
            Debug.LogError($"{P} EXCEPTION: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            exit = 2;
        }
        finally
        {
            try
            {
                if (go != null)
                {
                    var particles = go.GetComponent<SdfParticleSystem>();
                    if (particles != null) Invoke(particles, "OnDestroy");
                    var world = go.GetComponent<SdfWorld>();
                    if (world != null) Invoke(world, "OnDestroy");
                    UnityEngine.Object.DestroyImmediate(go);
                }
            }
            catch (Exception ce)
            {
                Debug.LogError($"{P} cleanup exception: {ce}");
            }
        }

        EditorApplication.Exit(exit);
    }
}
