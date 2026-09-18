// Behavioural check of the base collider (AliceSDF_Collider) on the Basic
// sample: the body sampling, the wall / floor classification, the dead band,
// and that the subclass override is what the base evaluates
using System;
using UnityEngine;
using AliceSDF;
using AliceSDF.Samples;

static class Behaviour
{
    public static int Run()
    {
        int fails = 0;
        void Check(bool ok, string what) { Console.WriteLine((ok ? "  ok   " : "  FAIL ") + what); if (!ok) fails++; }
        float dt = 1f / 90f, eye = 1.6f;

        // Basic: ground + sphere r 1.5 at (0, 1.5, 0)
        var b = new SampleBasic_Collider();
        Check(Math.Abs(b.Evaluate(new Vector3(3f, 1.5f, 0f)) - 1.5f) < 1e-6f, "the override is evaluated: 3 m from the sphere axis is 1.5 m from it");
        Check(b.PlayerPushOut(new Vector3(3f, 0f, 0f), eye, dt) == Vector3.zero, "standing on the floor away from the sphere: no push");
        // The floor itself is never pushed against (feet 5 cm in the ground)
        Check(b.PlayerPushOut(new Vector3(3f, -0.05f, 0f), eye, dt) == Vector3.zero, "feet 5 cm under the floor: a floor contact is left to the floor collider (no bobbing)");
        // Standing 1.2 m from the axis: the chest sample (y 1.2) is inside the sphere
        var stand = new Vector3(1.2f, 0f, 0f);
        var deepest = b.DeepestBodySample(stand, eye);
        Check(deepest.y > 0.5f && deepest.y < 2.5f, $"the deepest body sample is mid-body (y = {deepest.y:F2})");
        var push = b.PlayerPushOut(stand, eye, dt);
        Check(push.x > 0f && push.y == 0f && Math.Abs(push.z) < 1e-6f, "pushed straight back along +x, never lifted");
        var q = stand; int frames = 0;
        for (; frames < 600; frames++)
        {
            var d = b.PlayerPushOut(q, eye, dt);
            if (d == Vector3.zero) break;
            q = q + d;
        }
        Check(frames < 600, $"the push reached exactly zero in {frames} frames (dead band)");
        Check(q.x >= 1.5f + 0.1f - 0.02f, $"pushed clear of the sphere plus the margin (x = {q.x:F3})");
        // Directly under the sphere's top (on its column) the only way out is
        // up: a floor-like contact, so the base collider leaves it alone
        Check(b.PlayerPushOut(new Vector3(0f, 2.9f, 0f), eye, dt) == Vector3.zero, "feet on top of the sphere: floor-like, no push");

        // Cosmic / Mix: animTime moves the shapes (the collider follows the shader)
        var cos = new SampleCosmic_Collider();
        float atRest = cos.Evaluate(new Vector3(18f, 0f, 0f));
        cos.animTime = 10f;
        float later = cos.Evaluate(new Vector3(18f, 0f, 0f));
        Check(atRest < -2f && later > atRest + 1f, $"Cosmic: the planet is at (18, 0, 0) at t = 0 ({atRest:F2}) and has orbited away by t = 10 s ({later:F2})");
        var mix = new SampleMix_Collider();
        float onionRest = mix.Evaluate(new Vector3(16f, 0f, 0f));
        mix.animTime = 8f;
        float onionLater = mix.Evaluate(new Vector3(16f, 0f, 0f));
        Check(Math.Abs(onionLater - onionRest) > 0.1f, $"Mix: the onion shell orbits ({onionRest:F2} -> {onionLater:F2})");

        Console.WriteLine(fails == 0 ? "behaviour: all ok" : $"behaviour: {fails} FAIL");
        return fails;
    }
}
