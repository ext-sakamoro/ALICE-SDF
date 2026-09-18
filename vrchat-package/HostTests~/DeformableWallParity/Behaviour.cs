// Behavioural check of the wall collider: impacts (surface test, cooldown,
// slot reuse), recovery, the dented law, the body push and the desktop ray
using System;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Behaviour
{
    static readonly BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;
    static object Get(object o, string f) => typeof(SampleDeformableWall_Collider).GetField(f, F).GetValue(o);
    static void Set(object o, string f, object v) => typeof(SampleDeformableWall_Collider).GetField(f, F).SetValue(o, v);
    static void Call(object o, string m, params object[] a) => typeof(SampleDeformableWall_Collider).GetMethod(m, F).Invoke(o, a);

    static SampleDeformableWall_Collider Fresh()
    {
        var c = new SampleDeformableWall_Collider();
        Call(c, "Start");
        typeof(SampleDeformableWall_Collider).GetField("logEvents").SetValue(c, true); // event lines in the run output
        return c;
    }

    public static int Run()
    {
        int fails = 0;
        void Check(bool ok, string what) { Console.WriteLine((ok ? "  ok   " : "  FAIL ") + what); if (!ok) fails++; }
        float dt = 1f / 90f, eye = 1.6f;
        Time.time = 0f;

        // --- Fresh wall: box 10 x 5 x 0.4 on the ground ---
        var c = Fresh();
        Vector4[] pts() => (Vector4[])Get(c, "impactPoints");
        int count() => (int)Get(c, "impactCount");
        Check(count() == 0 && !c.HasUnsentChanges(), "fresh wall: no dents, nothing to send");
        Check(Math.Abs(c.EvaluateWallSdf(new Vector3(0f, 1f, 1.2f)) - 1.0f) < 1e-6f, "1 m in front of the face is 1 m away");
        Check(Math.Abs(c.EvaluateWallSdf(new Vector3(0f, 1f, 0f)) - (-0.2f)) < 1e-6f, "the middle of the wall is 0.2 m inside");
        Check(Math.Abs(c.EvaluateSdf(new Vector3(0f, 0.5f, 3f)) - 0.5f) < 1e-6f, "away from the wall the SDF is the ground");
        Check(c.PlayerPushOut(new Vector3(0f, 0f, 3f), eye, dt) == Vector3.zero, "standing 3 m away: no push");

        // --- Impacts ---
        var face = new Vector3(1f, 1.2f, 0.2f);   // on the +z face
        Check(!c.TryImpact(new Vector3(1f, 1.2f, 0.5f), 0, "test"), "a hand 0.3 m off the face does not dent");
        Check(c.TryImpact(face, 0, "test"), "a hand on the face dents it");
        Check(count() == 1 && pts()[0].w == 1f && c.HasUnsentChanges(), "dent #0 at full strength, marked for serialization");
        Check(!c.TryImpact(face, 0, "test"), "same hand within the cooldown is refused");
        Check(c.TryImpact(new Vector3(-2f, 0.8f, 0.2f), 1, "test"), "the other hand dents at once");
        Check(count() == 2, "two dents");
        // The dented law: the face point is now air (inside the dent), the
        // dent floor is 0.35 in
        Check(c.EvaluateWallSdf(face) > 0f, "the face point is now air (inside the dent)");
        // (the wall is only 0.4 thick: a full dent leaves 5 cm of material at the back)
        Check(c.EvaluateWallSdf(new Vector3(1f, 1.2f, 0.2f - 0.34f)) > 0f && c.EvaluateWallSdf(new Vector3(1f, 1.2f, 0.2f - 0.38f)) < 0f, "the dent is 0.35 m deep at full strength");
        // A hand following the dent inward does not drill: impacts are measured
        // against the undented face; hitting the same spot again refreshes it
        Time.time += 0.2f;
        Check(!c.TryImpact(new Vector3(1f, 1.2f, 0.2f - 0.35f), 0, "test") && count() == 2, "a hand on the floor of the dent (0.35 m in) does not dent again: no drilling");
        for (int i = 0; i < 90; i++) c.Decay(dt);
        Check(pts()[0].w < 0.7f, "the first dent has recovered a little");
        Check(c.TryImpact(new Vector3(1.05f, 1.2f, 0.2f), 0, "test") && count() == 2 && pts()[0].w == 1f, "hitting within half a radius of it refreshes it to full strength, no new slot");
        Time.time += 0.2f;
        Check(c.TryImpact(new Vector3(2.5f, 1.2f, 0.2f), 0, "test") && count() == 3, "a hit elsewhere is a third dent");

        // --- Recovery: strengths decay, dead trailing slots shrink the count ---
        Set(c, "stateDirty", false);
        for (int i = 0; i < 90; i++) c.Decay(dt);   // 1 s
        Check(Math.Abs(pts()[0].w - (float)Math.Exp(-0.5)) < 1e-3f, $"after 1 s a dent is at exp(-0.5) = {pts()[0].w:F3}");
        Check(c.HasUnsentChanges(), "recovery marks the state for serialization");
        for (int i = 0; i < 90 * 12; i++) c.Decay(dt);   // 12 more s: exp(-6.5) < 0.01
        Check(count() == 0 && c.LiveDents() == 0, "13 s later every dent has recovered and the count is 0");
        Check(Math.Abs(c.EvaluateWallSdf(face) - 0f) < 1e-6f, "the face is flat again");

        // --- Slot reuse: 16 dents fill the buffer, the 17th replaces the weakest ---
        var s = Fresh();
        var sp = (Vector4[])Get(s, "impactPoints");
        for (int i = 0; i < 16; i++) { sp[i] = new Vector4(-4f + 0.5f * i, 1f, 0.2f, 0.1f + 0.05f * i); }
        Set(s, "impactCount", 16);
        int slot = (int)typeof(SampleDeformableWall_Collider).GetMethod("RecordImpact", F).Invoke(s, new object[] { new Vector3(4.5f, 1f, 0.2f) });
        Check(slot == 0 && sp[0].w == 1f && sp[0].x == 4.5f, "with 16 live dents the weakest (slot 0) is replaced");

        // --- Player push: standing in the wall is pushed out sideways, never up ---
        var p = Fresh();
        var inside = new Vector3(0f, 0f, 0.1f);   // feet 0.1 m inside the +z half
        var push = p.PlayerPushOut(inside, eye, dt);
        Check(push.z > 0f && push.y == 0f && Math.Abs(push.x) < 1e-6f, "in the wall: pushed along +z, not lifted");
        var q = inside; int frames = 0;
        for (; frames < 600; frames++)
        {
            var d = p.PlayerPushOut(q, eye, dt);
            if (d == Vector3.zero) break;
            q = q + d;
        }
        Check(frames < 600 && q.z >= 0.2f + 0.1f - 0.01f, $"pushed clear to z = {q.z:F3} in {frames} frames (margin 0.1 off the face, dead band)");
        // A dent at foot level: the feet sample is in the air of the dent, so a
        // higher sample is the deepest and decides the push
        p.TryImpact(new Vector3(0f, 0.05f, 0.2f), 0, "test");
        var deepest = p.DeepestBodySample(new Vector3(0f, 0f, 0.25f), eye);
        Check(deepest.y > 0.3f, $"with a dent at the feet the deepest body sample is higher up (y = {deepest.y:F2})");

        // --- Desktop cursor ray ---
        var r = Fresh();
        float t = r.RaymarchWall(new Vector3(0f, 1f, 3f), new Vector3(0f, 0f, -1f), 4f);
        Check(Math.Abs(t - 2.8f) < 2e-3f, $"looking at the wall from 3 m the ray hits the face at t = {t:F3}");
        Check(r.RaymarchWall(new Vector3(0f, 1f, 3f), new Vector3(0f, 0f, 1f), 4f) < 0f, "looking away: no hit");
        Check(r.TryImpact(new Vector3(0f, 1f, 3f) + new Vector3(0f, 0f, -1f) * t, 1, "left click"), "a click on the face dents it");

        Console.WriteLine(fails == 0 ? "behaviour: all ok" : $"behaviour: {fails} FAIL");
        return fails;
    }
}
