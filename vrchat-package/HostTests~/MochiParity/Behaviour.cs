// Behavioural check of the hand-indexed refactor: grab -> split -> release -> merge
using System;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Behaviour
{
    static readonly BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;
    static object Get(object o, string f) => typeof(SampleMochi_Collider).GetField(f, F).GetValue(o);
    static void Call(object o, string m, params object[] a) => typeof(SampleMochi_Collider).GetMethod(m, F).Invoke(o, a);

    public static int Run()
    {
        var c = new SampleMochi_Collider();
        Call(c, "Start");
        int count() => (int)Get(c, "mochiCount");
        float[] r() => (float[])Get(c, "mochiR");
        Vector3[] pos() => (Vector3[])Get(c, "mochiPos");
        int[] grab() => (int[])Get(c, "grab");
        int fails = 0;
        void Check(bool ok, string what) { Console.WriteLine((ok ? "  ok   " : "  FAIL ") + what); if (!ok) fails++; }

        Check(count() == 5, "5 initial mochis (the owner spawns them)");
        bool dirty() => c.HasUnsentChanges();
        Check(dirty(), "spawning marked the state dirty for the first serialization");
        typeof(SampleMochi_Collider).GetField("stateDirty", F).SetValue(c, false);
        typeof(SampleMochi_Collider).GetField("logEvents").SetValue(c, true); // event lines in the run output
        // Player collision must ignore the ground plane: standing on the floor
        // (feet 5 cm under y=0) away from every mochi is not a penetration.
        float margin = (float)typeof(SampleMochi_Collider).GetField("collisionMargin").GetValue(c);
        Check(c.EvaluateMochiSdf(new Vector3(2f, -0.05f, 2f)) >= margin, "floor away from mochis: no collision push");
        Check(c.EvaluateSdf(new Vector3(2f, -0.05f, 2f)) < 0f, "rendered SDF still contains the ground");
        Check(c.EvaluateMochiSdf(pos()[0]) < 0f, "inside mochi 0 is still a penetration");
        // Hand 0 rests inside mochi 0 (r=0.35 at (-0.6,0.35,0.5)); dwell 0.08 s at 90 fps = 8 frames
        var origin = pos()[0];
        for (int i = 0; i < 10; i++) Call(c, "ProcessHand", origin, 0);
        Check(grab()[0] == 0, "hand 0 grabbed mochi 0 after dwell");
        Check(dirty(), "moving the held mochi marks the state dirty");
        // Other hand cannot take the same mochi
        for (int i = 0; i < 10; i++) Call(c, "ProcessHand", origin, 1);
        Check(grab()[1] == -1, "hand 1 refused the held mochi");
        // Pull to 3 r: split (>2.5 r), not released (<4 r)
        float r0 = r()[0];
        var pulled = origin + new Vector3(3f * r0, 0, 0);
        Call(c, "ProcessHand", pulled, 0);
        Check(count() == 6, "split spawned a 6th mochi");
        float half = r0 * 0.7937005f;
        Check(Math.Abs(r()[0] - half) < 1e-6f && Math.Abs(r()[5] - half) < 1e-6f, "both halves are r*cbrt(0.5)");
        Check(pos()[0] == pulled && pos()[5] == origin, "held half follows hand, other half stays at origin");
        Check(grab()[0] == 0, "still held after split");
        // Pull to 5 * (new r) from origin: release
        var far = origin + new Vector3(5f * half, 0, 0);
        Call(c, "ProcessHand", far, 0);
        Check(grab()[0] == -1, "released beyond 4 r");
        // Gravity: the released half (centre y=0.35 > r=0.278) settles toward y=r
        float yBefore = pos()[0].y;
        for (int i = 0; i < 900; i++) Call(c, "ApplyGravity");
        Check(pos()[0].y < yBefore && Math.Abs(pos()[0].y - half) < 2e-3f, "settled to y = r after 10 s");
        // Merge: move the free half at origin onto mochi 3 (r=0.40 at (-0.9,0.4,-0.2)) -> volume conserved
        float v = r()[5] * r()[5] * r()[5] + r()[3] * r()[3] * r()[3];
        var p = pos(); p[5] = p[3]; // same array instance, mutate in place
        p[0] = new Vector3(10f, 0.3f, 10f); // park the released half: it had drifted within merge range of mochi 1
        Call(c, "CheckMerge");
        Check(count() == 5, "merge removed one mochi");
        float rm = r()[3];
        Check(Math.Abs(rm * rm * rm - v) < 1e-5f, "merged volume conserved");
        Check(pos()[3].y >= rm - 1e-6f, "merged mochi not under the floor");
        // Player collision on a fresh scene: body sampled from feet to eyes,
        // the floor is VRChat's (y clamped to 0 like its floor collider would)
        var pc = new SampleMochi_Collider();
        Call(pc, "Start");
        float eye = 1.6f, dt = 1f / 90f;
        Check(pc.PlayerPushOut(new Vector3(2f, 0f, 2f), eye, dt) == Vector3.zero, "floor away from mochis: no push");
        // Standing 0.25 m to +x of mochi 0 (r=0.35 at (-0.6,0.35,0.5)): the knee-high
        // sample is inside, the foot sample only 8 cm from the underside
        var m0 = ((Vector3[])Get(pc, "mochiPos"))[0]; float r0m = ((float[])Get(pc, "mochiR"))[0];
        var stand = new Vector3(m0.x + 0.25f, 0f, m0.z);
        var first = pc.PlayerPushOut(stand, eye, dt);
        Check(first.x > 0f && first.y == 0f && Math.Abs(first.z) < first.x * 0.2f, "walked in from +x: pushed back along +x, never lifted");
        var q = stand; int frames = 0;
        for (; frames < 600; frames++)
        {
            var d = pc.PlayerPushOut(q, eye, dt);
            if (d == Vector3.zero) break;
            q = q + d; q = new Vector3(q.x, 0f, q.z); // floor
        }
        float xz = (float)Math.Sqrt((q.x - m0.x) * (q.x - m0.x) + (q.z - m0.z) * (q.z - m0.z));
        Check(frames < 600, $"push reached exactly zero in {frames} frames (dead band, no endless teleport)");
        Check(xz >= r0m - 0.02f, $"pushed clear of the mochi (xz {xz:F3} >= r {r0m:F2})");
        Check(q.y == 0f && Math.Abs(q.z - m0.z) < 0.02f, "stayed on the floor and on the approach line");
        // The mochi gives way by the mass ratio (r 0.35, water density: 180 kg vs 60 kg player)
        var yc = new SampleMochi_Collider();
        Call(yc, "Start");
        var ypos = (Vector3[])Get(yc, "mochiPos");
        var before = ypos[0];
        var ypush = yc.PlayerPushOut(stand, eye, dt);
        float share = yc.YieldMochi(stand, eye, ypush);
        float expect = 60f / (60f + 1000f * 4.18879f * 0.35f * 0.35f * 0.35f);
        Check(Math.Abs(share - expect) < 1e-4f, $"mochi share of the push is the mass ratio ({share:F3})");
        var moved = before - ypos[0];
        Check(moved.x > 0f && Math.Abs(moved.x - ypush.x * share) < 1e-6f && moved.y == 0f, "mochi slid away from the player by its share, on the floor");
        // Held in a hand, it does not yield
        ((int[])Get(yc, "grab"))[0] = 0;
        var held = ypos[0];
        Check(yc.YieldMochi(stand, eye, ypush) == 0f && ypos[0] == held, "a grabbed mochi does not yield");
        // Desktop cursor: a click along -z at mochi 1 (r 0.30 at (0.5,0.30,0.3)) from z = 2
        var dc = new SampleMochi_Collider();
        Call(dc, "Start");
        var dpos = (Vector3[])Get(dc, "mochiPos");
        var dgrab = (int[])Get(dc, "grab");
        var o = new Vector3(0.5f, 0.3f, 2f); var fwd = new Vector3(0f, 0f, -1f);
        float th = dc.RaymarchMochi(o, fwd, 4f);
        Check(Math.Abs(th - 1.4f) < 0.01f, $"ray hits the mochi surface at t = {th:F3} (2 - 0.3 - 0.3)");
        float cd = dc.CursorDistance(o, fwd);
        Check(Math.Abs(cd - 1.7f) < 1e-4f, $"cursor sits at the centre's depth along the ray ({cd:F3})");
        Check(dc.CursorDistance(new Vector3(5f, 0.3f, 2f), fwd) < 0f, "a click past every mochi has no cursor");
        var cursor = o + fwd * cd;
        for (int i = 0; i < 10; i++) Call(dc, "ProcessHand", cursor, 1);
        Check(dgrab[1] == 1, "the cursor grabbed mochi 1 after the dwell");
        var dragged = cursor + new Vector3(0.2f, 0f, 0f);
        Call(dc, "ProcessHand", dragged, 1);
        Check(dpos[1] == dragged, "dragging the cursor moves the mochi");
        Call(dc, "ReleaseHand", 1);
        Check(dgrab[1] == -1, "releasing the button drops it");
        // Grab button: split the held mochi without pulling
        for (int i = 0; i < 10; i++) Call(dc, "ProcessHand", cursor, 1);
        int before6 = (int)Get(dc, "mochiCount");
        float r1 = ((float[])Get(dc, "mochiR"))[1];
        bool did = (bool)typeof(SampleMochi_Collider).GetMethod("SplitHeld", F).Invoke(dc, new object[] { 1, "grab button" });
        Check(did && (int)Get(dc, "mochiCount") == before6 + 1, "grab button split the held mochi without a pull");
        Check(Math.Abs(((float[])Get(dc, "mochiR"))[1] - r1 * 0.7937005f) < 1e-6f, "held piece shrank to r*cbrt(0.5)");
        Check(dpos[before6] == ((Vector3[])Get(dc, "grabOrigin"))[1], "other half is left at the grab origin");
        // A piece at the minimum size refuses
        ((float[])Get(dc, "mochiR"))[1] = 0.12f;
        did = (bool)typeof(SampleMochi_Collider).GetMethod("SplitHeld", F).Invoke(dc, new object[] { 1, "grab button" });
        Check(!did, "a mochi at the minimum size does not split");
        Call(dc, "ReleaseHand", 1);
        // Directly on the mochi's column: the only way out is up, never down into the floor
        var under = pc.PlayerPushOut(new Vector3(m0.x, 0f, m0.z), eye, dt);
        Check(under != Vector3.zero && under.y >= 0f, "under the centre: push is not downward");
        Console.WriteLine(fails == 0 ? "behaviour: all ok" : $"behaviour: {fails} FAIL");
        return fails;
    }
}
