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

        Check(count() == 5, "5 initial mochis");
        // Hand 0 rests inside mochi 0 (r=0.35 at (-0.6,0.35,0.5)); dwell 0.08 s at 90 fps = 8 frames
        var origin = pos()[0];
        for (int i = 0; i < 10; i++) Call(c, "ProcessHand", origin, 0);
        Check(grab()[0] == 0, "hand 0 grabbed mochi 0 after dwell");
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
        Console.WriteLine(fails == 0 ? "behaviour: all ok" : $"behaviour: {fails} FAIL");
        return fails;
    }
}
