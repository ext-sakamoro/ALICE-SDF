// Behavioural check of the collider's law-side logic: sculpt (cooldown,
// stroke, buffer wrap), the surface search, the contact classification
// (step / wall / buried) and the desktop cursor ray
using System;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Behaviour
{
    static readonly BindingFlags F = BindingFlags.NonPublic | BindingFlags.Instance;
    static object Get(object o, string f) => typeof(SampleTerrainSculpt_Collider).GetField(f, F).GetValue(o);
    static void Set(object o, string f, object v) => typeof(SampleTerrainSculpt_Collider).GetField(f, F).SetValue(o, v);
    static void Call(object o, string m, params object[] a) => typeof(SampleTerrainSculpt_Collider).GetMethod(m, F).Invoke(o, a);

    static SampleTerrainSculpt_Collider Fresh()
    {
        var c = new SampleTerrainSculpt_Collider();
        Call(c, "Start");
        typeof(SampleTerrainSculpt_Collider).GetField("logEvents").SetValue(c, true); // event lines in the run output
        return c;
    }

    public static int Run()
    {
        int fails = 0;
        void Check(bool ok, string what) { Console.WriteLine((ok ? "  ok   " : "  FAIL ") + what); if (!ok) fails++; }
        float dt = 1f / 90f;
        Time.time = 0f;

        // --- Flat terrain: standing on the floor is never a contact ---
        var c = Fresh();
        int count() => (int)Get(c, "sculptCount");
        Check(count() == 0 && !c.HasUnsentChanges(), "fresh terrain: no sculpts, nothing to send");
        Check(Math.Abs(c.EvaluateSdf(new Vector3(0f, 0.5f, 0f)) - 0.5f) < 1e-6f, "flat terrain is the plane y = 0");
        Check(Math.Abs(c.SurfaceHeight(new Vector3(3f, 0f, 3f))) < 2e-3f, "surface under the feet on flat ground is y = 0");
        Check(c.ContactKind(new Vector3(3f, -0.05f, 3f)) == 0, "feet 5 cm under the floor (controller skin): no contact, no bobbing");
        Check(c.PlayerPushOut(new Vector3(3f, -0.05f, 3f), dt) == Vector3.zero, "no push on flat ground");

        // --- Sculpting: cooldown, stroke, far hand ---
        Check(c.TrySculpt(new Vector3(0f, 0f, 0f), true, 0, "test"), "add at the surface is recorded");
        Check(count() == 1 && c.HasUnsentChanges(), "one sculpt stored and marked for serialization");
        Check(((bool[])Get(c, "stroking"))[0], "a stroke started (ownership taken once)");
        Check(!c.TrySculpt(new Vector3(0.1f, 0f, 0f), true, 0, "test"), "same hand within the cooldown is refused");
        Time.time += 0.2f;
        Check(c.TrySculpt(new Vector3(0.3f, 0.2f, 0f), true, 0, "test"), "after the cooldown the hand sculpts again");
        Check(count() == 2, "two sculpts stored");
        Check(!c.TrySculpt(new Vector3(0f, 1.0f, 0f), true, 0, "test") && count() == 2, "a hand far from the surface does nothing");
        Check(!((bool[])Get(c, "stroking"))[0], "leaving the surface ends the stroke");
        Time.time += 0.2f;
        Check(c.TrySculpt(new Vector3(-1f, 0f, 0.5f), false, 1, "test"), "dig at the surface is recorded");
        Check(count() == 3 && ((Vector4[])Get(c, "sculptData"))[2].w < 0f, "the dig is stored with a negative radius");

        // --- Geometry after add(0,0,0) add(0.3,0.2,0) dig(-1,0,0.5), r 0.3 ---
        float hole = c.SurfaceHeight(new Vector3(-1f, 0f, 0.5f));
        Check(Math.Abs(hole - (-0.3f)) < 5e-3f, $"under the dig the surface is the hole bottom, y = {hole:F3} (you fall in)");
        Check(c.ContactKind(new Vector3(-1f, 0f, 0.5f)) == 0, "feet above the hole: no push, gravity and the support do it");
        float top = c.SurfaceHeight(new Vector3(0.3f, 0f, 0f));
        Check(Math.Abs(top - 0.5f) < 5e-3f, $"under the stacked hill the surface is its top, y = {top:F3}");
        Check(Math.Abs(c.SurfaceHeight(new Vector3(0.3f, 2f, 0f)) - 0.5f) < 5e-3f, "the search from high above finds the same top");
        var rec = typeof(SampleTerrainSculpt_Collider).GetMethod("RecordSculpt", F);
        Check(c.ContactKind(new Vector3(0.3f, 0f, 0f)) == 2, "feet under 0.5 m of hill: buried");
        var lift = c.PlayerPushOut(new Vector3(0.3f, 0f, 0f), dt);
        Check(lift.x == 0f && lift.z == 0f && Math.Abs(lift.y - 0.51f) < 5e-3f, $"buried feet are lifted straight onto the top (+{lift.y:F3})");
        Check(Math.Abs(c.SupportHeight(new Vector3(0.3f, 0f, 0f)) - 0.5f) < 5e-3f, "the support goes to the top as well");
        // A bump lower than the step limit is left to the support and the controller
        var s = Fresh();
        rec.Invoke(s, new object[] { new Vector3(0f, 0f, 0f), 0.3f });
        Check(s.EvaluateSdf(new Vector3(0.25f, 0f, 0f)) < 0f, "feet in the flank of a single 0.3 m bump are inside it");
        Check(s.ContactKind(new Vector3(0.25f, 0f, 0f)) == 0, "but 0.17 m of bump above them is a step, not a wall: no push");
        float bump = s.SupportHeight(new Vector3(0.25f, 0f, 0f));
        // sphere flank 0.166 plus the k = 0.25 fillet with the plane
        Check(bump > 0.166f && bump < 0.3f, $"the support rises onto the bump under the feet (y = {bump:F3})");
        // At the edge: the foot centre is off the bump but a foot-width sample is on it
        var edge = new Vector3(0.40f, 0f, 0f);
        Check(s.SurfaceHeight(edge) < 0.06f && s.SupportHeight(edge) > 0.10f, $"foot centre past the edge (surface {s.SurfaceHeight(edge):F3}) still stands on the bump (support {s.SupportHeight(edge):F3})");
        Check(Math.Abs(s.SupportHeight(new Vector3(0.6f, 0f, 0f))) < 0.02f, "a foot-width further out the support is the ground");

        // --- Wall: a 0.9 m column of three spheres ---
        var w = Fresh();
        rec.Invoke(w, new object[] { new Vector3(2f, 0f, 0f), 0.3f });
        rec.Invoke(w, new object[] { new Vector3(2f, 0.3f, 0f), 0.3f });
        rec.Invoke(w, new object[] { new Vector3(2f, 0.6f, 0f), 0.3f });
        var feet = new Vector3(1.75f, 0f, 0f);
        Check(w.ContactKind(feet) == 1, "feet inside the column's flank with 0.9 m above: a wall");
        var push = w.PlayerPushOut(feet, dt);
        Check(push.x < 0f && push.y == 0f && Math.Abs(push.z) < 1e-6f, "pushed back along -x, never lifted");
        Check(w.SupportHeight(feet) == feet.y, "against a wall the support stays at the feet");
        var q = feet; int frames = 0;
        for (; frames < 600; frames++)
        {
            var d = w.PlayerPushOut(q, dt);
            if (d == Vector3.zero) break;
            q = q + d;
        }
        Check(frames < 600, $"push reached exactly zero in {frames} frames (dead band, no endless teleport)");
        // The stop is the fillet at the column's foot (sdf slightly negative,
        // a few cm of blend above the feet): a step for the support, not a wall
        Check(q.x < 1.75f && w.ContactKind(q) == 0 && w.SupportHeight(q) - q.y < 0.3f, $"pushed clear of the wall (x {q.x:F3}, sdf {w.EvaluateSdf(q):F3}, {w.SupportHeight(q) - q.y:F3} m of fillet above the feet; the column top under the foot edge is not taken)");
        // Walking off the column top: the foot edge still on it keeps the floor there
        Set(w, "supportPlaced", true); Set(w, "lastSupportY", 0.9f);
        Check(w.SupportHeight(q) > 0.7f, $"with the last floor at the column top, the same foot edge keeps the player on it (sphere top at that x: {w.SupportHeight(q):F3})");
        Set(w, "supportPlaced", false);

        // --- Desktop cursor ray ---
        var r = Fresh();
        rec.Invoke(r, new object[] { new Vector3(0.3f, 0.2f, 0f), 0.3f });
        float tFlat = r.RaymarchTerrain(new Vector3(5f, 1f, 5f), Vector3.down, 6f);
        Check(Math.Abs(tFlat - 1f) < 5e-3f, $"looking down at flat ground the cursor is 1 m away ({tFlat:F3})");
        float tHill = r.RaymarchTerrain(new Vector3(0.3f, 2f, 0f), Vector3.down, 6f);
        Check(Math.Abs(tHill - 1.5f) < 5e-3f, $"looking down at the hill the cursor stops on its top ({tHill:F3})");
        Check(r.RaymarchTerrain(new Vector3(0.3f, 5f, 0f), Vector3.down, 3f) < 0f, "terrain beyond cursorMaxDist: no cursor");

        // --- Buffer wrap: at capacity 48, 50 sculpts keep the newest 48 ---
        var b = Fresh();
        typeof(SampleTerrainSculpt_Collider).GetField("sculptCapacity").SetValue(b, 48);
        Check(b.Capacity() == 48, "capacity follows the Inspector value");
        typeof(SampleTerrainSculpt_Collider).GetField("sculptCapacity").SetValue(s, 500);
        Check(s.Capacity() == SampleTerrainSculpt_Collider.MaxSculpts && SampleTerrainSculpt_Collider.MaxSculpts == 128, "capacity is clamped to the 128-slot array");
        for (int i = 0; i < 50; i++)
            rec.Invoke(b, new object[] { new Vector3(i, 0f, 0f), 0.3f });
        Check((int)Get(b, "sculptCount") == 48 && (int)Get(b, "nextSlot") == 2, "50 sculpts: 48 stored, next slot wrapped to 2");
        Check(((Vector4[])Get(b, "sculptData"))[0].x == 48f, "slot 0 holds the 49th sculpt (oldest overwritten)");
        // Skipping the most recent operations, across the wrap: the last three
        // are x = 47 (slot 47), 48 (slot 0), 49 (slot 1)
        Check(b.EvaluateSdf(new Vector3(47f, 0f, 0f)) < 0f && b.EvaluateSdfSkipping(new Vector3(47f, 0f, 0f), 3) >= 0f, "skipping 3 removes the hill at x = 47 across the buffer wrap");
        Check(b.EvaluateSdfSkipping(new Vector3(46f, 0f, 0f), 3) < 0f, "the hill at x = 46 (4th newest) stays");
        Check(b.EvaluateSdfSkipping(new Vector3(48f, 0f, 0f), 1) < 0f && b.EvaluateSdfSkipping(new Vector3(48f, 0f, 0f), 2) >= 0f, "x = 48 (2nd newest) survives skip 1 and goes with skip 2");
        // A held desktop button casts against the surface as it was at the
        // press: the cursor does not climb the hill it just made
        var dk = Fresh();
        var eye = new Vector3(0f, 1.6f, 0f); var down = new Vector3(0f, -0.6f, 0.8f); down = down / down.magnitude;
        float t1 = dk.RaymarchTerrainSkipping(eye, down, 6f, 0);
        var hit1 = eye + down * t1;
        Check(dk.TrySculptSkipping(hit1, true, 0, "press", 0), "first click sculpts at the view hit");
        float t2 = dk.RaymarchTerrainSkipping(eye, down, 6f, 1);
        Check(Math.Abs(t2 - t1) < 1e-4f, $"with the press's op skipped the cursor stays at the original hit ({t2:F3} = {t1:F3})");
        Check(dk.RaymarchTerrainSkipping(eye, down, 6f, 0) < t1 - 0.1f, "against the new terrain it would have moved up the hill toward the eye");

        // --- Serialization flag ---
        Set(b, "stateDirty", false);
        rec.Invoke(b, new object[] { new Vector3(0f, 0f, 0f), 0.3f });
        Check(b.HasUnsentChanges(), "a sculpt marks the state dirty again");

        Console.WriteLine(fails == 0 ? "behaviour: all ok" : $"behaviour: {fails} FAIL");
        return fails;
    }
}
