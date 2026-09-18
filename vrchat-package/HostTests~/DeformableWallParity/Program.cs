// Parity: C# SampleDeformableWall_Collider.EvaluateSdf vs the alice_sdf
// golden (examples/vrchat_deformable_wall_golden.rs), then the behaviour
// scenario. Exit 1 on drift.
using System;
using System.IO;
using System.Globalization;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Program
{
    // The dents of examples/vrchat_deformable_wall_golden.rs, in slot order
    // (w = strength); keep the two in sync
    public static readonly Vector4[] Dents =
    {
        new Vector4( 1.0f, 1.2f,  0.2f, 1.0f),
        new Vector4(-2.0f, 0.8f, -0.2f, 0.6f),
        new Vector4( 3.0f, 2.0f,  0.2f, 0.3f),
        new Vector4( 0.0f, 3.5f,  0.2f, 0.05f),
    };

    static int Main(string[] args)
    {
        var c = new SampleDeformableWall_Collider();
        // Start() is private (Unity message); invoke it like Unity would
        typeof(SampleDeformableWall_Collider).GetMethod("Start", BindingFlags.NonPublic | BindingFlags.Instance).Invoke(c, null);
        var F = BindingFlags.NonPublic | BindingFlags.Instance;
        var pts = (Vector4[])typeof(SampleDeformableWall_Collider).GetField("impactPoints", F).GetValue(c);
        for (int i = 0; i < Dents.Length; i++) pts[i] = Dents[i];
        typeof(SampleDeformableWall_Collider).GetField("impactCount", F).SetValue(c, Dents.Length);

        float maxErr = 0f; int n = 0; string worst = "";
        foreach (var line in File.ReadLines(args[0]))
        {
            var f = line.Split(' ');
            var p = new Vector3(Parse(f[0]), Parse(f[1]), Parse(f[2]));
            float golden = Parse(f[3]);
            float d = c.EvaluateSdf(p);
            float err = Math.Abs(d - golden);
            if (err > maxErr) { maxErr = err; worst = $"p=({p.x},{p.y},{p.z}) cs={d:F7} rust={golden:F7}"; }
            n++;
        }
        Console.WriteLine($"points={n} max_abs_err={maxErr:E3} worst: {worst}");
        // 1e-5: both sides are f32 with the same law; only summation order differs
        int bf = Behaviour.Run();
        return (maxErr <= 1e-5f && bf == 0) ? 0 : 1;
    }
    static float Parse(string s) => float.Parse(s, CultureInfo.InvariantCulture);
}
