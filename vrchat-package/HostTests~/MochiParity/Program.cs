// Parity: C# SampleMochi_Collider.EvaluateSdf vs the alice_sdf golden
// (examples/vrchat_mochi_golden.rs), then the behaviour scenario. Exit 1 on drift.
using System;
using System.IO;
using System.Globalization;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Program
{
    static int Main(string[] args)
    {
        var c = new SampleMochi_Collider();
        typeof(SampleMochi_Collider).GetField("groundOffset").SetValue(c, Vector3.zero);   // no transform on the host: the law anchored at the world origin, as the golden
        // Start() is private (Unity message); invoke it like Unity would
        typeof(SampleMochi_Collider).GetMethod("Start", BindingFlags.NonPublic | BindingFlags.Instance).Invoke(c, null);

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
        // 1e-5: both sides are f32 with the same law; only summation order / pow differ
        int bf = Behaviour.Run();
        return (maxErr <= 1e-5f && bf == 0) ? 0 : 1;
    }
    static float Parse(string s) => float.Parse(s, CultureInfo.InvariantCulture);
}
