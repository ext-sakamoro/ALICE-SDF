// Parity: C# SampleTerrainSculpt_Collider.EvaluateSdf vs the alice_sdf golden
// (examples/vrchat_terrain_sculpt_golden.rs), then the behaviour scenario.
// Exit 1 on drift.
using System;
using System.IO;
using System.Globalization;
using System.Reflection;
using UnityEngine;
using AliceSDF.Samples;

static class Program
{
    // The sculpts of examples/vrchat_terrain_sculpt_golden.rs, in order
    // (w > 0 adds a hill, w < 0 digs a hole); keep the two in sync
    public static readonly Vector4[] Sculpts =
    {
        new Vector4( 0.0f,  0.0f,  0.0f,  0.3f),
        new Vector4( 0.3f,  0.2f,  0.0f,  0.3f),
        new Vector4(-1.0f,  0.0f,  0.5f, -0.3f),
        new Vector4(-1.2f, -0.1f,  0.6f, -0.3f),
        new Vector4( 1.5f,  0.1f, -1.0f,  0.3f),
        new Vector4( 0.0f,  0.0f,  1.5f, -0.3f),
    };

    static int Main(string[] args)
    {
        var c = new SampleTerrainSculpt_Collider();
        typeof(SampleTerrainSculpt_Collider).GetField("groundOffset").SetValue(c, Vector3.zero);   // no transform on the host: the law anchored at the world origin, as the golden
        // The golden and these scenarios are the sphere brush with the original blend factors (the Inspector default is the block brush)
        typeof(SampleTerrainSculpt_Collider).GetField("blockBrush").SetValue(c, false);
        typeof(SampleTerrainSculpt_Collider).GetField("sculptRadius").SetValue(c, 0.3f);
        typeof(SampleTerrainSculpt_Collider).GetField("addSmooth").SetValue(c, 0.25f);
        typeof(SampleTerrainSculpt_Collider).GetField("subSmooth").SetValue(c, 0.15f);
        // Start() is private (Unity message); invoke it like Unity would
        typeof(SampleTerrainSculpt_Collider).GetMethod("Start", BindingFlags.NonPublic | BindingFlags.Instance).Invoke(c, null);
        var record = typeof(SampleTerrainSculpt_Collider).GetMethod("RecordSculpt", BindingFlags.NonPublic | BindingFlags.Instance);
        foreach (var s in Sculpts)
            record.Invoke(c, new object[] { new Vector3(s.x, s.y, s.z), s.w });

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
