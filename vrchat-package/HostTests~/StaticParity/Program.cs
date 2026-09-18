// Parity of one static sample collider's Evaluate vs its alice_sdf golden:
//   StaticParity <golden.txt> <example name>
// where the example name is vrchat_<sample>_golden (the parity script passes
// it). Then the base collider's push scenario. Exit 1 on drift.
using System;
using System.IO;
using System.Globalization;
using UnityEngine;
using AliceSDF;
using AliceSDF.Samples;

static class Program
{
    static int Main(string[] args)
    {
        string example = args.Length > 1 ? args[1] : "vrchat_basic_golden";
        AliceSDF_Collider c;
        switch (example)
        {
            case "vrchat_basic_golden":   c = new SampleBasic_Collider(); break;
            case "vrchat_cosmic_golden":  c = new SampleCosmic_Collider(); break;
            case "vrchat_fractal_golden": c = new SampleFractal_Collider(); break;
            case "vrchat_mix_golden":     c = new SampleMix_Collider(); break;
            default:
                Console.Error.WriteLine("unknown sample " + example);
                return 1;
        }
        // animTime stays 0: the golden is the t = 0 snapshot

        float maxErr = 0f; int n = 0; string worst = "";
        foreach (var line in File.ReadLines(args[0]))
        {
            var f = line.Split(' ');
            var p = new Vector3(Parse(f[0]), Parse(f[1]), Parse(f[2]));
            float golden = Parse(f[3]);
            float d = c.Evaluate(p);
            float err = Math.Abs(d - golden);
            if (err > maxErr) { maxErr = err; worst = $"p=({p.x},{p.y},{p.z}) cs={d:F7} rust={golden:F7}"; }
            n++;
        }
        Console.WriteLine($"{example}: points={n} max_abs_err={maxErr:E3} worst: {worst}");
        // 1e-5: both sides are f32 with the same law; summation order and the
        // sin / cos implementation (Mathf vs alice_det_math) differ by ulps
        int bf = Behaviour.Run();
        return (maxErr <= 1e-5f && bf == 0) ? 0 : 1;
    }
    static float Parse(string s) => float.Parse(s, CultureInfo.InvariantCulture);
}
