// Writes the Kit-renamed copies of the base collider and the seven sample
// colliders to the directory given as the first argument (default: gen/ next
// to this file). The rename rules mirror KitProductBuilder.BuildCore in
// vrchat-package/Editor: keep the two in step.
using System;
using System.IO;

namespace KitCompile
{
    internal static class Program
    {
        // sample directory, file, sample class, product class (KitProductBuilder.Scripts)
        private static readonly string[][] Scripts =
        {
            new[] { "SampleMochi",          "SampleMochi_Collider.cs",          "SampleMochi_Collider",          "AliceMochi" },
            new[] { "SampleDeformableWall", "SampleDeformableWall_Collider.cs", "SampleDeformableWall_Collider", "AliceWall" },
            new[] { "SampleTerrainSculpt",  "SampleTerrainSculpt_Collider.cs",  "SampleTerrainSculpt_Collider",  "AliceTerrain" },
            new[] { "SampleBasic",          "SampleBasic_Collider.cs",          "SampleBasic_Collider",          "AliceDecorBasic" },
            new[] { "SampleCosmic",         "SampleCosmic_Collider.cs",         "SampleCosmic_Collider",         "AliceDecorCosmic" },
            new[] { "SampleFractal",        "SampleFractal_Collider.cs",        "SampleFractal_Collider",        "AliceDecorFractal" },
            new[] { "SampleMix",            "SampleMix_Collider.cs",            "SampleMix_Collider",            "AliceDecorMix" },
        };

        private static int Main(string[] args)
        {
            string here = AppContext.BaseDirectory;
            // bin/<config>/<tfm>/ -> HostTests~/KitCompile
            string projectDir = Path.GetFullPath(Path.Combine(here, "..", "..", ".."));
            string package = Path.GetFullPath(Path.Combine(projectDir, "..", ".."));
            string gallery = Path.Combine(package, "Samples~", "SDF Gallery");
            string outDir = args.Length > 0 ? args[0] : Path.Combine(projectDir, "gen");
            if (!Directory.Exists(gallery)) { Console.Error.WriteLine("KitCompile: samples not found at " + gallery); return 1; }
            Directory.CreateDirectory(outDir);
            foreach (var stale in Directory.GetFiles(outDir, "*.cs")) File.Delete(stale);

            string baseCs = File.ReadAllText(Path.Combine(package, "Runtime", "Udon", "AliceSDF_Collider.cs"));
            baseCs = baseCs.Replace("namespace AliceSDF\r\n", "namespace AliceSDFKit\r\n").Replace("namespace AliceSDF\n", "namespace AliceSDFKit\n");
            if (!baseCs.Contains("namespace AliceSDFKit")) { Console.Error.WriteLine("KitCompile: base collider namespace not renamed"); return 1; }
            File.WriteAllText(Path.Combine(outDir, "AliceSDF_Collider.cs"), baseCs);

            foreach (var s in Scripts)
            {
                string cs = File.ReadAllText(Path.Combine(gallery, s[0], s[1]));
                cs = cs.Replace("namespace AliceSDF.Samples", "namespace AliceSDFKit").Replace(s[2], s[3]);
                if (!cs.Contains("class " + s[3])) { Console.Error.WriteLine("KitCompile: " + s[1] + " has no class " + s[2] + " to rename"); return 1; }
                File.WriteAllText(Path.Combine(outDir, s[3] + ".cs"), cs);
            }
            Console.WriteLine("KitCompile: wrote " + (Scripts.Length + 1) + " renamed scripts to " + outDir);
            return 0;
        }
    }
}
