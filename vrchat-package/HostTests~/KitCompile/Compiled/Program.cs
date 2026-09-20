// The eight Kit types must exist under AliceSDFKit with their SDF entry
// point (the four decor pieces derive from the base collider and override
// Evaluate, the three interactive ones expose EvaluateSdf) and nothing may
// be left in AliceSDF / AliceSDF.Samples
using System;
using System.Reflection;

namespace KitCompiled
{
    internal static class Program
    {
        private static int Main()
        {
            var asm = Assembly.GetExecutingAssembly();
            int bad = 0;
            var baseType = asm.GetType("AliceSDFKit.AliceSDF_Collider");
            if (baseType == null) { Console.Error.WriteLine("KitCompiled: AliceSDFKit.AliceSDF_Collider missing"); bad++; }
            foreach (var name in new[] { "AliceDecorBasic", "AliceDecorCosmic", "AliceDecorFractal", "AliceDecorMix" })
            {
                var t = asm.GetType("AliceSDFKit." + name);
                if (t == null) { Console.Error.WriteLine("KitCompiled: AliceSDFKit." + name + " missing"); bad++; continue; }
                if (baseType != null && !baseType.IsAssignableFrom(t)) { Console.Error.WriteLine("KitCompiled: " + name + " does not derive from AliceSDF_Collider"); bad++; }
                var ev = t.GetMethod("Evaluate", BindingFlags.Public | BindingFlags.Instance);
                if (ev == null || ev.DeclaringType != t) { Console.Error.WriteLine("KitCompiled: " + name + " does not override Evaluate"); bad++; }
            }
            foreach (var name in new[] { "AliceMochi", "AliceWall", "AliceTerrain" })
            {
                var t = asm.GetType("AliceSDFKit." + name);
                if (t == null) { Console.Error.WriteLine("KitCompiled: AliceSDFKit." + name + " missing"); bad++; continue; }
                if (t.GetMethod("EvaluateSdf", BindingFlags.Public | BindingFlags.Instance) == null) { Console.Error.WriteLine("KitCompiled: " + name + " has no public EvaluateSdf"); bad++; }
            }
            foreach (var t in asm.GetTypes())
                if (t.Namespace != null && (t.Namespace == "AliceSDF" || t.Namespace.StartsWith("AliceSDF.")))
                { Console.Error.WriteLine("KitCompiled: type left outside AliceSDFKit: " + t.FullName); bad++; }
            Console.WriteLine("KitCompiled: " + (bad == 0 ? "8 Kit types present under AliceSDFKit" : bad + " problem(s)"));
            return bad == 0 ? 0 : 1;
        }
    }
}
