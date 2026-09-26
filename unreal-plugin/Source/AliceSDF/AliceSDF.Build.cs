// ALICE-SDF Unreal Engine 5 Plugin
// Author: Moroya Sakamoto

using UnrealBuildTool;
using System.IO;

public class AliceSDF : ModuleRules
{
	public AliceSDF(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(new string[] {
			Path.Combine(ModuleDirectory, "Public"),
		});

		PrivateIncludePaths.AddRange(new string[] {
			Path.Combine(ModuleDirectory, "Private"),
		});

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"RenderCore",
			"RHI",
			"Renderer",
			"MeshDescription",
			"StaticMeshDescription",
		});

		PrivateDependencyModuleNames.AddRange(new string[] {
			"Projects",
		});

		// Link ALICE-SDF native library
		string LibDir = Path.Combine(ModuleDirectory, "..", "..", "ThirdParty", "AliceSDF", "lib");
		string IncDir = Path.Combine(ModuleDirectory, "..", "..", "ThirdParty", "AliceSDF", "include");

		PublicIncludePaths.Add(IncDir);

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string Dll = Path.Combine(LibDir, "Win64", "alice_sdf.dll");
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "Win64", "alice_sdf.lib"));
			// Delay load, or the Windows loader resolves every alice_sdf_*
			// import while loading UnrealEditor-AliceSDF.dll and fails with
			// error 126 when the native library is not next to it — which is
			// what a UAT-packaged plugin looked like until 3.2.0 (the module
			// loads it itself in StartupModule).
			PublicDelayLoadDLLs.Add("alice_sdf.dll");
			// Next to the module (packaged plugin) and in ThirdParty (source
			// plugin / the distributed zip): FAliceSdfModule looks in both.
			RuntimeDependencies.Add("$(BinaryOutputDir)/alice_sdf.dll", Dll);
			RuntimeDependencies.Add(Dll, StagedFileType.NonUFS);
		}
		else if (Target.Platform == UnrealTargetPlatform.Mac)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "Mac", "libalice_sdf.dylib"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/libalice_sdf.dylib",
				Path.Combine(LibDir, "Mac", "libalice_sdf.dylib"));
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "Linux", "libalice_sdf.so"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/libalice_sdf.so",
				Path.Combine(LibDir, "Linux", "libalice_sdf.so"));
		}

		PublicDefinitions.Add("WITH_ALICE_SDF=1");

		// The version the plugin was authored against, read from the .uplugin so
		// the two cannot drift. FAliceSdfModule compares it with what the native
		// library reports and says so when a stale library is loaded.
		string UPluginPath = Path.Combine(ModuleDirectory, "..", "..", "AliceSDF.uplugin");
		string ExpectedVersion = "unknown";
		if (File.Exists(UPluginPath))
		{
			var Match = System.Text.RegularExpressions.Regex.Match(
				File.ReadAllText(UPluginPath), @"""VersionName""\s*:\s*""([^""]+)""");
			if (Match.Success)
			{
				ExpectedVersion = Match.Groups[1].Value;
			}
		}
		PublicDefinitions.Add("ALICE_SDF_EXPECTED_VERSION=\"" + ExpectedVersion + "\"");
	}
}
