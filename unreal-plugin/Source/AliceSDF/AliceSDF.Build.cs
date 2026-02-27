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
			PublicAdditionalLibraries.Add(Path.Combine(LibDir, "Win64", "alice_sdf.lib"));
			RuntimeDependencies.Add("$(BinaryOutputDir)/alice_sdf.dll",
				Path.Combine(LibDir, "Win64", "alice_sdf.dll"));
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
	}
}
