// ALICE-SDF Module Implementation
// Author: Moroya Sakamoto

#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "ShaderCore.h"
#include "alice_sdf.h"

class FAliceSdfModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		FString BaseDir = IPluginManager::Get().FindPlugin(TEXT("AliceSDF"))->GetBaseDir();

		// Register shader directory for GPU particle compute shaders
		FString ShaderDir = FPaths::Combine(*BaseDir, TEXT("Shaders"));
		AddShaderSourceDirectoryMapping(TEXT("/Plugin/AliceSDF"), ShaderDir);

#if PLATFORM_MAC
		const FString LibName = TEXT("libalice_sdf.dylib");
		const FString LibSubDir = TEXT("Mac");
#elif PLATFORM_WINDOWS
		const FString LibName = TEXT("alice_sdf.dll");
		const FString LibSubDir = TEXT("Win64");
#elif PLATFORM_LINUX
		const FString LibName = TEXT("libalice_sdf.so");
		const FString LibSubDir = TEXT("Linux");
#endif

		// A UAT-packaged plugin has no ThirdParty/ — the library is staged
		// next to the module binary (AliceSDF.Build.cs RuntimeDependencies).
		// A source plugin (the distributed zip) has it in ThirdParty/. Try
		// both before giving up; the Win64 import is delay-loaded, so the
		// first alice_sdf_* call is what would fault if neither worked.
		const TArray<FString> Candidates = {
			FPaths::Combine(*BaseDir, TEXT("Binaries"), *LibSubDir, *LibName),
			FPaths::Combine(*BaseDir, TEXT("ThirdParty"), TEXT("AliceSDF"), TEXT("lib"), *LibSubDir, *LibName),
		};

		FString LoadedFrom;
		for (const FString& LibPath : Candidates)
		{
			if (!FPaths::FileExists(LibPath))
			{
				continue;
			}
			const FString LibDir = FPaths::GetPath(LibPath);
			FPlatformProcess::PushDllDirectory(*LibDir);
			LibHandle = FPlatformProcess::GetDllHandle(*LibPath);
			FPlatformProcess::PopDllDirectory(*LibDir);
			if (LibHandle)
			{
				LoadedFrom = LibPath;
				break;
			}
			UE_LOG(LogTemp, Warning, TEXT("ALICE-SDF: found but could not load %s"), *LibPath);
		}

		if (LibHandle)
		{
			// What the library actually is. A plugin installed next to a stale
			// library used to load fine and behave like an older release (the
			// repository shipped a 1.7.2 binary until 3.2.0), so say the version
			// out loud and complain when it is not the one this code expects.
			const VersionInfo Version = alice_sdf_version();
			const FString Reported = FString::Printf(TEXT("%u.%u.%u"),
				Version.major, Version.minor, Version.patch);
			const FString Expected = TEXT(ALICE_SDF_EXPECTED_VERSION);
			UE_LOG(LogTemp, Log, TEXT("ALICE-SDF: native library %s loaded from %s"), *Reported, *LoadedFrom);

			TArray<FString> ExpectedParts;
			Expected.ParseIntoArray(ExpectedParts, TEXT("."));
			const bool bMajorMinorMatch = ExpectedParts.Num() >= 2
				&& FCString::Atoi(*ExpectedParts[0]) == static_cast<int32>(Version.major)
				&& FCString::Atoi(*ExpectedParts[1]) == static_cast<int32>(Version.minor);
			if (!bMajorMinorMatch)
			{
				UE_LOG(LogTemp, Error,
					TEXT("ALICE-SDF: the native library is %s but this plugin was built against %s. ")
					TEXT("Rebuild it (cargo build --release --features unreal) or take the one from the ")
					TEXT("matching release zip — laws and the C API change between versions."),
					*Reported, *Expected);
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("ALICE-SDF: native library %s not found (looked in %s)"),
				*LibName, *FString::Join(Candidates, TEXT(", ")));
		}
	}

	virtual void ShutdownModule() override
	{
		if (LibHandle)
		{
			FPlatformProcess::FreeDllHandle(LibHandle);
			LibHandle = nullptr;
		}
	}

private:
	void* LibHandle = nullptr;
};

IMPLEMENT_MODULE(FAliceSdfModule, AliceSDF)
