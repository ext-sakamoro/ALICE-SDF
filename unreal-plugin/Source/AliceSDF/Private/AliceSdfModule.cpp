// ALICE-SDF Module Implementation
// Author: Moroya Sakamoto

#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "ShaderCore.h"

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
			UE_LOG(LogTemp, Log, TEXT("ALICE-SDF: native library loaded from %s"), *LoadedFrom);
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
