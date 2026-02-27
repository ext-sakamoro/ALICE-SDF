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
		FString LibPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/AliceSDF/lib/Mac/libalice_sdf.dylib"));
#elif PLATFORM_WINDOWS
		FString LibPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/AliceSDF/lib/Win64/alice_sdf.dll"));
#elif PLATFORM_LINUX
		FString LibPath = FPaths::Combine(*BaseDir, TEXT("ThirdParty/AliceSDF/lib/Linux/libalice_sdf.so"));
#endif

		FPlatformProcess::PushDllDirectory(*FPaths::GetPath(LibPath));
		LibHandle = FPlatformProcess::GetDllHandle(*LibPath);
		FPlatformProcess::PopDllDirectory(*FPaths::GetPath(LibPath));

		if (!LibHandle)
		{
			UE_LOG(LogTemp, Error, TEXT("ALICE-SDF: Failed to load native library: %s"), *LibPath);
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
