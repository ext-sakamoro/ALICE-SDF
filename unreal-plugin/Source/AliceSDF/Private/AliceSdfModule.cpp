// ALICE-SDF Module Implementation
// Author: Moroya Sakamoto

#include "Modules/ModuleManager.h"

class FAliceSdfModule : public IModuleInterface
{
public:
	virtual void StartupModule() override
	{
		// Load the native library
		// The DLL/dylib/so is loaded automatically via RuntimeDependencies
	}

	virtual void ShutdownModule() override
	{
	}
};

IMPLEMENT_MODULE(FAliceSdfModule, AliceSDF)
