# ALICE-SDF Unreal Engine 5 Plugin

Integrate ALICE-SDF with Unreal Engine 5 for real-time SDF evaluation, HLSL shader generation, and Blueprint scripting.

## Installation

### 1. Build the native library

```bash
cd ALICE-SDF
cargo build --release --features "ffi hlsl glsl gpu"
```

### 2. Copy files to your UE5 project

```
YourProject/
  Plugins/
    AliceSDF/
      AliceSDF.uplugin
      Source/
        AliceSDF/
          AliceSDF.Build.cs
          Public/
            AliceSdfComponent.h
          Private/
            AliceSdfComponent.cpp
            AliceSdfModule.cpp
      ThirdParty/
        AliceSDF/
          include/
            alice_sdf.h          <- from ALICE-SDF/include/
          lib/
            Win64/
              alice_sdf.dll      <- from target/release/
              alice_sdf.lib
            Mac/
              libalice_sdf.dylib <- from target/release/
            Linux/
              libalice_sdf.so    <- from target/release/
```

### 3. Regenerate project files

Open your `.uproject` in UE5 or run:

```bash
# Windows
UnrealBuildTool.exe -projectfiles -project="YourProject.uproject"

# macOS
mono UnrealBuildTool.exe -projectfiles -project="YourProject.uproject"
```

## Usage

### Blueprint

1. Add `AliceSdfComponent` to any Actor
2. Call `CreateSphere`, `CreateBox`, etc. from Blueprint
3. Use `EvalDistance` for collision queries
4. Use `GenerateHlsl` to get shader code for Custom Material Expressions

### C++

```cpp
#include "AliceSdfComponent.h"

void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    UAliceSdfComponent* Sdf = FindComponentByClass<UAliceSdfComponent>();
    Sdf->CreateSphere(1.0f);
    Sdf->Compile();

    float Distance = Sdf->EvalDistance(GetActorLocation());

    FString Hlsl = Sdf->GenerateHlsl();
}
```

### Custom Material Expression

```cpp
// Get HLSL code from SDF component
FString HlslCode = SdfComponent->GenerateHlsl();

// Paste into a Custom node in your Material graph
// Input: WorldPosition (from World Position node)
// Output: float (signed distance)
```

## Supported Platforms

| Platform | Library | Status |
|----------|---------|--------|
| Windows x64 | `alice_sdf.dll` | Supported |
| macOS (arm64/x64) | `libalice_sdf.dylib` | Supported |
| Linux x64 | `libalice_sdf.so` | Supported |

## API Reference

See [UNREAL_ENGINE.md](../docs/UNREAL_ENGINE.md) for the complete integration guide.

---

Author: Moroya Sakamoto
