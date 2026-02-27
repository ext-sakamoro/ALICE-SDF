// ALICE-SDF GPU Particle Actor Implementation
// Author: Moroya Sakamoto
//
// Full GPU SDF particle system for UE5.
// Compute Shader evaluates SDF + physics, renders via instanced indirect draw.

#include "AliceSdfParticleActor.h"
#include "AliceSdfParticleComponent.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "RHICommandList.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RHIStaticStates.h"
#include "PipelineStateCache.h"
#include "Camera/PlayerCameraManager.h"
#include "Engine/LocalPlayer.h"
#include "GameFramework/PlayerController.h"

// =============================================================================
// Compute Shader Declarations
// =============================================================================

// Base class for all SDF particle compute shaders
class FSdfParticleCS : public FGlobalShader
{
public:
	DECLARE_INLINE_TYPE_LAYOUT(FSdfParticleCS, NonVirtual);

	FSdfParticleCS() = default;
	FSdfParticleCS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		Particles.Bind(Initializer.ParameterMap, TEXT("Particles"));
		ParticleCount.Bind(Initializer.ParameterMap, TEXT("ParticleCount"));
		DeltaTime.Bind(Initializer.ParameterMap, TEXT("DeltaTime"));
		Time.Bind(Initializer.ParameterMap, TEXT("Time"));
		FlowSpeed.Bind(Initializer.ParameterMap, TEXT("FlowSpeed"));
		SurfaceAttraction.Bind(Initializer.ParameterMap, TEXT("SurfaceAttraction"));
		NoiseStrength.Bind(Initializer.ParameterMap, TEXT("NoiseStrength"));
		MaxDistance.Bind(Initializer.ParameterMap, TEXT("MaxDistance"));
		SpawnRadius.Bind(Initializer.ParameterMap, TEXT("SpawnRadius"));
		SliceIndex.Bind(Initializer.ParameterMap, TEXT("SliceIndex"));
		SliceCount.Bind(Initializer.ParameterMap, TEXT("SliceCount"));
	}

	void SetCommonParameters(FRHIBatchedShaderParameters& BatchedParams,
		FRHIUnorderedAccessView* InParticleUAV,
		uint32 InParticleCount, float InDeltaTime, float InTime,
		float InFlowSpeed, float InSurfaceAttraction, float InNoiseStrength,
		float InMaxDistance, float InSpawnRadius,
		uint32 InSliceIndex, uint32 InSliceCount)
	{
		if (Particles.IsBound())
			BatchedParams.SetUAVParameter(Particles.GetBaseIndex(), InParticleUAV);
		SetShaderValue(BatchedParams, ParticleCount, InParticleCount);
		SetShaderValue(BatchedParams, DeltaTime, InDeltaTime);
		SetShaderValue(BatchedParams, Time, InTime);
		SetShaderValue(BatchedParams, FlowSpeed, InFlowSpeed);
		SetShaderValue(BatchedParams, SurfaceAttraction, InSurfaceAttraction);
		SetShaderValue(BatchedParams, NoiseStrength, InNoiseStrength);
		SetShaderValue(BatchedParams, MaxDistance, InMaxDistance);
		SetShaderValue(BatchedParams, SpawnRadius, InSpawnRadius);
		SetShaderValue(BatchedParams, SliceIndex, InSliceIndex);
		SetShaderValue(BatchedParams, SliceCount, InSliceCount);
	}

	LAYOUT_FIELD(FShaderResourceParameter, Particles);
	LAYOUT_FIELD(FShaderParameter, ParticleCount);
	LAYOUT_FIELD(FShaderParameter, DeltaTime);
	LAYOUT_FIELD(FShaderParameter, Time);
	LAYOUT_FIELD(FShaderParameter, FlowSpeed);
	LAYOUT_FIELD(FShaderParameter, SurfaceAttraction);
	LAYOUT_FIELD(FShaderParameter, NoiseStrength);
	LAYOUT_FIELD(FShaderParameter, MaxDistance);
	LAYOUT_FIELD(FShaderParameter, SpawnRadius);
	LAYOUT_FIELD(FShaderParameter, SliceIndex);
	LAYOUT_FIELD(FShaderParameter, SliceCount);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};

// =============================================================================
// Cosmic Compute Shaders
// =============================================================================

class FSdfParticleCS_CosmicInit : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_CosmicInit);
public:
	FSdfParticleCS_CosmicInit() = default;
	FSdfParticleCS_CosmicInit(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
	}
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_CosmicInit,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Cosmic.usf", "CSInit", SF_Compute);

class FSdfParticleCS_CosmicMain : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_CosmicMain);
public:
	FSdfParticleCS_CosmicMain() = default;
	FSdfParticleCS_CosmicMain(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
		SunRadius.Bind(Initializer.ParameterMap, TEXT("SunRadius"));
		PlanetRadius.Bind(Initializer.ParameterMap, TEXT("PlanetRadius"));
		PlanetDistance.Bind(Initializer.ParameterMap, TEXT("PlanetDistance"));
		SmoothnessParam.Bind(Initializer.ParameterMap, TEXT("Smoothness"));
		RingMajorRadius.Bind(Initializer.ParameterMap, TEXT("RingMajorRadius"));
		RingMinorRadius.Bind(Initializer.ParameterMap, TEXT("RingMinorRadius"));
	}

	LAYOUT_FIELD(FShaderParameter, SunRadius);
	LAYOUT_FIELD(FShaderParameter, PlanetRadius);
	LAYOUT_FIELD(FShaderParameter, PlanetDistance);
	LAYOUT_FIELD(FShaderParameter, SmoothnessParam);
	LAYOUT_FIELD(FShaderParameter, RingMajorRadius);
	LAYOUT_FIELD(FShaderParameter, RingMinorRadius);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_CosmicMain,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Cosmic.usf", "CSMain", SF_Compute);

// =============================================================================
// Terrain Compute Shaders
// =============================================================================

class FSdfParticleCS_TerrainInit : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_TerrainInit);
public:
	FSdfParticleCS_TerrainInit() = default;
	FSdfParticleCS_TerrainInit(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer) {}
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_TerrainInit,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Terrain.usf", "CSInit", SF_Compute);

class FSdfParticleCS_TerrainMain : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_TerrainMain);
public:
	FSdfParticleCS_TerrainMain() = default;
	FSdfParticleCS_TerrainMain(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
		TerrainHeightParam.Bind(Initializer.ParameterMap, TEXT("TerrainHeight"));
		TerrainScaleParam.Bind(Initializer.ParameterMap, TEXT("TerrainScale"));
		WaterLevelParam.Bind(Initializer.ParameterMap, TEXT("WaterLevel"));
		RockSizeParam.Bind(Initializer.ParameterMap, TEXT("RockSize"));
	}

	LAYOUT_FIELD(FShaderParameter, TerrainHeightParam);
	LAYOUT_FIELD(FShaderParameter, TerrainScaleParam);
	LAYOUT_FIELD(FShaderParameter, WaterLevelParam);
	LAYOUT_FIELD(FShaderParameter, RockSizeParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_TerrainMain,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Terrain.usf", "CSMain", SF_Compute);

// =============================================================================
// Abstract Compute Shaders
// =============================================================================

class FSdfParticleCS_AbstractInit : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_AbstractInit);
public:
	FSdfParticleCS_AbstractInit() = default;
	FSdfParticleCS_AbstractInit(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer) {}
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_AbstractInit,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Abstract.usf", "CSInit", SF_Compute);

class FSdfParticleCS_AbstractMain : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_AbstractMain);
public:
	FSdfParticleCS_AbstractMain() = default;
	FSdfParticleCS_AbstractMain(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
		GyroidScaleParam.Bind(Initializer.ParameterMap, TEXT("GyroidScale"));
		GyroidThicknessParam.Bind(Initializer.ParameterMap, TEXT("GyroidThickness"));
		MetaballRadiusParam.Bind(Initializer.ParameterMap, TEXT("MetaballRadius"));
		MorphAmountParam.Bind(Initializer.ParameterMap, TEXT("MorphAmount"));
	}

	LAYOUT_FIELD(FShaderParameter, GyroidScaleParam);
	LAYOUT_FIELD(FShaderParameter, GyroidThicknessParam);
	LAYOUT_FIELD(FShaderParameter, MetaballRadiusParam);
	LAYOUT_FIELD(FShaderParameter, MorphAmountParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_AbstractMain,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Abstract.usf", "CSMain", SF_Compute);

// =============================================================================
// Fractal Compute Shaders
// =============================================================================

class FSdfParticleCS_FractalInit : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_FractalInit);
public:
	FSdfParticleCS_FractalInit() = default;
	FSdfParticleCS_FractalInit(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
		BoxSizeParam.Bind(Initializer.ParameterMap, TEXT("BoxSize"));
		HoleSizeParam.Bind(Initializer.ParameterMap, TEXT("HoleSize"));
		RepeatScaleParam.Bind(Initializer.ParameterMap, TEXT("RepeatScale"));
		TwistAmountParam.Bind(Initializer.ParameterMap, TEXT("TwistAmount"));
		CamPosParam.Bind(Initializer.ParameterMap, TEXT("CamPos"));
		CamForwardParam.Bind(Initializer.ParameterMap, TEXT("CamForward"));
		CamRightParam.Bind(Initializer.ParameterMap, TEXT("CamRight"));
		CamUpParam.Bind(Initializer.ParameterMap, TEXT("CamUp"));
		ZoomLevelParam.Bind(Initializer.ParameterMap, TEXT("ZoomLevel"));
	}

	LAYOUT_FIELD(FShaderParameter, BoxSizeParam);
	LAYOUT_FIELD(FShaderParameter, HoleSizeParam);
	LAYOUT_FIELD(FShaderParameter, RepeatScaleParam);
	LAYOUT_FIELD(FShaderParameter, TwistAmountParam);
	LAYOUT_FIELD(FShaderParameter, CamPosParam);
	LAYOUT_FIELD(FShaderParameter, CamForwardParam);
	LAYOUT_FIELD(FShaderParameter, CamRightParam);
	LAYOUT_FIELD(FShaderParameter, CamUpParam);
	LAYOUT_FIELD(FShaderParameter, ZoomLevelParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_FractalInit,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Fractal.usf", "CSInit", SF_Compute);

class FSdfParticleCS_FractalMain : public FSdfParticleCS
{
	DECLARE_GLOBAL_SHADER(FSdfParticleCS_FractalMain);
public:
	FSdfParticleCS_FractalMain() = default;
	FSdfParticleCS_FractalMain(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FSdfParticleCS(Initializer)
	{
		BoxSizeParam.Bind(Initializer.ParameterMap, TEXT("BoxSize"));
		HoleSizeParam.Bind(Initializer.ParameterMap, TEXT("HoleSize"));
		RepeatScaleParam.Bind(Initializer.ParameterMap, TEXT("RepeatScale"));
		TwistAmountParam.Bind(Initializer.ParameterMap, TEXT("TwistAmount"));
		FractalIterationsParam.Bind(Initializer.ParameterMap, TEXT("FractalIterations"));
		CamPosParam.Bind(Initializer.ParameterMap, TEXT("CamPos"));
		CamForwardParam.Bind(Initializer.ParameterMap, TEXT("CamForward"));
		CamRightParam.Bind(Initializer.ParameterMap, TEXT("CamRight"));
		CamUpParam.Bind(Initializer.ParameterMap, TEXT("CamUp"));
		ZoomLevelParam.Bind(Initializer.ParameterMap, TEXT("ZoomLevel"));
	}

	LAYOUT_FIELD(FShaderParameter, BoxSizeParam);
	LAYOUT_FIELD(FShaderParameter, HoleSizeParam);
	LAYOUT_FIELD(FShaderParameter, RepeatScaleParam);
	LAYOUT_FIELD(FShaderParameter, TwistAmountParam);
	LAYOUT_FIELD(FShaderParameter, FractalIterationsParam);
	LAYOUT_FIELD(FShaderParameter, CamPosParam);
	LAYOUT_FIELD(FShaderParameter, CamForwardParam);
	LAYOUT_FIELD(FShaderParameter, CamRightParam);
	LAYOUT_FIELD(FShaderParameter, CamUpParam);
	LAYOUT_FIELD(FShaderParameter, ZoomLevelParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleCS_FractalMain,
	"/Plugin/AliceSDF/Private/SdfParticleCompute_Fractal.usf", "CSMain", SF_Compute);

// =============================================================================
// Actor Implementation
// =============================================================================

AAliceSdfParticleActor::AAliceSdfParticleActor()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);

	ParticleRenderComponent = CreateDefaultSubobject<UAliceSdfParticleComponent>(TEXT("ParticleRenderer"));
	ParticleRenderComponent->SetupAttachment(Root);
}

void AAliceSdfParticleActor::BeginPlay()
{
	Super::BeginPlay();

	// Register view extension BEFORE initializing particles
	// so it's ready to render as soon as GPU resources are created
	if (ParticleRenderComponent)
	{
		ParticleRenderComponent->CreateViewExtension();
	}

	InitializeParticles();
}

void AAliceSdfParticleActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	// Destroy view extension BEFORE releasing GPU resources
	if (ParticleRenderComponent)
	{
		ParticleRenderComponent->DestroyViewExtension();
	}

	ReleaseGPU();
	Super::EndPlay(EndPlayReason);
}

void AAliceSdfParticleActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!bGpuInitialized) return;

	// FPS tracking
	FrameCounter++;
	FpsTimer += DeltaTime;
	if (FpsTimer >= 1.0f)
	{
		CurrentFPS = FrameCounter / FpsTimer;
		FrameCounter = 0;
		FpsTimer = 0.0f;
	}

	// Scene change check
	if (SceneType != CurrentSceneType)
	{
		SwitchScene(SceneType);
	}

	// Dispatch compute shader
	DispatchComputeShader(DeltaTime);

	// Render particles
	RenderParticles();
}

// =============================================================================
// GPU Resource Management
// =============================================================================

void AAliceSdfParticleActor::CreateGPUResources()
{
	const int32 BufferSize = PARTICLE_STRIDE * ParticleCount;
	const int32 Stride = PARTICLE_STRIDE;

	// Must create RHI resources on the render thread
	ENQUEUE_RENDER_COMMAND(AliceSdfCreateGPUResources)(
		[this, BufferSize, Stride](FRHICommandListImmediate& RHICmdList)
	{
		FRHIBufferCreateDesc BufferDesc = FRHIBufferCreateDesc::CreateStructured(
			TEXT("AliceSdfParticleBuffer"), BufferSize, Stride);
		BufferDesc.AddUsage(EBufferUsageFlags::UnorderedAccess | EBufferUsageFlags::ShaderResource);
		BufferDesc.DetermineInitialState();

		ParticleBuffer = RHICmdList.CreateBuffer(BufferDesc);

		ParticleSRV = RHICmdList.CreateShaderResourceView(
			ParticleBuffer, FRHIViewDesc::CreateBufferSRV().SetTypeFromBuffer(ParticleBuffer));
		ParticleUAV = RHICmdList.CreateUnorderedAccessView(
			ParticleBuffer, FRHIViewDesc::CreateBufferUAV().SetTypeFromBuffer(ParticleBuffer));
	});

	// Wait for render thread to finish creating resources
	FlushRenderingCommands();

	bGpuInitialized = true;
}

void AAliceSdfParticleActor::InitializeParticles()
{
	if (bGpuInitialized)
	{
		ReleaseGPU();
	}

	CreateGPUResources();
	CurrentSceneType = SceneType;

	// Dispatch init kernel
	ENQUEUE_RENDER_COMMAND(AliceSdfParticleInit)(
		[this](FRHICommandListImmediate& RHICmdList)
	{
		const int32 ThreadGroups = FMath::CeilToInt32(
			static_cast<float>(ParticleCount) / THREAD_GROUP_SIZE);

		switch (CurrentSceneType)
		{
		case EAliceSdfGpuScene::Cosmic:
		{
			TShaderMapRef<FSdfParticleCS_CosmicInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Terrain:
		{
			TShaderMapRef<FSdfParticleCS_TerrainInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Abstract:
		{
			TShaderMapRef<FSdfParticleCS_AbstractInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Fractal:
		{
			TShaderMapRef<FSdfParticleCS_FractalInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);

			// Fractal-specific params
			SetShaderValue(BatchedParams, CS->BoxSizeParam, BoxSize);
			SetShaderValue(BatchedParams, CS->HoleSizeParam, HoleSize);
			SetShaderValue(BatchedParams, CS->RepeatScaleParam, RepeatScale);
			SetShaderValue(BatchedParams, CS->TwistAmountParam, TwistAmount);
			SetShaderValue(BatchedParams, CS->CamPosParam, FVector3f::ZeroVector);
			SetShaderValue(BatchedParams, CS->CamForwardParam, FVector3f(1, 0, 0));
			SetShaderValue(BatchedParams, CS->CamRightParam, FVector3f(0, 1, 0));
			SetShaderValue(BatchedParams, CS->CamUpParam, FVector3f(0, 0, 1));
			SetShaderValue(BatchedParams, CS->ZoomLevelParam, 1.0f);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		}

		// UAV→SRV barrier: ensure compute writes are visible to render shader (Metal-critical)
		FRHITransitionInfo InitBarrier(ParticleBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics);
		RHICmdList.Transition(MakeArrayView(&InitBarrier, 1));
	});

	// Wait for init compute to complete before first render frame
	FlushRenderingCommands();

	UE_LOG(LogTemp, Log,
		TEXT("ALICE-SDF GPU Particles: Initialized %s particles (Scene: %d)"),
		*FString::Printf(TEXT("%d"), ParticleCount),
		static_cast<int32>(SceneType));
}

void AAliceSdfParticleActor::ReleaseGPU()
{
	if (ParticleRenderComponent)
	{
		ParticleRenderComponent->ClearParticleResources();
	}
	ParticleUAV = nullptr;
	ParticleSRV = nullptr;
	ParticleBuffer = nullptr;
	bGpuInitialized = false;
}

void AAliceSdfParticleActor::SwitchScene(EAliceSdfGpuScene NewScene)
{
	CurrentSceneType = NewScene;
	SceneType = NewScene;
	Reinitialize();

	UE_LOG(LogTemp, Log, TEXT("ALICE-SDF GPU Particles: Switched to scene %d"),
		static_cast<int32>(NewScene));
}

void AAliceSdfParticleActor::Reinitialize()
{
	if (!bGpuInitialized) return;

	// Re-dispatch init kernel
	ENQUEUE_RENDER_COMMAND(AliceSdfParticleReinit)(
		[this](FRHICommandListImmediate& RHICmdList)
	{
		const int32 ThreadGroups = FMath::CeilToInt32(
			static_cast<float>(ParticleCount) / THREAD_GROUP_SIZE);

		switch (CurrentSceneType)
		{
		case EAliceSdfGpuScene::Cosmic:
		{
			TShaderMapRef<FSdfParticleCS_CosmicInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Terrain:
		{
			TShaderMapRef<FSdfParticleCS_TerrainInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Abstract:
		{
			TShaderMapRef<FSdfParticleCS_AbstractInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);
			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Fractal:
		{
			TShaderMapRef<FSdfParticleCS_FractalInit> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, ParticleUAV,
				ParticleCount, 0, 0, FlowSpeed, SurfaceAttraction,
				NoiseStrength, MaxDistance, SpawnRadius, 0, 1);

			SetShaderValue(BatchedParams, CS->BoxSizeParam, BoxSize);
			SetShaderValue(BatchedParams, CS->HoleSizeParam, HoleSize);
			SetShaderValue(BatchedParams, CS->RepeatScaleParam, RepeatScale);
			SetShaderValue(BatchedParams, CS->TwistAmountParam, TwistAmount);
			SetShaderValue(BatchedParams, CS->CamPosParam, FVector3f::ZeroVector);
			SetShaderValue(BatchedParams, CS->CamForwardParam, FVector3f(1, 0, 0));
			SetShaderValue(BatchedParams, CS->CamRightParam, FVector3f(0, 1, 0));
			SetShaderValue(BatchedParams, CS->CamUpParam, FVector3f(0, 0, 1));
			SetShaderValue(BatchedParams, CS->ZoomLevelParam, 1.0f);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		}

		// UAV→SRV barrier after reinit compute dispatch (Metal-critical)
		FRHITransitionInfo ReinitBarrier(ParticleBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics);
		RHICmdList.Transition(MakeArrayView(&ReinitBarrier, 1));
	});
}

// =============================================================================
// Compute Shader Dispatch (Every Frame)
// =============================================================================

void AAliceSdfParticleActor::DispatchComputeShader(float DeltaTimeArg)
{
	if (!bGpuInitialized) return;

	const float CurrentTime = GetWorld()->GetTimeSeconds();
	const int32 CurrentSliceIndex = SliceIndex;
	const int32 CurrentSliceCount = UpdateDivisions;

	// Advance slice
	if (UpdateDivisions > 1)
	{
		SliceIndex = (SliceIndex + 1) % UpdateDivisions;
	}
	else
	{
		SliceIndex = 0;
	}

	// Capture all params for render thread
	const float CapturedFlowSpeed = FlowSpeed;
	const float CapturedAttraction = SurfaceAttraction;
	const float CapturedNoise = NoiseStrength;
	const float CapturedMaxDist = MaxDistance;
	const float CapturedSpawnRadius = SpawnRadius;
	const int32 CapturedParticleCount = ParticleCount;
	const EAliceSdfGpuScene CapturedScene = CurrentSceneType;

	// Scene-specific captures
	const float CapturedSunRadius = SunRadius;
	const float CapturedPlanetRadius = PlanetRadius;
	const float CapturedPlanetDistance = PlanetDistance;
	const float CapturedSmoothness = Smoothness;

	const float CapturedTerrainHeight = TerrainHeight;
	const float CapturedTerrainScale = TerrainScale;
	const float CapturedWaterLevel = WaterLevel;
	const float CapturedRockSize = RockSize;

	const float CapturedGyroidScale = GyroidScale;
	const float CapturedGyroidThickness = GyroidThickness;
	const float CapturedMetaballRadius = MetaballRadius;
	const float CapturedMorphAmount = MorphAmount;

	const float CapturedBoxSize = BoxSize;
	const float CapturedHoleSize = HoleSize;
	const float CapturedRepeatScale = RepeatScale;
	const float CapturedTwistAmount = TwistAmount;
	const int32 CapturedFractalIter = FractalIterations;

	// Camera for fractal microscope mode
	FVector3f CamPos3f = FVector3f::ZeroVector;
	FVector3f CamFwd3f = FVector3f(1, 0, 0);
	FVector3f CamRight3f = FVector3f(0, 1, 0);
	FVector3f CamUp3f = FVector3f(0, 0, 1);
	float CapturedZoom = 1.0f;

	if (CapturedScene == EAliceSdfGpuScene::Fractal)
	{
		APlayerController* PC = GetWorld()->GetFirstPlayerController();
		if (PC && PC->PlayerCameraManager)
		{
			FVector CamLoc = PC->PlayerCameraManager->GetCameraLocation();
			FRotator CamRot = PC->PlayerCameraManager->GetCameraRotation();
			CamPos3f = FVector3f(CamLoc);
			CamFwd3f = FVector3f(CamRot.Vector());
			CamRight3f = FVector3f(FRotationMatrix(CamRot).GetUnitAxis(EAxis::Y));
			CamUp3f = FVector3f(FRotationMatrix(CamRot).GetUnitAxis(EAxis::Z));
		}
	}

	FUnorderedAccessViewRHIRef CapturedUAV = ParticleUAV;
	FBufferRHIRef CapturedBuffer = ParticleBuffer;

	ENQUEUE_RENDER_COMMAND(AliceSdfParticleUpdate)(
		[=](FRHICommandListImmediate& RHICmdList)
	{
		// Transition buffer from SRV (previous frame's render) to UAV (compute write)
		FRHITransitionInfo PreBarrier(CapturedBuffer, ERHIAccess::SRVGraphics, ERHIAccess::UAVCompute);
		RHICmdList.Transition(MakeArrayView(&PreBarrier, 1));

		const int32 ThreadGroups = FMath::CeilToInt32(
			static_cast<float>(CapturedParticleCount) / THREAD_GROUP_SIZE);

		switch (CapturedScene)
		{
		case EAliceSdfGpuScene::Cosmic:
		{
			TShaderMapRef<FSdfParticleCS_CosmicMain> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, CapturedUAV,
				CapturedParticleCount, DeltaTimeArg, CurrentTime,
				CapturedFlowSpeed, CapturedAttraction, CapturedNoise,
				CapturedMaxDist, CapturedSpawnRadius,
				CurrentSliceIndex, CurrentSliceCount);

			SetShaderValue(BatchedParams, CS->SunRadius, CapturedSunRadius);
			SetShaderValue(BatchedParams, CS->PlanetRadius, CapturedPlanetRadius);
			SetShaderValue(BatchedParams, CS->PlanetDistance, CapturedPlanetDistance);
			SetShaderValue(BatchedParams, CS->SmoothnessParam, CapturedSmoothness);
			SetShaderValue(BatchedParams, CS->RingMajorRadius, CapturedPlanetRadius * 1.8f);
			SetShaderValue(BatchedParams, CS->RingMinorRadius, 12.0f);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Terrain:
		{
			TShaderMapRef<FSdfParticleCS_TerrainMain> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, CapturedUAV,
				CapturedParticleCount, DeltaTimeArg, CurrentTime,
				CapturedFlowSpeed, CapturedAttraction, CapturedNoise,
				CapturedMaxDist, CapturedSpawnRadius,
				CurrentSliceIndex, CurrentSliceCount);

			SetShaderValue(BatchedParams, CS->TerrainHeightParam, CapturedTerrainHeight);
			SetShaderValue(BatchedParams, CS->TerrainScaleParam, CapturedTerrainScale);
			SetShaderValue(BatchedParams, CS->WaterLevelParam, CapturedWaterLevel);
			SetShaderValue(BatchedParams, CS->RockSizeParam, CapturedRockSize);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Abstract:
		{
			TShaderMapRef<FSdfParticleCS_AbstractMain> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, CapturedUAV,
				CapturedParticleCount, DeltaTimeArg, CurrentTime,
				CapturedFlowSpeed, CapturedAttraction, CapturedNoise,
				CapturedMaxDist, CapturedSpawnRadius,
				CurrentSliceIndex, CurrentSliceCount);

			SetShaderValue(BatchedParams, CS->GyroidScaleParam, CapturedGyroidScale);
			SetShaderValue(BatchedParams, CS->GyroidThicknessParam, CapturedGyroidThickness);
			SetShaderValue(BatchedParams, CS->MetaballRadiusParam, CapturedMetaballRadius);
			SetShaderValue(BatchedParams, CS->MorphAmountParam, CapturedMorphAmount);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		case EAliceSdfGpuScene::Fractal:
		{
			TShaderMapRef<FSdfParticleCS_FractalMain> CS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			SetComputePipelineState(RHICmdList, CS.GetComputeShader());
			FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
			CS->SetCommonParameters(BatchedParams, CapturedUAV,
				CapturedParticleCount, DeltaTimeArg, CurrentTime,
				CapturedFlowSpeed, CapturedAttraction, CapturedNoise,
				CapturedMaxDist, CapturedSpawnRadius,
				CurrentSliceIndex, CurrentSliceCount);

			SetShaderValue(BatchedParams, CS->BoxSizeParam, CapturedBoxSize);
			SetShaderValue(BatchedParams, CS->HoleSizeParam, CapturedHoleSize);
			SetShaderValue(BatchedParams, CS->RepeatScaleParam, CapturedRepeatScale);
			SetShaderValue(BatchedParams, CS->TwistAmountParam, CapturedTwistAmount);
			SetShaderValue(BatchedParams, CS->FractalIterationsParam, CapturedFractalIter);
			SetShaderValue(BatchedParams, CS->CamPosParam, CamPos3f);
			SetShaderValue(BatchedParams, CS->CamForwardParam, CamFwd3f);
			SetShaderValue(BatchedParams, CS->CamRightParam, CamRight3f);
			SetShaderValue(BatchedParams, CS->CamUpParam, CamUp3f);
			SetShaderValue(BatchedParams, CS->ZoomLevelParam, CapturedZoom);

			RHICmdList.SetBatchedShaderParameters(CS.GetComputeShader(), BatchedParams);
			RHICmdList.DispatchComputeShader(ThreadGroups, 1, 1);
			break;
		}
		}

		// Transition buffer from UAV (compute write) to SRV (graphics read)
		// Critical for Metal: ensures compute writes are visible to render shader
		FRHITransitionInfo PostBarrier(CapturedBuffer, ERHIAccess::UAVCompute, ERHIAccess::SRVGraphics);
		RHICmdList.Transition(MakeArrayView(&PostBarrier, 1));
	});
}

// =============================================================================
// Rendering — Delegates to ParticleRenderComponent
// =============================================================================

void AAliceSdfParticleActor::RenderParticles()
{
	if (!ParticleRenderComponent || !ParticleSRV)
		return;

	FLinearColor Color = GetColorForScene(CurrentSceneType);

	// Update component's rendering parameters each frame.
	// The FAliceSdfParticleViewExtension captures these in
	// BeginRenderViewFamily() and draws via RDG raster pass.
	// VP matrix and camera position are taken directly from
	// the engine's FSceneView for correct jitter/TAA.
	ParticleRenderComponent->SetParticleResources(
		ParticleSRV,
		ParticleCount,
		ParticleSize,
		Brightness,
		CoreGlow,
		Color);
}

FLinearColor AAliceSdfParticleActor::GetColorForScene(EAliceSdfGpuScene Scene) const
{
	switch (Scene)
	{
	case EAliceSdfGpuScene::Cosmic:  return CosmicColor;
	case EAliceSdfGpuScene::Terrain: return TerrainColor;
	case EAliceSdfGpuScene::Abstract: return AbstractColor;
	case EAliceSdfGpuScene::Fractal: return FractalColor;
	default: return CosmicColor;
	}
}
