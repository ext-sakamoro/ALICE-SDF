// ALICE-SDF Raymarching Actor Implementation
// Author: Moroya Sakamoto
//
// Object-space raymarching using custom global shaders (VS/PS).
// Renders SDF shapes on a bounding box cube mesh with SV_Depth output
// for correct scene depth integration.

#include "AliceSdfRaymarchActor.h"
#include "Engine/World.h"
#include "Engine/Engine.h"
#include "RHICommandList.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RHIStaticStates.h"
#include "PipelineStateCache.h"
#include "SceneView.h"
#include "SceneRendering.h"
#include "Camera/PlayerCameraManager.h"
#include "GameFramework/PlayerController.h"

// =============================================================================
// Global Shader: Fractal Raymarching (Menger Sponge)
// =============================================================================

class FSdfRaymarchFractalVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfRaymarchFractalVS);

public:
	FSdfRaymarchFractalVS() = default;
	FSdfRaymarchFractalVS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		LocalToWorldParam.Bind(Initializer.ParameterMap, TEXT("LocalToWorld"));
		WorldToClipParam.Bind(Initializer.ParameterMap, TEXT("WorldToClip"));
	}

	LAYOUT_FIELD(FShaderParameter, LocalToWorldParam);
	LAYOUT_FIELD(FShaderParameter, WorldToClipParam);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};
IMPLEMENT_GLOBAL_SHADER(FSdfRaymarchFractalVS,
	"/Plugin/AliceSDF/Private/SdfRaymarching_Fractal.usf", "MainVS", SF_Vertex);

class FSdfRaymarchFractalPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfRaymarchFractalPS);

public:
	FSdfRaymarchFractalPS() = default;
	FSdfRaymarchFractalPS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		LocalToWorldParam.Bind(Initializer.ParameterMap, TEXT("LocalToWorld"));
		WorldToClipParam.Bind(Initializer.ParameterMap, TEXT("WorldToClip"));
		CameraWorldPosParam.Bind(Initializer.ParameterMap, TEXT("CameraWorldPos"));
		BoxSizeParam.Bind(Initializer.ParameterMap, TEXT("BoxSize"));
		HoleSizeParam.Bind(Initializer.ParameterMap, TEXT("HoleSize"));
		RepeatScaleParam.Bind(Initializer.ParameterMap, TEXT("RepeatScale"));
		TwistAmountParam.Bind(Initializer.ParameterMap, TEXT("TwistAmount"));
		DetailScaleParam.Bind(Initializer.ParameterMap, TEXT("DetailScale"));
		MaxStepsParam.Bind(Initializer.ParameterMap, TEXT("MaxSteps"));
		MaxDistParam.Bind(Initializer.ParameterMap, TEXT("MaxDist"));
		SurfaceEpsilonParam.Bind(Initializer.ParameterMap, TEXT("SurfaceEpsilon"));
		TimeParam.Bind(Initializer.ParameterMap, TEXT("Time"));
	}

	LAYOUT_FIELD(FShaderParameter, LocalToWorldParam);
	LAYOUT_FIELD(FShaderParameter, WorldToClipParam);
	LAYOUT_FIELD(FShaderParameter, CameraWorldPosParam);
	LAYOUT_FIELD(FShaderParameter, BoxSizeParam);
	LAYOUT_FIELD(FShaderParameter, HoleSizeParam);
	LAYOUT_FIELD(FShaderParameter, RepeatScaleParam);
	LAYOUT_FIELD(FShaderParameter, TwistAmountParam);
	LAYOUT_FIELD(FShaderParameter, DetailScaleParam);
	LAYOUT_FIELD(FShaderParameter, MaxStepsParam);
	LAYOUT_FIELD(FShaderParameter, MaxDistParam);
	LAYOUT_FIELD(FShaderParameter, SurfaceEpsilonParam);
	LAYOUT_FIELD(FShaderParameter, TimeParam);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};
IMPLEMENT_GLOBAL_SHADER(FSdfRaymarchFractalPS,
	"/Plugin/AliceSDF/Private/SdfRaymarching_Fractal.usf", "MainPS", SF_Pixel);

// =============================================================================
// Global Shader: Cosmic Fractal (4 Demo Modes)
// =============================================================================

class FSdfRaymarchCosmicVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfRaymarchCosmicVS);

public:
	FSdfRaymarchCosmicVS() = default;
	FSdfRaymarchCosmicVS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		LocalToWorldParam.Bind(Initializer.ParameterMap, TEXT("LocalToWorld"));
		WorldToClipParam.Bind(Initializer.ParameterMap, TEXT("WorldToClip"));
	}

	LAYOUT_FIELD(FShaderParameter, LocalToWorldParam);
	LAYOUT_FIELD(FShaderParameter, WorldToClipParam);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};
IMPLEMENT_GLOBAL_SHADER(FSdfRaymarchCosmicVS,
	"/Plugin/AliceSDF/Private/SdfRaymarching_CosmicFractal.usf", "MainVS", SF_Vertex);

class FSdfRaymarchCosmicPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfRaymarchCosmicPS);

public:
	FSdfRaymarchCosmicPS() = default;
	FSdfRaymarchCosmicPS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		LocalToWorldParam.Bind(Initializer.ParameterMap, TEXT("LocalToWorld"));
		WorldToClipParam.Bind(Initializer.ParameterMap, TEXT("WorldToClip"));
		CameraWorldPosParam.Bind(Initializer.ParameterMap, TEXT("CameraWorldPos"));
		TimeParam.Bind(Initializer.ParameterMap, TEXT("Time"));
		DemoModeParam.Bind(Initializer.ParameterMap, TEXT("DemoMode"));
		SunRadiusParam.Bind(Initializer.ParameterMap, TEXT("SunRadius"));
		PlanetDistanceParam.Bind(Initializer.ParameterMap, TEXT("PlanetDistance"));
		Sphere1PosParam.Bind(Initializer.ParameterMap, TEXT("Sphere1Pos"));
		Sphere2PosParam.Bind(Initializer.ParameterMap, TEXT("Sphere2Pos"));
		FusionSmoothnessParam.Bind(Initializer.ParameterMap, TEXT("FusionSmoothness"));
		HoleCountParam.Bind(Initializer.ParameterMap, TEXT("HoleCount"));
		MorphTParam.Bind(Initializer.ParameterMap, TEXT("MorphT"));
		MaxStepsParam.Bind(Initializer.ParameterMap, TEXT("MaxSteps"));
		MaxDistParam.Bind(Initializer.ParameterMap, TEXT("MaxDist"));
		SurfaceEpsilonParam.Bind(Initializer.ParameterMap, TEXT("SurfaceEpsilon"));
		FogDensityParam.Bind(Initializer.ParameterMap, TEXT("FogDensity"));
	}

	LAYOUT_FIELD(FShaderParameter, LocalToWorldParam);
	LAYOUT_FIELD(FShaderParameter, WorldToClipParam);
	LAYOUT_FIELD(FShaderParameter, CameraWorldPosParam);
	LAYOUT_FIELD(FShaderParameter, TimeParam);
	LAYOUT_FIELD(FShaderParameter, DemoModeParam);
	LAYOUT_FIELD(FShaderParameter, SunRadiusParam);
	LAYOUT_FIELD(FShaderParameter, PlanetDistanceParam);
	LAYOUT_FIELD(FShaderParameter, Sphere1PosParam);
	LAYOUT_FIELD(FShaderParameter, Sphere2PosParam);
	LAYOUT_FIELD(FShaderParameter, FusionSmoothnessParam);
	LAYOUT_FIELD(FShaderParameter, HoleCountParam);
	LAYOUT_FIELD(FShaderParameter, MorphTParam);
	LAYOUT_FIELD(FShaderParameter, MaxStepsParam);
	LAYOUT_FIELD(FShaderParameter, MaxDistParam);
	LAYOUT_FIELD(FShaderParameter, SurfaceEpsilonParam);
	LAYOUT_FIELD(FShaderParameter, FogDensityParam);

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}
};
IMPLEMENT_GLOBAL_SHADER(FSdfRaymarchCosmicPS,
	"/Plugin/AliceSDF/Private/SdfRaymarching_CosmicFractal.usf", "MainPS", SF_Pixel);

// =============================================================================
// View Extension — Renders the bounding box with custom shaders
// =============================================================================

class FAliceSdfRaymarchViewExtension : public FSceneViewExtensionBase
{
public:
	FAliceSdfRaymarchViewExtension(const FAutoRegister& AutoRegister, AAliceSdfRaymarchActor* InOwner)
		: FSceneViewExtensionBase(AutoRegister)
		, Owner(InOwner)
	{
	}

	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {}
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}

	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override {}

	virtual void PostRenderBasePassDeferred_RenderThread(
		FRDGBuilder& GraphBuilder,
		FSceneView& InView,
		const FRenderTargetBindingSlots& RenderTargets,
		TRDGUniformBufferRef<FSceneTextureUniformParameters> SceneTextures) override
	{
		RenderRaymarch_RenderThread(GraphBuilder, InView, RenderTargets);
	}

	virtual bool IsActiveThisFrame_Internal(const FSceneViewExtensionContext& Context) const override
	{
		return Owner != nullptr && Owner->bRenderingActive;
	}

private:
	AAliceSdfRaymarchActor* Owner = nullptr;

	void RenderRaymarch_RenderThread(
		FRDGBuilder& GraphBuilder,
		FSceneView& InView,
		const FRenderTargetBindingSlots& RenderTargets);
};

void FAliceSdfRaymarchViewExtension::RenderRaymarch_RenderThread(
	FRDGBuilder& GraphBuilder,
	FSceneView& InView,
	const FRenderTargetBindingSlots& RenderTargets)
{
	if (!Owner || !Owner->VertexBuffer || !Owner->IndexBuffer)
		return;

	// Capture parameters from the actor (already on render thread via view extension)
	const EAliceSdfRaymarchShader Mode = Owner->ShaderMode;
	const float CurrentTime = Owner->AccumulatedTime;

	// Actor transform
	const FMatrix44f LocalToWorld44f(Owner->GetActorTransform().ToMatrixWithScale());
	const FMatrix44f ViewProj44f(InView.ViewMatrices.GetViewProjectionMatrix());
	const FVector3f CamPos(InView.ViewMatrices.GetViewOrigin());

	// Bounding box buffers
	FRHIBuffer* VB = Owner->VertexBuffer;
	FRHIBuffer* IB = Owner->IndexBuffer;

	GraphBuilder.AddPass(
		RDG_EVENT_NAME("AliceSDF_Raymarch"),
		ERDGPassFlags::Raster,
		[=, this](FRHICommandListImmediate& RHICmdList)
	{
		if (Mode == EAliceSdfRaymarchShader::Fractal)
		{
			// Fractal shader
			TShaderMapRef<FSdfRaymarchFractalVS> VS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			TShaderMapRef<FSdfRaymarchFractalPS> PS(GetGlobalShaderMap(GMaxRHIFeatureLevel));

			FGraphicsPipelineStateInitializer PSOInit;
			RHICmdList.ApplyCachedRenderTargets(PSOInit);
			PSOInit.BoundShaderState.VertexDeclarationRHI = GTileVertexDeclaration.VertexDeclarationRHI;
			PSOInit.BoundShaderState.VertexShaderRHI = VS.GetVertexShader();
			PSOInit.BoundShaderState.PixelShaderRHI = PS.GetPixelShader();
			PSOInit.PrimitiveType = PT_TriangleList;
			PSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
			PSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI();
			PSOInit.BlendState = TStaticBlendState<>::GetRHI();

			SetGraphicsPipelineState(RHICmdList, PSOInit, 0);

			// VS parameters
			{
				FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
				SetShaderValue(BatchedParams, VS->LocalToWorldParam, LocalToWorld44f);
				SetShaderValue(BatchedParams, VS->WorldToClipParam, ViewProj44f);
				RHICmdList.SetBatchedShaderParameters(VS.GetVertexShader(), BatchedParams);
			}

			// PS parameters
			{
				FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
				SetShaderValue(BatchedParams, PS->LocalToWorldParam, LocalToWorld44f);
				SetShaderValue(BatchedParams, PS->WorldToClipParam, ViewProj44f);
				SetShaderValue(BatchedParams, PS->CameraWorldPosParam, CamPos);
				SetShaderValue(BatchedParams, PS->BoxSizeParam, Owner->BoxSize);
				SetShaderValue(BatchedParams, PS->HoleSizeParam, Owner->HoleSize);
				SetShaderValue(BatchedParams, PS->RepeatScaleParam, Owner->RepeatScale);
				SetShaderValue(BatchedParams, PS->TwistAmountParam, Owner->TwistAmount);
				SetShaderValue(BatchedParams, PS->DetailScaleParam, Owner->DetailScale);
				SetShaderValue(BatchedParams, PS->MaxStepsParam, Owner->MaxSteps);
				SetShaderValue(BatchedParams, PS->MaxDistParam, Owner->MaxDist);
				SetShaderValue(BatchedParams, PS->SurfaceEpsilonParam, Owner->SurfaceEpsilon);
				SetShaderValue(BatchedParams, PS->TimeParam, CurrentTime);
				RHICmdList.SetBatchedShaderParameters(PS.GetPixelShader(), BatchedParams);
			}

			// Draw bounding box (12 triangles = 36 indices)
			RHICmdList.SetStreamSource(0, VB, 0);
			RHICmdList.DrawIndexedPrimitive(IB, 0, 0, 8, 0, 12, 1);
		}
		else
		{
			// CosmicFractal shader
			TShaderMapRef<FSdfRaymarchCosmicVS> VS(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			TShaderMapRef<FSdfRaymarchCosmicPS> PS(GetGlobalShaderMap(GMaxRHIFeatureLevel));

			FGraphicsPipelineStateInitializer PSOInit;
			RHICmdList.ApplyCachedRenderTargets(PSOInit);
			PSOInit.BoundShaderState.VertexDeclarationRHI = GTileVertexDeclaration.VertexDeclarationRHI;
			PSOInit.BoundShaderState.VertexShaderRHI = VS.GetVertexShader();
			PSOInit.BoundShaderState.PixelShaderRHI = PS.GetPixelShader();
			PSOInit.PrimitiveType = PT_TriangleList;
			PSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
			PSOInit.DepthStencilState = TStaticDepthStencilState<true, CF_DepthNearOrEqual>::GetRHI();
			PSOInit.BlendState = TStaticBlendState<>::GetRHI();

			SetGraphicsPipelineState(RHICmdList, PSOInit, 0);

			// VS parameters
			{
				FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
				SetShaderValue(BatchedParams, VS->LocalToWorldParam, LocalToWorld44f);
				SetShaderValue(BatchedParams, VS->WorldToClipParam, ViewProj44f);
				RHICmdList.SetBatchedShaderParameters(VS.GetVertexShader(), BatchedParams);
			}

			// PS parameters
			{
				FRHIBatchedShaderParameters& BatchedParams = RHICmdList.GetScratchShaderParameters();
				SetShaderValue(BatchedParams, PS->LocalToWorldParam, LocalToWorld44f);
				SetShaderValue(BatchedParams, PS->WorldToClipParam, ViewProj44f);
				SetShaderValue(BatchedParams, PS->CameraWorldPosParam, CamPos);
				SetShaderValue(BatchedParams, PS->TimeParam, CurrentTime);

				int32 DemoModeInt = static_cast<int32>(Owner->CosmicMode);
				SetShaderValue(BatchedParams, PS->DemoModeParam, DemoModeInt);

				SetShaderValue(BatchedParams, PS->SunRadiusParam, Owner->SunRadius);
				SetShaderValue(BatchedParams, PS->PlanetDistanceParam, Owner->PlanetDistance);

				// Fusion mode: sphere positions animated
				float s1x = FMath::Sin(CurrentTime * 0.5f) * 4.0f;
				float s2x = -FMath::Sin(CurrentTime * 0.5f) * 4.0f;
				SetShaderValue(BatchedParams, PS->Sphere1PosParam, FVector3f(s1x, 0, 0));
				SetShaderValue(BatchedParams, PS->Sphere2PosParam, FVector3f(s2x, 0, 0));
				SetShaderValue(BatchedParams, PS->FusionSmoothnessParam, Owner->FusionSmoothness);

				// Destruction mode: no runtime holes by default
				SetShaderValue(BatchedParams, PS->HoleCountParam, 0);

				SetShaderValue(BatchedParams, PS->MorphTParam, Owner->MorphT);
				SetShaderValue(BatchedParams, PS->MaxStepsParam, Owner->MaxSteps);
				SetShaderValue(BatchedParams, PS->MaxDistParam, Owner->MaxDist);
				SetShaderValue(BatchedParams, PS->SurfaceEpsilonParam, Owner->SurfaceEpsilon);
				SetShaderValue(BatchedParams, PS->FogDensityParam, Owner->FogDensity);

				RHICmdList.SetBatchedShaderParameters(PS.GetPixelShader(), BatchedParams);
			}

			// Draw bounding box
			RHICmdList.SetStreamSource(0, VB, 0);
			RHICmdList.DrawIndexedPrimitive(IB, 0, 0, 8, 0, 12, 1);
		}
	});
}

// =============================================================================
// Actor Implementation
// =============================================================================

AAliceSdfRaymarchActor::AAliceSdfRaymarchActor()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

void AAliceSdfRaymarchActor::BeginPlay()
{
	Super::BeginPlay();

	CreateBoundingBoxBuffers();

	// Register view extension for custom rendering
	ViewExtension = FSceneViewExtensionBase::NewExtension<FAliceSdfRaymarchViewExtension>(this);

	bRenderingActive = true;

	UE_LOG(LogTemp, Log, TEXT("ALICE-SDF Raymarch: Started (Mode=%d, BoundsSize=%.0f)"),
		static_cast<int32>(ShaderMode), BoundsSize);
}

void AAliceSdfRaymarchActor::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	bRenderingActive = false;

	// Release view extension
	ViewExtension.Reset();

	ReleaseBoundingBoxBuffers();

	Super::EndPlay(EndPlayReason);
}

void AAliceSdfRaymarchActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	AccumulatedTime += DeltaTime;

	// Auto-morph for CosmicFractal Mode 3
	if (ShaderMode == EAliceSdfRaymarchShader::CosmicFractal
		&& CosmicMode == EAliceSdfCosmicMode::Morph
		&& bAutoMorph)
	{
		MorphT += DeltaTime * MorphSpeed;
		if (MorphT >= 4.0f) MorphT -= 4.0f;
	}
}

// =============================================================================
// Bounding Box Mesh (Unit Cube)
// =============================================================================

void AAliceSdfRaymarchActor::CreateBoundingBoxBuffers()
{
	// 8 vertices of a unit cube (scaled by BoundsSize via actor transform)
	const float S = BoundsSize;
	const FVector3f Verts[8] = {
		FVector3f(-S, -S, -S), FVector3f( S, -S, -S),
		FVector3f( S,  S, -S), FVector3f(-S,  S, -S),
		FVector3f(-S, -S,  S), FVector3f( S, -S,  S),
		FVector3f( S,  S,  S), FVector3f(-S,  S,  S),
	};

	// 36 indices for 12 triangles (6 faces)
	const uint16 Indices[36] = {
		// Front (-Z)
		0, 2, 1,  0, 3, 2,
		// Back (+Z)
		4, 5, 6,  4, 6, 7,
		// Left (-X)
		0, 4, 7,  0, 7, 3,
		// Right (+X)
		1, 2, 6,  1, 6, 5,
		// Bottom (-Y)
		0, 1, 5,  0, 5, 4,
		// Top (+Y)
		2, 3, 7,  2, 7, 6,
	};

	ENQUEUE_RENDER_COMMAND(AliceSdfCreateBBoxBuffers)(
		[this, Verts, Indices](FRHICommandListImmediate& RHICmdList)
	{
		// Vertex buffer
		FRHIBufferCreateDesc VBDesc =
			FRHIBufferCreateDesc::CreateVertex(TEXT("AliceSdfRaymarchVB"), sizeof(Verts), sizeof(FVector3f));
		VBDesc.DetermineInitialState();
		VertexBuffer = RHICmdList.CreateBuffer(VBDesc);
		void* VBData = RHICmdList.LockBuffer(VertexBuffer, 0, sizeof(Verts), RLM_WriteOnly);
		FMemory::Memcpy(VBData, Verts, sizeof(Verts));
		RHICmdList.UnlockBuffer(VertexBuffer);

		// Index buffer
		FRHIBufferCreateDesc IBDesc =
			FRHIBufferCreateDesc::CreateIndex(TEXT("AliceSdfRaymarchIB"), sizeof(Indices), sizeof(uint16));
		IBDesc.DetermineInitialState();
		IndexBuffer = RHICmdList.CreateBuffer(IBDesc);
		void* IBData = RHICmdList.LockBuffer(IndexBuffer, 0, sizeof(Indices), RLM_WriteOnly);
		FMemory::Memcpy(IBData, Indices, sizeof(Indices));
		RHICmdList.UnlockBuffer(IndexBuffer);
	});

	FlushRenderingCommands();
}

void AAliceSdfRaymarchActor::ReleaseBoundingBoxBuffers()
{
	VertexBuffer = nullptr;
	IndexBuffer = nullptr;
}
