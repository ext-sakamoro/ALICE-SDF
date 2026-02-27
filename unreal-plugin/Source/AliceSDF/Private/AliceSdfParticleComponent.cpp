// ALICE-SDF GPU Particle Rendering Component Implementation
// Author: Moroya Sakamoto
//
// Uses FSceneViewExtensionBase + RDG raster pass for Metal-compatible rendering.
// RDG pass guarantees an active render command encoder on Metal/Vulkan/D3D12.
// No vertex buffer needed — quad corners computed from SV_VertexID in shader.

#include "AliceSdfParticleComponent.h"
#include "PrimitiveViewRelevance.h"
#include "PrimitiveSceneProxy.h"
#include "Engine/Engine.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RHICommandList.h"
#include "RHIStaticStates.h"
#include "PipelineStateCache.h"
#include "RenderGraphUtils.h"
#include "GlobalRenderResources.h"
#include "SceneViewExtension.h"
#include "ScreenPass.h"
#include "PostProcess/PostProcessMaterialInputs.h"
#include "ScreenRendering.h"

// =============================================================================
// Render Shaders (Vertex + Pixel)
// =============================================================================

class FSdfParticleVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfParticleVS);

public:
	FSdfParticleVS() = default;
	FSdfParticleVS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		ParticlesBufferParam.Bind(Initializer.ParameterMap, TEXT("ParticlesBuffer"));
		ViewProjectionMatrixParam.Bind(Initializer.ParameterMap, TEXT("ViewProjectionMatrix"));
		CameraPositionParam.Bind(Initializer.ParameterMap, TEXT("CameraPosition"));
		ParticleSizeParam.Bind(Initializer.ParameterMap, TEXT("ParticleSize"));
		BrightnessParam.Bind(Initializer.ParameterMap, TEXT("Brightness"));
		BaseColorParam.Bind(Initializer.ParameterMap, TEXT("BaseColor"));
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	LAYOUT_FIELD(FShaderResourceParameter, ParticlesBufferParam);
	LAYOUT_FIELD(FShaderParameter, ViewProjectionMatrixParam);
	LAYOUT_FIELD(FShaderParameter, CameraPositionParam);
	LAYOUT_FIELD(FShaderParameter, ParticleSizeParam);
	LAYOUT_FIELD(FShaderParameter, BrightnessParam);
	LAYOUT_FIELD(FShaderParameter, BaseColorParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticleVS,
	"/Plugin/AliceSDF/Private/SdfParticleRender.usf", "MainVS", SF_Vertex);

class FSdfParticlePS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FSdfParticlePS);

public:
	FSdfParticlePS() = default;
	FSdfParticlePS(const ShaderMetaType::CompiledShaderInitializerType& Initializer)
		: FGlobalShader(Initializer)
	{
		CoreGlowParam.Bind(Initializer.ParameterMap, TEXT("CoreGlow"));
	}

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
	}

	LAYOUT_FIELD(FShaderParameter, CoreGlowParam);
};
IMPLEMENT_GLOBAL_SHADER(FSdfParticlePS,
	"/Plugin/AliceSDF/Private/SdfParticleRender.usf", "MainPS", SF_Pixel);

// =============================================================================
// RDG Pass Parameters — defines render target bindings for the raster pass
// =============================================================================

BEGIN_SHADER_PARAMETER_STRUCT(FAliceSdfParticlePassParameters, )
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

// =============================================================================
// Render Data — thread-safe snapshot for render thread consumption
// =============================================================================

struct FParticleRenderData
{
	FShaderResourceViewRHIRef SRV;
	int32 ParticleCount = 0;
	float ParticleSize = 0.0f;
	float Brightness = 0.0f;
	float CoreGlow = 0.0f;
	FVector4f BaseColor = FVector4f(0, 0, 0, 1);
	bool bReady = false;
};

// =============================================================================
// Scene View Extension — hooks into UE5 render pipeline
// =============================================================================
// Renders particles via SubscribeToPostProcessingPass (Tonemap pass).
// RDG raster pass guarantees Metal/Vulkan render command encoder is active.
// Tonemap pass is always active (PassEnabled=1) unlike MotionBlur which
// may be skipped by the engine when motion blur is disabled.
// Data flow:
//   Game thread: BeginRenderViewFamily() → capture component state
//   Render thread: SubscribeToPostProcessingPass(Tonemap) → draw particles
// =============================================================================

class FAliceSdfParticleViewExtension : public FSceneViewExtensionBase
{
public:
	FAliceSdfParticleViewExtension(
		const FAutoRegister& AutoRegister,
		UAliceSdfParticleComponent* InOwner)
		: FSceneViewExtensionBase(AutoRegister)
		, Owner(InOwner)
	{
	}

	// -- Game thread callbacks --

	virtual void SetupViewFamily(FSceneViewFamily& InViewFamily) override {}
	virtual void SetupView(FSceneViewFamily& InViewFamily, FSceneView& InView) override {}

	virtual void BeginRenderViewFamily(FSceneViewFamily& InViewFamily) override
	{
		// One-time diagnostic log
		if (!bLoggedBeginRender)
		{
			bLoggedBeginRender = true;
			FVector ActorLoc = FVector::ZeroVector;
			if (Owner.IsValid() && Owner->GetOwner())
			{
				ActorLoc = Owner->GetOwner()->GetActorLocation();
			}
			UE_LOG(LogTemp, Log, TEXT("ALICE-SDF ViewExt: BeginRenderViewFamily Owner=%d SRV=%d Count=%d ActorPos=(%.1f, %.1f, %.1f)"),
				Owner.IsValid() ? 1 : 0,
				(Owner.IsValid() && Owner->ParticleSRV) ? 1 : 0,
				Owner.IsValid() ? Owner->ParticleCount : -1,
				ActorLoc.X, ActorLoc.Y, ActorLoc.Z);
		}

		// Game thread — capture latest render data from component
		if (Owner.IsValid() && Owner->ParticleSRV && Owner->ParticleCount > 0)
		{
			CachedData.SRV = Owner->ParticleSRV;
			CachedData.ParticleCount = Owner->ParticleCount;
			CachedData.ParticleSize = Owner->ParticleSize;
			CachedData.Brightness = Owner->Brightness;
			CachedData.CoreGlow = Owner->CoreGlow;
			CachedData.BaseColor = FVector4f(
				Owner->BaseColor.R, Owner->BaseColor.G,
				Owner->BaseColor.B, Owner->BaseColor.A);
			CachedData.bReady = true;
		}
		else
		{
			CachedData.bReady = false;
		}
	}

	// -- Post-process subscription (render thread) --

	virtual void SubscribeToPostProcessingPass(
		EPostProcessingPass Pass,
		const FSceneView& InView,
		FPostProcessingPassDelegateArray& InOutPassCallbacks,
		bool bIsPassEnabled) override
	{
		// Hook into Tonemap pass — always active (PassEnabled=1).
		// MotionBlur had PassEnabled=0 when engine skips motion blur.
		if (Pass == EPostProcessingPass::Tonemap)
		{
			// One-time diagnostic log
			if (!bLoggedSubscribe)
			{
				bLoggedSubscribe = true;
				UE_LOG(LogTemp, Log, TEXT("ALICE-SDF ViewExt: SubscribeToPostProcessingPass REGISTERED Tonemap callback (PassEnabled=%d)"),
					bIsPassEnabled ? 1 : 0);
			}

			InOutPassCallbacks.Add(
				FPostProcessingPassDelegate::CreateRaw(
					this, &FAliceSdfParticleViewExtension::RenderParticles_RenderThread));
		}
	}

	FScreenPassTexture RenderParticles_RenderThread(
		FRDGBuilder& GraphBuilder,
		const FSceneView& InView,
		const FPostProcessMaterialInputs& Inputs)
	{
		FScreenPassTexture SceneColor = Inputs.ReturnUntouchedSceneColorForPostProcessing(GraphBuilder);

		// One-time diagnostic log
		if (!bLoggedRender)
		{
			bLoggedRender = true;
			UE_LOG(LogTemp, Log, TEXT("ALICE-SDF ViewExt: RenderParticles_RenderThread bReady=%d Count=%d SceneColor=%d"),
				CachedData.bReady ? 1 : 0,
				CachedData.ParticleCount,
				SceneColor.IsValid() ? 1 : 0);
		}

		if (!CachedData.bReady || CachedData.ParticleCount <= 0 || !SceneColor.IsValid())
			return SceneColor;

		// Copy data for lambda capture (render thread safety)
		FParticleRenderData LocalData = CachedData;

		// Use engine's VP matrix and camera position for correct jitter/TAA
		FMatrix44f VP = FMatrix44f(InView.ViewMatrices.GetViewProjectionMatrix());
		FVector3f CamPos = FVector3f(InView.ViewMatrices.GetViewOrigin());
		FIntRect ViewRect = SceneColor.ViewRect;

		// Setup RDG raster pass targeting SceneColor
		auto* PassParameters = GraphBuilder.AllocParameters<FAliceSdfParticlePassParameters>();
		PassParameters->RenderTargets[0] = FRenderTargetBinding(
			SceneColor.Texture, ERenderTargetLoadAction::ELoad);

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("AliceSdfParticles"),
			PassParameters,
			ERDGPassFlags::Raster | ERDGPassFlags::NeverCull,
			[LocalData, VP, CamPos, ViewRect](FRHICommandList& RHICmdList)
		{
			// Get shaders
			TShaderMapRef<FSdfParticleVS> VertexShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
			TShaderMapRef<FSdfParticlePS> PixelShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));

			// One-time shader diagnostic (static to survive across frames)
			static bool bLoggedShaderDiag = false;
			if (!bLoggedShaderDiag)
			{
				bLoggedShaderDiag = true;
				UE_LOG(LogTemp, Log, TEXT("ALICE-SDF RDG Lambda: VS=%d PS=%d SRV=%d Count=%d ViewRect=[%d,%d]-[%d,%d] CamPos=(%.1f,%.1f,%.1f) SRV_Bound=%d VP_Bound=%d CamPos_Bound=%d Size_Bound=%d"),
					VertexShader.IsValid() ? 1 : 0,
					PixelShader.IsValid() ? 1 : 0,
					LocalData.SRV.IsValid() ? 1 : 0,
					LocalData.ParticleCount,
					ViewRect.Min.X, ViewRect.Min.Y,
					ViewRect.Max.X, ViewRect.Max.Y,
					CamPos.X, CamPos.Y, CamPos.Z,
					VertexShader->ParticlesBufferParam.IsBound() ? 1 : 0,
					VertexShader->ViewProjectionMatrixParam.IsBound() ? 1 : 0,
					VertexShader->CameraPositionParam.IsBound() ? 1 : 0,
					VertexShader->ParticleSizeParam.IsBound() ? 1 : 0);
			}

			if (!VertexShader.IsValid() || !PixelShader.IsValid())
			{
				return;
			}

			// Set viewport
			RHICmdList.SetViewport(
				ViewRect.Min.X, ViewRect.Min.Y, 0.0f,
				ViewRect.Max.X, ViewRect.Max.Y, 1.0f);

			// Setup graphics pipeline state
			FGraphicsPipelineStateInitializer GraphicsPSOInit;
			RHICmdList.ApplyCachedRenderTargets(GraphicsPSOInit);

			GraphicsPSOInit.RasterizerState = TStaticRasterizerState<FM_Solid, CM_None>::GetRHI();
			GraphicsPSOInit.DepthStencilState = TStaticDepthStencilState<false, CF_Always>::GetRHI();
			// Additive blending: SrcAlpha + One (matches Unity's Blend SrcAlpha One)
			GraphicsPSOInit.BlendState = TStaticBlendState<
				CW_RGBA, BO_Add, BF_SourceAlpha, BF_One,
				BO_Add, BF_Zero, BF_One>::GetRHI();

			GraphicsPSOInit.PrimitiveType = PT_TriangleList;

			// GTileVertexDeclaration + GScreenRectangleVertexBuffer for Metal compatibility.
			// Metal requires a vertex buffer bound to stream 0 when the vertex declaration
			// has attributes, even if the VS only uses SV_VertexID. Without a bound buffer,
			// Metal silently drops the draw call.
			GraphicsPSOInit.BoundShaderState.VertexDeclarationRHI =
				GTileVertexDeclaration.VertexDeclarationRHI;
			GraphicsPSOInit.BoundShaderState.VertexShaderRHI =
				VertexShader.GetVertexShader();
			GraphicsPSOInit.BoundShaderState.PixelShaderRHI =
				PixelShader.GetPixelShader();

			SetGraphicsPipelineState(RHICmdList, GraphicsPSOInit, 0);

			// Set VS parameters via batched API (UE5 5.7)
			FRHIVertexShader* VSRHI = VertexShader.GetVertexShader();
			{
				FRHIBatchedShaderParameters& VSParams = RHICmdList.GetScratchShaderParameters();

				if (VertexShader->ParticlesBufferParam.IsBound())
				{
					VSParams.SetShaderResourceViewParameter(
						VertexShader->ParticlesBufferParam.GetBaseIndex(),
						LocalData.SRV);
				}

				SetShaderValue(VSParams, VertexShader->ViewProjectionMatrixParam, VP);
				SetShaderValue(VSParams, VertexShader->CameraPositionParam, CamPos);
				SetShaderValue(VSParams, VertexShader->ParticleSizeParam, LocalData.ParticleSize);
				SetShaderValue(VSParams, VertexShader->BrightnessParam, LocalData.Brightness);
				SetShaderValue(VSParams, VertexShader->BaseColorParam, LocalData.BaseColor);

				RHICmdList.SetBatchedShaderParameters(VSRHI, VSParams);
			}

			// Set PS parameters via batched API
			FRHIPixelShader* PSRHI = PixelShader.GetPixelShader();
			{
				FRHIBatchedShaderParameters& PSParams = RHICmdList.GetScratchShaderParameters();
				SetShaderValue(PSParams, PixelShader->CoreGlowParam, LocalData.CoreGlow);
				RHICmdList.SetBatchedShaderParameters(PSRHI, PSParams);
			}

			// Bind dummy vertex buffer to stream 0 — Metal requires this even though
			// the VS only uses SV_VertexID. GScreenRectangleVertexBuffer is a standard
			// UE5 global resource that's always valid.
			RHICmdList.SetStreamSource(0, GScreenRectangleVertexBuffer.VertexBufferRHI, 0);

			RHICmdList.DrawPrimitive(
				0,                           // BaseVertexIndex
				2,                           // NumPrimitives (2 tri = 1 quad)
				LocalData.ParticleCount      // NumInstances (1 per particle)
			);

			// One-time draw confirmation
			static bool bLoggedDraw = false;
			if (!bLoggedDraw)
			{
				bLoggedDraw = true;
				UE_LOG(LogTemp, Log, TEXT("ALICE-SDF RDG Lambda: DrawPrimitive EXECUTED NumPrimitives=2 NumInstances=%d (instanced)"),
					LocalData.ParticleCount);
			}
		});

		return SceneColor;
	}

protected:
	virtual bool IsActiveThisFrame_Internal(
		const FSceneViewExtensionContext& Context) const override
	{
		return Owner.IsValid() && Owner->ParticleSRV && Owner->ParticleCount > 0;
	}

private:
	TWeakObjectPtr<UAliceSdfParticleComponent> Owner;
	FParticleRenderData CachedData;

	// One-time diagnostic flags (log once, not every frame)
	bool bLoggedBeginRender = false;
	bool bLoggedSubscribe = false;
	bool bLoggedRender = false;
};

// =============================================================================
// Scene Proxy (provides bounds + view relevance for UE5 culling)
// =============================================================================

class FAliceSdfParticleSceneProxy : public FPrimitiveSceneProxy
{
public:
	FAliceSdfParticleSceneProxy(const UAliceSdfParticleComponent* InComponent)
		: FPrimitiveSceneProxy(InComponent)
	{
		bWillEverBeLit = false;
		bVerifyUsedMaterials = false;
	}

	virtual SIZE_T GetTypeHash() const override
	{
		static size_t UniquePointer;
		return reinterpret_cast<size_t>(&UniquePointer);
	}

	virtual uint32 GetMemoryFootprint() const override
	{
		return sizeof(*this) + GetAllocatedSize();
	}

	virtual FPrimitiveViewRelevance GetViewRelevance(const FSceneView* View) const override
	{
		FPrimitiveViewRelevance Result;
		Result.bDrawRelevance = IsShown(View);
		Result.bShadowRelevance = false;
		Result.bDynamicRelevance = true;
		Result.bStaticRelevance = false;
		Result.bRenderInMainPass = true;
		Result.bUsesLightingChannels = false;
		Result.bRenderCustomDepth = false;
		return Result;
	}

	virtual void GetDynamicMeshElements(
		const TArray<const FSceneView*>& Views,
		const FSceneViewFamily& ViewFamily,
		uint32 VisibilityMap,
		FMeshElementCollector& Collector) const override
	{
		// Rendering is done via FAliceSdfParticleViewExtension (RDG raster pass).
		// SceneProxy provides bounds + view relevance only.
	}
};

// =============================================================================
// Component Implementation
// =============================================================================

UAliceSdfParticleComponent::UAliceSdfParticleComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
	SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
	SetGenerateOverlapEvents(false);
	CastShadow = false;
	bUseAsOccluder = false;
}

void UAliceSdfParticleComponent::SetParticleResources(
	FShaderResourceViewRHIRef InSRV,
	int32 InParticleCount,
	float InParticleSize,
	float InBrightness,
	float InCoreGlow,
	FLinearColor InBaseColor)
{
	ParticleSRV = InSRV;
	ParticleCount = InParticleCount;
	ParticleSize = InParticleSize;
	Brightness = InBrightness;
	CoreGlow = InCoreGlow;
	BaseColor = InBaseColor;
	MarkRenderStateDirty();
}

void UAliceSdfParticleComponent::SetCameraData(
	const FMatrix& InViewProjection,
	const FVector& InCameraPosition)
{
	ViewProjectionMatrix = InViewProjection;
	CameraPosition = InCameraPosition;
}

void UAliceSdfParticleComponent::ClearParticleResources()
{
	ParticleSRV = nullptr;
	ParticleCount = 0;
	MarkRenderStateDirty();
}

void UAliceSdfParticleComponent::CreateViewExtension()
{
	if (!ViewExtension.IsValid())
	{
		ViewExtension = FSceneViewExtensions::NewExtension<FAliceSdfParticleViewExtension>(this);
	}
}

void UAliceSdfParticleComponent::DestroyViewExtension()
{
	ViewExtension.Reset();
}

FPrimitiveSceneProxy* UAliceSdfParticleComponent::CreateSceneProxy()
{
	if (!ParticleSRV || ParticleCount <= 0)
		return nullptr;

	return new FAliceSdfParticleSceneProxy(this);
}

FBoxSphereBounds UAliceSdfParticleComponent::CalcBounds(const FTransform& LocalToWorld) const
{
	// Large bounds to ensure particles are always visible (UE5 cm scale)
	float Extent = 15000.0f;
	return FBoxSphereBounds(FVector::ZeroVector, FVector(Extent), Extent).TransformBy(LocalToWorld);
}
