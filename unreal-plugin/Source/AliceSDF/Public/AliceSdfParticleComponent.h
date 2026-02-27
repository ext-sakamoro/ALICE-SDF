// ALICE-SDF GPU Particle Rendering Component for UE5
// Author: Moroya Sakamoto
//
// Custom PrimitiveComponent for billboard instanced draw.
// Uses FSceneViewExtensionBase for Metal-compatible rendering via RDG raster pass.

#pragma once

#include "CoreMinimal.h"
#include "Components/PrimitiveComponent.h"
#include "RHIResources.h"
#include "AliceSdfParticleComponent.generated.h"

class FAliceSdfParticleViewExtension;

UCLASS()
class ALICESDF_API UAliceSdfParticleComponent : public UPrimitiveComponent
{
	GENERATED_BODY()

public:
	UAliceSdfParticleComponent();

	// Called by AAliceSdfParticleActor each frame to pass GPU resources + camera
	void SetParticleResources(
		FShaderResourceViewRHIRef InSRV,
		int32 InParticleCount,
		float InParticleSize,
		float InBrightness,
		float InCoreGlow,
		FLinearColor InBaseColor);

	// Set camera data for billboard rendering
	void SetCameraData(const FMatrix& InViewProjection, const FVector& InCameraPosition);

	void ClearParticleResources();

	// View extension lifecycle (call from Actor BeginPlay/EndPlay)
	void CreateViewExtension();
	void DestroyViewExtension();

	// UPrimitiveComponent interface
	virtual FPrimitiveSceneProxy* CreateSceneProxy() override;
	virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;

	// Rendering parameters (updated each frame from Actor)
	FShaderResourceViewRHIRef ParticleSRV;
	int32 ParticleCount = 0;
	float ParticleSize = 0.1f;
	float Brightness = 2.0f;
	float CoreGlow = 0.3f;
	FLinearColor BaseColor = FLinearColor(0.3f, 0.9f, 1.0f, 1.0f);

	// Camera data
	FMatrix ViewProjectionMatrix = FMatrix::Identity;
	FVector CameraPosition = FVector::ZeroVector;

private:
	TSharedPtr<FAliceSdfParticleViewExtension, ESPMode::ThreadSafe> ViewExtension;
};
