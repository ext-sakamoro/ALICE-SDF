// ALICE-SDF Fractal Showcase — Infinite Detail Demo for UE5
// Author: Moroya Sakamoto
//
// Ported from Unity SDF Universe: FractalDemo.cs
// High-resolution Menger Sponge demonstrating Nanite's ability to handle
// millions of triangles from fractal geometry.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfFractalShowcase.generated.h"

UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfFractalShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfFractalShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Scale for the fractal */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Fractal")
	float FractalScale = 600.0f;

	/** Mesh resolution (higher = more fractal detail, 512 recommended) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Fractal",
		meta = (ClampMin = "128", ClampMax = "1024"))
	int32 ShapeResolution = 512;

	/** Slow rotation speed (degrees/sec) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Fractal")
	float RotationSpeed = 5.0f;

	/** Enable slow turntable rotation */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Fractal")
	bool bRotate = true;

	/** Material */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Fractal")
	TObjectPtr<UMaterialInterface> FractalMaterial;

	// ========================================================================
	// Actions
	// ========================================================================

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Fractal")
	void BuildFractal();

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Fractal")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Fractal|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Fractal|Info")
	float BuildTimeSeconds = 0.0f;

private:
	UPROPERTY()
	TObjectPtr<AAliceSdfNaniteActor> FractalActor;

	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
