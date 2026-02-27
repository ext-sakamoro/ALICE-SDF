// ALICE-SDF Nanite Showcase Actor for UE5
// Author: Moroya Sakamoto
//
// Places 12 ALICE-SDF shapes on pedestals in a circular arrangement.
// Each shape is rendered with Nanite at high resolution.
// Total polygon count: 25M+ triangles, all handled by Nanite at 60fps.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfNaniteShowcase.generated.h"

/**
 * AAliceSdfNaniteShowcase
 *
 * Spawns 12 AAliceSdfNaniteActor instances on rotating pedestals,
 * forming a museum-style showcase of ALICE-SDF x Nanite capabilities.
 *
 * Total: 25M+ triangles rendered at 60fps via Nanite.
 *
 * Usage:
 *   1. Place this actor in a level
 *   2. Click "Build All Shapes" in the Details panel
 *   3. Press Play — pedestals rotate slowly
 */
UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfNaniteShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfNaniteShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Radius of the circular arrangement (Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	float CircleRadius = 800.0f;

	/** Resolution for all showcase shapes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase",
		meta = (ClampMin = "64", ClampMax = "512"))
	int32 ShapeResolution = 256;

	/** Pedestal rotation speed (degrees per second) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	float RotationSpeed = 15.0f;

	/** Height of pedestals (Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	float PedestalHeight = 80.0f;

	/** Scale for shapes (SDF units to Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	float ShapeScale = 100.0f;

	/** Optional material for all shapes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	TObjectPtr<UMaterialInterface> ShapeMaterial;

	/** Optional material for pedestals */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	TObjectPtr<UMaterialInterface> PedestalMaterial;

	/** Enable turntable rotation during play */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Showcase")
	bool bRotatePedestals = true;

	// ========================================================================
	// Actions
	// ========================================================================

	/** Build all 12 shapes and place them on pedestals */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Showcase")
	void BuildAllShapes();

	/** Remove all spawned shapes and pedestals */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Showcase")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Showcase|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Showcase|Info")
	float TotalBuildTimeSeconds = 0.0f;

private:
	static constexpr int32 NUM_SHAPES = 12;

	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
