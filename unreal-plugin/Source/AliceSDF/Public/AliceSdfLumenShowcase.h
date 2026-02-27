// ALICE-SDF Lumen Showcase Actor for UE5
// Author: Moroya Sakamoto
//
// Creates a gallery room with SDF shapes and colored lights to demonstrate
// Lumen global illumination: color bleeding, indirect lighting, and soft shadows.
//
// Usage:
//   1. Place this actor in a level
//   2. Click "Build Gallery" in the Details panel
//   3. Observe Lumen GI color bleeding in editor or Play mode

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfLumenShowcase.generated.h"

/**
 * AAliceSdfLumenShowcase
 *
 * Builds a gallery room from engine primitives, places ALICE-SDF Nanite shapes
 * inside, and spawns colored point lights. Lumen's real-time GI produces
 * color bleeding, indirect bounces, and soft shadows on the SDF geometry.
 *
 * The room has 3 walls + floor + ceiling with a skylight opening.
 * 4 SDF shapes (Cathedral, CoralReef, TPMSSphere, Crystal) sit on pedestals.
 * Colored lights (warm, cool, accent) illuminate the scene.
 */
UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfLumenShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfLumenShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Room width (X axis, Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	float RoomWidth = 3600.0f;

	/** Room depth (Y axis, Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	float RoomDepth = 2400.0f;

	/** Room height (Z axis, Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	float RoomHeight = 1500.0f;

	/** Resolution for gallery shapes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen",
		meta = (ClampMin = "64", ClampMax = "512"))
	int32 ShapeResolution = 256;

	/** Scale for shapes (SDF units to Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	float ShapeScale = 300.0f;

	/** Light intensity for colored point lights */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen",
		meta = (ClampMin = "1000", ClampMax = "300000"))
	float LightIntensity = 45000.0f;

	/** Light attenuation radius */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	float LightRadius = 2400.0f;

	/** Optional material for floor */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	TObjectPtr<UMaterialInterface> FloorMaterial;

	/** Optional material for walls */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	TObjectPtr<UMaterialInterface> WallMaterial;

	/** Optional material for SDF shapes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen")
	TObjectPtr<UMaterialInterface> ShapeMaterial;

	// ========================================================================
	// Lumen Settings
	// ========================================================================

	/** Spawn a PostProcessVolume with Lumen-optimal settings */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen|GI")
	bool bSpawnPostProcessVolume = true;

	/** Spawn a SkyLight for environment indirect lighting */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen|GI")
	bool bSpawnSkyLight = true;

	/** SkyLight intensity (lower = more visible color bleed from point lights) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen|GI",
		meta = (ClampMin = "0.1", ClampMax = "5.0"))
	float SkyLightIntensity = 0.5f;

	/** Lumen Final Gather Quality (1=default, 2=high, 4=ultra) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen|GI",
		meta = (ClampMin = "0.5", ClampMax = "4.0"))
	float LumenFinalGatherQuality = 2.0f;

	/** Lumen Scene Detail (higher = more detail for small objects) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Lumen|GI",
		meta = (ClampMin = "0.5", ClampMax = "4.0"))
	float LumenSceneDetail = 2.0f;

	// ========================================================================
	// Actions
	// ========================================================================

	/** Build the gallery room with shapes and lights */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Lumen")
	void BuildGallery();

	/** Remove all gallery elements */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Lumen")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Lumen|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Lumen|Info")
	float BuildTimeSeconds = 0.0f;

private:
	static constexpr int32 NUM_GALLERY_SHAPES = 4;

	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
