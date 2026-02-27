// ALICE-SDF Terrain Showcase — Procedural Landscape Demo for UE5
// Author: Moroya Sakamoto
//
// Ported from Unity SDF Universe: SdfCompute_Terrain
// FBM-style terrain, water surface, scattered rocks, and floating islands.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfTerrainShowcase.generated.h"

UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfTerrainShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfTerrainShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Terrain scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float TerrainScale = 900.0f;

	/** Water scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float WaterScale = 900.0f;

	/** Water height offset from terrain */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float WaterHeight = 90.0f;

	/** Number of scattered rocks (Unity: 5) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain",
		meta = (ClampMin = "0", ClampMax = "12"))
	int32 NumRocks = 5;

	/** Rock scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float RockScale = 120.0f;

	/** Scatter radius for rocks */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float RockScatterRadius = 750.0f;

	/** Mesh resolution */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain",
		meta = (ClampMin = "64", ClampMax = "512"))
	int32 ShapeResolution = 256;

	/** Terrain material */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	TObjectPtr<UMaterialInterface> TerrainMaterial;

	/** Water material */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	TObjectPtr<UMaterialInterface> WaterMaterial;

	/** Rock material */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	TObjectPtr<UMaterialInterface> RockMaterial;

	// --- Floating Islands (Unity: 3 flattened spheres with FBM noise) ---

	/** Number of floating islands (Unity: 3) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain",
		meta = (ClampMin = "0", ClampMax = "6"))
	int32 NumIslands = 3;

	/** Floating island scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float IslandScale = 180.0f;

	/** Orbit radius for floating islands */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float IslandOrbitRadius = 1050.0f;

	/** Height offset for floating islands */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	float IslandHeight = 360.0f;

	/** Island material */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Terrain")
	TObjectPtr<UMaterialInterface> IslandMaterial;

	// ========================================================================
	// Actions
	// ========================================================================

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Terrain")
	void BuildTerrain();

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Terrain")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Terrain|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Terrain|Info")
	float BuildTimeSeconds = 0.0f;

private:
	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
