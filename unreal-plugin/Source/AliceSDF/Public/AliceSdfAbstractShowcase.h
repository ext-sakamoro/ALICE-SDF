// ALICE-SDF Abstract Showcase — Generative Art Demo for UE5
// Author: Moroya Sakamoto
//
// Ported from Unity SDF Universe: SdfCompute_Abstract
// Center: Bounded Gyroid
// 6 morphing Metaballs orbiting
// 3 rotating Torus Rings
// 4 Schwarz P Corners
// 8 floating rotating Cubes

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfAbstractShowcase.generated.h"

UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfAbstractShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfAbstractShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Center (Bounded Gyroid)
	// ========================================================================

	/** Scale for central bounded gyroid */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CenterScale = 360.0f;

	// ========================================================================
	// Metaballs (Unity: 6 morphing sphere-octahedron blobs)
	// ========================================================================

	/** Orbit radius for metaballs */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float MetaballOrbitRadius = 600.0f;

	/** Metaball scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float MetaballScale = 150.0f;

	/** Metaball orbit speed (degrees/sec) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float MetaballOrbitSpeed = 12.0f;

	// ========================================================================
	// Torus Rings (Unity: 3 rotating torus rings)
	// ========================================================================

	/** Scale for torus rings */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float RingScale = 390.0f;

	/** Ring rotation speed (degrees/sec) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float RingRotationSpeed = 15.0f;

	// ========================================================================
	// Schwarz P Corners (Unity: 4 at diagonal positions)
	// ========================================================================

	/** Distance from center for corner pieces */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CornerDistance = 900.0f;

	/** Scale for corner Schwarz P */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CornerScale = 210.0f;

	// ========================================================================
	// Floating Cubes (Unity: 8 orbiting rotating cubes)
	// ========================================================================

	/** Orbit radius for cubes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CubeOrbitRadius = 1200.0f;

	/** Cube scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CubeScale = 105.0f;

	/** Cube orbit speed (degrees/sec) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CubeOrbitSpeed = 8.0f;

	/** Cube self-rotation speed (degrees/sec) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	float CubeSelfRotation = 20.0f;

	// ========================================================================
	// Common
	// ========================================================================

	/** Mesh resolution */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract",
		meta = (ClampMin = "64", ClampMax = "512"))
	int32 ShapeResolution = 256;

	/** Material for shapes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Abstract")
	TObjectPtr<UMaterialInterface> ShapeMaterial;

	// ========================================================================
	// Actions
	// ========================================================================

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Abstract")
	void BuildAbstractScene();

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Abstract")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Abstract|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Abstract|Info")
	float BuildTimeSeconds = 0.0f;

private:
	float MetaballAngle = 0.0f;
	float CubeAngle = 0.0f;

	UPROPERTY()
	TArray<TObjectPtr<AAliceSdfNaniteActor>> MetaballActors;

	UPROPERTY()
	TArray<TObjectPtr<AAliceSdfNaniteActor>> RingActors;

	UPROPERTY()
	TArray<TObjectPtr<AAliceSdfNaniteActor>> CubeActors;

	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
