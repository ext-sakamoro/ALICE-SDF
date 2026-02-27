// ALICE-SDF Cosmic Showcase — Solar System Demo for UE5
// Author: Moroya Sakamoto
//
// Ported from Unity SDF Universe: CosmicDemo.cs
// Sun, 2 planets, ring, moon, asteroid belt — all orbiting with Tick animation.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfNaniteActor.h"
#include "AliceSdfCosmicShowcase.generated.h"

UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfCosmicShowcase : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfCosmicShowcase();

	virtual void OnConstruction(const FTransform& Transform) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Sun scale (SDF units to Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float SunScale = 450.0f;

	/** Planet scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float PlanetScale = 240.0f;

	/** Orbit radius for planet 1 (Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float Planet1Orbit = 1500.0f;

	/** Orbit radius for planet 2 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float Planet2Orbit = 2400.0f;

	/** Moon orbit radius (relative to planet 1) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float MoonOrbit = 450.0f;

	/** Moon scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float MoonScale = 120.0f;

	/** Asteroid belt radius */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float AsteroidBeltRadius = 1050.0f;

	/** Number of asteroids in belt */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic",
		meta = (ClampMin = "4", ClampMax = "20"))
	int32 NumAsteroids = 8;

	/** Asteroid scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float AsteroidScale = 75.0f;

	/** Planet 1 orbital speed (degrees per second) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float Planet1Speed = 12.0f;

	/** Planet 2 orbital speed */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float Planet2Speed = 7.0f;

	/** Moon orbital speed */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float MoonSpeed = 45.0f;

	/** Sun self-rotation speed */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	float SunRotationSpeed = 5.0f;

	/** Mesh resolution */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic",
		meta = (ClampMin = "64", ClampMax = "512"))
	int32 ShapeResolution = 192;

	/** Optional material for sun */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	TObjectPtr<UMaterialInterface> SunMaterial;

	/** Optional material for planets */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Cosmic")
	TObjectPtr<UMaterialInterface> PlanetMaterial;

	// ========================================================================
	// Actions
	// ========================================================================

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Cosmic")
	void BuildSolarSystem();

	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Cosmic")
	void ClearAll();

	// ========================================================================
	// Info
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Cosmic|Info")
	int32 TotalTriangles = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Cosmic|Info")
	float BuildTimeSeconds = 0.0f;

private:
	// Orbital state
	float Planet1Angle = 0.0f;
	float Planet2Angle = 120.0f;
	float MoonAngle = 0.0f;

	// References for Tick animation
	UPROPERTY()
	TObjectPtr<AAliceSdfNaniteActor> SunActor;

	UPROPERTY()
	TObjectPtr<AAliceSdfNaniteActor> Planet1Actor;

	UPROPERTY()
	TObjectPtr<AAliceSdfNaniteActor> Planet2Actor;

	UPROPERTY()
	TObjectPtr<AAliceSdfNaniteActor> MoonActor;

	UPROPERTY()
	TArray<TObjectPtr<AActor>> SpawnedActors;
};
