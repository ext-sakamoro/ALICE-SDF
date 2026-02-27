// ALICE-SDF Raymarching Actor for UE5
// Author: Moroya Sakamoto
//
// Object-space raymarching on a bounding box mesh.
// Two shader modes:
//   - Fractal: Menger Sponge with infinite zoom and twist
//   - CosmicFractal: 4 demo modes (Normal, Fusion, Destruction, Morph)
//
// The bounding box mesh is rendered with custom VS/PS global shaders.
// MainPS performs per-pixel raymarching and outputs SV_Depth for
// correct depth integration with the rest of the scene.
//
// Usage:
//   1. Place this actor in a level
//   2. Select ShaderMode (Fractal or CosmicFractal)
//   3. Press Play — the SDF shape renders via raymarching

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RenderResource.h"
#include "RHIResources.h"
#include "SceneViewExtension.h"
#include "AliceSdfRaymarchActor.generated.h"

/**
 * Which raymarching shader to use
 */
UENUM(BlueprintType)
enum class EAliceSdfRaymarchShader : uint8
{
	/** Menger Sponge fractal with twist and infinite zoom */
	Fractal        UMETA(DisplayName = "Fractal (Menger Sponge)"),
	/** 4-mode cosmic demo: Normal, Fusion, Destruction, Morph */
	CosmicFractal  UMETA(DisplayName = "Cosmic Fractal (4 Modes)")
};

/**
 * CosmicFractal demo modes (only used when ShaderMode == CosmicFractal)
 */
UENUM(BlueprintType)
enum class EAliceSdfCosmicMode : uint8
{
	/** Sun + orbiting fractal planet + ring + moon */
	Normal      UMETA(DisplayName = "Normal (Solar System)"),
	/** Two spheres with dynamic smooth union */
	Fusion      UMETA(DisplayName = "Fusion (Metaballs)"),
	/** Box with fractal + runtime-subtracted holes */
	Destruction UMETA(DisplayName = "Destruction (Holes)"),
	/** Sphere -> Box -> Torus -> Menger interpolation */
	Morph       UMETA(DisplayName = "Morph (Shape Interpolation)")
};

// Forward declaration
class FAliceSdfRaymarchViewExtension;

/**
 * AAliceSdfRaymarchActor
 *
 * Renders SDF shapes via per-pixel raymarching on a bounding box mesh.
 * Uses custom global shaders (VS/PS) with SV_Depth output for
 * correct scene depth integration.
 *
 * Ported from Unity SDF Universe:
 *   - SdfSurface_Raymarching.shader -> Fractal mode
 *   - CosmicFractal.shader -> CosmicFractal mode (4 demos)
 */
UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfRaymarchActor : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfRaymarchActor();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Shader Selection
	// ========================================================================

	/** Which raymarching shader to use */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Shader")
	EAliceSdfRaymarchShader ShaderMode = EAliceSdfRaymarchShader::Fractal;

	/** CosmicFractal demo mode (ignored when ShaderMode == Fractal) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Shader")
	EAliceSdfCosmicMode CosmicMode = EAliceSdfCosmicMode::Normal;

	// ========================================================================
	// Raymarching Parameters
	// ========================================================================

	/** Maximum raymarching steps per pixel */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Quality",
		meta = (ClampMin = "32", ClampMax = "512"))
	int32 MaxSteps = 128;

	/** Maximum ray distance (SDF units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Quality",
		meta = (ClampMin = "10.0", ClampMax = "500.0"))
	float MaxDist = 100.0f;

	/** Surface hit threshold */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Quality",
		meta = (ClampMin = "0.0001", ClampMax = "0.01"))
	float SurfaceEpsilon = 0.001f;

	/** Distance fog density */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Quality",
		meta = (ClampMin = "0.0", ClampMax = "0.1"))
	float FogDensity = 0.02f;

	// ========================================================================
	// Fractal Parameters (ShaderMode == Fractal)
	// ========================================================================

	/** Outer box half-size (SDF units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Fractal",
		meta = (ClampMin = "1.0", ClampMax = "20.0"))
	float BoxSize = 5.0f;

	/** Cross-section hole radius */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Fractal",
		meta = (ClampMin = "0.01", ClampMax = "2.0"))
	float HoleSize = 0.5f;

	/** Repetition scale (opRepeat period) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Fractal",
		meta = (ClampMin = "0.5", ClampMax = "10.0"))
	float RepeatScale = 3.33f;

	/** Y-axis twist amount (0 = no twist) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Fractal",
		meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float TwistAmount = 0.0f;

	/** FBM procedural texture scale */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Fractal",
		meta = (ClampMin = "0.1", ClampMax = "5.0"))
	float DetailScale = 1.0f;

	// ========================================================================
	// Cosmic Parameters (ShaderMode == CosmicFractal)
	// ========================================================================

	/** Sun radius (Mode 0: Normal) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic",
		meta = (ClampMin = "1.0", ClampMax = "20.0"))
	float SunRadius = 5.0f;

	/** Planet orbit distance (Mode 0: Normal) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic",
		meta = (ClampMin = "5.0", ClampMax = "50.0"))
	float PlanetDistance = 15.0f;

	/** Smooth union factor for fusion mode (Mode 1) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic",
		meta = (ClampMin = "0.1", ClampMax = "5.0"))
	float FusionSmoothness = 1.5f;

	/** Morph parameter 0-3 cycling through 4 shapes (Mode 3) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic",
		meta = (ClampMin = "0.0", ClampMax = "4.0"))
	float MorphT = 0.0f;

	/** Auto-advance morph parameter */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic")
	bool bAutoMorph = true;

	/** Morph speed when auto-advancing */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Cosmic",
		meta = (ClampMin = "0.1", ClampMax = "2.0"))
	float MorphSpeed = 0.3f;

	// ========================================================================
	// Bounding Box
	// ========================================================================

	/** Half-size of the bounding box mesh (Unreal units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Raymarch|Bounds",
		meta = (ClampMin = "100.0", ClampMax = "10000.0"))
	float BoundsSize = 1000.0f;

	// ========================================================================
	// Info (Read-Only)
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Raymarch|Info")
	bool bRenderingActive = false;

private:
	// View extension for custom rendering
	TSharedPtr<FAliceSdfRaymarchViewExtension, ESPMode::ThreadSafe> ViewExtension;

	// Bounding box vertex/index buffers
	FBufferRHIRef VertexBuffer;
	FBufferRHIRef IndexBuffer;

	// Accumulated time for shader animation
	float AccumulatedTime = 0.0f;

	// Internal methods
	void CreateBoundingBoxBuffers();
	void ReleaseBoundingBoxBuffers();

	friend class FAliceSdfRaymarchViewExtension;
};
