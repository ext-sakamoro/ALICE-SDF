// ALICE-SDF GPU Particle Actor for UE5
// Author: Moroya Sakamoto
//
// Full GPU particle visualization of SDF surfaces.
// Zero CPU data transfer — Compute Shader + Instanced Indirect Draw.
//
// Matches Unity SDF Universe's GPU particle pipeline:
//   1. Compute Shader: SDF evaluation + physics simulation
//   2. Vertex Shader: Billboard transform (camera-facing quads)
//   3. Fragment Shader: Additive glow rendering
//
// Performance: 10M+ particles at 60 FPS

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "RenderResource.h"
#include "RHIResources.h"
#include "AliceSdfParticleComponent.h"
#include "AliceSdfParticleActor.generated.h"

/**
 * GPU Scene Type — each uses a different Compute Shader
 */
UENUM(BlueprintType)
enum class EAliceSdfGpuScene : uint8
{
	/** Sun + Planet + Ring + Moon + Asteroids */
	Cosmic    UMETA(DisplayName = "Cosmic (Solar System)"),
	/** FBM Noise Terrain + Water + Floating Islands */
	Terrain   UMETA(DisplayName = "Terrain (FBM Landscape)"),
	/** Gyroid + Metaballs + Rotating Torus + Schwarz P */
	Abstract  UMETA(DisplayName = "Abstract (Gyroid + Metaballs)"),
	/** Menger Sponge with Microscope Mode (Infinite Zoom) */
	Fractal   UMETA(DisplayName = "Fractal (Menger Sponge)")
};

/**
 * AAliceSdfParticleActor
 *
 * Full GPU SDF particle system — matches Unity SDF Universe.
 * Compute Shader evaluates SDF + physics, rendering via instanced indirect draw.
 *
 * Usage:
 *   1. Place in level
 *   2. Select SceneType (Cosmic, Terrain, Abstract, Fractal)
 *   3. Press Play — particles flow across SDF surfaces
 */
UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfParticleActor : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfParticleActor();

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void Tick(float DeltaTime) override;

	// ========================================================================
	// Scene Selection
	// ========================================================================

	/** SDF scene to simulate (each has its own Compute Shader) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Scene")
	EAliceSdfGpuScene SceneType = EAliceSdfGpuScene::Cosmic;

	// ========================================================================
	// Particle Count
	// ========================================================================

	/** Number of GPU particles (10K to 10M) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Particles",
		meta = (ClampMin = "10000", ClampMax = "10000000"))
	int32 ParticleCount = 100000;

	// ========================================================================
	// Physics Parameters
	// ========================================================================

	/** Particle flow speed along SDF surfaces (cm/s) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Physics",
		meta = (ClampMin = "10.0", ClampMax = "1000.0"))
	float FlowSpeed = 300.0f;

	/** Attraction force toward SDF surface */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Physics",
		meta = (ClampMin = "10.0", ClampMax = "1000.0"))
	float SurfaceAttraction = 200.0f;

	/** Turbulence noise strength */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Physics",
		meta = (ClampMin = "0.0", ClampMax = "100.0"))
	float NoiseStrength = 10.0f;

	// ========================================================================
	// Rendering
	// ========================================================================

	/** Size of each billboard particle (cm) — DEBUG: 200cm for visibility test */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering",
		meta = (ClampMin = "1.0", ClampMax = "500.0"))
	float ParticleSize = 200.0f;

	/** Brightness multiplier */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering",
		meta = (ClampMin = "0.5", ClampMax = "10.0"))
	float Brightness = 2.0f;

	/** Core glow intensity */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering",
		meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float CoreGlow = 0.3f;

	/** Particle color for Cosmic scene */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering")
	FLinearColor CosmicColor = FLinearColor(0.3f, 0.9f, 1.0f, 1.0f);

	/** Particle color for Terrain scene */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering")
	FLinearColor TerrainColor = FLinearColor(0.4f, 1.0f, 0.5f, 1.0f);

	/** Particle color for Abstract scene */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering")
	FLinearColor AbstractColor = FLinearColor(1.0f, 0.5f, 0.8f, 1.0f);

	/** Particle color for Fractal scene */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Rendering")
	FLinearColor FractalColor = FLinearColor(0.2f, 0.8f, 1.0f, 1.0f);

	// ========================================================================
	// Spawn
	// ========================================================================

	/** Initial spawn radius (cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Spawn")
	float SpawnRadius = 4000.0f;

	/** Maximum distance before particle respawn (cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Spawn")
	float MaxDistance = 10000.0f;

	// ========================================================================
	// Time Slicing
	// ========================================================================

	/** Update divisions (1=full, 3=1/3 per frame, 10=1/10 per frame) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Performance",
		meta = (ClampMin = "1", ClampMax = "10"))
	int32 UpdateDivisions = 1;

	// ========================================================================
	// Cosmic Scene Parameters
	// ========================================================================

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Cosmic")
	float SunRadius = 800.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Cosmic")
	float PlanetRadius = 250.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Cosmic")
	float PlanetDistance = 1800.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Cosmic")
	float Smoothness = 150.0f;

	// ========================================================================
	// Terrain Scene Parameters
	// ========================================================================

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Terrain")
	float TerrainHeight = 1000.0f;

	/** Noise frequency (smaller = larger terrain features) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Terrain")
	float TerrainScale = 0.01f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Terrain")
	float WaterLevel = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Terrain")
	float RockSize = 150.0f;

	// ========================================================================
	// Abstract Scene Parameters
	// ========================================================================

	/** Gyroid frequency (smaller = larger pattern) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Abstract")
	float GyroidScale = 0.005f;

	/** Gyroid surface thickness (cm) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Abstract")
	float GyroidThickness = 30.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Abstract")
	float MetaballRadius = 200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Abstract",
		meta = (ClampMin = "0.0", ClampMax = "1.0"))
	float MorphAmount = 0.5f;

	// ========================================================================
	// Fractal Scene Parameters
	// ========================================================================

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Fractal",
		meta = (ClampMin = "1000.0", ClampMax = "20000.0"))
	float BoxSize = 5000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Fractal",
		meta = (ClampMin = "50.0", ClampMax = "1000.0"))
	float HoleSize = 200.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Fractal",
		meta = (ClampMin = "500.0", ClampMax = "5000.0"))
	float RepeatScale = 1500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Fractal",
		meta = (ClampMin = "0.0", ClampMax = "0.002"))
	float TwistAmount = 0.0002f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Particle|Fractal",
		meta = (ClampMin = "1", ClampMax = "5"))
	int32 FractalIterations = 3;

	// ========================================================================
	// Actions
	// ========================================================================

	/** Initialize GPU resources and spawn particles */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Particle")
	void InitializeParticles();

	/** Switch to a different GPU scene at runtime */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF Particle")
	void SwitchScene(EAliceSdfGpuScene NewScene);

	/** Reinitialize all particles */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF Particle")
	void Reinitialize();

	/** Release all GPU resources */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Particle")
	void ReleaseGPU();

	// ========================================================================
	// Info (Read-Only)
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Particle|Info")
	float CurrentFPS = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Particle|Info")
	float GpuDispatchTimeMs = 0.0f;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Particle|Info")
	bool bGpuInitialized = false;

private:
	// Rendering Component
	UPROPERTY()
	TObjectPtr<UAliceSdfParticleComponent> ParticleRenderComponent;

	// GPU Resources
	FBufferRHIRef ParticleBuffer;
	FShaderResourceViewRHIRef ParticleSRV;
	FUnorderedAccessViewRHIRef ParticleUAV;

	// Time slicing state
	int32 SliceIndex = 0;

	// FPS tracking
	float FpsTimer = 0.0f;
	int32 FrameCounter = 0;

	// Scene tracking
	EAliceSdfGpuScene CurrentSceneType = EAliceSdfGpuScene::Cosmic;

	// Internal methods
	void CreateGPUResources();
	void DispatchComputeShader(float DeltaTime);
	void RenderParticles();
	FLinearColor GetColorForScene(EAliceSdfGpuScene Scene) const;

	static constexpr int32 THREAD_GROUP_SIZE = 256;
	static constexpr int32 PARTICLE_STRIDE = 32; // float3 + float3 + float + float = 8 floats * 4 bytes
};
