// ALICE-SDF Nanite Demo Actor for UE5
// Author: Moroya Sakamoto
//
// Generates high-polygon SDF meshes and renders them with Nanite.
// ALICE-SDF's Marching Cubes outputs millions of triangles that
// Nanite virtualizes for real-time 60fps rendering.
//
// Usage:
//   1. Place this actor in a level
//   2. Select ShapeType in the Details panel
//   3. Click "Rebuild Mesh" (or adjust MeshResolution)
//   4. The Nanite-enabled mesh appears immediately

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfComponent.h"
#include "AliceSdfNaniteActor.generated.h"

/**
 * Shape presets for the Nanite demo
 */
UENUM(BlueprintType)
enum class EAliceSdfNaniteShape : uint8
{
	/** Sphere with internal Gyroid lattice — quarter cutaway reveals structure */
	TPMSSphere       UMETA(DisplayName = "TPMS Gyroid Sphere"),

	/** Interlocking tori with smooth union, twist, and noise */
	OrganicSculpture UMETA(DisplayName = "Organic Sculpture"),

	/** Cluster of octahedrons and diamonds */
	Crystal          UMETA(DisplayName = "Crystal Formation"),

	/** Classical column with fluting, capital, and base */
	ArchColumn       UMETA(DisplayName = "Architectural Column"),

	/** SAO-inspired floating island with internal gyroid tunnels */
	SAOFloat         UMETA(DisplayName = "SAO Floating Island"),

	// --- Ported from VRChat / Unity demos ---

	/** Menger Sponge fractal: Box - Repeat(Cross) + Twist */
	MengerSponge     UMETA(DisplayName = "Menger Sponge Fractal"),

	/** Solar system: Sun + Planet(Ring) + Moon (SmoothUnion) */
	CosmicSystem     UMETA(DisplayName = "Cosmic Solar System"),

	/** Soft-body squishy blobs: 5 spheres with strong SmoothUnion */
	MochiBlobs       UMETA(DisplayName = "Mochi Blobs"),

	/** Fractal sphere + torus ring + onion shell */
	FractalPlanet    UMETA(DisplayName = "Fractal Planet Mix"),

	/** Gyroid + Schwarz P combined minimal surfaces */
	AbstractSurface  UMETA(DisplayName = "Abstract Minimal Surfaces"),

	// --- Nanite + Lumen showcase ---

	/** Domed cathedral with arched windows and columns — Lumen GI demo */
	Cathedral        UMETA(DisplayName = "SDF Cathedral"),

	/** Triple-TPMS ultra-dense coral — Nanite stress test */
	CoralReef        UMETA(DisplayName = "Coral Reef"),

	// --- Scene objects (used by Showcases) ---

	/** Glowing sun with surface turbulence */
	CosmicSun        UMETA(DisplayName = "Cosmic Sun"),

	/** Planet with tilted ring (Saturn-like) */
	CosmicRingedPlanet UMETA(DisplayName = "Cosmic Ringed Planet"),

	/** Small rocky asteroid */
	CosmicAsteroid   UMETA(DisplayName = "Cosmic Asteroid"),

	/** Terrain ground with FBM-like noise hills */
	TerrainGround    UMETA(DisplayName = "Terrain Ground"),

	/** Flat water surface with subtle ripples */
	TerrainWater     UMETA(DisplayName = "Terrain Water"),

	/** Rocky boulder with noise displacement */
	TerrainRock      UMETA(DisplayName = "Terrain Rock"),

	/** Floating island — flattened sphere + FBM noise + cave carving */
	FloatingIsland   UMETA(DisplayName = "Floating Island"),

	/** Bounded gyroid — gyroid clipped to sphere (Abstract center) */
	BoundedGyroid    UMETA(DisplayName = "Bounded Gyroid"),

	/** Smooth metaball — sphere with noise accent */
	Metaball         UMETA(DisplayName = "Metaball"),

	/** Schwarz P surface clipped to sphere (Abstract corner) */
	SchwarzPCorner   UMETA(DisplayName = "Schwarz P Corner"),

	/** Simple box with noise — floating cube for Abstract scene */
	FloatingCube     UMETA(DisplayName = "Floating Cube"),

	/** Torus ring — rotating ring for Abstract scene */
	AbstractRing     UMETA(DisplayName = "Abstract Ring"),
};

/**
 * AAliceSdfNaniteActor
 *
 * Converts ALICE-SDF shapes into Nanite-enabled UStaticMesh at editor time.
 * The pipeline: SDF tree -> Marching Cubes -> OBJ -> FMeshDescription -> Nanite.
 *
 * At Resolution=256, shapes produce ~500K-2M triangles.
 * At Resolution=512, shapes produce ~2M-8M triangles.
 * Nanite handles all of this without manual LOD setup.
 */
UCLASS(ClassGroup=(AliceSDF), meta=(BlueprintSpawnableComponent))
class ALICESDF_API AAliceSdfNaniteActor : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfNaniteActor();

	virtual void OnConstruction(const FTransform& Transform) override;

#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

	// ========================================================================
	// Parameters
	// ========================================================================

	/** Which shape to generate */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Nanite")
	EAliceSdfNaniteShape ShapeType = EAliceSdfNaniteShape::TPMSSphere;

	/** Marching Cubes resolution (128=preview, 256=balanced, 512+=ultra) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Nanite",
		meta = (ClampMin = "32", ClampMax = "1024"))
	int32 MeshResolution = 256;

	/** Bounding box half-size for mesh generation (SDF units) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Nanite",
		meta = (ClampMin = "0.5", ClampMax = "10.0"))
	float Bounds = 2.0f;

	/** Scale factor: SDF units to Unreal units (100 = 1 SDF unit = 1m) */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Nanite")
	float WorldScale = 100.0f;

	/** Optional material override */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Nanite")
	TObjectPtr<UMaterialInterface> OverrideMaterial;

	// ========================================================================
	// Actions
	// ========================================================================

	/** Rebuild the Nanite mesh from current parameters */
	UFUNCTION(BlueprintCallable, CallInEditor, Category = "ALICE SDF Nanite")
	void RebuildMesh();

	// ========================================================================
	// Info (read-only)
	// ========================================================================

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite|Info")
	int32 TriangleCount = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite|Info")
	int32 VertexCount = 0;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite|Info")
	float BuildTimeSeconds = 0.0f;

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite")
	TObjectPtr<UStaticMeshComponent> MeshComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite")
	TObjectPtr<UAliceSdfComponent> MainShape;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Nanite")
	TObjectPtr<UAliceSdfComponent> CsgShape;

private:
	// Shape builders
	void BuildShapeSdf();
	void BuildTPMSSphere();
	void BuildOrganicSculpture();
	void BuildCrystal();
	void BuildArchColumn();
	void BuildSAOFloat();
	void BuildMengerSponge();
	void BuildCosmicSystem();
	void BuildMochiBlobs();
	void BuildFractalPlanet();
	void BuildAbstractSurface();
	void BuildCathedral();
	void BuildCoralReef();
	void BuildCosmicSun();
	void BuildCosmicRingedPlanet();
	void BuildCosmicAsteroid();
	void BuildTerrainGround();
	void BuildTerrainWater();
	void BuildTerrainRock();
	void BuildFloatingIsland();
	void BuildBoundedGyroid();
	void BuildMetaball();
	void BuildSchwarzPCorner();
	void BuildFloatingCube();
	void BuildAbstractRing();

	// Mesh pipeline
	UStaticMesh* CreateNaniteStaticMesh();
	bool ParseObjFile(const FString& Path,
		TArray<FVector3f>& OutPositions,
		TArray<FVector3f>& OutNormals,
		TArray<uint32>& OutPosIndices,
		TArray<uint32>& OutNormIndices);

	UPROPERTY()
	TObjectPtr<UStaticMesh> GeneratedMesh;
};
