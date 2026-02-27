// ALICE-SDF Terrain Showcase Implementation
// Author: Moroya Sakamoto

#include "AliceSdfTerrainShowcase.h"
#include "Engine/World.h"

AAliceSdfTerrainShowcase::AAliceSdfTerrainShowcase()
{
	PrimaryActorTick.bCanEverTick = false;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfTerrainShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildTerrain();
	}
}

// ============================================================================
// Helper
// ============================================================================

static AAliceSdfNaniteActor* SpawnTerrainShape(
	UWorld* World, const FVector& Location,
	EAliceSdfNaniteShape Shape, int32 Resolution, float Scale,
	float Bounds, UMaterialInterface* Material, const FString& Label)
{
	FActorSpawnParameters Params;
	Params.SpawnCollisionHandlingOverride =
		ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	AAliceSdfNaniteActor* Actor = World->SpawnActor<AAliceSdfNaniteActor>(
		AAliceSdfNaniteActor::StaticClass(), Location,
		FRotator::ZeroRotator, Params);

	if (Actor)
	{
		Actor->SetActorLabel(Label);
		Actor->ShapeType = Shape;
		Actor->MeshResolution = Resolution;
		Actor->WorldScale = Scale;
		Actor->Bounds = Bounds;
		if (Material)
		{
			Actor->OverrideMaterial = Material;
		}
		Actor->RebuildMesh();
	}
	return Actor;
}

// ============================================================================
// BuildTerrain
// ============================================================================

void AAliceSdfTerrainShowcase::BuildTerrain()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	const FVector Base = GetActorLocation();

	// --- Ground terrain ---
	AAliceSdfNaniteActor* Ground = SpawnTerrainShape(World, Base,
		EAliceSdfNaniteShape::TerrainGround, ShapeResolution, TerrainScale,
		3.0f, TerrainMaterial, TEXT("Terrain_Ground"));
	if (Ground)
	{
		TotalTriangles += Ground->TriangleCount;
		SpawnedActors.Add(Ground);
	}

	// --- Water surface ---
	AAliceSdfNaniteActor* Water = SpawnTerrainShape(World,
		Base + FVector(0.0f, 0.0f, WaterHeight),
		EAliceSdfNaniteShape::TerrainWater, FMath::Max(128, ShapeResolution / 2),
		WaterScale, 3.0f, WaterMaterial, TEXT("Terrain_Water"));
	if (Water)
	{
		TotalTriangles += Water->TriangleCount;
		SpawnedActors.Add(Water);
	}

	// --- Scattered rocks ---
	for (int32 i = 0; i < NumRocks; i++)
	{
		// Pseudo-random placement using golden angle
		const float GoldenAngle = PI * (3.0f - FMath::Sqrt(5.0f));
		const float Angle = GoldenAngle * i;
		const float Radius = RockScatterRadius * FMath::Sqrt(static_cast<float>(i + 1) / NumRocks);

		const FVector RockPos = Base + FVector(
			FMath::Cos(Angle) * Radius,
			FMath::Sin(Angle) * Radius,
			15.0f + 10.0f * (i % 3));

		const float ThisRockScale = RockScale * (0.5f + 0.5f * ((i * 7 + 3) % 5) / 4.0f);

		AAliceSdfNaniteActor* Rock = SpawnTerrainShape(World, RockPos,
			EAliceSdfNaniteShape::TerrainRock,
			FMath::Max(64, ShapeResolution / 3),
			ThisRockScale, 2.0f, RockMaterial,
			FString::Printf(TEXT("Terrain_Rock_%d"), i));

		if (Rock)
		{
			TotalTriangles += Rock->TriangleCount;
			SpawnedActors.Add(Rock);
		}
	}

	// --- Floating islands (Unity: 3 flattened spheres at orbit positions) ---
	for (int32 i = 0; i < NumIslands; i++)
	{
		const float Angle = (2.0f * PI * i) / FMath::Max(1, NumIslands);
		const FVector IslandPos = Base + FVector(
			FMath::Cos(Angle) * IslandOrbitRadius,
			FMath::Sin(Angle) * IslandOrbitRadius,
			IslandHeight + 20.0f * (i % 2));

		AAliceSdfNaniteActor* Island = SpawnTerrainShape(World, IslandPos,
			EAliceSdfNaniteShape::FloatingIsland,
			FMath::Max(128, ShapeResolution / 2),
			IslandScale, 2.0f,
			IslandMaterial ? IslandMaterial : TerrainMaterial,
			FString::Printf(TEXT("Terrain_Island_%d"), i));

		if (Island)
		{
			TotalTriangles += Island->TriangleCount;
			SpawnedActors.Add(Island);
		}
	}

	BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Terrain Showcase Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Ground + Water + %d rocks + %d floating islands"), NumRocks, NumIslands);
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %d"), TotalTriangles);
	UE_LOG(LogTemp, Log,
		TEXT("  Build time: %.2fs"), BuildTimeSeconds);
}

// ============================================================================
// ClearAll
// ============================================================================

void AAliceSdfTerrainShowcase::ClearAll()
{
	for (AActor* Actor : SpawnedActors)
	{
		if (Actor && IsValid(Actor))
		{
			Actor->Destroy();
		}
	}
	SpawnedActors.Empty();
	TotalTriangles = 0;
}
