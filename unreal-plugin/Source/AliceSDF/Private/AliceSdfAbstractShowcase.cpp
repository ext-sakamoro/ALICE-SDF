// ALICE-SDF Abstract Showcase Implementation
// Author: Moroya Sakamoto
//
// Ported from Unity SDF Universe: SdfCompute_Abstract
// Matches Unity's exact composition:
//   Center: Bounded Gyroid (gyroid clipped to sphere)
//   6 Metaballs (sphere+octahedron morph, orbiting)
//   3 Torus Rings (different sizes, rotating)
//   4 Schwarz P Corners (at diagonal positions)
//   8 Floating Cubes (orbiting, self-rotating)

#include "AliceSdfAbstractShowcase.h"
#include "Engine/World.h"

AAliceSdfAbstractShowcase::AAliceSdfAbstractShowcase()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfAbstractShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildAbstractScene();
	}
}

// ============================================================================
// Helper
// ============================================================================

static AAliceSdfNaniteActor* SpawnAbstractShape(
	UWorld* World, const FVector& Location,
	EAliceSdfNaniteShape Shape, int32 Resolution, float Scale,
	UMaterialInterface* Material, const FString& Label)
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
		if (Material)
		{
			Actor->OverrideMaterial = Material;
		}
		Actor->RebuildMesh();
	}
	return Actor;
}

// ============================================================================
// BuildAbstractScene — Unity SdfCompute_Abstract composition
// ============================================================================

void AAliceSdfAbstractShowcase::BuildAbstractScene()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	const FVector Base = GetActorLocation();

	// --- 1. Center: Bounded Gyroid (Unity: max(gyroid, sphere)) ---
	AAliceSdfNaniteActor* Center = SpawnAbstractShape(World, Base,
		EAliceSdfNaniteShape::BoundedGyroid, ShapeResolution, CenterScale,
		ShapeMaterial, TEXT("Abstract_Center_Gyroid"));
	if (Center)
	{
		TotalTriangles += Center->TriangleCount;
		SpawnedActors.Add(Center);
	}

	// --- 2. Six Metaballs (Unity: 6 morphing sphere-octahedron) ---
	MetaballActors.Empty();
	for (int32 i = 0; i < 6; i++)
	{
		const float Angle = (2.0f * PI * i) / 6.0f
			+ FMath::DegreesToRadians(MetaballAngle);
		const float Height = 30.0f * FMath::Sin(Angle * 2.0f);

		const FVector MetaPos = Base + FVector(
			FMath::Cos(Angle) * MetaballOrbitRadius,
			FMath::Sin(Angle) * MetaballOrbitRadius,
			Height);

		AAliceSdfNaniteActor* Meta = SpawnAbstractShape(World, MetaPos,
			EAliceSdfNaniteShape::Metaball,
			FMath::Max(64, ShapeResolution / 2),
			MetaballScale,
			ShapeMaterial,
			FString::Printf(TEXT("Abstract_Metaball_%d"), i));

		if (Meta)
		{
			TotalTriangles += Meta->TriangleCount;
			SpawnedActors.Add(Meta);
			MetaballActors.Add(Meta);
		}
	}

	// --- 3. Three Torus Rings (Unity: sizes 10+j*2, thickness 0.3+j*0.1) ---
	RingActors.Empty();
	static const FRotator RingTilts[] = {
		FRotator(30.0f, 0.0f, 0.0f),
		FRotator(0.0f, 0.0f, 45.0f),
		FRotator(60.0f, 30.0f, 0.0f),
	};

	for (int32 j = 0; j < 3; j++)
	{
		AAliceSdfNaniteActor* Ring = SpawnAbstractShape(World, Base,
			EAliceSdfNaniteShape::AbstractRing,
			FMath::Max(128, ShapeResolution / 2),
			RingScale + j * 25.0f,
			ShapeMaterial,
			FString::Printf(TEXT("Abstract_Ring_%d"), j));

		if (Ring)
		{
			Ring->SetActorRotation(RingTilts[j]);
			TotalTriangles += Ring->TriangleCount;
			SpawnedActors.Add(Ring);
			RingActors.Add(Ring);
		}
	}

	// --- 4. Four Schwarz P Corners (Unity: at (±15, ±15, ±15)) ---
	static const FVector CornerDirs[] = {
		FVector( 1.0f,  1.0f,  1.0f),
		FVector(-1.0f,  1.0f, -1.0f),
		FVector( 1.0f, -1.0f, -1.0f),
		FVector(-1.0f, -1.0f,  1.0f),
	};

	for (int32 k = 0; k < 4; k++)
	{
		const FVector CornerPos = Base + CornerDirs[k] * CornerDistance;

		AAliceSdfNaniteActor* Corner = SpawnAbstractShape(World, CornerPos,
			EAliceSdfNaniteShape::SchwarzPCorner,
			FMath::Max(128, ShapeResolution / 2),
			CornerScale,
			ShapeMaterial,
			FString::Printf(TEXT("Abstract_SchwarzP_%d"), k));

		if (Corner)
		{
			TotalTriangles += Corner->TriangleCount;
			SpawnedActors.Add(Corner);
		}
	}

	// --- 5. Eight Floating Cubes (Unity: orbit r=18, varying sizes) ---
	CubeActors.Empty();
	for (int32 c = 0; c < 8; c++)
	{
		const float Angle = (2.0f * PI * c) / 8.0f
			+ FMath::DegreesToRadians(CubeAngle);
		const float Height = 40.0f * ((c % 2 == 0) ? 1.0f : -1.0f);

		const FVector CubePos = Base + FVector(
			FMath::Cos(Angle) * CubeOrbitRadius,
			FMath::Sin(Angle) * CubeOrbitRadius,
			Height);

		// Unity: sizes vary per cube (1.0 + i*0.3)
		const float ThisCubeScale = CubeScale * (0.7f + 0.3f * c / 7.0f);

		AAliceSdfNaniteActor* Cube = SpawnAbstractShape(World, CubePos,
			EAliceSdfNaniteShape::FloatingCube,
			FMath::Max(64, ShapeResolution / 3),
			ThisCubeScale,
			ShapeMaterial,
			FString::Printf(TEXT("Abstract_Cube_%d"), c));

		if (Cube)
		{
			TotalTriangles += Cube->TriangleCount;
			SpawnedActors.Add(Cube);
			CubeActors.Add(Cube);
		}
	}

	BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Abstract Showcase Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Center(Gyroid) + 6 Metaballs + 3 Rings + 4 SchwarzP + 8 Cubes"));
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %d"), TotalTriangles);
	UE_LOG(LogTemp, Log,
		TEXT("  Build time: %.2fs"), BuildTimeSeconds);
}

// ============================================================================
// Tick — Orbit metaballs and cubes, rotate rings
// ============================================================================

void AAliceSdfAbstractShowcase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	const FVector Base = GetActorLocation();

	// Metaball orbit
	if (MetaballActors.Num() > 0)
	{
		MetaballAngle += MetaballOrbitSpeed * DeltaTime;

		for (int32 i = 0; i < MetaballActors.Num(); i++)
		{
			if (!MetaballActors[i] || !IsValid(MetaballActors[i])) continue;

			const float Angle = (2.0f * PI * i) / MetaballActors.Num()
				+ FMath::DegreesToRadians(MetaballAngle);
			const float Height = 30.0f * FMath::Sin(Angle * 2.0f);

			MetaballActors[i]->SetActorLocation(Base + FVector(
				FMath::Cos(Angle) * MetaballOrbitRadius,
				FMath::Sin(Angle) * MetaballOrbitRadius,
				Height));
		}
	}

	// Ring rotation
	for (int32 j = 0; j < RingActors.Num(); j++)
	{
		if (!RingActors[j] || !IsValid(RingActors[j])) continue;

		// Each ring rotates around a different axis
		const float Speed = RingRotationSpeed * (1.0f + j * 0.3f);
		FRotator DeltaRot;
		if (j == 0) DeltaRot = FRotator(Speed * DeltaTime, 0.0f, 0.0f);
		else if (j == 1) DeltaRot = FRotator(0.0f, Speed * DeltaTime, 0.0f);
		else DeltaRot = FRotator(0.0f, 0.0f, Speed * DeltaTime);
		RingActors[j]->AddActorLocalRotation(DeltaRot);
	}

	// Cube orbit + self-rotation
	if (CubeActors.Num() > 0)
	{
		CubeAngle += CubeOrbitSpeed * DeltaTime;

		for (int32 c = 0; c < CubeActors.Num(); c++)
		{
			if (!CubeActors[c] || !IsValid(CubeActors[c])) continue;

			const float Angle = (2.0f * PI * c) / CubeActors.Num()
				+ FMath::DegreesToRadians(CubeAngle);
			const float Height = 40.0f * ((c % 2 == 0) ? 1.0f : -1.0f);

			CubeActors[c]->SetActorLocation(Base + FVector(
				FMath::Cos(Angle) * CubeOrbitRadius,
				FMath::Sin(Angle) * CubeOrbitRadius,
				Height));

			CubeActors[c]->AddActorLocalRotation(
				FRotator(CubeSelfRotation * DeltaTime,
					CubeSelfRotation * 0.7f * DeltaTime,
					CubeSelfRotation * 0.5f * DeltaTime));
		}
	}
}

// ============================================================================
// ClearAll
// ============================================================================

void AAliceSdfAbstractShowcase::ClearAll()
{
	for (AActor* Actor : SpawnedActors)
	{
		if (Actor && IsValid(Actor))
		{
			Actor->Destroy();
		}
	}
	SpawnedActors.Empty();
	MetaballActors.Empty();
	RingActors.Empty();
	CubeActors.Empty();
	TotalTriangles = 0;
}
