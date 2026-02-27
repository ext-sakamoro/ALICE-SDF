// ALICE-SDF Fractal Showcase Implementation
// Author: Moroya Sakamoto

#include "AliceSdfFractalShowcase.h"
#include "Engine/World.h"

AAliceSdfFractalShowcase::AAliceSdfFractalShowcase()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfFractalShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildFractal();
	}
}

// ============================================================================
// BuildFractal — High-resolution Menger Sponge
// ============================================================================

void AAliceSdfFractalShowcase::BuildFractal()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	const FVector Base = GetActorLocation();

	FActorSpawnParameters Params;
	Params.SpawnCollisionHandlingOverride =
		ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	FractalActor = World->SpawnActor<AAliceSdfNaniteActor>(
		AAliceSdfNaniteActor::StaticClass(), Base,
		FRotator::ZeroRotator, Params);

	if (FractalActor)
	{
		FractalActor->SetActorLabel(TEXT("Fractal_MengerSponge"));
		FractalActor->ShapeType = EAliceSdfNaniteShape::MengerSponge;
		FractalActor->MeshResolution = ShapeResolution;
		FractalActor->WorldScale = FractalScale;
		FractalActor->Bounds = 3.0f;
		if (FractalMaterial)
		{
			FractalActor->OverrideMaterial = FractalMaterial;
		}

		FractalActor->RebuildMesh();

		TotalTriangles = FractalActor->TriangleCount;
		SpawnedActors.Add(FractalActor);
	}

	BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Fractal Showcase Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Menger Sponge at Resolution=%d"), ShapeResolution);
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %d"), TotalTriangles);
	UE_LOG(LogTemp, Log,
		TEXT("  Build time: %.2fs"), BuildTimeSeconds);
	UE_LOG(LogTemp, Log,
		TEXT("  Nanite handles all %d triangles without LOD setup"), TotalTriangles);
}

// ============================================================================
// Tick — Slow turntable rotation
// ============================================================================

void AAliceSdfFractalShowcase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!bRotate) return;

	if (FractalActor && IsValid(FractalActor))
	{
		FractalActor->AddActorLocalRotation(
			FRotator(0.0f, RotationSpeed * DeltaTime, 0.0f));
	}
}

// ============================================================================
// ClearAll
// ============================================================================

void AAliceSdfFractalShowcase::ClearAll()
{
	for (AActor* Actor : SpawnedActors)
	{
		if (Actor && IsValid(Actor))
		{
			Actor->Destroy();
		}
	}
	SpawnedActors.Empty();
	FractalActor = nullptr;
	TotalTriangles = 0;
}
