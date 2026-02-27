// ALICE-SDF Cosmic Showcase Implementation
// Author: Moroya Sakamoto

#include "AliceSdfCosmicShowcase.h"
#include "Engine/World.h"

AAliceSdfCosmicShowcase::AAliceSdfCosmicShowcase()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfCosmicShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildSolarSystem();
	}
}

// ============================================================================
// Helper: Spawn a NaniteActor with given shape type
// ============================================================================

static AAliceSdfNaniteActor* SpawnShape(
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
// BuildSolarSystem
// ============================================================================

void AAliceSdfCosmicShowcase::BuildSolarSystem()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	const FVector Base = GetActorLocation();

	// --- Sun (center) ---
	SunActor = SpawnShape(World, Base,
		EAliceSdfNaniteShape::CosmicSun, ShapeResolution, SunScale,
		SunMaterial, TEXT("Cosmic_Sun"));
	if (SunActor)
	{
		TotalTriangles += SunActor->TriangleCount;
		SpawnedActors.Add(SunActor);
	}

	// --- Planet 1 (ringed) ---
	const float P1Rad = FMath::DegreesToRadians(Planet1Angle);
	const FVector P1Pos = Base + FVector(
		FMath::Cos(P1Rad) * Planet1Orbit, FMath::Sin(P1Rad) * Planet1Orbit, 0.0f);

	Planet1Actor = SpawnShape(World, P1Pos,
		EAliceSdfNaniteShape::CosmicRingedPlanet, ShapeResolution, PlanetScale,
		PlanetMaterial, TEXT("Cosmic_Planet1"));
	if (Planet1Actor)
	{
		TotalTriangles += Planet1Actor->TriangleCount;
		SpawnedActors.Add(Planet1Actor);
	}

	// --- Moon (orbits planet 1) ---
	const float MRad = FMath::DegreesToRadians(MoonAngle);
	const FVector MoonPos = P1Pos + FVector(
		FMath::Cos(MRad) * MoonOrbit, FMath::Sin(MRad) * MoonOrbit, 20.0f);

	MoonActor = SpawnShape(World, MoonPos,
		EAliceSdfNaniteShape::CosmicAsteroid, FMath::Max(64, ShapeResolution / 2), MoonScale,
		PlanetMaterial, TEXT("Cosmic_Moon"));
	if (MoonActor)
	{
		TotalTriangles += MoonActor->TriangleCount;
		SpawnedActors.Add(MoonActor);
	}

	// --- Planet 2 (distant, no ring — use sphere shape) ---
	const float P2Rad = FMath::DegreesToRadians(Planet2Angle);
	const FVector P2Pos = Base + FVector(
		FMath::Cos(P2Rad) * Planet2Orbit, FMath::Sin(P2Rad) * Planet2Orbit, 50.0f);

	Planet2Actor = SpawnShape(World, P2Pos,
		EAliceSdfNaniteShape::CosmicAsteroid, ShapeResolution, PlanetScale * 0.6f,
		PlanetMaterial, TEXT("Cosmic_Planet2"));
	if (Planet2Actor)
	{
		TotalTriangles += Planet2Actor->TriangleCount;
		SpawnedActors.Add(Planet2Actor);
	}

	// --- Asteroid belt ---
	for (int32 i = 0; i < NumAsteroids; i++)
	{
		const float Angle = (2.0f * PI * i) / NumAsteroids;
		const float Jitter = (i % 3 - 1) * 30.0f;
		const FVector AstPos = Base + FVector(
			FMath::Cos(Angle) * (AsteroidBeltRadius + Jitter),
			FMath::Sin(Angle) * (AsteroidBeltRadius + Jitter),
			(i % 2 == 0 ? 15.0f : -15.0f));

		AAliceSdfNaniteActor* Ast = SpawnShape(World, AstPos,
			EAliceSdfNaniteShape::CosmicAsteroid,
			FMath::Max(64, ShapeResolution / 3),
			AsteroidScale * (0.6f + 0.4f * (i % 3) / 2.0f),
			PlanetMaterial,
			FString::Printf(TEXT("Cosmic_Asteroid_%d"), i));

		if (Ast)
		{
			TotalTriangles += Ast->TriangleCount;
			SpawnedActors.Add(Ast);
		}
	}

	BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Cosmic Showcase Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Sun + %d planets + moon + %d asteroids"), 2, NumAsteroids);
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %d"), TotalTriangles);
	UE_LOG(LogTemp, Log,
		TEXT("  Build time: %.2fs"), BuildTimeSeconds);
}

// ============================================================================
// Tick — Orbital animation
// ============================================================================

void AAliceSdfCosmicShowcase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	const FVector Base = GetActorLocation();

	// Sun self-rotation
	if (SunActor && IsValid(SunActor))
	{
		SunActor->AddActorLocalRotation(
			FRotator(0.0f, SunRotationSpeed * DeltaTime, 0.0f));
	}

	// Planet 1 orbit
	Planet1Angle += Planet1Speed * DeltaTime;
	if (Planet1Actor && IsValid(Planet1Actor))
	{
		const float P1Rad = FMath::DegreesToRadians(Planet1Angle);
		Planet1Actor->SetActorLocation(Base + FVector(
			FMath::Cos(P1Rad) * Planet1Orbit,
			FMath::Sin(P1Rad) * Planet1Orbit, 0.0f));
		Planet1Actor->AddActorLocalRotation(
			FRotator(0.0f, Planet1Speed * 2.0f * DeltaTime, 0.0f));
	}

	// Moon orbit around planet 1
	MoonAngle += MoonSpeed * DeltaTime;
	if (MoonActor && IsValid(MoonActor) && Planet1Actor && IsValid(Planet1Actor))
	{
		const FVector P1Loc = Planet1Actor->GetActorLocation();
		const float MRad = FMath::DegreesToRadians(MoonAngle);
		MoonActor->SetActorLocation(P1Loc + FVector(
			FMath::Cos(MRad) * MoonOrbit,
			FMath::Sin(MRad) * MoonOrbit, 20.0f));
	}

	// Planet 2 orbit
	Planet2Angle += Planet2Speed * DeltaTime;
	if (Planet2Actor && IsValid(Planet2Actor))
	{
		const float P2Rad = FMath::DegreesToRadians(Planet2Angle);
		Planet2Actor->SetActorLocation(Base + FVector(
			FMath::Cos(P2Rad) * Planet2Orbit,
			FMath::Sin(P2Rad) * Planet2Orbit, 50.0f));
		Planet2Actor->AddActorLocalRotation(
			FRotator(0.0f, Planet2Speed * 2.0f * DeltaTime, 0.0f));
	}
}

// ============================================================================
// ClearAll
// ============================================================================

void AAliceSdfCosmicShowcase::ClearAll()
{
	for (AActor* Actor : SpawnedActors)
	{
		if (Actor && IsValid(Actor))
		{
			Actor->Destroy();
		}
	}
	SpawnedActors.Empty();
	SunActor = nullptr;
	Planet1Actor = nullptr;
	Planet2Actor = nullptr;
	MoonActor = nullptr;
	TotalTriangles = 0;
}
