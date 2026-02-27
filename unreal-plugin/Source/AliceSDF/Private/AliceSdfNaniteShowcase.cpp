// ALICE-SDF Nanite Showcase Actor Implementation
// Author: Moroya Sakamoto

#include "AliceSdfNaniteShowcase.h"
#include "Components/StaticMeshComponent.h"
#include "Components/TextRenderComponent.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"

AAliceSdfNaniteShowcase::AAliceSdfNaniteShowcase()
{
	PrimaryActorTick.bCanEverTick = true;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfNaniteShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildAllShapes();
	}
}

// ============================================================================
// BuildAllShapes — Spawn 12 shapes on pedestals in a circle
// ============================================================================

void AAliceSdfNaniteShowcase::BuildAllShapes()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	// Shape names for labels
	static const TCHAR* ShapeNames[] = {
		// Original 5
		TEXT("TPMS Gyroid Sphere"),
		TEXT("Organic Sculpture"),
		TEXT("Crystal Formation"),
		TEXT("Architectural Column"),
		TEXT("SAO Floating Island"),
		// Ported from VRChat / Unity
		TEXT("Menger Sponge"),
		TEXT("Cosmic System"),
		TEXT("Mochi Blobs"),
		TEXT("Fractal Planet"),
		TEXT("Abstract Surface"),
		// Nanite + Lumen showcase
		TEXT("SDF Cathedral"),
		TEXT("Coral Reef"),
	};

	static const EAliceSdfNaniteShape ShapeTypes[] = {
		// Original 5
		EAliceSdfNaniteShape::TPMSSphere,
		EAliceSdfNaniteShape::OrganicSculpture,
		EAliceSdfNaniteShape::Crystal,
		EAliceSdfNaniteShape::ArchColumn,
		EAliceSdfNaniteShape::SAOFloat,
		// Ported from VRChat / Unity
		EAliceSdfNaniteShape::MengerSponge,
		EAliceSdfNaniteShape::CosmicSystem,
		EAliceSdfNaniteShape::MochiBlobs,
		EAliceSdfNaniteShape::FractalPlanet,
		EAliceSdfNaniteShape::AbstractSurface,
		// Nanite + Lumen showcase
		EAliceSdfNaniteShape::Cathedral,
		EAliceSdfNaniteShape::CoralReef,
	};

	const FVector BaseLocation = GetActorLocation();

	for (int32 i = 0; i < NUM_SHAPES; i++)
	{
		// Position in circle
		const float Angle = (2.0f * PI * i) / NUM_SHAPES;
		const FVector Offset(
			FMath::Cos(Angle) * CircleRadius,
			FMath::Sin(Angle) * CircleRadius,
			0.0f
		);
		const FVector ShapeLocation = BaseLocation + Offset;

		// --- Pedestal ---
		AActor* PedestalActor = World->SpawnActor<AActor>(
			AActor::StaticClass(), ShapeLocation, FRotator::ZeroRotator);
		if (PedestalActor)
		{
			PedestalActor->SetActorLabel(FString::Printf(TEXT("Pedestal_%d"), i));

			UStaticMeshComponent* PedestalMesh =
				NewObject<UStaticMeshComponent>(PedestalActor);
			PedestalActor->SetRootComponent(PedestalMesh);

			// Use engine's basic cylinder as pedestal
			UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(
				nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
			if (CylinderMesh)
			{
				PedestalMesh->SetStaticMesh(CylinderMesh);
				PedestalMesh->SetRelativeScale3D(FVector(1.5f, 1.5f, PedestalHeight / 100.0f));
				PedestalMesh->SetRelativeLocation(FVector(0, 0, -PedestalHeight * 0.5f));
				if (PedestalMaterial)
				{
					PedestalMesh->SetMaterial(0, PedestalMaterial);
				}
			}
			PedestalMesh->RegisterComponent();
			SpawnedActors.Add(PedestalActor);
		}

		// --- Text Label ---
		AActor* LabelActor = World->SpawnActor<AActor>(
			AActor::StaticClass(),
			ShapeLocation + FVector(0, 0, -PedestalHeight - 30.0f),
			FRotator::ZeroRotator);
		if (LabelActor)
		{
			UTextRenderComponent* TextComp =
				NewObject<UTextRenderComponent>(LabelActor);
			LabelActor->SetRootComponent(TextComp);
			TextComp->SetText(FText::FromString(ShapeNames[i]));
			TextComp->SetTextRenderColor(FColor::White);
			TextComp->SetHorizontalAlignment(EHTA_Center);
			TextComp->SetWorldSize(20.0f);
			TextComp->RegisterComponent();
			SpawnedActors.Add(LabelActor);
		}

		// --- ALICE-SDF Nanite Shape ---
		FActorSpawnParameters SpawnParams;
		SpawnParams.SpawnCollisionHandlingOverride =
			ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

		AAliceSdfNaniteActor* ShapeActor = World->SpawnActor<AAliceSdfNaniteActor>(
			AAliceSdfNaniteActor::StaticClass(),
			ShapeLocation + FVector(0, 0, ShapeScale * 0.5f),
			FRotator::ZeroRotator,
			SpawnParams);

		if (ShapeActor)
		{
			ShapeActor->SetActorLabel(FString::Printf(TEXT("AliceSDF_%s"),
				ShapeNames[i]));
			ShapeActor->ShapeType = ShapeTypes[i];
			ShapeActor->MeshResolution = ShapeResolution;
			ShapeActor->WorldScale = ShapeScale;
			if (ShapeMaterial)
			{
				ShapeActor->OverrideMaterial = ShapeMaterial;
			}

			// Build the Nanite mesh
			ShapeActor->RebuildMesh();

			TotalTriangles += ShapeActor->TriangleCount;
			SpawnedActors.Add(ShapeActor);
		}

		UE_LOG(LogTemp, Log, TEXT("ALICE-SDF Showcase: Built %s (%d/%d)"),
			ShapeNames[i], i + 1, NUM_SHAPES);
	}

	TotalBuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Nanite Showcase Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Shapes: %d"), NUM_SHAPES);
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %s"),
		*FString::Printf(TEXT("%d"), TotalTriangles));
	UE_LOG(LogTemp, Log,
		TEXT("  Total build time: %.2fs"), TotalBuildTimeSeconds);
	UE_LOG(LogTemp, Log,
		TEXT("  All rendered at 60fps via Nanite"));
}

// ============================================================================
// ClearAll — Remove all spawned actors
// ============================================================================

void AAliceSdfNaniteShowcase::ClearAll()
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

// ============================================================================
// Tick — Turntable rotation
// ============================================================================

void AAliceSdfNaniteShowcase::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!bRotatePedestals) return;

	for (AActor* Actor : SpawnedActors)
	{
		if (AAliceSdfNaniteActor* ShapeActor = Cast<AAliceSdfNaniteActor>(Actor))
		{
			ShapeActor->AddActorLocalRotation(
				FRotator(0.0f, RotationSpeed * DeltaTime, 0.0f));
		}
	}
}
