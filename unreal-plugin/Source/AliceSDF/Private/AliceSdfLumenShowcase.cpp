// ALICE-SDF Lumen Showcase Actor Implementation
// Author: Moroya Sakamoto

#include "AliceSdfLumenShowcase.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/TextRenderComponent.h"
#include "Engine/PostProcessVolume.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"

AAliceSdfLumenShowcase::AAliceSdfLumenShowcase()
{
	PrimaryActorTick.bCanEverTick = false;

	USceneComponent* Root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
	SetRootComponent(Root);
}

// ============================================================================
// OnConstruction — Auto-build when placed in level
// ============================================================================

void AAliceSdfLumenShowcase::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (SpawnedActors.Num() == 0)
	{
		BuildGallery();
	}
}

// ============================================================================
// BuildGallery — Build room, shapes, and colored lights
// ============================================================================

void AAliceSdfLumenShowcase::BuildGallery()
{
	ClearAll();

	UWorld* World = GetWorld();
	if (!World) return;

	const double StartTime = FPlatformTime::Seconds();
	TotalTriangles = 0;

	const FVector Base = GetActorLocation();
	const float HalfW = RoomWidth * 0.5f;
	const float HalfD = RoomDepth * 0.5f;

	// ======================================================================
	// Room geometry (engine basic shapes)
	// ======================================================================

	UStaticMesh* CubeMesh = LoadObject<UStaticMesh>(
		nullptr, TEXT("/Engine/BasicShapes/Cube.Cube"));
	UStaticMesh* PlaneMesh = LoadObject<UStaticMesh>(
		nullptr, TEXT("/Engine/BasicShapes/Plane.Plane"));

	auto SpawnWall = [&](const FVector& Pos, const FVector& Scale, const FString& Name)
	{
		AActor* WallActor = World->SpawnActor<AActor>(
			AActor::StaticClass(), Base + Pos, FRotator::ZeroRotator);
		if (!WallActor || !CubeMesh) return;

		WallActor->SetActorLabel(Name);

		UStaticMeshComponent* MeshComp =
			NewObject<UStaticMeshComponent>(WallActor);
		MeshComp->SetupAttachment(WallActor->GetRootComponent());
		MeshComp->SetStaticMesh(CubeMesh);
		MeshComp->SetRelativeScale3D(Scale);
		if (WallMaterial)
		{
			MeshComp->SetMaterial(0, WallMaterial);
		}
		MeshComp->RegisterComponent();
		SpawnedActors.Add(WallActor);
	};

	// Floor
	{
		AActor* FloorActor = World->SpawnActor<AActor>(
			AActor::StaticClass(), Base, FRotator::ZeroRotator);
		if (FloorActor && PlaneMesh)
		{
			FloorActor->SetActorLabel(TEXT("Lumen_Floor"));

			UStaticMeshComponent* FloorMesh =
				NewObject<UStaticMeshComponent>(FloorActor);
			FloorMesh->SetupAttachment(FloorActor->GetRootComponent());
			FloorMesh->SetStaticMesh(PlaneMesh);
			FloorMesh->SetRelativeScale3D(FVector(
				RoomWidth / 100.0f, RoomDepth / 100.0f, 1.0f));
			if (FloorMaterial)
			{
				FloorMesh->SetMaterial(0, FloorMaterial);
			}
			FloorMesh->RegisterComponent();
			SpawnedActors.Add(FloorActor);
		}
	}

	// Wall thickness
	const float WallThick = 20.0f;

	// Back wall (Y = -HalfD)
	SpawnWall(
		FVector(0.0f, -HalfD, RoomHeight * 0.5f),
		FVector(RoomWidth / 100.0f, WallThick / 100.0f, RoomHeight / 100.0f),
		TEXT("Lumen_BackWall"));

	// Left wall (X = -HalfW)
	SpawnWall(
		FVector(-HalfW, 0.0f, RoomHeight * 0.5f),
		FVector(WallThick / 100.0f, RoomDepth / 100.0f, RoomHeight / 100.0f),
		TEXT("Lumen_LeftWall"));

	// Right wall (X = +HalfW)
	SpawnWall(
		FVector(HalfW, 0.0f, RoomHeight * 0.5f),
		FVector(WallThick / 100.0f, RoomDepth / 100.0f, RoomHeight / 100.0f),
		TEXT("Lumen_RightWall"));

	// Ceiling (with skylight opening — slightly smaller than floor)
	{
		AActor* CeilActor = World->SpawnActor<AActor>(
			AActor::StaticClass(),
			Base + FVector(0.0f, 0.0f, RoomHeight),
			FRotator::ZeroRotator);
		if (CeilActor && PlaneMesh)
		{
			CeilActor->SetActorLabel(TEXT("Lumen_Ceiling"));

			UStaticMeshComponent* CeilMesh =
				NewObject<UStaticMeshComponent>(CeilActor);
			CeilMesh->SetupAttachment(CeilActor->GetRootComponent());
			CeilMesh->SetStaticMesh(PlaneMesh);
			CeilMesh->SetRelativeScale3D(FVector(
				RoomWidth / 100.0f, RoomDepth / 100.0f, 1.0f));
			CeilMesh->SetRelativeRotation(FRotator(180.0f, 0.0f, 0.0f));
			if (WallMaterial)
			{
				CeilMesh->SetMaterial(0, WallMaterial);
			}
			CeilMesh->RegisterComponent();
			SpawnedActors.Add(CeilActor);
		}
	}

	// ======================================================================
	// SDF Shapes — 4 shapes on pedestals in a line
	// ======================================================================

	static const TCHAR* ShapeNames[] = {
		TEXT("Cathedral"),
		TEXT("Coral Reef"),
		TEXT("TPMS Sphere"),
		TEXT("Crystal"),
	};

	static const EAliceSdfNaniteShape ShapeTypes[] = {
		EAliceSdfNaniteShape::Cathedral,
		EAliceSdfNaniteShape::CoralReef,
		EAliceSdfNaniteShape::TPMSSphere,
		EAliceSdfNaniteShape::Crystal,
	};

	UStaticMesh* CylinderMesh = LoadObject<UStaticMesh>(
		nullptr, TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));

	const float PedestalHeight = 60.0f;
	const float Spacing = RoomWidth / (NUM_GALLERY_SHAPES + 1);

	for (int32 i = 0; i < NUM_GALLERY_SHAPES; i++)
	{
		const float XPos = -HalfW + Spacing * (i + 1);
		const FVector ShapePos = Base + FVector(XPos, 0.0f, PedestalHeight);

		// Pedestal
		AActor* PedestalActor = World->SpawnActor<AActor>(
			AActor::StaticClass(), ShapePos, FRotator::ZeroRotator);
		if (PedestalActor && CylinderMesh)
		{
			PedestalActor->SetActorLabel(
				FString::Printf(TEXT("Lumen_Pedestal_%d"), i));

			UStaticMeshComponent* PedMesh =
				NewObject<UStaticMeshComponent>(PedestalActor);
			PedMesh->SetupAttachment(PedestalActor->GetRootComponent());
			PedMesh->SetStaticMesh(CylinderMesh);
			PedMesh->SetRelativeScale3D(FVector(1.2f, 1.2f, PedestalHeight / 100.0f));
			PedMesh->SetRelativeLocation(FVector(0, 0, -PedestalHeight * 0.5f));
			if (FloorMaterial)
			{
				PedMesh->SetMaterial(0, FloorMaterial);
			}
			PedMesh->RegisterComponent();
			SpawnedActors.Add(PedestalActor);
		}

		// Text label
		AActor* LabelActor = World->SpawnActor<AActor>(
			AActor::StaticClass(),
			ShapePos + FVector(0.0f, 0.0f, -PedestalHeight - 20.0f),
			FRotator::ZeroRotator);
		if (LabelActor)
		{
			UTextRenderComponent* TextComp =
				NewObject<UTextRenderComponent>(LabelActor);
			TextComp->SetupAttachment(LabelActor->GetRootComponent());
			TextComp->SetText(FText::FromString(ShapeNames[i]));
			TextComp->SetTextRenderColor(FColor::White);
			TextComp->SetHorizontalAlignment(EHTA_Center);
			TextComp->SetWorldSize(16.0f);
			TextComp->RegisterComponent();
			SpawnedActors.Add(LabelActor);
		}

		// SDF Shape
		FActorSpawnParameters SpawnParams;
		SpawnParams.SpawnCollisionHandlingOverride =
			ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

		AAliceSdfNaniteActor* ShapeActor = World->SpawnActor<AAliceSdfNaniteActor>(
			AAliceSdfNaniteActor::StaticClass(),
			ShapePos + FVector(0.0f, 0.0f, ShapeScale * 0.5f),
			FRotator::ZeroRotator,
			SpawnParams);

		if (ShapeActor)
		{
			ShapeActor->SetActorLabel(
				FString::Printf(TEXT("Lumen_%s"), ShapeNames[i]));
			ShapeActor->ShapeType = ShapeTypes[i];
			ShapeActor->MeshResolution = ShapeResolution;
			ShapeActor->WorldScale = ShapeScale;
			if (ShapeMaterial)
			{
				ShapeActor->OverrideMaterial = ShapeMaterial;
			}

			ShapeActor->RebuildMesh();

			TotalTriangles += ShapeActor->TriangleCount;
			SpawnedActors.Add(ShapeActor);
		}

		UE_LOG(LogTemp, Log, TEXT("ALICE-SDF Lumen Gallery: Built %s (%d/%d)"),
			ShapeNames[i], i + 1, NUM_GALLERY_SHAPES);
	}

	// ======================================================================
	// Colored Point Lights for Lumen GI color bleeding
	// ======================================================================

	struct LightDef
	{
		FVector Offset;
		FLinearColor Color;
		const TCHAR* Name;
	};

	const LightDef Lights[] = {
		// Warm light (left side) — red/orange bounce on walls
		{ FVector(-HalfW * 0.6f, -HalfD * 0.5f, RoomHeight * 0.8f),
		  FLinearColor(1.0f, 0.6f, 0.2f), TEXT("Lumen_WarmLight") },

		// Cool light (right side) — blue bounce on SDF shapes
		{ FVector(HalfW * 0.6f, -HalfD * 0.5f, RoomHeight * 0.8f),
		  FLinearColor(0.2f, 0.5f, 1.0f), TEXT("Lumen_CoolLight") },

		// Accent light (center top) — green tint for visible color bleed
		{ FVector(0.0f, 0.0f, RoomHeight * 0.9f),
		  FLinearColor(0.3f, 1.0f, 0.4f), TEXT("Lumen_AccentLight") },

		// Fill light (front) — soft white from open side
		{ FVector(0.0f, HalfD * 0.8f, RoomHeight * 0.5f),
		  FLinearColor(1.0f, 0.95f, 0.9f), TEXT("Lumen_FillLight") },
	};

	for (const LightDef& Def : Lights)
	{
		AActor* LightActor = World->SpawnActor<AActor>(
			AActor::StaticClass(),
			Base + Def.Offset,
			FRotator::ZeroRotator);
		if (LightActor)
		{
			LightActor->SetActorLabel(Def.Name);

			UPointLightComponent* LightComp =
				NewObject<UPointLightComponent>(LightActor);
			LightComp->SetupAttachment(LightActor->GetRootComponent());
			LightComp->SetIntensity(LightIntensity);
			LightComp->SetAttenuationRadius(LightRadius);
			LightComp->SetLightColor(Def.Color);
			LightComp->SetCastShadows(true);
			// Inverse squared falloff produces more realistic Lumen GI
			LightComp->SetIntensityUnits(ELightUnits::Candelas);
			LightComp->RegisterComponent();
			SpawnedActors.Add(LightActor);
		}
	}

	// ======================================================================
	// SkyLight — Environment indirect lighting for Lumen
	// ======================================================================

	if (bSpawnSkyLight)
	{
		AActor* SkyActor = World->SpawnActor<AActor>(
			AActor::StaticClass(),
			Base + FVector(0.0f, 0.0f, RoomHeight * 0.5f),
			FRotator::ZeroRotator);
		if (SkyActor)
		{
			SkyActor->SetActorLabel(TEXT("Lumen_SkyLight"));

			USkyLightComponent* SkyComp =
				NewObject<USkyLightComponent>(SkyActor);
			SkyComp->SetupAttachment(SkyActor->GetRootComponent());
			SkyComp->SetIntensity(SkyLightIntensity);
			SkyComp->SetCastShadows(true);
			// Captured scene provides indirect ambient for Lumen
			SkyComp->SourceType = ESkyLightSourceType::SLS_CapturedScene;
			SkyComp->RegisterComponent();
			SkyComp->RecaptureSky();
			SpawnedActors.Add(SkyActor);
		}
	}

	// ======================================================================
	// PostProcessVolume — Lumen GI quality settings
	// ======================================================================

	if (bSpawnPostProcessVolume)
	{
		APostProcessVolume* PPV = World->SpawnActor<APostProcessVolume>(
			APostProcessVolume::StaticClass(),
			Base + FVector(0.0f, 0.0f, RoomHeight * 0.5f),
			FRotator::ZeroRotator);
		if (PPV)
		{
			PPV->SetActorLabel(TEXT("Lumen_PostProcess"));
			PPV->bUnbound = true;

			FPostProcessSettings& Settings = PPV->Settings;

			// Lumen GI settings
			Settings.bOverride_DynamicGlobalIlluminationMethod = true;
			Settings.DynamicGlobalIlluminationMethod = EDynamicGlobalIlluminationMethod::Lumen;

			// Lumen Reflection settings
			Settings.bOverride_ReflectionMethod = true;
			Settings.ReflectionMethod = EReflectionMethod::Lumen;

			// Lumen Final Gather Quality
			Settings.bOverride_LumenFinalGatherQuality = true;
			Settings.LumenFinalGatherQuality = LumenFinalGatherQuality;

			// Lumen Scene Detail
			Settings.bOverride_LumenSceneDetail = true;
			Settings.LumenSceneDetail = LumenSceneDetail;

			// Lumen Scene Lighting Quality
			Settings.bOverride_LumenSceneLightingQuality = true;
			Settings.LumenSceneLightingQuality = 2.0f;

			SpawnedActors.Add(PPV);

			UE_LOG(LogTemp, Log,
				TEXT("ALICE-SDF Lumen: PostProcessVolume spawned (GI=Lumen, Reflections=Lumen, FGQ=%.1f)"),
				LumenFinalGatherQuality);
		}
	}

	BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

	UE_LOG(LogTemp, Log,
		TEXT("=== ALICE-SDF Lumen Gallery Complete ==="));
	UE_LOG(LogTemp, Log,
		TEXT("  Shapes: %d, Lights: 4, SkyLight: %s, PPV: %s"),
		NUM_GALLERY_SHAPES,
		bSpawnSkyLight ? TEXT("Yes") : TEXT("No"),
		bSpawnPostProcessVolume ? TEXT("Yes") : TEXT("No"));
	UE_LOG(LogTemp, Log,
		TEXT("  Total triangles: %d"), TotalTriangles);
	UE_LOG(LogTemp, Log,
		TEXT("  Build time: %.2fs"), BuildTimeSeconds);
	UE_LOG(LogTemp, Log,
		TEXT("  Lumen GI active — observe color bleeding on walls and shapes"));
}

// ============================================================================
// ClearAll — Remove all gallery elements
// ============================================================================

void AAliceSdfLumenShowcase::ClearAll()
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
