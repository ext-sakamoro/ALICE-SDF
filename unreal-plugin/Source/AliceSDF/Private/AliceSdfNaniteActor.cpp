// ALICE-SDF Nanite Demo Actor Implementation
// Author: Moroya Sakamoto

#include "AliceSdfNaniteActor.h"
#include "Engine/StaticMesh.h"
#include "MeshDescription.h"
#include "StaticMeshAttributes.h"
#include "Components/StaticMeshComponent.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"

AAliceSdfNaniteActor::AAliceSdfNaniteActor()
{
	PrimaryActorTick.bCanEverTick = false;

	MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("NaniteMesh"));
	SetRootComponent(MeshComponent);

	MainShape = CreateDefaultSubobject<UAliceSdfComponent>(TEXT("MainShape"));
	CsgShape = CreateDefaultSubobject<UAliceSdfComponent>(TEXT("CsgShape"));
}

void AAliceSdfNaniteActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);

	if (!GeneratedMesh)
	{
		RebuildMesh();
	}
}

#if WITH_EDITOR
void AAliceSdfNaniteActor::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	const FName PropertyName = PropertyChangedEvent.GetPropertyName();
	if (PropertyName == GET_MEMBER_NAME_CHECKED(AAliceSdfNaniteActor, ShapeType) ||
		PropertyName == GET_MEMBER_NAME_CHECKED(AAliceSdfNaniteActor, MeshResolution) ||
		PropertyName == GET_MEMBER_NAME_CHECKED(AAliceSdfNaniteActor, Bounds) ||
		PropertyName == GET_MEMBER_NAME_CHECKED(AAliceSdfNaniteActor, WorldScale) ||
		PropertyName == GET_MEMBER_NAME_CHECKED(AAliceSdfNaniteActor, OverrideMaterial))
	{
		RebuildMesh();
	}
}
#endif

// ============================================================================
// RebuildMesh — Main entry point
// ============================================================================

void AAliceSdfNaniteActor::RebuildMesh()
{
#if WITH_EDITOR
	const double StartTime = FPlatformTime::Seconds();

	// 1. Build SDF tree
	BuildShapeSdf();

	// 2. Convert to Nanite StaticMesh
	UStaticMesh* NewMesh = CreateNaniteStaticMesh();
	if (NewMesh)
	{
		GeneratedMesh = NewMesh;
		MeshComponent->SetStaticMesh(NewMesh);
		MeshComponent->SetWorldScale3D(FVector(WorldScale));

		if (OverrideMaterial)
		{
			MeshComponent->SetMaterial(0, OverrideMaterial);
		}

		BuildTimeSeconds = static_cast<float>(FPlatformTime::Seconds() - StartTime);

		UE_LOG(LogTemp, Log,
			TEXT("ALICE-SDF Nanite: %s — %d verts, %d tris, %.2fs (Resolution=%d)"),
			*UEnum::GetValueAsString(ShapeType),
			VertexCount, TriangleCount, BuildTimeSeconds, MeshResolution);
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("ALICE-SDF Nanite: Failed to build mesh"));
	}
#else
	UE_LOG(LogTemp, Warning, TEXT("ALICE-SDF Nanite: Mesh generation requires editor"));
#endif
}

// ============================================================================
// Shape Builders
// ============================================================================

void AAliceSdfNaniteActor::BuildShapeSdf()
{
	switch (ShapeType)
	{
	case EAliceSdfNaniteShape::TPMSSphere:       BuildTPMSSphere(); break;
	case EAliceSdfNaniteShape::OrganicSculpture:  BuildOrganicSculpture(); break;
	case EAliceSdfNaniteShape::Crystal:           BuildCrystal(); break;
	case EAliceSdfNaniteShape::ArchColumn:        BuildArchColumn(); break;
	case EAliceSdfNaniteShape::SAOFloat:          BuildSAOFloat(); break;
	case EAliceSdfNaniteShape::MengerSponge:     BuildMengerSponge(); break;
	case EAliceSdfNaniteShape::CosmicSystem:     BuildCosmicSystem(); break;
	case EAliceSdfNaniteShape::MochiBlobs:       BuildMochiBlobs(); break;
	case EAliceSdfNaniteShape::FractalPlanet:    BuildFractalPlanet(); break;
	case EAliceSdfNaniteShape::AbstractSurface:  BuildAbstractSurface(); break;
	case EAliceSdfNaniteShape::Cathedral:        BuildCathedral(); break;
	case EAliceSdfNaniteShape::CoralReef:        BuildCoralReef(); break;
	case EAliceSdfNaniteShape::CosmicSun:        BuildCosmicSun(); break;
	case EAliceSdfNaniteShape::CosmicRingedPlanet: BuildCosmicRingedPlanet(); break;
	case EAliceSdfNaniteShape::CosmicAsteroid:   BuildCosmicAsteroid(); break;
	case EAliceSdfNaniteShape::TerrainGround:    BuildTerrainGround(); break;
	case EAliceSdfNaniteShape::TerrainWater:     BuildTerrainWater(); break;
	case EAliceSdfNaniteShape::TerrainRock:      BuildTerrainRock(); break;
	case EAliceSdfNaniteShape::FloatingIsland:   BuildFloatingIsland(); break;
	case EAliceSdfNaniteShape::BoundedGyroid:    BuildBoundedGyroid(); break;
	case EAliceSdfNaniteShape::Metaball:         BuildMetaball(); break;
	case EAliceSdfNaniteShape::SchwarzPCorner:   BuildSchwarzPCorner(); break;
	case EAliceSdfNaniteShape::FloatingCube:     BuildFloatingCube(); break;
	case EAliceSdfNaniteShape::AbstractRing:     BuildAbstractRing(); break;
	}
}

// ----------------------------------------------------------------------------
// Shape 0: TPMS Gyroid Sphere — internal lattice with quarter cutaway
// ----------------------------------------------------------------------------
void AAliceSdfNaniteActor::BuildTPMSSphere()
{
	// Fine gyroid lattice
	MainShape->CreateGyroid(4.0f, 0.03f);

	// Constrain to sphere
	CsgShape->CreateSphere(1.2f);
	MainShape->IntersectWith(CsgShape);

	// Quarter cutaway to expose internal structure
	CsgShape->CreateBox(FVector(2.0f, 2.0f, 2.0f));
	CsgShape->TranslateSdf(FVector(1.2f, 1.2f, 0.0f));
	MainShape->SubtractFrom(CsgShape);
}

// ----------------------------------------------------------------------------
// Shape 1: Organic Sculpture — Borromean tori + twist + noise
// ----------------------------------------------------------------------------
void AAliceSdfNaniteActor::BuildOrganicSculpture()
{
	// Three interlocking tori (Borromean rings-inspired)
	MainShape->CreateTorus(1.0f, 0.3f);

	CsgShape->CreateTorus(0.8f, 0.25f);
	CsgShape->RotateSdf(FRotator(90.0f, 0.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.3f);

	CsgShape->CreateTorus(0.6f, 0.2f);
	CsgShape->RotateSdf(FRotator(0.0f, 0.0f, 90.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.25f);

	// Organic bulge at center
	CsgShape->CreateSphere(0.5f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, 0.3f));
	MainShape->SmoothUnionWith(CsgShape, 0.2f);

	// Organic distortion
	MainShape->ApplyTwist(0.5f);
	MainShape->ApplyNoise(0.02f, 8.0f, 42);
}

// ----------------------------------------------------------------------------
// Shape 2: Crystal Formation — octahedron + diamond cluster
// ----------------------------------------------------------------------------
void AAliceSdfNaniteActor::BuildCrystal()
{
	// Central octahedron
	MainShape->CreateOctahedron(0.8f);

	// Surrounding diamond crystals at various angles
	CsgShape->CreateDiamond(0.3f, 0.9f);
	CsgShape->TranslateSdf(FVector(0.6f, 0.2f, 0.3f));
	CsgShape->RotateSdf(FRotator(10.0f, 20.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.05f);

	CsgShape->CreateDiamond(0.25f, 0.7f);
	CsgShape->TranslateSdf(FVector(-0.4f, 0.5f, 0.1f));
	CsgShape->RotateSdf(FRotator(-15.0f, 0.0f, 25.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.05f);

	CsgShape->CreateDiamond(0.2f, 0.6f);
	CsgShape->TranslateSdf(FVector(0.2f, -0.5f, 0.4f));
	CsgShape->RotateSdf(FRotator(5.0f, -30.0f, 10.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.04f);

	CsgShape->CreateOctahedron(0.3f);
	CsgShape->TranslateSdf(FVector(-0.3f, -0.3f, 0.6f));
	CsgShape->RotateSdf(FRotator(15.0f, 30.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.03f);

	CsgShape->CreateDiamond(0.15f, 0.5f);
	CsgShape->TranslateSdf(FVector(0.5f, -0.2f, -0.3f));
	CsgShape->RotateSdf(FRotator(-20.0f, 45.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.03f);
}

// ----------------------------------------------------------------------------
// Shape 3: Architectural Column — fluted Doric column with capital
// ----------------------------------------------------------------------------
void AAliceSdfNaniteActor::BuildArchColumn()
{
	// Main shaft
	MainShape->CreateCylinder(0.35f, 1.2f);

	// Fluting: polar-repeated small cylinders subtracted
	CsgShape->CreateCylinder(0.06f, 1.3f);
	CsgShape->TranslateSdf(FVector(0.32f, 0.0f, 0.0f));
	CsgShape->ApplyPolarRepeat(16);
	MainShape->SubtractFrom(CsgShape);

	// Capital (abacus)
	CsgShape->CreateBox(FVector(0.45f, 0.45f, 0.06f));
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, 1.28f));
	MainShape->SmoothUnionWith(CsgShape, 0.03f);

	// Echinus (curved capital transition)
	CsgShape->CreateTorus(0.38f, 0.05f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, 1.18f));
	MainShape->SmoothUnionWith(CsgShape, 0.04f);

	// Base plinth
	CsgShape->CreateCylinder(0.42f, 0.08f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -1.28f));
	MainShape->SmoothUnionWith(CsgShape, 0.03f);

	// Base torus ring
	CsgShape->CreateTorus(0.40f, 0.03f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -1.18f));
	MainShape->SmoothUnionWith(CsgShape, 0.02f);
}

// ----------------------------------------------------------------------------
// Shape 4: SAO Floating Island — hemisphere + noise terrain + gyroid caves
// ----------------------------------------------------------------------------
void AAliceSdfNaniteActor::BuildSAOFloat()
{
	// Base sphere
	MainShape->CreateSphere(1.0f);

	// Cut bottom half -> dome
	CsgShape->CreateBox(FVector(2.0f, 2.0f, 1.0f));
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -1.5f));
	MainShape->SubtractFrom(CsgShape);

	// Terrain noise on surface
	MainShape->ApplyNoise(0.08f, 4.0f, 42);

	// Carve internal gyroid tunnels (visible from underside)
	CsgShape->CreateGyroid(3.0f, 0.08f);
	MainShape->SmoothSubtractFrom(CsgShape, 0.05f);

	// Clean up edges — bound to slightly larger sphere
	CsgShape->CreateSphere(1.15f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, 0.05f));
	MainShape->IntersectWith(CsgShape);
}

// ============================================================================
// Shape 5: Menger Sponge — Box - Repeat(Cross) + Twist
// (Ported from VRChat SampleFractal + Unity Fractal Demo)
// ============================================================================
void AAliceSdfNaniteActor::BuildMengerSponge()
{
	// Build 3D cross (union of 3 thin boxes) into CsgShape
	CsgShape->CreateBox(FVector(50.0f, 0.4f, 0.4f));   // barX
	MainShape->CreateBox(FVector(0.4f, 50.0f, 0.4f));   // barY
	CsgShape->UnionWith(MainShape);
	MainShape->CreateBox(FVector(0.4f, 0.4f, 50.0f));   // barZ
	CsgShape->UnionWith(MainShape);

	// Infinite repeat cross pattern — space folding magic
	CsgShape->ApplyRepeat(FVector(3.0f, 3.0f, 3.0f));

	// Base box — subtract repeated cross (Menger Sponge)
	MainShape->CreateBox(FVector(1.5f, 1.5f, 1.5f));
	MainShape->SubtractFrom(CsgShape);

	// Organic twist
	MainShape->ApplyTwist(0.02f);
}

// ============================================================================
// Shape 6: Cosmic Solar System — Sun + Planet(Ring) + Moon
// (Ported from VRChat SampleCosmic + Unity CosmicDemo)
// ============================================================================
void AAliceSdfNaniteActor::BuildCosmicSystem()
{
	// Sun (center sphere)
	MainShape->CreateSphere(0.6f);

	// Planet (orbiting)
	CsgShape->CreateSphere(0.25f);
	CsgShape->TranslateSdf(FVector(1.5f, 0.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.15f);

	// Ring around planet (tilted torus)
	CsgShape->CreateTorus(0.45f, 0.025f);
	CsgShape->RotateSdf(FRotator(15.0f, 0.0f, 10.0f));
	CsgShape->TranslateSdf(FVector(1.5f, 0.0f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.08f);

	// Moon
	CsgShape->CreateSphere(0.1f);
	CsgShape->TranslateSdf(FVector(1.5f + 0.6f, 0.15f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.06f);

	// Planet 2 (distant)
	CsgShape->CreateSphere(0.18f);
	CsgShape->TranslateSdf(FVector(-1.0f, 0.0f, 0.8f));
	MainShape->SmoothUnionWith(CsgShape, 0.1f);

	// Asteroid belt (6 small spheres)
	for (int32 i = 0; i < 6; i++)
	{
		const float Angle = static_cast<float>(i) * PI * 2.0f / 6.0f;
		const float BeltR = 1.1f;
		CsgShape->CreateSphere(0.06f + 0.02f * i);
		CsgShape->TranslateSdf(FVector(
			FMath::Cos(Angle) * BeltR,
			(i % 2 == 0 ? 0.1f : -0.1f),
			FMath::Sin(Angle) * BeltR));
		MainShape->SmoothUnionWith(CsgShape, 0.04f);
	}
}

// ============================================================================
// Shape 7: Mochi Blobs — Soft-body squishy spheres
// (Ported from VRChat SampleMochi)
// ============================================================================
void AAliceSdfNaniteActor::BuildMochiBlobs()
{
	// 5 spheres with strong smooth union (k=0.5) for soft-body look
	MainShape->CreateSphere(0.35f);
	MainShape->TranslateSdf(FVector(-0.6f, 0.0f, 0.5f));

	CsgShape->CreateSphere(0.3f);
	CsgShape->TranslateSdf(FVector(0.5f, 0.0f, 0.3f));
	MainShape->SmoothUnionWith(CsgShape, 0.5f);

	CsgShape->CreateSphere(0.28f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -0.4f));
	MainShape->SmoothUnionWith(CsgShape, 0.5f);

	CsgShape->CreateSphere(0.22f);
	CsgShape->TranslateSdf(FVector(-0.3f, 0.25f, -0.1f));
	MainShape->SmoothUnionWith(CsgShape, 0.5f);

	CsgShape->CreateSphere(0.2f);
	CsgShape->TranslateSdf(FVector(0.4f, -0.2f, -0.2f));
	MainShape->SmoothUnionWith(CsgShape, 0.5f);

	// Ground contact bulge
	CsgShape->CreateSphere(0.6f);
	CsgShape->TranslateSdf(FVector(0.0f, -0.5f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.3f);
}

// ============================================================================
// Shape 8: Fractal Planet Mix — Menger sphere + torus ring + onion shell
// (Ported from VRChat SampleMix + Unity CosmicFractalDemo)
// ============================================================================
void AAliceSdfNaniteActor::BuildFractalPlanet()
{
	// Step 1: Build 3D cross into CsgShape
	CsgShape->CreateBox(FVector(50.0f, 0.35f, 0.35f));
	MainShape->CreateBox(FVector(0.35f, 50.0f, 0.35f));
	CsgShape->UnionWith(MainShape);
	MainShape->CreateBox(FVector(0.35f, 0.35f, 50.0f));
	CsgShape->UnionWith(MainShape);

	// Step 2: Infinite repeat and subtract from box
	CsgShape->ApplyRepeat(FVector(2.5f, 2.5f, 2.5f));
	MainShape->CreateBox(FVector(1.3f, 1.3f, 1.3f));
	MainShape->SubtractFrom(CsgShape);

	// Step 3: Clip to sphere (fractal planet)
	CsgShape->CreateSphere(1.2f);
	MainShape->IntersectWith(CsgShape);

	// Step 4: Orbital torus ring
	CsgShape->CreateTorus(1.6f, 0.04f);
	CsgShape->RotateSdf(FRotator(15.0f, 0.0f, 10.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.05f);

	// Step 5: Outer onion shell (hollow sphere)
	CsgShape->CreateSphere(1.8f);
	CsgShape->ApplyOnion(0.02f);
	MainShape->SmoothUnionWith(CsgShape, 0.03f);
}

// ============================================================================
// Shape 9: Abstract Minimal Surfaces — Gyroid + Schwarz P combined
// (Ported from Unity Abstract Demo)
// ============================================================================
void AAliceSdfNaniteActor::BuildAbstractSurface()
{
	// Gyroid surface
	MainShape->CreateGyroid(3.0f, 0.04f);

	// Combine with Schwarz P
	CsgShape->CreateSchwarzP(3.0f, 0.04f);
	MainShape->SmoothUnionWith(CsgShape, 0.08f);

	// Clip to sphere
	CsgShape->CreateSphere(1.0f);
	MainShape->IntersectWith(CsgShape);

	// Orbiting torus
	CsgShape->CreateTorus(1.3f, 0.06f);
	CsgShape->RotateSdf(FRotator(30.0f, 0.0f, 20.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.08f);

	// Metaball accents
	CsgShape->CreateSphere(0.25f);
	CsgShape->TranslateSdf(FVector(0.9f, 0.5f, 0.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.15f);

	CsgShape->CreateSphere(0.18f);
	CsgShape->TranslateSdf(FVector(-0.7f, -0.6f, 0.4f));
	MainShape->SmoothUnionWith(CsgShape, 0.15f);
}

// ============================================================================
// Shape 10: SDF Cathedral — Dome + arched windows + columns (Lumen GI demo)
// ============================================================================
void AAliceSdfNaniteActor::BuildCathedral()
{
	// Outer dome (hemisphere)
	MainShape->CreateSphere(1.5f);

	// Hollow interior
	CsgShape->CreateSphere(1.35f);
	MainShape->SubtractFrom(CsgShape);

	// Cut bottom half to make dome shape
	CsgShape->CreateBox(FVector(2.0f, 2.0f, 1.5f));
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -1.5f));
	MainShape->SubtractFrom(CsgShape);

	// Arched windows (capsule shapes polar-repeated)
	CsgShape->CreateCapsule(FVector(0.0f, 0.0f, -0.15f), FVector(0.0f, 0.0f, 0.15f), 0.1f);
	CsgShape->TranslateSdf(FVector(1.43f, 0.0f, 0.5f));
	CsgShape->ApplyPolarRepeat(8);
	MainShape->SubtractFrom(CsgShape);

	// Rose window at top (smaller holes near apex)
	CsgShape->CreateSphere(0.06f);
	CsgShape->TranslateSdf(FVector(1.43f, 0.0f, 0.95f));
	CsgShape->ApplyPolarRepeat(8);
	MainShape->SubtractFrom(CsgShape);

	// Support columns at base
	CsgShape->CreateCylinder(0.04f, 0.55f);
	CsgShape->TranslateSdf(FVector(1.35f, 0.0f, 0.0f));
	CsgShape->ApplyPolarRepeat(16);
	MainShape->SmoothUnionWith(CsgShape, 0.02f);

	// Floor disc
	CsgShape->CreateCylinder(1.45f, 0.02f);
	MainShape->UnionWith(CsgShape);

	// Entrance arch (cutout on one side)
	CsgShape->CreateCapsule(FVector(0.0f, 0.0f, -0.3f), FVector(0.0f, 0.0f, 0.3f), 0.2f);
	CsgShape->TranslateSdf(FVector(1.43f, 0.0f, 0.15f));
	MainShape->SubtractFrom(CsgShape);
}

// ============================================================================
// Shape 11: Coral Reef — Triple TPMS for extreme Nanite density
// ============================================================================
void AAliceSdfNaniteActor::BuildCoralReef()
{
	// Gyroid base
	MainShape->CreateGyroid(5.0f, 0.02f);

	// Layer Schwarz P
	CsgShape->CreateSchwarzP(4.0f, 0.025f);
	MainShape->SmoothUnionWith(CsgShape, 0.04f);

	// Layer Diamond surface
	CsgShape->CreateDiamondSurface(3.5f, 0.02f);
	MainShape->SmoothUnionWith(CsgShape, 0.04f);

	// Clip to organic blob
	CsgShape->CreateSphere(0.9f);
	MainShape->IntersectWith(CsgShape);

	// Surface noise for organic feel
	MainShape->ApplyNoise(0.02f, 6.0f, 7);

	// Rock base
	CsgShape->CreateSphere(0.5f);
	CsgShape->TranslateSdf(FVector(0.0f, 0.0f, -0.7f));
	CsgShape->ApplyNoise(0.05f, 3.0f, 13);
	MainShape->SmoothUnionWith(CsgShape, 0.1f);
}

// ============================================================================
// Scene Objects — Cosmic
// ============================================================================

void AAliceSdfNaniteActor::BuildCosmicSun()
{
	// Turbulent star surface
	MainShape->CreateSphere(1.0f);
	MainShape->ApplyNoise(0.12f, 3.0f, 42);

	// Solar prominences (small spheres on surface)
	CsgShape->CreateSphere(0.25f);
	CsgShape->TranslateSdf(FVector(0.9f, 0.2f, 0.3f));
	MainShape->SmoothUnionWith(CsgShape, 0.3f);

	CsgShape->CreateSphere(0.2f);
	CsgShape->TranslateSdf(FVector(-0.3f, 0.85f, -0.2f));
	MainShape->SmoothUnionWith(CsgShape, 0.25f);

	CsgShape->CreateSphere(0.18f);
	CsgShape->TranslateSdf(FVector(0.1f, -0.4f, 0.8f));
	MainShape->SmoothUnionWith(CsgShape, 0.25f);
}

void AAliceSdfNaniteActor::BuildCosmicRingedPlanet()
{
	// Planet body
	MainShape->CreateSphere(0.6f);

	// Surface detail
	MainShape->ApplyNoise(0.02f, 5.0f, 7);

	// Tilted ring (Saturn-like)
	CsgShape->CreateTorus(1.0f, 0.03f);
	CsgShape->RotateSdf(FRotator(20.0f, 0.0f, 10.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.05f);

	// Outer ring
	CsgShape->CreateTorus(1.2f, 0.02f);
	CsgShape->RotateSdf(FRotator(20.0f, 0.0f, 10.0f));
	MainShape->SmoothUnionWith(CsgShape, 0.03f);
}

void AAliceSdfNaniteActor::BuildCosmicAsteroid()
{
	// Irregular rocky shape
	MainShape->CreateOctahedron(0.6f);
	MainShape->ApplyNoise(0.1f, 4.0f, 13);

	// Smaller fragment
	CsgShape->CreateSphere(0.3f);
	CsgShape->TranslateSdf(FVector(0.4f, 0.2f, 0.1f));
	CsgShape->ApplyNoise(0.08f, 5.0f, 7);
	MainShape->SmoothUnionWith(CsgShape, 0.08f);
}

// ============================================================================
// Scene Objects — Terrain
// ============================================================================

void AAliceSdfNaniteActor::BuildTerrainGround()
{
	// Flat slab as terrain base
	MainShape->CreateBox(FVector(1.5f, 1.5f, 0.15f));

	// Terrain hills via noise displacement
	MainShape->ApplyNoise(0.15f, 2.0f, 42);

	// Second layer of finer detail
	MainShape->ApplyNoise(0.04f, 6.0f, 7);
}

void AAliceSdfNaniteActor::BuildTerrainWater()
{
	// Thin flat disc for water surface
	MainShape->CreateCylinder(1.5f, 0.01f);

	// Subtle ripple
	MainShape->ApplyNoise(0.005f, 8.0f, 3);
}

void AAliceSdfNaniteActor::BuildTerrainRock()
{
	// Rounded rock
	MainShape->CreateSphere(0.5f);
	MainShape->ApplyNoise(0.08f, 3.0f, 17);

	// Slight squash
	CsgShape->CreateBox(FVector(1.0f, 1.0f, 0.6f));
	MainShape->IntersectWith(CsgShape);
}

// ============================================================================
// Scene Objects — Floating Island (Unity Terrain: 3 floating islands)
// ============================================================================

void AAliceSdfNaniteActor::BuildFloatingIsland()
{
	// Flattened sphere (ip.y *= 3.0 in Unity — vertically squashed)
	MainShape->CreateEllipsoid(FVector(1.0f, 0.35f, 1.0f));

	// FBM-like terrain noise (2 octaves of noise like Unity)
	MainShape->ApplyNoise(0.1f, 3.0f, 42);
	MainShape->ApplyNoise(0.03f, 7.0f, 7);
}

// ============================================================================
// Scene Objects — Bounded Gyroid (Unity Abstract: center piece)
// ============================================================================

void AAliceSdfNaniteActor::BuildBoundedGyroid()
{
	// Unity: max(sdGyroid(p, _GyroidScale, _GyroidThickness), sdSphere(p, 8.0))
	// Gyroid bounded by sphere intersection
	MainShape->CreateGyroid(3.0f, 0.04f);

	// Clip to sphere
	CsgShape->CreateSphere(1.0f);
	MainShape->IntersectWith(CsgShape);
}

// ============================================================================
// Scene Objects — Metaball (Unity Abstract: 6 morphing metaballs)
// ============================================================================

void AAliceSdfNaniteActor::BuildMetaball()
{
	// Unity: lerp(sdSphere, sdOctahedron, morphT)
	// Approximate morphed sphere with organic noise
	MainShape->CreateSphere(0.5f);

	// Octahedral influence (smooth union for morphing feel)
	CsgShape->CreateOctahedron(0.45f);
	MainShape->SmoothUnionWith(CsgShape, 0.15f);

	// Subtle organic surface
	MainShape->ApplyNoise(0.015f, 5.0f, 7);
}

// ============================================================================
// Scene Objects — Schwarz P Corner (Unity Abstract: 4 corners)
// ============================================================================

void AAliceSdfNaniteActor::BuildSchwarzPCorner()
{
	// Unity: max(sdSchwarzP(p, scale, thickness), sdSphere(p, 5.0))
	// Schwarz P surface clipped to sphere
	MainShape->CreateSchwarzP(3.0f, 0.04f);

	CsgShape->CreateSphere(0.8f);
	MainShape->IntersectWith(CsgShape);
}

// ============================================================================
// Scene Objects — Floating Cube (Unity Abstract: 8 rotating cubes)
// ============================================================================

void AAliceSdfNaniteActor::BuildFloatingCube()
{
	// Unity: sdBox(rp, vec3(size)) — simple rotating box
	MainShape->CreateBox(FVector(0.4f, 0.4f, 0.4f));

	// Slight surface noise for visual interest
	MainShape->ApplyNoise(0.01f, 6.0f, 13);
}

// ============================================================================
// Scene Objects — Abstract Ring (Unity Abstract: 3 rotating torus rings)
// ============================================================================

void AAliceSdfNaniteActor::BuildAbstractRing()
{
	// Unity: sdTorus(rp, vec2(majorR, minorR))
	MainShape->CreateTorus(1.0f, 0.05f);
}

// ============================================================================
// OBJ Parser
// ============================================================================

bool AAliceSdfNaniteActor::ParseObjFile(
	const FString& Path,
	TArray<FVector3f>& OutPositions,
	TArray<FVector3f>& OutNormals,
	TArray<uint32>& OutPosIndices,
	TArray<uint32>& OutNormIndices)
{
	TArray<FString> Lines;
	if (!FFileHelper::LoadFileToStringArray(Lines, *Path))
	{
		return false;
	}

	OutPositions.Reserve(Lines.Num() / 3);
	OutNormals.Reserve(Lines.Num() / 3);

	for (const FString& Line : Lines)
	{
		if (Line.IsEmpty() || Line[0] == TEXT('#'))
		{
			continue;
		}

		TArray<FString> Tokens;
		Line.ParseIntoArrayWS(Tokens);
		if (Tokens.Num() < 2) continue;

		if (Tokens[0] == TEXT("v") && Tokens.Num() >= 4)
		{
			// Vertex position
			OutPositions.Add(FVector3f(
				FCString::Atof(*Tokens[1]),
				FCString::Atof(*Tokens[2]),
				FCString::Atof(*Tokens[3])
			));
		}
		else if (Tokens[0] == TEXT("vn") && Tokens.Num() >= 4)
		{
			// Vertex normal
			OutNormals.Add(FVector3f(
				FCString::Atof(*Tokens[1]),
				FCString::Atof(*Tokens[2]),
				FCString::Atof(*Tokens[3])
			));
		}
		else if (Tokens[0] == TEXT("f") && Tokens.Num() >= 4)
		{
			// Face: f v1 v2 v3 ... or f v1//vn1 v2//vn2 ... or f v1/vt1/vn1 ...
			TArray<uint32> FacePosIdx;
			TArray<uint32> FaceNormIdx;

			for (int32 i = 1; i < Tokens.Num(); i++)
			{
				TArray<FString> Parts;
				Tokens[i].ParseIntoArray(Parts, TEXT("/"), false);

				if (Parts.Num() >= 1 && !Parts[0].IsEmpty())
				{
					FacePosIdx.Add(FCString::Atoi(*Parts[0]) - 1); // OBJ is 1-indexed
				}
				if (Parts.Num() >= 3 && !Parts[2].IsEmpty())
				{
					FaceNormIdx.Add(FCString::Atoi(*Parts[2]) - 1);
				}
			}

			// Fan triangulation for polygons with 4+ vertices
			const bool bFaceHasNormals = (FaceNormIdx.Num() == FacePosIdx.Num());
			for (int32 i = 1; i < FacePosIdx.Num() - 1; i++)
			{
				OutPosIndices.Add(FacePosIdx[0]);
				OutPosIndices.Add(FacePosIdx[i]);
				OutPosIndices.Add(FacePosIdx[i + 1]);

				if (bFaceHasNormals)
				{
					OutNormIndices.Add(FaceNormIdx[0]);
					OutNormIndices.Add(FaceNormIdx[i]);
					OutNormIndices.Add(FaceNormIdx[i + 1]);
				}
			}
		}
	}

	return OutPositions.Num() > 0 && OutPosIndices.Num() >= 3;
}

// ============================================================================
// CreateNaniteStaticMesh — The core pipeline
// ============================================================================

UStaticMesh* AAliceSdfNaniteActor::CreateNaniteStaticMesh()
{
	// --- Step 1: Export SDF to temp OBJ ---
	const FString TempDir = FPaths::ProjectSavedDir() / TEXT("AliceSDF_Temp");
	IPlatformFile& PlatformFile = FPlatformFileManager::Get().GetPlatformFile();
	PlatformFile.CreateDirectory(*TempDir);

	const FString ObjPath = TempDir / FString::Printf(TEXT("nanite_%s.obj"), *GetName());

	if (!MainShape->ExportObj(ObjPath, MeshResolution, Bounds))
	{
		UE_LOG(LogTemp, Error, TEXT("ALICE-SDF: Failed to export OBJ to %s"), *ObjPath);
		return nullptr;
	}

	// --- Step 2: Parse OBJ ---
	TArray<FVector3f> Positions;
	TArray<FVector3f> Normals;
	TArray<uint32> PosIndices;
	TArray<uint32> NormIndices;

	if (!ParseObjFile(ObjPath, Positions, Normals, PosIndices, NormIndices))
	{
		UE_LOG(LogTemp, Error, TEXT("ALICE-SDF: Failed to parse OBJ"));
		PlatformFile.DeleteFile(*ObjPath);
		return nullptr;
	}

	VertexCount = Positions.Num();
	TriangleCount = PosIndices.Num() / 3;

	if (TriangleCount == 0)
	{
		PlatformFile.DeleteFile(*ObjPath);
		return nullptr;
	}

	const bool bHasNormals = Normals.Num() > 0 && NormIndices.Num() == PosIndices.Num();

	// --- Step 3: Build FMeshDescription ---
	UStaticMesh* StaticMesh = NewObject<UStaticMesh>(this, NAME_None, RF_Transient);
	StaticMesh->GetStaticMaterials().Add(FStaticMaterial());
	StaticMesh->SetNumSourceModels(1);

	FMeshDescription* MeshDesc = StaticMesh->CreateMeshDescription(0);
	FStaticMeshAttributes Attrs(*MeshDesc);
	Attrs.Register();

	// Reserve space
	MeshDesc->ReserveNewVertices(Positions.Num());
	MeshDesc->ReserveNewVertexInstances(PosIndices.Num());
	MeshDesc->ReserveNewPolygons(TriangleCount);

	// Vertex positions
	TArray<FVertexID> VertexIDs;
	VertexIDs.Reserve(Positions.Num());
	TVertexAttributesRef<FVector3f> VertexPositions = Attrs.GetVertexPositions();

	for (const FVector3f& Pos : Positions)
	{
		FVertexID VID = MeshDesc->CreateVertex();
		VertexPositions[VID] = Pos;
		VertexIDs.Add(VID);
	}

	// Polygon group
	FPolygonGroupID PolyGroupID = MeshDesc->CreatePolygonGroup();

	// Attribute references
	TVertexInstanceAttributesRef<FVector3f> VertNormals = Attrs.GetVertexInstanceNormals();
	TVertexInstanceAttributesRef<FVector2f> VertUVs = Attrs.GetVertexInstanceUVs();

	// Create triangles with vertex instances
	for (int32 i = 0; i < PosIndices.Num(); i += 3)
	{
		TArray<FVertexInstanceID> TriVerts;
		TriVerts.Reserve(3);
		bool bValid = true;

		for (int32 j = 0; j < 3; j++)
		{
			const uint32 PIdx = PosIndices[i + j];
			if (PIdx >= static_cast<uint32>(VertexIDs.Num()))
			{
				bValid = false;
				break;
			}

			FVertexInstanceID VIID = MeshDesc->CreateVertexInstance(VertexIDs[PIdx]);

			// Normal
			if (bHasNormals)
			{
				const uint32 NIdx = NormIndices[i + j];
				if (NIdx < static_cast<uint32>(Normals.Num()))
				{
					VertNormals[VIID] = Normals[NIdx];
				}
			}

			// Box-projected UV from position and normal
			const FVector3f& Pos = Positions[PIdx];
			FVector3f N = bHasNormals && NormIndices[i + j] < static_cast<uint32>(Normals.Num())
				? Normals[NormIndices[i + j]]
				: FVector3f(0.0f, 0.0f, 1.0f);
			const FVector3f AbsN(FMath::Abs(N.X), FMath::Abs(N.Y), FMath::Abs(N.Z));

			FVector2f UV;
			if (AbsN.X >= AbsN.Y && AbsN.X >= AbsN.Z)
				UV = FVector2f(Pos.Y, Pos.Z);
			else if (AbsN.Y >= AbsN.X && AbsN.Y >= AbsN.Z)
				UV = FVector2f(Pos.X, Pos.Z);
			else
				UV = FVector2f(Pos.X, Pos.Y);
			VertUVs[VIID] = UV * 0.5f + FVector2f(0.5f, 0.5f);

			TriVerts.Add(VIID);
		}

		if (bValid && TriVerts.Num() == 3)
		{
			MeshDesc->CreatePolygon(PolyGroupID, TriVerts);
		}
	}

	StaticMesh->CommitMeshDescription(0);

	// --- Step 4: Build settings ---
	FStaticMeshSourceModel& SrcModel = StaticMesh->GetSourceModel(0);
	SrcModel.BuildSettings.bRecomputeNormals = !bHasNormals;
	SrcModel.BuildSettings.bRecomputeTangents = true;
	SrcModel.BuildSettings.bGenerateLightmapUVs = true;

	// --- Step 5: Enable Nanite (Windows/Linux only; macOS Metal lacks support) ---
#if PLATFORM_MAC
	StaticMesh->NaniteSettings.bEnabled = false;
#else
	StaticMesh->NaniteSettings.bEnabled = true;
#endif

	// --- Step 6: Build (generates Nanite data, LODs, etc.) ---
	StaticMesh->Build(false);
	StaticMesh->CreateBodySetup();

	// --- Step 7: Cleanup temp file ---
	PlatformFile.DeleteFile(*ObjPath);

	return StaticMesh;
}
