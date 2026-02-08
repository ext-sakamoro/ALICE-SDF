// ALICE-SDF Demo Actor Implementation
// Author: Moroya Sakamoto

#include "AliceSdfDemoActor.h"

AAliceSdfDemoActor::AAliceSdfDemoActor()
{
	PrimaryActorTick.bCanEverTick = false;

	MainShape = CreateDefaultSubobject<UAliceSdfComponent>(TEXT("MainShape"));
	CsgShape = CreateDefaultSubobject<UAliceSdfComponent>(TEXT("CsgShape"));
}

void AAliceSdfDemoActor::BeginPlay()
{
	Super::BeginPlay();

	UE_LOG(LogTemp, Log, TEXT("=== ALICE-SDF Demo Actor (Demo %d) ==="), DemoIndex);

	switch (DemoIndex)
	{
	case 0: RunDemo0_BasicShapes(); break;
	case 1: RunDemo1_CSGOperations(); break;
	case 2: RunDemo2_TPMS(); break;
	case 3: RunDemo3_Modifiers(); break;
	case 4: RunDemo4_ShaderGeneration(); break;
	default:
		UE_LOG(LogTemp, Warning, TEXT("Unknown demo index %d. Set DemoIndex to 0-4."), DemoIndex);
		RunDemo0_BasicShapes();
		break;
	}
}

// ============================================================================
// Demo 0: Basic Shapes — Create and evaluate primitives
// ============================================================================
void AAliceSdfDemoActor::RunDemo0_BasicShapes()
{
	UE_LOG(LogTemp, Log, TEXT("--- Demo 0: Basic Shapes ---"));

	// Sphere
	MainShape->CreateSphere(1.0f);
	float d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Sphere(r=1) at origin: %.4f (expected: -1.0)"), d);

	d = MainShape->EvalDistanceLocal(FVector(1, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Sphere(r=1) at surface: %.4f (expected: 0.0)"), d);

	// Box
	MainShape->CreateBox(FVector(0.5f, 0.5f, 0.5f));
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Box(0.5) at origin: %.4f (expected: -0.5)"), d);

	// Torus
	MainShape->CreateTorus(1.0f, 0.25f);
	d = MainShape->EvalDistanceLocal(FVector(1, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Torus at (1,0,0): %.4f (expected: -0.25)"), d);

	// Heart
	MainShape->CreateHeart(1.0f);
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Heart at origin: %.4f (expected: <0)"), d);

	// BoxFrame
	MainShape->CreateBoxFrame(FVector(1.0f, 1.0f, 1.0f), 0.1f);
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("BoxFrame at origin: %.4f (expected: >0, hollow)"), d);

	UE_LOG(LogTemp, Log, TEXT("Node count: %d"), MainShape->NodeCount);
	UE_LOG(LogTemp, Log, TEXT("Is compiled: %s"), MainShape->bIsCompiled ? TEXT("true") : TEXT("false"));
}

// ============================================================================
// Demo 1: CSG Operations — Combine shapes with boolean operations
// ============================================================================
void AAliceSdfDemoActor::RunDemo1_CSGOperations()
{
	UE_LOG(LogTemp, Log, TEXT("--- Demo 1: CSG Operations ---"));

	// Sphere - Box with smooth subtraction
	MainShape->CreateSphere(1.0f);
	CsgShape->CreateBox(FVector(0.7f, 0.7f, 0.7f));

	UE_LOG(LogTemp, Log, TEXT("Before CSG: MainShape nodes=%d"), MainShape->NodeCount);

	MainShape->SmoothSubtractFrom(CsgShape, 0.1f);
	UE_LOG(LogTemp, Log, TEXT("After SmoothSubtract: nodes=%d"), MainShape->NodeCount);

	float d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Sphere-Box at origin: %.4f"), d);

	// Chamfer union
	MainShape->CreateSphere(1.0f);
	CsgShape->CreateSphere(0.8f);
	CsgShape->TranslateSdf(FVector(1.2f, 0, 0));
	MainShape->ChamferUnionWith(CsgShape, 0.15f);
	UE_LOG(LogTemp, Log, TEXT("ChamferUnion nodes=%d"), MainShape->NodeCount);

	// XOR
	MainShape->CreateSphere(1.0f);
	CsgShape->CreateBox(FVector(0.8f, 0.8f, 0.8f));
	MainShape->XorWith(CsgShape);
	UE_LOG(LogTemp, Log, TEXT("XOR nodes=%d"), MainShape->NodeCount);

	// Morph
	MainShape->CreateSphere(1.0f);
	CsgShape->CreateBox(FVector(0.8f, 0.8f, 0.8f));
	MainShape->MorphWith(CsgShape, 0.5f);
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Morph(0.5) at origin: %.4f"), d);
}

// ============================================================================
// Demo 2: TPMS — Triply Periodic Minimal Surfaces
// ============================================================================
void AAliceSdfDemoActor::RunDemo2_TPMS()
{
	UE_LOG(LogTemp, Log, TEXT("--- Demo 2: TPMS Surfaces ---"));

	// Gyroid
	MainShape->CreateGyroid(2.0f, 0.05f);
	float d = MainShape->EvalDistanceLocal(FVector(0.5f, 0.5f, 0.5f));
	UE_LOG(LogTemp, Log, TEXT("Gyroid(scale=2, t=0.05) at (0.5,0.5,0.5): %.4f"), d);

	// Schwarz P
	MainShape->CreateSchwarzP(2.0f, 0.05f);
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("SchwarzP at origin: %.4f"), d);

	// Diamond Surface
	MainShape->CreateDiamondSurface(2.0f, 0.05f);
	UE_LOG(LogTemp, Log, TEXT("DiamondSurface created, nodes=%d"), MainShape->NodeCount);

	// Neovius
	MainShape->CreateNeovius(2.0f, 0.05f);
	UE_LOG(LogTemp, Log, TEXT("Neovius created, nodes=%d"), MainShape->NodeCount);

	// Lidinoid
	MainShape->CreateLidinoid(2.0f, 0.05f);
	UE_LOG(LogTemp, Log, TEXT("Lidinoid created, nodes=%d"), MainShape->NodeCount);

	// All 9 TPMS available: Gyroid, SchwarzP, DiamondSurface,
	// Neovius, Lidinoid, IWP, FRD, FischerKochS, PMY
	UE_LOG(LogTemp, Log, TEXT("All 9 TPMS surfaces available via CreateXxx()"));
}

// ============================================================================
// Demo 3: Modifiers — Transform and modify shapes
// ============================================================================
void AAliceSdfDemoActor::RunDemo3_Modifiers()
{
	UE_LOG(LogTemp, Log, TEXT("--- Demo 3: Modifiers ---"));

	// Twisted box
	MainShape->CreateBox(FVector(0.5f, 0.5f, 2.0f));
	MainShape->ApplyTwist(2.0f);
	float d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Twisted box at origin: %.4f"), d);

	// Onion sphere (shell)
	MainShape->CreateSphere(1.0f);
	MainShape->ApplyOnion(0.05f);
	d = MainShape->EvalDistanceLocal(FVector(0, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Onion sphere at origin: %.4f (expected: >0, hollow)"), d);

	// Repeated cylinder
	MainShape->CreateCylinder(0.2f, 0.5f);
	MainShape->ApplyRepeatFinite(FIntVector(3, 3, 1), FVector(1.0f, 1.0f, 1.0f));
	UE_LOG(LogTemp, Log, TEXT("Repeated cylinder nodes=%d"), MainShape->NodeCount);

	// Polar repeat torus
	MainShape->CreateBox(FVector(0.1f, 0.1f, 0.5f));
	MainShape->TranslateSdf(FVector(1.0f, 0, 0));
	MainShape->ApplyPolarRepeat(8);
	UE_LOG(LogTemp, Log, TEXT("Polar repeat (8x) nodes=%d"), MainShape->NodeCount);

	// Octant mirror
	MainShape->CreateSphere(0.3f);
	MainShape->TranslateSdf(FVector(1.0f, 0.5f, 0.2f));
	MainShape->ApplyOctantMirror();
	UE_LOG(LogTemp, Log, TEXT("Octant mirror nodes=%d"), MainShape->NodeCount);

	// Noise displacement
	MainShape->CreateSphere(1.0f);
	MainShape->ApplyNoise(0.1f, 5.0f, 42);
	d = MainShape->EvalDistanceLocal(FVector(1.0f, 0, 0));
	UE_LOG(LogTemp, Log, TEXT("Noisy sphere at surface: %.4f"), d);
}

// ============================================================================
// Demo 4: Shader Generation — Generate HLSL for Custom Material Expression
// ============================================================================
void AAliceSdfDemoActor::RunDemo4_ShaderGeneration()
{
	UE_LOG(LogTemp, Log, TEXT("--- Demo 4: Shader Generation ---"));

	// Create a shape
	MainShape->CreateSphere(1.0f);
	CsgShape->CreateBox(FVector(0.7f, 0.7f, 0.7f));
	MainShape->SmoothSubtractFrom(CsgShape, 0.1f);
	MainShape->ApplyTwist(1.0f);

	// Generate HLSL
	FString Hlsl = MainShape->GenerateHlsl();
	if (Hlsl.Len() > 0)
	{
		UE_LOG(LogTemp, Log, TEXT("HLSL generated: %d characters"), Hlsl.Len());
		UE_LOG(LogTemp, Log, TEXT("First 200 chars:\n%s"), *Hlsl.Left(200));
		UE_LOG(LogTemp, Log, TEXT(""));
		UE_LOG(LogTemp, Log, TEXT("To use in UE5:"));
		UE_LOG(LogTemp, Log, TEXT("  1. Create a Material"));
		UE_LOG(LogTemp, Log, TEXT("  2. Add a Custom node"));
		UE_LOG(LogTemp, Log, TEXT("  3. Paste the HLSL code above"));
		UE_LOG(LogTemp, Log, TEXT("  4. Connect WorldPosition / 100.0 as input (cm -> m)"));
		UE_LOG(LogTemp, Log, TEXT("  5. Output to Opacity Mask"));
	}
	else
	{
		UE_LOG(LogTemp, Warning, TEXT("HLSL generation failed. Ensure 'hlsl' feature is enabled."));
	}

	// Mesh export demo
	FString ExportPath = FPaths::ProjectSavedDir() / TEXT("AliceSDF_Demo.obj");
	if (MainShape->ExportObj(ExportPath, MeshResolution, 2.0f))
	{
		UE_LOG(LogTemp, Log, TEXT("Mesh exported to: %s"), *ExportPath);
	}
}
