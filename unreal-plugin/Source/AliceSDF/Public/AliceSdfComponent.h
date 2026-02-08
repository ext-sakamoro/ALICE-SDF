// ALICE-SDF UE5 Component
// Author: Moroya Sakamoto

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "alice_sdf.h"
#include "AliceSdfComponent.generated.h"

/**
 * UAliceSdfComponent
 *
 * Full-featured SDF component for Unreal Engine 5.
 * 66 primitives, 24 CSG operations, 17 modifiers, HLSL generation, mesh export.
 *
 * Usage:
 *   1. Add this component to any Actor
 *   2. Call CreateSphere/CreateBox/etc. to define the SDF shape
 *   3. Use CSG operations (UnionWith, SmoothSubtractFrom, etc.) to combine shapes
 *   4. Call Compile() for fast evaluation (auto-compiled by default)
 *   5. Use EvalDistance() for collision or GenerateHlsl() for materials
 */
UCLASS(ClassGroup=(Rendering), meta=(BlueprintSpawnableComponent))
class ALICESDF_API UAliceSdfComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UAliceSdfComponent();

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

public:
	// ========================================================================
	// Primitives — Basic
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateSphere(float Radius = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateBox(FVector HalfExtents);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCylinder(float Radius = 1.0f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateTorus(float MajorRadius = 1.0f, float MinorRadius = 0.25f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCapsule(FVector PointA, FVector PointB, float Radius = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreatePlane(FVector Normal, float Distance = 0.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCone(float Radius = 1.0f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateEllipsoid(FVector Radii);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateRoundedCone(float R1 = 0.5f, float R2 = 0.2f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreatePyramid(float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateOctahedron(float Size = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateHexPrism(float HexRadius = 1.0f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateLink(float HalfLength = 0.5f, float R1 = 0.5f, float R2 = 0.15f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateRoundedBox(FVector HalfExtents, float RoundRadius = 0.1f);

	// ========================================================================
	// Primitives — Advanced
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCappedCone(float HalfHeight = 1.0f, float R1 = 0.5f, float R2 = 0.2f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCappedTorus(float MajorRadius = 1.0f, float MinorRadius = 0.25f, float CapAngle = 1.57f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateRoundedCylinder(float Radius = 0.5f, float RoundRadius = 0.1f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateTriangularPrism(float Width = 1.0f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCutSphere(float Radius = 1.0f, float CutHeight = 0.3f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateDeathStar(float Ra = 1.0f, float Rb = 0.8f, float D = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateHeart(float Size = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateBarrel(float Radius = 0.5f, float HalfHeight = 1.0f, float Bulge = 0.2f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateDiamond(float Radius = 0.5f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateEgg(float Ra = 0.5f, float Rb = 0.3f);

	// ========================================================================
	// Primitives — Platonic & Archimedean
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|Platonic")
	void CreateTetrahedron(float Size = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|Platonic")
	void CreateDodecahedron(float Radius = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|Platonic")
	void CreateIcosahedron(float Radius = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|Platonic")
	void CreateTruncatedOctahedron(float Radius = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|Platonic")
	void CreateTruncatedIcosahedron(float Radius = 1.0f);

	// ========================================================================
	// Primitives — TPMS (Triply Periodic Minimal Surfaces)
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateGyroid(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateSchwarzP(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateDiamondSurface(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateNeovius(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateLidinoid(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateIWP(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateFRD(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreateFischerKochS(float Scale = 1.0f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|TPMS")
	void CreatePMY(float Scale = 1.0f, float Thickness = 0.1f);

	// ========================================================================
	// Primitives — Structural
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateBoxFrame(FVector HalfExtents, float Edge = 0.05f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateTube(float OuterRadius = 0.5f, float Thickness = 0.1f, float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateChamferedCube(FVector HalfExtents, float Chamfer = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateStairs(float StepWidth = 0.3f, float StepHeight = 0.2f, float NumSteps = 5.0f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateHelix(float MajorRadius = 1.0f, float MinorRadius = 0.1f, float Pitch = 0.5f, float HalfHeight = 2.0f);

	// ========================================================================
	// Primitives — 2D/Extruded & Additional
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateTriangle(FVector A, FVector B, FVector C);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateBezier(FVector A, FVector B, FVector C, float Radius = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCutHollowSphere(float Radius = 1.0f, float CutHeight = 0.3f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateSolidAngle(float Angle = 0.5f, float Radius = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateRhombus(float La = 0.5f, float Lb = 0.3f, float HalfHeight = 0.5f, float RoundRadius = 0.05f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateHorseshoe(float Angle = 1.0f, float Radius = 0.5f, float HalfLength = 0.3f, float Width = 0.1f, float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateVesica(float Radius = 0.5f, float HalfDist = 0.3f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateInfiniteCylinder(float Radius = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateInfiniteCone(float Angle = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateSuperEllipsoid(FVector HalfExtents, float E1 = 1.0f, float E2 = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateRoundedX(float Width = 0.5f, float RoundRadius = 0.1f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreatePie(float Angle = 1.0f, float Radius = 1.0f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateTrapezoid(float R1 = 0.5f, float R2 = 0.3f, float TrapHeight = 0.5f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateParallelogram(float Width = 0.5f, float ParaHeight = 0.5f, float Skew = 0.2f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateTunnel(float Width = 0.5f, float Height2D = 0.5f, float HalfDepth = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateUnevenCapsule(float R1 = 0.3f, float R2 = 0.15f, float CapHeight = 0.5f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateArcShape(float Aperture = 1.0f, float Radius = 0.5f, float Thickness = 0.1f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateMoon(float D = 0.5f, float Ra = 0.5f, float Rb = 0.3f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateCrossShape(float Length = 0.5f, float Thickness = 0.2f, float RoundRadius = 0.05f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateBlobbyCross(float Size = 0.5f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateParabolaSegment(float Width = 0.5f, float ParaHeight = 0.5f, float HalfDepth = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateRegularPolygon(float Radius = 0.5f, float NSides = 6.0f, float HalfHeight = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives|2D")
	void CreateStarPolygon(float Radius = 0.5f, float NPoints = 5.0f, float M = 2.0f, float HalfHeight = 0.5f);

	// ========================================================================
	// Boolean Operations — Standard
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void UnionWith(UAliceSdfComponent* Other);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void IntersectWith(UAliceSdfComponent* Other);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void SubtractFrom(UAliceSdfComponent* Other);

	// ========================================================================
	// Boolean Operations — Smooth
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Smooth")
	void SmoothUnionWith(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Smooth")
	void SmoothIntersectWith(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Smooth")
	void SmoothSubtractFrom(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	// ========================================================================
	// Boolean Operations — Chamfer
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Chamfer")
	void ChamferUnionWith(UAliceSdfComponent* Other, float Radius = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Chamfer")
	void ChamferIntersectWith(UAliceSdfComponent* Other, float Radius = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Chamfer")
	void ChamferSubtractFrom(UAliceSdfComponent* Other, float Radius = 0.1f);

	// ========================================================================
	// Boolean Operations — Stairs (terraced blend)
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Stairs")
	void StairsUnionWith(UAliceSdfComponent* Other, float Radius = 0.2f, float Steps = 4.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Stairs")
	void StairsIntersectWith(UAliceSdfComponent* Other, float Radius = 0.2f, float Steps = 4.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Stairs")
	void StairsSubtractFrom(UAliceSdfComponent* Other, float Radius = 0.2f, float Steps = 4.0f);

	// ========================================================================
	// Boolean Operations — Columns
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Columns")
	void ColumnsUnionWith(UAliceSdfComponent* Other, float Radius = 0.2f, float Count = 4.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Columns")
	void ColumnsIntersectWith(UAliceSdfComponent* Other, float Radius = 0.2f, float Count = 4.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations|Columns")
	void ColumnsSubtractFrom(UAliceSdfComponent* Other, float Radius = 0.2f, float Count = 4.0f);

	// ========================================================================
	// Boolean Operations — Advanced
	// ========================================================================

	/** XOR (symmetric difference) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void XorWith(UAliceSdfComponent* Other);

	/** Morph (linear interpolation between two shapes) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void MorphWith(UAliceSdfComponent* Other, float T = 0.5f);

	/** Pipe: cylindrical surface at intersection */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void PipeWith(UAliceSdfComponent* Other, float Radius = 0.1f);

	/** Engrave shape B into shape A */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void EngraveWith(UAliceSdfComponent* Other, float Depth = 0.1f);

	/** Groove: cut a groove of shape B into shape A */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void GrooveWith(UAliceSdfComponent* Other, float Ra = 0.2f, float Rb = 0.1f);

	/** Tongue: add a tongue protrusion */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void TongueWith(UAliceSdfComponent* Other, float Ra = 0.2f, float Rb = 0.1f);

	// ========================================================================
	// Transforms
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void TranslateSdf(FVector Offset);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void RotateSdf(FRotator Rotation);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void ScaleSdf(float Factor);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void ScaleSdfNonUniform(FVector Scale);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void RotateEulerSdf(FVector EulerRadians);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void RotateQuatSdf(FQuat Rotation);

	// ========================================================================
	// Modifiers
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRound(float Radius = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyOnion(float Thickness = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyTwist(float Strength = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyBend(float Curvature = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRepeat(FVector Spacing);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRepeatFinite(FIntVector Count, FVector Spacing);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyMirror(bool X = true, bool Y = false, bool Z = false);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyElongate(FVector Amount);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRevolution(float Offset = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyExtrude(float HalfHeight = 1.0f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyNoise(float Amplitude = 0.1f, float Frequency = 5.0f, int32 Seed = 42);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyTaper(float Factor = 0.5f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyDisplacement(float Strength = 0.1f);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyPolarRepeat(int32 Count = 6);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyOctantMirror();

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplySweepBezier(FVector2D P0, FVector2D P1, FVector2D P2);

	/** Assign a material ID to this SDF subtree */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void SetMaterialId(int32 MaterialId = 0);

	// ========================================================================
	// Compilation & Evaluation
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Evaluation")
	bool Compile();

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	float EvalDistance(FVector WorldPosition) const;

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	float EvalDistanceLocal(FVector LocalPosition) const;

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Evaluation")
	TArray<float> EvalDistanceBatch(const TArray<FVector>& Points) const;

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	bool IsPointInside(FVector WorldPosition) const;

	// ========================================================================
	// Shader Generation
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateHlsl() const;

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateGlsl() const;

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateWgsl() const;

	// ========================================================================
	// Mesh Export
	// ========================================================================

	/** Generate mesh and export to OBJ */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Mesh")
	bool ExportObj(const FString& FilePath, int32 Resolution = 128, float Bounds = 2.0f);

	/** Generate mesh and export to GLB (binary glTF) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Mesh")
	bool ExportGlb(const FString& FilePath, int32 Resolution = 128, float Bounds = 2.0f);

	/** Generate mesh and export to USDA (Universal Scene Description) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Mesh")
	bool ExportUsda(const FString& FilePath, int32 Resolution = 128, float Bounds = 2.0f);

	/** Generate mesh and export to FBX */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Mesh")
	bool ExportFbx(const FString& FilePath, int32 Resolution = 128, float Bounds = 2.0f);

	// ========================================================================
	// File I/O
	// ========================================================================

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|IO")
	bool SaveToFile(const FString& FilePath);

	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|IO")
	bool LoadFromFile(const FString& FilePath);

	// ========================================================================
	// Properties
	// ========================================================================

	UPROPERTY(BlueprintReadOnly, Category = "ALICE SDF")
	bool bIsCompiled = false;

	UPROPERTY(BlueprintReadOnly, Category = "ALICE SDF")
	int32 NodeCount = 0;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF")
	bool bAutoCompile = true;

	/** Get the raw SDF handle (for advanced C++ usage) */
	SdfHandle GetSdfHandle() const { return SdfNodeHandle; }

private:
	void FreeHandles();
	void UpdateNodeCount();
	void AutoCompileIfNeeded();
	void SetNewShape(SdfHandle NewHandle);
	void ApplyModifier(SdfHandle NewHandle);
	void ApplyBinaryOp(UAliceSdfComponent* Other, SdfHandle Result);
	FString ShaderResultToString(StringResult Result) const;

	SdfHandle SdfNodeHandle = nullptr;
	CompiledHandle CompiledSdfHandle = nullptr;
};
