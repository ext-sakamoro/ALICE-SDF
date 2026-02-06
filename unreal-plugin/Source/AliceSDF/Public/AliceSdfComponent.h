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
 * Provides SDF evaluation, shader generation, and mesh conversion
 * as an Unreal Engine 5 ActorComponent.
 *
 * Usage:
 *   1. Add this component to any Actor
 *   2. Call CreateSphere/CreateBox/etc. to define the SDF shape
 *   3. Call Compile() for fast evaluation
 *   4. Use EvalDistance() for collision queries or GenerateHlsl() for materials
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
	// Primitives
	// ========================================================================

	/** Create a sphere SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateSphere(float Radius = 1.0f);

	/** Create a box SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateBox(FVector HalfExtents);

	/** Create a cylinder SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCylinder(float Radius = 1.0f, float HalfHeight = 1.0f);

	/** Create a torus SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateTorus(float MajorRadius = 1.0f, float MinorRadius = 0.25f);

	/** Create a capsule SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreateCapsule(FVector PointA, FVector PointB, float Radius = 0.5f);

	/** Create a plane SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Primitives")
	void CreatePlane(FVector Normal, float Distance = 0.0f);

	// ========================================================================
	// Boolean Operations
	// ========================================================================

	/** Union with another SDF component */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void UnionWith(UAliceSdfComponent* Other);

	/** Intersection with another SDF component */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void IntersectWith(UAliceSdfComponent* Other);

	/** Subtract another SDF component */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void SubtractFrom(UAliceSdfComponent* Other);

	/** Smooth union with another SDF component */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void SmoothUnionWith(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	/** Smooth intersection with another SDF component */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void SmoothIntersectWith(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	/** Smooth subtraction */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Operations")
	void SmoothSubtractFrom(UAliceSdfComponent* Other, float Smoothness = 0.2f);

	// ========================================================================
	// Transforms
	// ========================================================================

	/** Translate the SDF */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void TranslateSdf(FVector Offset);

	/** Rotate the SDF by Euler angles (degrees) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void RotateSdf(FRotator Rotation);

	/** Scale the SDF uniformly */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Transforms")
	void ScaleSdf(float Factor);

	// ========================================================================
	// Modifiers
	// ========================================================================

	/** Apply rounding to edges */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRound(float Radius = 0.1f);

	/** Apply shell (onion) modifier */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyOnion(float Thickness = 0.1f);

	/** Apply twist modifier */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyTwist(float Strength = 1.0f);

	/** Apply bend modifier */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyBend(float Curvature = 1.0f);

	/** Apply infinite repetition */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Modifiers")
	void ApplyRepeat(FVector Spacing);

	// ========================================================================
	// Compilation & Evaluation
	// ========================================================================

	/** Compile SDF for fast evaluation (call once at setup time) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Evaluation")
	bool Compile();

	/** Evaluate signed distance at a world-space point */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	float EvalDistance(FVector WorldPosition) const;

	/** Evaluate signed distance at a local-space point */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	float EvalDistanceLocal(FVector LocalPosition) const;

	/** Batch evaluate distances (returns array of distances) */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Evaluation")
	TArray<float> EvalDistanceBatch(const TArray<FVector>& Points) const;

	/** Check if a point is inside the SDF */
	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ALICE SDF|Evaluation")
	bool IsPointInside(FVector WorldPosition) const;

	// ========================================================================
	// Shader Generation
	// ========================================================================

	/** Generate HLSL shader code for Custom Material Expression */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateHlsl() const;

	/** Generate GLSL shader code */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateGlsl() const;

	/** Generate WGSL shader code */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|Shaders")
	FString GenerateWgsl() const;

	// ========================================================================
	// File I/O
	// ========================================================================

	/** Save SDF to .asdf file */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|IO")
	bool SaveToFile(const FString& FilePath);

	/** Load SDF from .asdf file */
	UFUNCTION(BlueprintCallable, Category = "ALICE SDF|IO")
	bool LoadFromFile(const FString& FilePath);

	// ========================================================================
	// Properties
	// ========================================================================

	/** Whether the SDF is compiled for fast evaluation */
	UPROPERTY(BlueprintReadOnly, Category = "ALICE SDF")
	bool bIsCompiled = false;

	/** Number of nodes in the SDF tree */
	UPROPERTY(BlueprintReadOnly, Category = "ALICE SDF")
	int32 NodeCount = 0;

	/** Auto-compile when shape changes */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF")
	bool bAutoCompile = true;

private:
	void FreeHandles();
	void UpdateNodeCount();
	void AutoCompileIfNeeded();
	FString ShaderResultToString(StringResult Result) const;

	SdfHandle SdfNodeHandle = nullptr;
	CompiledHandle CompiledSdfHandle = nullptr;
};
