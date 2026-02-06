// ALICE-SDF UE5 Component Implementation
// Author: Moroya Sakamoto

#include "AliceSdfComponent.h"

UAliceSdfComponent::UAliceSdfComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UAliceSdfComponent::BeginPlay()
{
	Super::BeginPlay();
}

void UAliceSdfComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	FreeHandles();
	Super::EndPlay(EndPlayReason);
}

void UAliceSdfComponent::FreeHandles()
{
	if (CompiledSdfHandle)
	{
		alice_sdf_free_compiled(CompiledSdfHandle);
		CompiledSdfHandle = nullptr;
		bIsCompiled = false;
	}
	if (SdfNodeHandle)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = nullptr;
		NodeCount = 0;
	}
}

void UAliceSdfComponent::UpdateNodeCount()
{
	if (SdfNodeHandle)
	{
		NodeCount = static_cast<int32>(alice_sdf_node_count(SdfNodeHandle));
	}
	else
	{
		NodeCount = 0;
	}
}

void UAliceSdfComponent::AutoCompileIfNeeded()
{
	if (bAutoCompile && SdfNodeHandle)
	{
		Compile();
	}
}

// ============================================================================
// Primitives
// ============================================================================

void UAliceSdfComponent::CreateSphere(float Radius)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_sphere(Radius);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::CreateBox(FVector HalfExtents)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_box(HalfExtents.X, HalfExtents.Y, HalfExtents.Z);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::CreateCylinder(float Radius, float HalfHeight)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_cylinder(Radius, HalfHeight);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::CreateTorus(float MajorRadius, float MinorRadius)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_torus(MajorRadius, MinorRadius);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::CreateCapsule(FVector PointA, FVector PointB, float Radius)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_capsule(
		PointA.X, PointA.Y, PointA.Z,
		PointB.X, PointB.Y, PointB.Z,
		Radius
	);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::CreatePlane(FVector Normal, float Distance)
{
	FreeHandles();
	SdfNodeHandle = alice_sdf_plane(Normal.X, Normal.Y, Normal.Z, Distance);
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

// ============================================================================
// Boolean Operations
// ============================================================================

void UAliceSdfComponent::UnionWith(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_union(SdfNodeHandle, Other->SdfNodeHandle);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::IntersectWith(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_intersection(SdfNodeHandle, Other->SdfNodeHandle);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::SubtractFrom(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_subtract(SdfNodeHandle, Other->SdfNodeHandle);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::SmoothUnionWith(UAliceSdfComponent* Other, float Smoothness)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_smooth_union(SdfNodeHandle, Other->SdfNodeHandle, Smoothness);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::SmoothIntersectWith(UAliceSdfComponent* Other, float Smoothness)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_smooth_intersection(SdfNodeHandle, Other->SdfNodeHandle, Smoothness);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::SmoothSubtractFrom(UAliceSdfComponent* Other, float Smoothness)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_smooth_subtract(SdfNodeHandle, Other->SdfNodeHandle, Smoothness);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

// ============================================================================
// Transforms
// ============================================================================

void UAliceSdfComponent::TranslateSdf(FVector Offset)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_translate(SdfNodeHandle, Offset.X, Offset.Y, Offset.Z);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::RotateSdf(FRotator Rotation)
{
	if (!SdfNodeHandle) return;

	// Convert degrees to radians
	float Rx = FMath::DegreesToRadians(Rotation.Pitch);
	float Ry = FMath::DegreesToRadians(Rotation.Yaw);
	float Rz = FMath::DegreesToRadians(Rotation.Roll);

	SdfHandle Result = alice_sdf_rotate_euler(SdfNodeHandle, Rx, Ry, Rz);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::ScaleSdf(float Factor)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_scale(SdfNodeHandle, Factor);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

// ============================================================================
// Modifiers
// ============================================================================

void UAliceSdfComponent::ApplyRound(float Radius)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_round(SdfNodeHandle, Radius);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::ApplyOnion(float Thickness)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_onion(SdfNodeHandle, Thickness);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::ApplyTwist(float Strength)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_twist(SdfNodeHandle, Strength);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::ApplyBend(float Curvature)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_bend(SdfNodeHandle, Curvature);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

void UAliceSdfComponent::ApplyRepeat(FVector Spacing)
{
	if (!SdfNodeHandle) return;

	SdfHandle Result = alice_sdf_repeat(SdfNodeHandle, Spacing.X, Spacing.Y, Spacing.Z);
	if (Result)
	{
		alice_sdf_free(SdfNodeHandle);
		SdfNodeHandle = Result;
		UpdateNodeCount();
		AutoCompileIfNeeded();
	}
}

// ============================================================================
// Compilation & Evaluation
// ============================================================================

bool UAliceSdfComponent::Compile()
{
	if (!SdfNodeHandle) return false;

	if (CompiledSdfHandle)
	{
		alice_sdf_free_compiled(CompiledSdfHandle);
		CompiledSdfHandle = nullptr;
	}

	CompiledSdfHandle = alice_sdf_compile(SdfNodeHandle);
	bIsCompiled = (CompiledSdfHandle != nullptr);
	return bIsCompiled;
}

float UAliceSdfComponent::EvalDistance(FVector WorldPosition) const
{
	// Convert world to local space
	FVector LocalPos = GetComponentTransform().InverseTransformPosition(WorldPosition);
	return EvalDistanceLocal(LocalPos);
}

float UAliceSdfComponent::EvalDistanceLocal(FVector LocalPosition) const
{
	if (bIsCompiled && CompiledSdfHandle)
	{
		return alice_sdf_eval_compiled(CompiledSdfHandle,
			LocalPosition.X, LocalPosition.Y, LocalPosition.Z);
	}
	if (SdfNodeHandle)
	{
		return alice_sdf_eval(SdfNodeHandle,
			LocalPosition.X, LocalPosition.Y, LocalPosition.Z);
	}
	return MAX_FLT;
}

TArray<float> UAliceSdfComponent::EvalDistanceBatch(const TArray<FVector>& Points) const
{
	TArray<float> Distances;
	Distances.SetNum(Points.Num());

	if (Points.Num() == 0) return Distances;

	if (bIsCompiled && CompiledSdfHandle)
	{
		// Convert FVector array to flat float array
		TArray<float> FlatPoints;
		FlatPoints.SetNum(Points.Num() * 3);
		for (int32 i = 0; i < Points.Num(); i++)
		{
			FVector LocalPos = GetComponentTransform().InverseTransformPosition(Points[i]);
			FlatPoints[i * 3 + 0] = LocalPos.X;
			FlatPoints[i * 3 + 1] = LocalPos.Y;
			FlatPoints[i * 3 + 2] = LocalPos.Z;
		}

		alice_sdf_eval_compiled_batch(
			CompiledSdfHandle,
			FlatPoints.GetData(),
			Distances.GetData(),
			static_cast<uint32>(Points.Num())
		);
	}
	else
	{
		for (int32 i = 0; i < Points.Num(); i++)
		{
			Distances[i] = EvalDistance(Points[i]);
		}
	}

	return Distances;
}

bool UAliceSdfComponent::IsPointInside(FVector WorldPosition) const
{
	return EvalDistance(WorldPosition) < 0.0f;
}

// ============================================================================
// Shader Generation
// ============================================================================

FString UAliceSdfComponent::ShaderResultToString(StringResult Result) const
{
	if (Result.result == SdfResult_Ok && Result.data)
	{
		FString Code = FString(UTF8_TO_TCHAR(Result.data));
		alice_sdf_free_string(Result.data);
		return Code;
	}
	return FString();
}

FString UAliceSdfComponent::GenerateHlsl() const
{
	if (!SdfNodeHandle) return FString();
	return ShaderResultToString(alice_sdf_to_hlsl(SdfNodeHandle));
}

FString UAliceSdfComponent::GenerateGlsl() const
{
	if (!SdfNodeHandle) return FString();
	return ShaderResultToString(alice_sdf_to_glsl(SdfNodeHandle));
}

FString UAliceSdfComponent::GenerateWgsl() const
{
	if (!SdfNodeHandle) return FString();
	return ShaderResultToString(alice_sdf_to_wgsl(SdfNodeHandle));
}

// ============================================================================
// File I/O
// ============================================================================

bool UAliceSdfComponent::SaveToFile(const FString& FilePath)
{
	if (!SdfNodeHandle) return false;

	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	SdfResult Result = alice_sdf_save(SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath));
	return Result == SdfResult_Ok;
}

bool UAliceSdfComponent::LoadFromFile(const FString& FilePath)
{
	FreeHandles();

	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	SdfNodeHandle = alice_sdf_load(TCHAR_TO_UTF8(*AbsPath));

	if (SdfNodeHandle)
	{
		UpdateNodeCount();
		AutoCompileIfNeeded();
		return true;
	}
	return false;
}
