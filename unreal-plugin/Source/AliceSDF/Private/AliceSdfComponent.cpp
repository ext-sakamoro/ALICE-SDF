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

// ============================================================================
// Internal Helpers
// ============================================================================

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
	NodeCount = SdfNodeHandle ? static_cast<int32>(alice_sdf_node_count(SdfNodeHandle)) : 0;
}

void UAliceSdfComponent::AutoCompileIfNeeded()
{
	if (bAutoCompile && SdfNodeHandle) { Compile(); }
}

void UAliceSdfComponent::SetNewShape(SdfHandle NewHandle)
{
	FreeHandles();
	SdfNodeHandle = NewHandle;
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::ApplyModifier(SdfHandle NewHandle)
{
	if (!NewHandle) return;
	alice_sdf_free(SdfNodeHandle);
	SdfNodeHandle = NewHandle;
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

void UAliceSdfComponent::ApplyBinaryOp(UAliceSdfComponent* Other, SdfHandle Result)
{
	if (!Result) return;
	alice_sdf_free(SdfNodeHandle);
	SdfNodeHandle = Result;
	UpdateNodeCount();
	AutoCompileIfNeeded();
}

// ============================================================================
// Primitives — Basic
// ============================================================================

void UAliceSdfComponent::CreateSphere(float Radius)
{ SetNewShape(alice_sdf_sphere(Radius)); }

void UAliceSdfComponent::CreateBox(FVector H)
{ SetNewShape(alice_sdf_box(H.X, H.Y, H.Z)); }

void UAliceSdfComponent::CreateCylinder(float Radius, float HalfHeight)
{ SetNewShape(alice_sdf_cylinder(Radius, HalfHeight)); }

void UAliceSdfComponent::CreateTorus(float MajorRadius, float MinorRadius)
{ SetNewShape(alice_sdf_torus(MajorRadius, MinorRadius)); }

void UAliceSdfComponent::CreateCapsule(FVector A, FVector B, float Radius)
{ SetNewShape(alice_sdf_capsule(A.X, A.Y, A.Z, B.X, B.Y, B.Z, Radius)); }

void UAliceSdfComponent::CreatePlane(FVector Normal, float Distance)
{ SetNewShape(alice_sdf_plane(Normal.X, Normal.Y, Normal.Z, Distance)); }

void UAliceSdfComponent::CreateCone(float Radius, float HalfHeight)
{ SetNewShape(alice_sdf_cone(Radius, HalfHeight)); }

void UAliceSdfComponent::CreateEllipsoid(FVector Radii)
{ SetNewShape(alice_sdf_ellipsoid(Radii.X, Radii.Y, Radii.Z)); }

void UAliceSdfComponent::CreateRoundedCone(float R1, float R2, float HalfHeight)
{ SetNewShape(alice_sdf_rounded_cone(R1, R2, HalfHeight)); }

void UAliceSdfComponent::CreatePyramid(float HalfHeight)
{ SetNewShape(alice_sdf_pyramid(HalfHeight)); }

void UAliceSdfComponent::CreateOctahedron(float Size)
{ SetNewShape(alice_sdf_octahedron(Size)); }

void UAliceSdfComponent::CreateHexPrism(float HexRadius, float HalfHeight)
{ SetNewShape(alice_sdf_hex_prism(HexRadius, HalfHeight)); }

void UAliceSdfComponent::CreateLink(float HalfLength, float R1, float R2)
{ SetNewShape(alice_sdf_link(HalfLength, R1, R2)); }

void UAliceSdfComponent::CreateRoundedBox(FVector H, float RoundRadius)
{ SetNewShape(alice_sdf_rounded_box(H.X, H.Y, H.Z, RoundRadius)); }

// ============================================================================
// Primitives — Advanced
// ============================================================================

void UAliceSdfComponent::CreateCappedCone(float HalfHeight, float R1, float R2)
{ SetNewShape(alice_sdf_capped_cone(HalfHeight, R1, R2)); }

void UAliceSdfComponent::CreateCappedTorus(float MajorRadius, float MinorRadius, float CapAngle)
{ SetNewShape(alice_sdf_capped_torus(MajorRadius, MinorRadius, CapAngle)); }

void UAliceSdfComponent::CreateRoundedCylinder(float Radius, float RoundRadius, float HalfHeight)
{ SetNewShape(alice_sdf_rounded_cylinder(Radius, RoundRadius, HalfHeight)); }

void UAliceSdfComponent::CreateTriangularPrism(float Width, float HalfDepth)
{ SetNewShape(alice_sdf_triangular_prism(Width, HalfDepth)); }

void UAliceSdfComponent::CreateCutSphere(float Radius, float CutHeight)
{ SetNewShape(alice_sdf_cut_sphere(Radius, CutHeight)); }

void UAliceSdfComponent::CreateDeathStar(float Ra, float Rb, float D)
{ SetNewShape(alice_sdf_death_star(Ra, Rb, D)); }

void UAliceSdfComponent::CreateHeart(float Size)
{ SetNewShape(alice_sdf_heart(Size)); }

void UAliceSdfComponent::CreateBarrel(float Radius, float HalfHeight, float Bulge)
{ SetNewShape(alice_sdf_barrel(Radius, HalfHeight, Bulge)); }

void UAliceSdfComponent::CreateDiamond(float Radius, float HalfHeight)
{ SetNewShape(alice_sdf_diamond(Radius, HalfHeight)); }

void UAliceSdfComponent::CreateEgg(float Ra, float Rb)
{ SetNewShape(alice_sdf_egg(Ra, Rb)); }

// ============================================================================
// Primitives — Platonic & Archimedean
// ============================================================================

void UAliceSdfComponent::CreateTetrahedron(float Size)
{ SetNewShape(alice_sdf_tetrahedron(Size)); }

void UAliceSdfComponent::CreateDodecahedron(float Radius)
{ SetNewShape(alice_sdf_dodecahedron(Radius)); }

void UAliceSdfComponent::CreateIcosahedron(float Radius)
{ SetNewShape(alice_sdf_icosahedron(Radius)); }

void UAliceSdfComponent::CreateTruncatedOctahedron(float Radius)
{ SetNewShape(alice_sdf_truncated_octahedron(Radius)); }

void UAliceSdfComponent::CreateTruncatedIcosahedron(float Radius)
{ SetNewShape(alice_sdf_truncated_icosahedron(Radius)); }

// ============================================================================
// Primitives — TPMS
// ============================================================================

void UAliceSdfComponent::CreateGyroid(float Scale, float Thickness)
{ SetNewShape(alice_sdf_gyroid(Scale, Thickness)); }

void UAliceSdfComponent::CreateSchwarzP(float Scale, float Thickness)
{ SetNewShape(alice_sdf_schwarz_p(Scale, Thickness)); }

void UAliceSdfComponent::CreateDiamondSurface(float Scale, float Thickness)
{ SetNewShape(alice_sdf_diamond_surface(Scale, Thickness)); }

void UAliceSdfComponent::CreateNeovius(float Scale, float Thickness)
{ SetNewShape(alice_sdf_neovius(Scale, Thickness)); }

void UAliceSdfComponent::CreateLidinoid(float Scale, float Thickness)
{ SetNewShape(alice_sdf_lidinoid(Scale, Thickness)); }

void UAliceSdfComponent::CreateIWP(float Scale, float Thickness)
{ SetNewShape(alice_sdf_iwp(Scale, Thickness)); }

void UAliceSdfComponent::CreateFRD(float Scale, float Thickness)
{ SetNewShape(alice_sdf_frd(Scale, Thickness)); }

void UAliceSdfComponent::CreateFischerKochS(float Scale, float Thickness)
{ SetNewShape(alice_sdf_fischer_koch_s(Scale, Thickness)); }

void UAliceSdfComponent::CreatePMY(float Scale, float Thickness)
{ SetNewShape(alice_sdf_pmy(Scale, Thickness)); }

// ============================================================================
// Primitives — Structural
// ============================================================================

void UAliceSdfComponent::CreateBoxFrame(FVector H, float Edge)
{ SetNewShape(alice_sdf_box_frame(H.X, H.Y, H.Z, Edge)); }

void UAliceSdfComponent::CreateTube(float OuterRadius, float Thickness, float HalfHeight)
{ SetNewShape(alice_sdf_tube(OuterRadius, Thickness, HalfHeight)); }

void UAliceSdfComponent::CreateChamferedCube(FVector H, float Chamfer)
{ SetNewShape(alice_sdf_chamfered_cube(H.X, H.Y, H.Z, Chamfer)); }

void UAliceSdfComponent::CreateStairs(float StepWidth, float StepHeight, float NumSteps, float HalfDepth)
{ SetNewShape(alice_sdf_stairs(StepWidth, StepHeight, NumSteps, HalfDepth)); }

void UAliceSdfComponent::CreateHelix(float MajorRadius, float MinorRadius, float Pitch, float HalfHeight)
{ SetNewShape(alice_sdf_helix(MajorRadius, MinorRadius, Pitch, HalfHeight)); }

// ============================================================================
// Primitives — 2D/Extruded & Additional
// ============================================================================

void UAliceSdfComponent::CreateTriangle(FVector A, FVector B, FVector C)
{ SetNewShape(alice_sdf_triangle(A.X, A.Y, A.Z, B.X, B.Y, B.Z, C.X, C.Y, C.Z)); }

void UAliceSdfComponent::CreateBezier(FVector A, FVector B, FVector C, float Radius)
{ SetNewShape(alice_sdf_bezier(A.X, A.Y, A.Z, B.X, B.Y, B.Z, C.X, C.Y, C.Z, Radius)); }

void UAliceSdfComponent::CreateCutHollowSphere(float Radius, float CutHeight, float Thickness)
{ SetNewShape(alice_sdf_cut_hollow_sphere(Radius, CutHeight, Thickness)); }

void UAliceSdfComponent::CreateSolidAngle(float Angle, float Radius)
{ SetNewShape(alice_sdf_solid_angle(Angle, Radius)); }

void UAliceSdfComponent::CreateRhombus(float La, float Lb, float HalfHeight, float RoundRadius)
{ SetNewShape(alice_sdf_rhombus(La, Lb, HalfHeight, RoundRadius)); }

void UAliceSdfComponent::CreateHorseshoe(float Angle, float Radius, float HalfLength, float Width, float Thickness)
{ SetNewShape(alice_sdf_horseshoe(Angle, Radius, HalfLength, Width, Thickness)); }

void UAliceSdfComponent::CreateVesica(float Radius, float HalfDist)
{ SetNewShape(alice_sdf_vesica(Radius, HalfDist)); }

void UAliceSdfComponent::CreateInfiniteCylinder(float Radius)
{ SetNewShape(alice_sdf_infinite_cylinder(Radius)); }

void UAliceSdfComponent::CreateInfiniteCone(float Angle)
{ SetNewShape(alice_sdf_infinite_cone(Angle)); }

void UAliceSdfComponent::CreateSuperEllipsoid(FVector H, float E1, float E2)
{ SetNewShape(alice_sdf_superellipsoid(H.X, H.Y, H.Z, E1, E2)); }

void UAliceSdfComponent::CreateRoundedX(float Width, float RoundRadius, float HalfHeight)
{ SetNewShape(alice_sdf_rounded_x(Width, RoundRadius, HalfHeight)); }

void UAliceSdfComponent::CreatePie(float Angle, float Radius, float HalfHeight)
{ SetNewShape(alice_sdf_pie(Angle, Radius, HalfHeight)); }

void UAliceSdfComponent::CreateTrapezoid(float R1, float R2, float TrapHeight, float HalfDepth)
{ SetNewShape(alice_sdf_trapezoid(R1, R2, TrapHeight, HalfDepth)); }

void UAliceSdfComponent::CreateParallelogram(float Width, float ParaHeight, float Skew, float HalfDepth)
{ SetNewShape(alice_sdf_parallelogram(Width, ParaHeight, Skew, HalfDepth)); }

void UAliceSdfComponent::CreateTunnel(float Width, float Height2D, float HalfDepth)
{ SetNewShape(alice_sdf_tunnel(Width, Height2D, HalfDepth)); }

void UAliceSdfComponent::CreateUnevenCapsule(float R1, float R2, float CapHeight, float HalfDepth)
{ SetNewShape(alice_sdf_uneven_capsule(R1, R2, CapHeight, HalfDepth)); }

void UAliceSdfComponent::CreateArcShape(float Aperture, float Radius, float Thickness, float HalfHeight)
{ SetNewShape(alice_sdf_arc_shape(Aperture, Radius, Thickness, HalfHeight)); }

void UAliceSdfComponent::CreateMoon(float D, float Ra, float Rb, float HalfHeight)
{ SetNewShape(alice_sdf_moon(D, Ra, Rb, HalfHeight)); }

void UAliceSdfComponent::CreateCrossShape(float Length, float Thickness, float RoundRadius, float HalfHeight)
{ SetNewShape(alice_sdf_cross_shape(Length, Thickness, RoundRadius, HalfHeight)); }

void UAliceSdfComponent::CreateBlobbyCross(float Size, float HalfHeight)
{ SetNewShape(alice_sdf_blobby_cross(Size, HalfHeight)); }

void UAliceSdfComponent::CreateParabolaSegment(float Width, float ParaHeight, float HalfDepth)
{ SetNewShape(alice_sdf_parabola_segment(Width, ParaHeight, HalfDepth)); }

void UAliceSdfComponent::CreateRegularPolygon(float Radius, float NSides, float HalfHeight)
{ SetNewShape(alice_sdf_regular_polygon(Radius, NSides, HalfHeight)); }

void UAliceSdfComponent::CreateStarPolygon(float Radius, float NPoints, float M, float HalfHeight)
{ SetNewShape(alice_sdf_star_polygon(Radius, NPoints, M, HalfHeight)); }

// ============================================================================
// Boolean Operations
// ============================================================================


void UAliceSdfComponent::UnionWith(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_union(SdfNodeHandle, Other->SdfNodeHandle));
}

void UAliceSdfComponent::IntersectWith(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_intersection(SdfNodeHandle, Other->SdfNodeHandle));
}

void UAliceSdfComponent::SubtractFrom(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_subtract(SdfNodeHandle, Other->SdfNodeHandle));
}

void UAliceSdfComponent::SmoothUnionWith(UAliceSdfComponent* Other, float K)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_smooth_union(SdfNodeHandle, Other->SdfNodeHandle, K));
}

void UAliceSdfComponent::SmoothIntersectWith(UAliceSdfComponent* Other, float K)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_smooth_intersection(SdfNodeHandle, Other->SdfNodeHandle, K));
}

void UAliceSdfComponent::SmoothSubtractFrom(UAliceSdfComponent* Other, float K)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_smooth_subtract(SdfNodeHandle, Other->SdfNodeHandle, K));
}

void UAliceSdfComponent::ChamferUnionWith(UAliceSdfComponent* Other, float R)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_chamfer_union(SdfNodeHandle, Other->SdfNodeHandle, R));
}

void UAliceSdfComponent::ChamferIntersectWith(UAliceSdfComponent* Other, float R)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_chamfer_intersection(SdfNodeHandle, Other->SdfNodeHandle, R));
}

void UAliceSdfComponent::ChamferSubtractFrom(UAliceSdfComponent* Other, float R)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_chamfer_subtract(SdfNodeHandle, Other->SdfNodeHandle, R));
}

void UAliceSdfComponent::StairsUnionWith(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_stairs_union(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::StairsIntersectWith(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_stairs_intersection(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::StairsSubtractFrom(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_stairs_subtract(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::ColumnsUnionWith(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_columns_union(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::ColumnsIntersectWith(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_columns_intersection(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::ColumnsSubtractFrom(UAliceSdfComponent* Other, float R, float N)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_columns_subtract(SdfNodeHandle, Other->SdfNodeHandle, R, N));
}

void UAliceSdfComponent::XorWith(UAliceSdfComponent* Other)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_xor(SdfNodeHandle, Other->SdfNodeHandle));
}

void UAliceSdfComponent::MorphWith(UAliceSdfComponent* Other, float T)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_morph(SdfNodeHandle, Other->SdfNodeHandle, T));
}

void UAliceSdfComponent::PipeWith(UAliceSdfComponent* Other, float R)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_pipe(SdfNodeHandle, Other->SdfNodeHandle, R));
}

void UAliceSdfComponent::EngraveWith(UAliceSdfComponent* Other, float R)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_engrave(SdfNodeHandle, Other->SdfNodeHandle, R));
}

void UAliceSdfComponent::GrooveWith(UAliceSdfComponent* Other, float Ra, float Rb)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_groove(SdfNodeHandle, Other->SdfNodeHandle, Ra, Rb));
}

void UAliceSdfComponent::TongueWith(UAliceSdfComponent* Other, float Ra, float Rb)
{
	if (!SdfNodeHandle || !Other || !Other->SdfNodeHandle) return;
	ApplyBinaryOp(Other, alice_sdf_tongue(SdfNodeHandle, Other->SdfNodeHandle, Ra, Rb));
}

// ============================================================================
// Transforms
// ============================================================================

void UAliceSdfComponent::TranslateSdf(FVector Offset)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_translate(SdfNodeHandle, Offset.X, Offset.Y, Offset.Z));
}

void UAliceSdfComponent::RotateSdf(FRotator Rotation)
{
	if (!SdfNodeHandle) return;
	float Rx = FMath::DegreesToRadians(Rotation.Pitch);
	float Ry = FMath::DegreesToRadians(Rotation.Yaw);
	float Rz = FMath::DegreesToRadians(Rotation.Roll);
	ApplyModifier(alice_sdf_rotate_euler(SdfNodeHandle, Rx, Ry, Rz));
}

void UAliceSdfComponent::ScaleSdf(float Factor)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_scale(SdfNodeHandle, Factor));
}

void UAliceSdfComponent::ScaleSdfNonUniform(FVector S)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_scale_non_uniform(SdfNodeHandle, S.X, S.Y, S.Z));
}

void UAliceSdfComponent::RotateEulerSdf(FVector EulerRadians)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_rotate_euler(SdfNodeHandle, EulerRadians.X, EulerRadians.Y, EulerRadians.Z));
}

void UAliceSdfComponent::RotateQuatSdf(FQuat Rotation)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_rotate(SdfNodeHandle, Rotation.X, Rotation.Y, Rotation.Z, Rotation.W));
}

// ============================================================================
// Modifiers
// ============================================================================

void UAliceSdfComponent::ApplyRound(float Radius)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_round(SdfNodeHandle, Radius));
}

void UAliceSdfComponent::ApplyOnion(float Thickness)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_onion(SdfNodeHandle, Thickness));
}

void UAliceSdfComponent::ApplyTwist(float Strength)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_twist(SdfNodeHandle, Strength));
}

void UAliceSdfComponent::ApplyBend(float Curvature)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_bend(SdfNodeHandle, Curvature));
}

void UAliceSdfComponent::ApplyRepeat(FVector Spacing)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_repeat(SdfNodeHandle, Spacing.X, Spacing.Y, Spacing.Z));
}

void UAliceSdfComponent::ApplyRepeatFinite(FIntVector Count, FVector Spacing)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_repeat_finite(SdfNodeHandle, static_cast<uint32_t>(Count.X), static_cast<uint32_t>(Count.Y), static_cast<uint32_t>(Count.Z), Spacing.X, Spacing.Y, Spacing.Z));
}

void UAliceSdfComponent::ApplyMirror(bool X, bool Y, bool Z)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_mirror(SdfNodeHandle, X ? 1u : 0u, Y ? 1u : 0u, Z ? 1u : 0u));
}

void UAliceSdfComponent::ApplyElongate(FVector Amount)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_elongate(SdfNodeHandle, Amount.X, Amount.Y, Amount.Z));
}

void UAliceSdfComponent::ApplyRevolution(float Offset)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_revolution(SdfNodeHandle, Offset));
}

void UAliceSdfComponent::ApplyExtrude(float HalfHeight)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_extrude(SdfNodeHandle, HalfHeight));
}

void UAliceSdfComponent::ApplyNoise(float Amplitude, float Frequency, int32 Seed)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_noise(SdfNodeHandle, Amplitude, Frequency, static_cast<uint32>(Seed)));
}

void UAliceSdfComponent::ApplyTaper(float Factor)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_taper(SdfNodeHandle, Factor));
}

void UAliceSdfComponent::ApplyDisplacement(float Strength)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_displacement(SdfNodeHandle, Strength));
}

void UAliceSdfComponent::ApplyPolarRepeat(int32 Count)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_polar_repeat(SdfNodeHandle, static_cast<uint32>(Count)));
}

void UAliceSdfComponent::ApplyOctantMirror()
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_octant_mirror(SdfNodeHandle));
}

void UAliceSdfComponent::ApplySweepBezier(FVector2D P0, FVector2D P1, FVector2D P2)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_sweep_bezier(SdfNodeHandle, P0.X, P0.Y, P1.X, P1.Y, P2.X, P2.Y));
}

void UAliceSdfComponent::SetMaterialId(int32 MaterialId)
{
	if (!SdfNodeHandle) return;
	ApplyModifier(alice_sdf_with_material(SdfNodeHandle, static_cast<uint32>(MaterialId)));
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
	FVector LocalPos = GetComponentTransform().InverseTransformPosition(WorldPosition);
	return EvalDistanceLocal(LocalPos);
}

float UAliceSdfComponent::EvalDistanceLocal(FVector LocalPosition) const
{
	if (bIsCompiled && CompiledSdfHandle)
	{
		return alice_sdf_eval_compiled(CompiledSdfHandle, LocalPosition.X, LocalPosition.Y, LocalPosition.Z);
	}
	if (SdfNodeHandle)
	{
		return alice_sdf_eval(SdfNodeHandle, LocalPosition.X, LocalPosition.Y, LocalPosition.Z);
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
		TArray<float> FlatPoints;
		FlatPoints.SetNum(Points.Num() * 3);
		for (int32 i = 0; i < Points.Num(); i++)
		{
			FVector LocalPos = GetComponentTransform().InverseTransformPosition(Points[i]);
			FlatPoints[i * 3 + 0] = LocalPos.X;
			FlatPoints[i * 3 + 1] = LocalPos.Y;
			FlatPoints[i * 3 + 2] = LocalPos.Z;
		}
		BatchResult BatchRes = alice_sdf_eval_compiled_batch(
			CompiledSdfHandle, FlatPoints.GetData(), Distances.GetData(),
			static_cast<uint32>(Points.Num()));
		if (BatchRes.result != SdfResult_Ok)
		{
			UE_LOG(LogTemp, Warning, TEXT("ALICE-SDF: EvalDistanceBatch failed (result=%d)"), (int)BatchRes.result);
		}
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
// Mesh Export
// ============================================================================

bool UAliceSdfComponent::ExportObj(const FString& FilePath, int32 Resolution, float Bounds)
{
	if (!SdfNodeHandle) return false;
	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	return alice_sdf_export_obj(nullptr, SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath), Resolution, Bounds) == SdfResult_Ok;
}

bool UAliceSdfComponent::ExportGlb(const FString& FilePath, int32 Resolution, float Bounds)
{
	if (!SdfNodeHandle) return false;
	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	return alice_sdf_export_glb(nullptr, SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath), Resolution, Bounds) == SdfResult_Ok;
}

bool UAliceSdfComponent::ExportUsda(const FString& FilePath, int32 Resolution, float Bounds)
{
	if (!SdfNodeHandle) return false;
	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	return alice_sdf_export_usda(nullptr, SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath), Resolution, Bounds) == SdfResult_Ok;
}

bool UAliceSdfComponent::ExportFbx(const FString& FilePath, int32 Resolution, float Bounds)
{
	if (!SdfNodeHandle) return false;
	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	return alice_sdf_export_fbx(nullptr, SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath), Resolution, Bounds) == SdfResult_Ok;
}

// ============================================================================
// File I/O
// ============================================================================

bool UAliceSdfComponent::SaveToFile(const FString& FilePath)
{
	if (!SdfNodeHandle) return false;
	FString AbsPath = FPaths::ConvertRelativePathToFull(FilePath);
	return alice_sdf_save(SdfNodeHandle, TCHAR_TO_UTF8(*AbsPath)) == SdfResult_Ok;
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
