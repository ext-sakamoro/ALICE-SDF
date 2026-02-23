// ALICE-SDF Demo Actor for UE5
// Author: Moroya Sakamoto
//
// Drop this actor into any level to see ALICE-SDF in action.
// It creates several SDF shapes with CSG operations and logs the results.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AliceSdfComponent.h"
#include "AliceSdfDemoActor.generated.h"

/**
 * AAliceSdfDemoActor
 *
 * A showcase actor that demonstrates ALICE-SDF capabilities.
 * Drop into a level and press Play to see SDF evaluation, CSG operations,
 * shader generation, and mesh export in action.
 *
 * Check the Output Log for results.
 */
UCLASS(ClassGroup=(AliceSDF))
class ALICESDF_API AAliceSdfDemoActor : public AActor
{
	GENERATED_BODY()

public:
	AAliceSdfDemoActor();

protected:
	virtual void BeginPlay() override;

public:
	/** Main SDF shape component */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Demo")
	TObjectPtr<UAliceSdfComponent> MainShape;

	/** Second shape for CSG operations */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "ALICE SDF Demo")
	TObjectPtr<UAliceSdfComponent> CsgShape;

	/** Which demo to run on BeginPlay */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Demo")
	int32 DemoIndex = 0;

	/** Resolution for mesh export demo */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ALICE SDF Demo")
	int32 MeshResolution = 64;

private:
	void RunDemo0_BasicShapes();
	void RunDemo1_CSGOperations();
	void RunDemo2_TPMS();
	void RunDemo3_Modifiers();
	void RunDemo4_ShaderGeneration();
};
