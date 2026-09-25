// ALICE-SDF UE5 Plugin — editor-only helpers
// Author: Moroya Sakamoto
//
// AActor::SetActorLabel exists only in editor builds (WITH_EDITOR). The
// showcase actors label what they spawn so the Outliner is readable; in a
// cooked game there is no Outliner and no label API, so the call compiles
// away. Runtime builds were broken by the bare calls until 3.2.0
// (BuildPlugin on UE 5.7 never ran before the unreal-ue5 CI job).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

inline void AliceSdfSetActorLabel(AActor* Actor, const FString& Label)
{
#if WITH_EDITOR
	if (Actor)
	{
		Actor->SetActorLabel(Label);
	}
#else
	(void)Actor;
	(void)Label;
#endif
}
