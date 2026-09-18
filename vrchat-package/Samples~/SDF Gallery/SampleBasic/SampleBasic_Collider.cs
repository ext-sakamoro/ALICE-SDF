// ALICE-SDF Sample: Basic Collider (Ground + Sphere)
using UnityEngine;

#if UDONSHARP
using VRC.SDKBase;
using UdonSharp;
#endif

namespace AliceSDF.Samples
{
#if UDONSHARP
    [UdonBehaviourSyncMode(BehaviourSyncMode.None)]
    public class SampleBasic_Collider : AliceSDF_Collider
#else
    public class SampleBasic_Collider : AliceSDF_Collider
#endif
    {
        // override, not new: UdonSharp resolves a virtual call to the most
        // derived method, so the base collider pushes against this SDF (with
        // new it pushed against the base class demo sphere)
        public override float Evaluate(Vector3 p)
        {
            float ground = p.y;
            float sphere = (p - new Vector3(0f, 1.5f, 0f)).magnitude - 1.5f;
            return Mathf.Min(ground, sphere);
        }
    }
}
