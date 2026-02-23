// =============================================================================
// AliceSDF_LOD.cginc - VRChat GPU Budget Control (Deep Fried)
// =============================================================================
// Dynamically adjusts raymarching quality based on:
//   - Camera distance (far objects need fewer steps)
//   - VRChat performance rank targets
//
// "VRChatで60fps死守。数学に妥協はない、GPUに妥協する。"
//
// Author: Moroya Sakamoto
// =============================================================================

#ifndef ALICE_SDF_LOD
#define ALICE_SDF_LOD

// Performance tier constants (VRChat GPU time budget)
// Excellent: < 5ms, Good: < 8ms, Medium: < 11ms
#define ALICE_LOD_TIER_HIGH   0
#define ALICE_LOD_TIER_MED    1
#define ALICE_LOD_TIER_LOW    2

// Step count presets per LOD tier
#define ALICE_STEPS_HIGH     128
#define ALICE_STEPS_MED       64
#define ALICE_STEPS_LOW       32

// Epsilon presets per LOD tier
#define ALICE_EPS_HIGH    0.0001
#define ALICE_EPS_MED     0.001
#define ALICE_EPS_LOW     0.005

// Distance thresholds for automatic LOD switching
#define ALICE_LOD_DIST_NEAR   20.0
#define ALICE_LOD_DIST_MED    60.0

// =============================================================================
// Automatic LOD Selection
// =============================================================================

// Returns LOD tier based on camera distance to object center
int aliceLodTier(float cameraDist)
{
    if (cameraDist < ALICE_LOD_DIST_NEAR) return ALICE_LOD_TIER_HIGH;
    if (cameraDist < ALICE_LOD_DIST_MED)  return ALICE_LOD_TIER_MED;
    return ALICE_LOD_TIER_LOW;
}

// Get max raymarching steps for current LOD tier
int aliceLodSteps(int tier)
{
    if (tier == ALICE_LOD_TIER_HIGH) return ALICE_STEPS_HIGH;
    if (tier == ALICE_LOD_TIER_MED)  return ALICE_STEPS_MED;
    return ALICE_STEPS_LOW;
}

// Get surface epsilon for current LOD tier
float aliceLodEpsilon(int tier)
{
    if (tier == ALICE_LOD_TIER_HIGH) return ALICE_EPS_HIGH;
    if (tier == ALICE_LOD_TIER_MED)  return ALICE_EPS_MED;
    return ALICE_EPS_LOW;
}

// Get step multiplier for relaxed sphere tracing
// At lower LOD, we take larger steps (less precise but faster)
float aliceLodStepScale(int tier)
{
    if (tier == ALICE_LOD_TIER_HIGH) return 0.9;  // Conservative
    if (tier == ALICE_LOD_TIER_MED)  return 1.0;  // Standard
    return 1.2;                                     // Aggressive (overshooting)
}

// =============================================================================
// Adaptive Raymarching
// =============================================================================

// Raymarch with automatic LOD adjustment
// Returns: float4(hitPos.xyz, distance) or float4(0,0,0,-1) on miss
float4 aliceRaymarchLOD(float3 rayOrigin, float3 rayDir, float maxDist)
{
    // Calculate camera distance to determine LOD
    float cameraDist = length(rayOrigin - _WorldSpaceCameraPos);
    int tier = aliceLodTier(cameraDist);

    int maxSteps = aliceLodSteps(tier);
    float eps = aliceLodEpsilon(tier);
    float stepScale = aliceLodStepScale(tier);

    float t = 0.0;

    for (int i = 0; i < maxSteps; i++)
    {
        float3 p = rayOrigin + rayDir * t;
        float d = map(p);

        if (d < eps)
        {
            return float4(p, t);
        }

        t += d * stepScale;

        if (t > maxDist) break;
    }

    return float4(0, 0, 0, -1);
}

// =============================================================================
// AO with LOD (fewer samples when far)
// =============================================================================

float aliceAO_LOD(float3 pos, float3 nor, int tier)
{
    int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                  (tier == ALICE_LOD_TIER_MED)  ? 3 : 2;

    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < aoSteps; i++)
    {
        float h = 0.01 + 0.12 * float(i) / float(max(aoSteps - 1, 1));
        float d = map(pos + h * nor);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return saturate(1.0 - 3.0 * occ);
}

// =============================================================================
// Soft Shadow with LOD (fewer steps when far)
// =============================================================================

#define ALICE_SHADOW_STEPS_HIGH  48
#define ALICE_SHADOW_STEPS_MED   24
#define ALICE_SHADOW_STEPS_LOW   12

float aliceSoftShadow_LOD(float3 ro, float3 rd, float mint, float maxt, float softness, int tier)
{
    int maxSteps = (tier == ALICE_LOD_TIER_HIGH) ? ALICE_SHADOW_STEPS_HIGH :
                   (tier == ALICE_LOD_TIER_MED)  ? ALICE_SHADOW_STEPS_MED :
                                                   ALICE_SHADOW_STEPS_LOW;
    float res = 1.0;
    float t = mint;
    float ph = 1e20;
    for (int i = 0; i < 48; i++)
    {
        if (i >= maxSteps) break;
        float h = map(ro + rd * t);
        if (h < 0.0001)
            return 0.0;
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, softness * d / max(0.0, t - y));
        ph = h;
        t += h;
        if (t > maxt) break;
    }
    return saturate(res);
}

// =============================================================================
// GI Approximation (2-bounce indirect light estimation)
// =============================================================================

float3 aliceGI_LOD(float3 pos, float3 nor, float3 lightDir, float3 lightColor, int tier)
{
    // 1st bounce: sample in light direction reflected off surface
    float3 bounceDir = reflect(-lightDir, nor);
    float bounceDist = (tier == ALICE_LOD_TIER_HIGH) ? 0.5 : 1.0;
    float3 bouncePos = pos + nor * 0.1 + bounceDir * bounceDist;
    float d = map(bouncePos);
    float occlusion = saturate(d / bounceDist);

    // 2nd bounce: hemisphere sampling (approximated by normal-biased sample)
    float3 hemiPos = pos + nor * 0.3;
    float d2 = map(hemiPos);
    float hemiOcc = saturate(d2 / 0.3);

    // Combine: indirect color is surface color tinted by light, attenuated by occlusion
    float3 indirect = lightColor * 0.15 * occlusion * hemiOcc;

    // Sky contribution (top hemisphere gets more light)
    float skyAmount = saturate(nor.y * 0.5 + 0.5);
    indirect += float3(0.05, 0.08, 0.12) * skyAmount * hemiOcc;

    return indirect;
}

#endif // ALICE_SDF_LOD
