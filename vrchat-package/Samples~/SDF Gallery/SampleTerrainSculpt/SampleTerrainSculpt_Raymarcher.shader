// =============================================================================
// ALICE-SDF Sample: Terrain Sculpting (Dig & Build)
// =============================================================================
// A flat ground that players can sculpt in real-time. VR: left hand adds
// terrain (SmoothUnion), right hand digs (SmoothSubtraction). Desktop: the
// left button raises, the right button digs, at the view cursor.
// Both rendering and collision use the exact same SDF — dig a hole and you
// actually fall in. Build a hill and you can climb it.
//
// This is impossible with traditional VRChat meshes because MeshColliders
// cannot be updated at runtime. ALICE-SDF evaluates the same formula for
// both pixels and physics, so visual = collision at all times.
//
// Sculpt data is sent from UdonSharp via Material.SetVectorArray.
// Up to 128 stored sculpt operations (circular buffer, Sculpt Capacity on the collider).
//
// Rendering notes (same fixes as the Mochi sample):
//   - Cull Off: the player walks inside the volume cube, so its back faces
//     must run the raymarch too
//   - closest-approach acceptance: a ray grazing a hill's silhouette runs
//     out of steps a hair short of the surface; treated as a miss it wrote
//     the far depth and the world behind showed through as a thin line
//   - AO samples the hard (k = 0) union: the smooth blend zone under-reports
//     distance and read as a dark ring around every hill and hole
//   - the ground plane is at _Origin.y (set by the collider from
//     transform.position + groundOffset); the march stops where the ray
//     leaves the volume cube, so the ground is drawn only under the cube
//   - far-ground fast path (exact where it applies): a ray that stays
//     farther than the blend inflation from every sculpt sphere can only
//     meet the plane, so it takes the analytic plane hit (+y normal, AO 1)
//     and marches the shadow ray only if it can pass near a sculpt
//
// Author: Moroya Sakamoto
// =============================================================================

Shader "AliceSDF/Samples/TerrainSculpt"
{
    Properties
    {
        [Header(Terrain Colors)]
        _GrassColor ("Grass", Color) = (0.35, 0.55, 0.25, 1.0)
        _DirtColor ("Dirt", Color) = (0.55, 0.40, 0.25, 1.0)
        _RockColor ("Rock (Underground)", Color) = (0.40, 0.38, 0.35, 1.0)

        [Header(Texture)]
        // Driven by the collider's Look section; triplanar in world space
        [NoScaleOffset] _GroundTex ("Surface Texture", 2D) = "white" {}
        _GroundTexScale ("Surface Texture Tiles per Metre", Float) = 0.5
        _GroundTexStrength ("Surface Texture Strength", Range(0, 1)) = 0.0

        [Header(Placement)]
        // Driven by the collider: the ground plane is at _Origin.y
        _Origin ("Ground Origin (xyz)", Vector) = (0, 0, 0, 0)

        [Header(Cursor)]
        _AddCursorColor ("Add Cursor (Left Hand / Left Click)", Color) = (0.3, 0.6, 1.0, 1.0)
        _SubCursorColor ("Dig Cursor (Right Hand / Right Click)", Color) = (1.0, 0.3, 0.2, 1.0)
        // Driven every frame by SampleTerrainSculpt_Collider (xyz = position, w = 1 visible)
        _LeftHand ("Add Cursor Position", Vector) = (0, 0, 0, 0)
        _RightHand ("Dig Cursor Position", Vector) = (0, 0, 0, 0)

        [Header(Raymarching)]
        _MaxDist ("Max Distance", Float) = 100.0

        [Header(Lighting)]
        _LightDir ("Light Direction", Vector) = (1.0, 1.5, -0.5, 0.0)
        _ShadowEnabled ("Enable Soft Shadow", Int) = 1
        _ShadowSoftness ("Shadow Softness", Range(1, 128)) = 16.0
        _ShadowMaxDist ("Shadow Max Distance", Float) = 10.0

        [Header(Sculpting)]
        _AddSmooth ("Add Smoothness", Float) = 0.25
        _SubSmooth ("Dig Smoothness", Float) = 0.15
        _SculptRadius ("Sculpt Radius (for cursor)", Float) = 0.5
        // Driven by the collider: 1 = block brush (cubes), 0 = spheres
        _BrushShape ("Brush Shape (0 sphere, 1 block)", Float) = 1

        [Header(Fog)]
        _FogColor ("Fog Color", Color) = (0.70, 0.80, 0.90, 1.0)
        _FogDensity ("Fog Density", Float) = 0.004
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            Cull Off
            ZWrite On
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 3.0
            #pragma multi_compile_instancing
            #include "UnityCG.cginc"
            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_Include.cginc"

            // Inspector properties
            float4 _GrassColor, _DirtColor, _RockColor, _FogColor;
            float4 _AddCursorColor, _SubCursorColor;
            sampler2D _GroundTex;
            float _GroundTexScale, _GroundTexStrength;
            float4 _Origin;
            float _MaxDist;
            float _AddSmooth, _SubSmooth, _SculptRadius, _BrushShape;
            float4 _LightDir;
            int _ShadowEnabled;
            float _ShadowSoftness, _ShadowMaxDist, _FogDensity;

            // Closest-approach acceptance, metres per metre of ray length
            // (~1 px at a 60 deg / 1000 px view)
            #define NEAR_MISS_PER_M 0.002

            // Dynamic sculpt data (set from UdonSharp)
            // xyz = world position, w = radius (positive = add, negative = dig)
            float4 _SculptData[128];
            float _SculptCount;

            // Hand cursor positions (xyz = pos, w = 1 if near terrain, 0 if not)
            float4 _LeftHand;
            float4 _RightHand;

            struct appdata {
                float4 vertex : POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };
            struct v2f {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
                float3 rayDir : TEXCOORD1;
                float3 objectCenter : TEXCOORD2;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            // =================================================================
            // SDF: Ground Plane + Dynamic Sculpt Operations
            // Every stored sculpt is folded, in slot order, exactly as the
            // collider does. No distance culling: an earlier version skipped a
            // sphere farther than r + 2k from the point and returned the plane
            // distance instead, which above a tall stack is larger than the
            // true distance, so rays overshot into the terrain and stopped
            // inside it (black cavities and a floating cap on every column).
            // =================================================================
            // One sculpt operation's distance: a cube of half-side r (block
            // brush) or a sphere of radius r, identical to the collider's Brush()
            float brush(float3 d, float r)
            {
                return _BrushShape > 0.5 ? sdBox(d, float3(r, r, r)) : sdSphere(d, r);
            }

            float map(float3 p)
            {
                // Base terrain: flat ground at _Origin.y
                float terrain = p.y - _Origin.y;

                // Apply sculpt operations in order
                // Deep Fried: skip sculpts whose sphere of influence is too far
                int count = (int)_SculptCount;
                for (int i = 0; i < 128; i++)
                {
                    if (i >= count) break;

                    float3 sp = _SculptData[i].xyz;
                    float rw = _SculptData[i].w;
                    float3 delta = p - sp;

                    if (rw > 0.001)
                    {
                        // Add terrain (left hand): SmoothUnion
                        float hill = brush(delta, rw);
                        terrain = opSmoothUnion(terrain, hill, _AddSmooth);
                    }
                    else if (rw < -0.001)
                    {
                        // Dig terrain (right hand): SmoothSubtraction
                        float hole = brush(delta, -rw);
                        terrain = opSmoothSubtraction(terrain, hole, _SubSmooth);
                    }
                }

                return terrain;
            }

            // The same terrain with hard (k = 0) operations: exact outside the
            // spheres and the plane, so only real geometry occludes. The smooth
            // blend zone reports distances up to k/4 short, which the AO
            // integral (h - d) read as occlusion: a dark ring at the foot of
            // every hill and around every hole.
            float mapOccluder(float3 p)
            {
                float terrain = p.y - _Origin.y;
                int count = (int)_SculptCount;
                for (int i = 0; i < 128; i++)
                {
                    if (i >= count) break;
                    float3 sp = _SculptData[i].xyz;
                    float rw = _SculptData[i].w;
                    float3 delta = p - sp;
                    if (rw > 0.001)
                        terrain = min(terrain, brush(delta, rw));
                    else if (rw < -0.001)
                        terrain = max(terrain, -brush(delta, -rw));
                }
                return terrain;
            }

            #include "Packages/com.alice.sdf/Runtime/Shaders/AliceSDF_LOD.cginc"

            // Central-difference half-width for the normal, wider than the
            // surface epsilon so it averages over the hit tolerance
            float normalEps(int tier) {
                if (tier == ALICE_LOD_TIER_HIGH) return 0.001;
                if (tier == ALICE_LOD_TIER_MED)  return 0.003;
                return 0.01;
            }

            // Normal via central differences
            float3 calcN(float3 p, float e) {
                return normalize(float3(
                    map(p + float3(e,0,0)) - map(p - float3(e,0,0)),
                    map(p + float3(0,e,0)) - map(p - float3(0,e,0)),
                    map(p + float3(0,0,e)) - map(p - float3(0,0,e))
                ));
            }

            // AO against the hard union, sample count by LOD tier
            float terrainAO(float3 p, float3 n, int tier) {
                #ifdef ALICE_MOBILE_BUDGET
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 2 : 1;
                #else
                int aoSteps = (tier == ALICE_LOD_TIER_HIGH) ? 5 :
                              (tier == ALICE_LOD_TIER_MED)  ? 3 : 2;
                #endif
                float occ = 0.0;
                float sca = 1.0;
                for (int i = 0; i < 5; i++) {
                    if (i >= aoSteps) break;
                    float h = 0.01 + 0.12 * float(i) / float(max(aoSteps - 1, 1));
                    occ += (h - mapOccluder(p + h * n)) * sca;
                    sca *= 0.95;
                }
                return saturate(1.0 - 3.0 * occ);
            }

            // Simple noise for texture variation
            float hash2d(float2 p) {
                p = frac(p * float2(0.3183, 0.3671));
                p *= 17.0;
                return frac(p.x * p.y * (p.x + p.y));
            }

            // Triplanar sample: the three axis projections of q blended by the
            // normal, so hills and holes show the texture without UVs
            float3 triplanar(sampler2D tex, float3 q, float3 n)
            {
                float3 w = abs(n);
                w = w / max(w.x + w.y + w.z, 1e-4);
                float3 cx = tex2D(tex, q.yz).rgb;
                float3 cy = tex2D(tex, q.xz).rgb;
                float3 cz = tex2D(tex, q.xy).rgb;
                return cx * w.x + cy * w.y + cz * w.z;
            }

            // Height + normal based terrain coloring
            float3 terrainColor(float3 p, float3 n) {
                float h = p.y - _Origin.y;
                float top = saturate(n.y); // 1 = flat top, 0 = vertical/underside

                // Texture variation
                float noise = hash2d(p.xz * 3.0) * 0.12;

                // Surface color: flat = grass, steep = dirt
                float3 grass = _GrassColor.rgb * (0.94 + noise);
                float3 dirt = _DirtColor.rgb * (0.94 + noise);
                float3 rock = _RockColor.rgb * (0.92 + noise);

                float3 surfaceCol = lerp(dirt, grass, top);
                if (_GroundTexStrength > 0.0)
                    surfaceCol = lerp(surfaceCol, triplanar(_GroundTex, (p - _Origin.xyz) * _GroundTexScale, n), _GroundTexStrength);

                // Underground blend: deeper = more rock
                float underground = smoothstep(0.0, -0.5, h);

                return lerp(surfaceCol, rock, underground);
            }

            // Cursor overlay: colored glow near each hand
            float3 cursorOverlay(float3 p) {
                float3 overlay = float3(0, 0, 0);

                if (_LeftHand.w > 0.5) {
                    float d = length(p - _LeftHand.xyz);
                    float glow = smoothstep(_SculptRadius * 1.5, _SculptRadius * 0.2, d);
                    overlay += _AddCursorColor.rgb * glow * 0.45;
                }
                if (_RightHand.w > 0.5) {
                    float d = length(p - _RightHand.xyz);
                    float glow = smoothstep(_SculptRadius * 1.5, _SculptRadius * 0.2, d);
                    overlay += _SubCursorColor.rgb * glow * 0.45;
                }

                return overlay;
            }

            // =================================================================
            // Far-ground fast path (exact where it applies, see the header)
            // =================================================================
            // Blend inflation: a chain of polynomial smooth unions lowers the
            // nearest-sphere distance by well under one k in practice (k / 4
            // for the first blend, geometrically less for each further one)
            float sculptInflation()
            {
                return max(_AddSmooth, _SubSmooth) * 1.5 + 0.2;
            }

            // True if the segment ro + rd * [0, len] passes within (|r_i| + infl)
            // of any sculpt centre
            bool segmentNearSculpts(float3 ro, float3 rd, float len, float infl)
            {
                int count = (int)_SculptCount;
                for (int i = 0; i < 128; i++)
                {
                    if (i >= count) break;
                    float3 c = _SculptData[i].xyz;
                    float tc = clamp(dot(c - ro, rd), 0.0, len);
                    float3 q = ro + rd * tc - c;
                    // a block's bounding sphere is sqrt(3) times its half-side
                    float R = abs(_SculptData[i].w) * (_BrushShape > 0.5 ? 1.7321 : 1.0) + infl;
                    if (dot(q, q) < R * R) return true;
                }
                return false;
            }

            // Soft shadow of the bare plane: the same loop as
            // aliceSoftShadow_LOD with map replaced by the plane distance
            float planeSoftShadow_LOD(float3 ro, float3 rd, float mint, float maxt, float softness, int tier)
            {
                int maxSteps = (tier == ALICE_LOD_TIER_HIGH) ? ALICE_SHADOW_STEPS_HIGH :
                               (tier == ALICE_LOD_TIER_MED)  ? ALICE_SHADOW_STEPS_MED :
                                                               ALICE_SHADOW_STEPS_LOW;
                if (maxSteps <= 0) return 1.0;
                float res = 1.0;
                float t = mint;
                float ph = 1e20;
                for (int i = 0; i < 48; i++)
                {
                    if (i >= maxSteps) break;
                    float h = (ro + rd * t).y - _Origin.y;
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

            // Distance along the ray to where it leaves this object's unit
            // cube (object space slab test, mapped back to a world distance)
            float exitDistance(float3 ro, float3 rd)
            {
                float3 roObj = mul(unity_WorldToObject, float4(ro, 1.0)).xyz;
                float3 rdObj = mul((float3x3)unity_WorldToObject, rd);
                float3 inv = 1.0 / (abs(rdObj) < 1e-6 ? (rdObj < 0.0 ? -1e-6 : 1e-6) : rdObj);
                float3 t0 = (-0.5 - roObj) * inv;
                float3 t1 = ( 0.5 - roObj) * inv;
                float3 tmax = max(t0, t1);
                float tObj = min(tmax.x, min(tmax.y, tmax.z));
                float3 exitObj = roObj + rdObj * tObj;
                float3 exitWorld = mul(unity_ObjectToWorld, float4(exitObj, 1.0)).xyz;
                return length(exitWorld - ro);
            }

            v2f vert(appdata v) {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.rayDir = o.worldPos - _WorldSpaceCameraPos;
                o.objectCenter = mul(unity_ObjectToWorld, float4(0,0,0,1)).xyz;
                return o;
            }

            struct FragOutput { fixed4 color : SV_Target; float depth : SV_Depth; };

            FragOutput frag(v2f i) {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i);
                float3 ro = _WorldSpaceCameraPos;
                float3 rd = normalize(i.rayDir);
                // LOD: adapt steps based on camera distance
                float camDist = length(i.objectCenter - ro);
                int tier = aliceLodTier(camDist);
                int maxSteps = aliceLodSteps(tier);
                float eps = aliceLodEpsilon(tier);
                float ss = aliceLodStepScale(tier);
                float t = 0.0;
                float tExit = exitDistance(ro, rd);
                FragOutput o;

                // Closest approach along the ray: a ray grazing a hill's
                // silhouette takes ever smaller steps and runs out of budget
                // a hair short of the surface. Dropped as a miss it wrote the
                // far depth and whatever the world had behind the volume showed
                // through as a thin line along every contour. If the budget
                // runs out within one pixel footprint, the closest point is the hit.
                float bestD = 1e10;
                float bestT = 0.0;
                bool hit = false;
                bool farGround = false;

                float infl = sculptInflation();
                bool nearSculpt = (ro.y - _Origin.y) < eps
                               || segmentNearSculpts(ro, rd, min(tExit, _MaxDist), infl);
                if (!nearSculpt) {
                    if (rd.y < -1e-6) {
                        // Stop eps short of the plane like the march does (its
                        // hit is the first sample with d < eps): a coplanar floor
                        // mesh would otherwise z-fight the exact plane
                        float tp = (_Origin.y - ro.y) / rd.y - eps;
                        if (tp > 0.0 && tp <= tExit && tp <= _MaxDist) { t = tp; hit = true; farGround = true; }
                    }
                } else {
                for (int k = 0; k < 128; k++) {
                    if (k >= maxSteps) break;
                    float d = map(ro + rd * t);

                    if (d < eps) { hit = true; break; }
                    if (d < bestD) { bestD = d; bestT = t; }

                    t += d * ss;
                    if (t > _MaxDist || t > tExit) break;
                }
                if (!hit && bestD < max(eps, bestT * NEAR_MISS_PER_M)) {
                    t = bestT;
                    hit = true;
                }
                }

                if (hit) {
                    float3 p = ro + rd * t;
                    float3 n = farGround ? float3(0.0, 1.0, 0.0) : calcN(p, normalEps(tier));
                    float3 lightDir = normalize(_LightDir.xyz);
                    float ao = farGround ? 1.0 : terrainAO(p, n, tier);

                    // Direct-light visibility (hill -> ground contact shadow)
                    float shadow = 1.0;
                    if (_ShadowEnabled > 0)
                    {
                        float3 sro = p + n * 0.02;
                        if (farGround && !segmentNearSculpts(sro, lightDir, _ShadowMaxDist,
                                                             infl + _ShadowMaxDist / max(_ShadowSoftness, 1.0)))
                            shadow = planeSoftShadow_LOD(sro, lightDir, 0.02, _ShadowMaxDist, _ShadowSoftness, tier);
                        else
                            shadow = aliceSoftShadow_LOD(sro, lightDir, 0.02,
                                                         _ShadowMaxDist, _ShadowSoftness, tier);
                    }

                    // Terrain color
                    float3 baseCol = terrainColor(p, n);

                    // Lighting
                    float diff = max(dot(n, lightDir), 0.0);
                    float3 fc = baseCol * (0.25 + diff * 0.75 * shadow) * ao;

                    // Hand / cursor overlay
                    fc += cursorOverlay(p);

                    // Fog
                    fc = lerp(_FogColor.rgb, fc, exp(-t * _FogDensity));

                    float4 cp = UnityWorldToClipPos(p);
                    o.color = fixed4(fc, 1.0);
                    #if defined(UNITY_REVERSED_Z)
                        o.depth = cp.z / cp.w;
                    #else
                        o.depth = (cp.z / cp.w) * 0.5 + 0.5;
                    #endif
                    return o;
                }

                o.color = fixed4(_FogColor.rgb, 1.0);
                #if defined(UNITY_REVERSED_Z)
                    o.depth = 0.0;
                #else
                    o.depth = 1.0;
                #endif
                return o;
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}
