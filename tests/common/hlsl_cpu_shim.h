// HLSL / BlinkScript host shim for the ALICE-SDF transpiler value oracle.
//
// The emitted `float sdf_eval(float3 p)` body is compiled by a *C++* compiler
// against this header and executed on the CPU, so the parse is done by clang
// and every intrinsic below is defined from the **HLSL intrinsic reference**
// (Microsoft `Direct3D 11 / Shader Model 5` docs and the HLSL specification),
// never by reading ALICE-SDF's Rust evaluator. A transpiler that emits the
// wrong intrinsic, the wrong operand order or a wrong constant therefore
// cannot agree with this shim by construction.
//
// Deliberate properties:
//
//   * **No standard headers.** `<cmath>` / `<cstdlib>` would put `abs`,
//     `sqrt`, `floor`, `fmod` ... into the global namespace and every call in
//     the emit would become ambiguous against the definitions below. All
//     scalar math goes through clang `__builtin_*`, all I/O through raw
//     `read`/`write` declared by hand.
//   * **`fmod` is truncated** (`x - y * trunc(x / y)`), which is what HLSL
//     specifies. The floor-modulo the CPU law uses is emitted *explicitly* by
//     `HlslLang::modulo_expr`, so a regression that swaps the two shows up as
//     a value drift rather than silently agreeing.
//   * **Swizzles are storage aliases**, matching HLSL semantics, so
//     `length(p.xz)` reads the same bytes the emit means. `swizzle_selftest`
//     in the Rust harness pins them against hand-computed values.
//
// Author: Moroya Sakamoto

#ifndef ALICE_SDF_HLSL_CPU_SHIM_H
#define ALICE_SDF_HLSL_CPU_SHIM_H

typedef unsigned int uint;

// ---------------------------------------------------------------------------
// Raw I/O (declared by hand so no libc header leaks math names in)
// ---------------------------------------------------------------------------
extern "C" long read(int, void *, unsigned long);
extern "C" long write(int, const void *, unsigned long);

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------
struct float2;
struct float3;
struct float4;

// A swizzle is a *view* over the parent's storage: `N` is the parent's
// component count, `A`/`B`/`C` the selected lanes.
template <int N, int A, int B> struct swz2 {
    float d[N];
    inline operator float2() const;
};
template <int N, int A, int B, int C> struct swz3 {
    float d[N];
    inline operator float3() const;
};

struct float2 {
    float x, y;
    float2() : x(0.0f), y(0.0f) {}
    float2(float a, float b) : x(a), y(b) {}
    // HLSL vectors are dynamically indexable.
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

// Integer vectors, for the `asuint` / `asint` bit-cast paths in the noise laws.
struct int3 {
    int x, y, z;
    int3() : x(0), y(0), z(0) {}
    int3(int a, int b, int c) : x(a), y(b), z(c) {}
    int &operator[](int i) { return (&x)[i]; }
    int operator[](int i) const { return (&x)[i]; }
};

struct uint3 {
    uint x, y, z;
    uint3() : x(0u), y(0u), z(0u) {}
    uint3(uint a, uint b, uint c) : x(a), y(b), z(c) {}
    uint &operator[](int i) { return (&x)[i]; }
    uint operator[](int i) const { return (&x)[i]; }
};

struct float3 {
    union {
        struct {
            float x, y, z;
        };
        swz2<3, 0, 1> xy;
        swz2<3, 0, 2> xz;
        swz2<3, 1, 2> yz;
        swz2<3, 1, 0> yx;
        swz2<3, 2, 0> zx;
        swz3<3, 0, 1, 2> xyz;
    };
    float3() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }
    float3(float a, float b, float c) {
        x = a;
        y = b;
        z = c;
    }
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct float4 {
    union {
        struct {
            float x, y, z, w;
        };
        swz2<4, 0, 1> xy;
        swz2<4, 0, 2> xz;
        swz2<4, 1, 2> yz;
        swz2<4, 2, 3> zw;
        swz3<4, 0, 1, 2> xyz;
        swz3<4, 1, 2, 3> yzw;
    };
    float4() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        w = 0.0f;
    }
    float4(float a, float b, float c, float d_) {
        x = a;
        y = b;
        z = c;
        w = d_;
    }
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

template <int N, int A, int B> inline swz2<N, A, B>::operator float2() const {
    return float2(d[A], d[B]);
}
template <int N, int A, int B, int C> inline swz3<N, A, B, C>::operator float3() const {
    return float3(d[A], d[B], d[C]);
}

// ---------------------------------------------------------------------------
// Arithmetic operators (component-wise, scalar broadcast) — HLSL §Operators
// ---------------------------------------------------------------------------
#define ALICE_V2_BINOP(OP)                                                                         \
    inline float2 operator OP(float2 a, float2 b) { return float2(a.x OP b.x, a.y OP b.y); }       \
    inline float2 operator OP(float2 a, float s) { return float2(a.x OP s, a.y OP s); }            \
    inline float2 operator OP(float s, float2 b) { return float2(s OP b.x, s OP b.y); }
#define ALICE_V3_BINOP(OP)                                                                         \
    inline float3 operator OP(float3 a, float3 b) {                                                \
        return float3(a.x OP b.x, a.y OP b.y, a.z OP b.z);                                         \
    }                                                                                              \
    inline float3 operator OP(float3 a, float s) { return float3(a.x OP s, a.y OP s, a.z OP s); }  \
    inline float3 operator OP(float s, float3 b) { return float3(s OP b.x, s OP b.y, s OP b.z); }
#define ALICE_V4_BINOP(OP)                                                                         \
    inline float4 operator OP(float4 a, float4 b) {                                                \
        return float4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w);                             \
    }                                                                                              \
    inline float4 operator OP(float4 a, float s) {                                                 \
        return float4(a.x OP s, a.y OP s, a.z OP s, a.w OP s);                                     \
    }                                                                                              \
    inline float4 operator OP(float s, float4 b) {                                                 \
        return float4(s OP b.x, s OP b.y, s OP b.z, s OP b.w);                                     \
    }

ALICE_V2_BINOP(+)
ALICE_V2_BINOP(-)
ALICE_V2_BINOP(*)
ALICE_V2_BINOP(/)
ALICE_V3_BINOP(+)
ALICE_V3_BINOP(-)
ALICE_V3_BINOP(*)
ALICE_V3_BINOP(/)
ALICE_V4_BINOP(+)
ALICE_V4_BINOP(-)
ALICE_V4_BINOP(*)
ALICE_V4_BINOP(/)

#undef ALICE_V2_BINOP
#undef ALICE_V3_BINOP
#undef ALICE_V4_BINOP

inline float2 operator-(float2 a) { return float2(-a.x, -a.y); }
inline float3 operator-(float3 a) { return float3(-a.x, -a.y, -a.z); }
inline float4 operator-(float4 a) { return float4(-a.x, -a.y, -a.z, -a.w); }

#define ALICE_ASSIGN_OP(T, OP)                                                                     \
    inline T &operator OP##=(T & a, T b) {                                                         \
        a = a OP b;                                                                                \
        return a;                                                                                  \
    }                                                                                              \
    inline T &operator OP##=(T & a, float s) {                                                     \
        a = a OP s;                                                                                \
        return a;                                                                                  \
    }
ALICE_ASSIGN_OP(float2, +)
ALICE_ASSIGN_OP(float2, -)
ALICE_ASSIGN_OP(float2, *)
ALICE_ASSIGN_OP(float2, /)
ALICE_ASSIGN_OP(float3, +)
ALICE_ASSIGN_OP(float3, -)
ALICE_ASSIGN_OP(float3, *)
ALICE_ASSIGN_OP(float3, /)
ALICE_ASSIGN_OP(float4, +)
ALICE_ASSIGN_OP(float4, -)
ALICE_ASSIGN_OP(float4, *)
ALICE_ASSIGN_OP(float4, /)
#undef ALICE_ASSIGN_OP

// ---------------------------------------------------------------------------
// Scalar intrinsics — HLSL intrinsic reference
// ---------------------------------------------------------------------------
inline float abs(float v) { return __builtin_fabsf(v); }
inline float sqrt(float v) { return __builtin_sqrtf(v); }
inline float floor(float v) { return __builtin_floorf(v); }
inline float ceil(float v) { return __builtin_ceilf(v); }
inline float trunc(float v) { return __builtin_truncf(v); }
inline float round(float v) { return __builtin_roundf(v); }
inline float sin(float v) { return __builtin_sinf(v); }
inline float cos(float v) { return __builtin_cosf(v); }
inline float tan(float v) { return __builtin_tanf(v); }
inline float asin(float v) { return __builtin_asinf(v); }
inline float acos(float v) { return __builtin_acosf(v); }
inline float atan(float v) { return __builtin_atanf(v); }
inline float atan2(float y, float x) { return __builtin_atan2f(y, x); }
inline float log(float v) { return __builtin_logf(v); }
inline float log2(float v) { return __builtin_log2f(v); }
inline float exp(float v) { return __builtin_expf(v); }
inline float exp2(float v) { return __builtin_exp2f(v); }
inline float pow(float a, float b) { return __builtin_powf(a, b); }
inline float rsqrt(float v) { return 1.0f / __builtin_sqrtf(v); }
inline float rcp(float v) { return 1.0f / v; }
inline float frac(float v) { return v - __builtin_floorf(v); }
inline bool isnan(float v) { return __builtin_isnan(v); }
inline bool isinf(float v) { return __builtin_isinf(v); }

// HLSL `fmod` is the *truncated* remainder: `x - y * trunc(x / y)`.
inline float fmod(float a, float b) { return a - b * __builtin_truncf(a / b); }

// HLSL `sign` returns -1 / 0 / +1 as a float.
inline float sign(float v) { return v < 0.0f ? -1.0f : (v > 0.0f ? 1.0f : 0.0f); }

inline float min(float a, float b) { return a < b ? a : b; }
inline float max(float a, float b) { return a > b ? a : b; }
inline float clamp(float v, float lo, float hi) { return min(max(v, lo), hi); }
inline float saturate(float v) { return clamp(v, 0.0f, 1.0f); }
// HLSL `lerp(a, b, s)` == `a + s * (b - a)`.
inline float lerp(float a, float b, float s) { return a + s * (b - a); }
inline float step(float y, float x) { return x >= y ? 1.0f : 0.0f; }
inline float mad(float a, float b, float c) { return a * b + c; }

// ---------------------------------------------------------------------------
// Bit reinterpretation — HLSL `asint` / `asuint` / `asfloat`
// ---------------------------------------------------------------------------
inline int asint(float v) {
    int out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline uint asuint(float v) {
    uint out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline float asfloat(uint v) {
    float out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline float asfloat(int v) {
    float out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline int asint(uint v) { return static_cast<int>(v); }
inline uint asuint(int v) { return static_cast<uint>(v); }

// Component-wise bit casts (HLSL applies `asuint` / `asint` per component).
inline uint3 asuint(float3 v) { return uint3(asuint(v.x), asuint(v.y), asuint(v.z)); }
inline int3 asint(float3 v) { return int3(asint(v.x), asint(v.y), asint(v.z)); }
inline float3 asfloat(uint3 v) { return float3(asfloat(v.x), asfloat(v.y), asfloat(v.z)); }
inline float3 asfloat(int3 v) { return float3(asfloat(v.x), asfloat(v.y), asfloat(v.z)); }

// ---------------------------------------------------------------------------
// Component-wise vector intrinsics
// ---------------------------------------------------------------------------
#define ALICE_V_UNARY(NAME)                                                                        \
    inline float2 NAME(float2 v) { return float2(NAME(v.x), NAME(v.y)); }                          \
    inline float3 NAME(float3 v) { return float3(NAME(v.x), NAME(v.y), NAME(v.z)); }               \
    inline float4 NAME(float4 v) { return float4(NAME(v.x), NAME(v.y), NAME(v.z), NAME(v.w)); }
ALICE_V_UNARY(abs)
ALICE_V_UNARY(sqrt)
ALICE_V_UNARY(floor)
ALICE_V_UNARY(ceil)
ALICE_V_UNARY(trunc)
ALICE_V_UNARY(round)
ALICE_V_UNARY(sin)
ALICE_V_UNARY(cos)
ALICE_V_UNARY(tan)
ALICE_V_UNARY(asin)
ALICE_V_UNARY(acos)
ALICE_V_UNARY(atan)
ALICE_V_UNARY(log)
ALICE_V_UNARY(exp)
ALICE_V_UNARY(sign)
ALICE_V_UNARY(frac)
ALICE_V_UNARY(saturate)
ALICE_V_UNARY(rsqrt)
#undef ALICE_V_UNARY

#define ALICE_V_BINARY(NAME)                                                                       \
    inline float2 NAME(float2 a, float2 b) { return float2(NAME(a.x, b.x), NAME(a.y, b.y)); }      \
    inline float2 NAME(float2 a, float s) { return float2(NAME(a.x, s), NAME(a.y, s)); }           \
    inline float2 NAME(float s, float2 b) { return float2(NAME(s, b.x), NAME(s, b.y)); }           \
    inline float3 NAME(float3 a, float3 b) {                                                       \
        return float3(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z));                             \
    }                                                                                              \
    inline float3 NAME(float3 a, float s) {                                                        \
        return float3(NAME(a.x, s), NAME(a.y, s), NAME(a.z, s));                                   \
    }                                                                                              \
    inline float3 NAME(float s, float3 b) {                                                        \
        return float3(NAME(s, b.x), NAME(s, b.y), NAME(s, b.z));                                   \
    }                                                                                              \
    inline float4 NAME(float4 a, float4 b) {                                                       \
        return float4(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z), NAME(a.w, b.w));             \
    }                                                                                              \
    inline float4 NAME(float4 a, float s) {                                                        \
        return float4(NAME(a.x, s), NAME(a.y, s), NAME(a.z, s), NAME(a.w, s));                     \
    }
ALICE_V_BINARY(min)
ALICE_V_BINARY(max)
ALICE_V_BINARY(pow)
ALICE_V_BINARY(fmod)
ALICE_V_BINARY(atan2)
ALICE_V_BINARY(step)
#undef ALICE_V_BINARY

inline float2 clamp(float2 v, float2 lo, float2 hi) {
    return float2(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y));
}
inline float2 clamp(float2 v, float lo, float hi) {
    return float2(clamp(v.x, lo, hi), clamp(v.y, lo, hi));
}
inline float3 clamp(float3 v, float3 lo, float3 hi) {
    return float3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
inline float3 clamp(float3 v, float lo, float hi) {
    return float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

inline float2 lerp(float2 a, float2 b, float s) { return a + s * (b - a); }
inline float2 lerp(float2 a, float2 b, float2 s) { return a + s * (b - a); }
inline float3 lerp(float3 a, float3 b, float s) { return a + s * (b - a); }
inline float3 lerp(float3 a, float3 b, float3 s) { return a + s * (b - a); }
inline float4 lerp(float4 a, float4 b, float s) { return a + s * (b - a); }

// ---------------------------------------------------------------------------
// Geometric intrinsics
// ---------------------------------------------------------------------------
inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline float length(float2 v) { return sqrt(dot(v, v)); }
inline float length(float3 v) { return sqrt(dot(v, v)); }
inline float length(float4 v) { return sqrt(dot(v, v)); }

inline float2 normalize(float2 v) { return v / length(v); }
inline float3 normalize(float3 v) { return v / length(v); }
inline float4 normalize(float4 v) { return v / length(v); }

inline float3 cross(float3 a, float3 b) {
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline bool any(float2 v) { return v.x != 0.0f || v.y != 0.0f; }
inline bool all(float2 v) { return v.x != 0.0f && v.y != 0.0f; }

#endif // ALICE_SDF_HLSL_CPU_SHIM_H
