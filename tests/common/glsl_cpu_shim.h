// GLSL host shim for the ALICE-SDF transpiler value oracle.
//
// An emitted GLSL function body (`float sdf_eval(vec3 p)`, `vec2 sdf_eval_scaled(vec3 p)`) is
// compiled by a *C++* compiler against this header and run on the CPU, so the parse is done by
// clang / gcc and every built-in below is defined from the **OpenGL Shading Language
// specification** (4.50, §8 Built-in Functions; GLSL ES 3.00 agrees on every one used here),
// never by reading ALICE-SDF's Rust evaluator. A transpiler that emits the wrong built-in, the
// wrong operand order or a wrong constant therefore cannot agree with this shim by
// construction.
//
// Deliberate properties (as in `hlsl_cpu_shim.h`, which this mirrors with GLSL's semantics):
//
//   * **No standard headers.** All scalar math goes through clang `__builtin_*`, all I/O through
//     raw `read` / `write` declared by hand, so no libc math name makes an emitted call
//     ambiguous.
//   * **GLSL, not HLSL, semantics where they differ:** `mod(x, y) = x − y·floor(x / y)`,
//     `mix(x, y, a) = x·(1 − a) + y·a`, `fract(x) = x − floor(x)`, `inversesqrt`, the two-argument
//     `atan(y, x)`, and the broadcast / concatenating constructors (`vec3(s)`, `vec3(v2, z)`).
//   * **Two things the harness rewrites rather than this header** (C++ cannot spell them):
//     the array constructor `const float a[N] = float[N](...);` becomes a brace initializer, and
//     `ALICE_PRECISE` is predefined empty (the emitted block tests `__VERSION__`, which is a
//     string in C++). Contraction is forbidden for the whole unit by `-ffp-contract=off`, which is
//     what `precise` asks of a driver.
//
// Author: Moroya Sakamoto

#ifndef ALICE_SDF_GLSL_CPU_SHIM_H
#define ALICE_SDF_GLSL_CPU_SHIM_H

typedef unsigned int uint;

// ---------------------------------------------------------------------------
// Raw I/O (declared by hand so no libc header leaks math names in)
// ---------------------------------------------------------------------------
extern "C" long read(int, void *, unsigned long);
extern "C" long write(int, const void *, unsigned long);

// fd 0 / 1 in binary mode on Windows (text mode stops reading at 0x1A; see hlsl_cpu_shim.h)
#ifdef _WIN32
extern "C" int _setmode(int, int);
#define ALICE_SDF_SET_BINARY_IO()                                                                  \
    do {                                                                                           \
        _setmode(0, 0x8000);                                                                       \
        _setmode(1, 0x8000);                                                                       \
    } while (0)
#else
#define ALICE_SDF_SET_BINARY_IO()                                                                  \
    do {                                                                                           \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// Vector types (GLSL §5.4 constructors, §5.5 swizzles)
// ---------------------------------------------------------------------------
struct vec2;
struct vec3;
struct vec4;

// A swizzle is a view over the parent's storage: `N` is the parent's component count, the
// template arguments the selected lanes (read-only here: the emitter never assigns to one).
template <int N, int A, int B> struct swz2 {
    float d[N];
    inline operator vec2() const;
};
template <int N, int A, int B, int C> struct swz3 {
    float d[N];
    inline operator vec3() const;
};

struct vec2 {
    union {
        struct {
            float x, y;
        };
        swz2<2, 1, 0> yx;
        swz2<2, 0, 0> xx;
        swz2<2, 1, 1> yy;
    };
    vec2() : x(0.0f), y(0.0f) {}
    vec2(float a, float b) : x(a), y(b) {}
    explicit vec2(float s) : x(s), y(s) {}
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct ivec3 {
    int x, y, z;
    ivec3() : x(0), y(0), z(0) {}
    ivec3(int a, int b, int c) : x(a), y(b), z(c) {}
    int &operator[](int i) { return (&x)[i]; }
    int operator[](int i) const { return (&x)[i]; }
};

struct uvec3 {
    uint x, y, z;
    uvec3() : x(0u), y(0u), z(0u) {}
    uvec3(uint a, uint b, uint c) : x(a), y(b), z(c) {}
    uint &operator[](int i) { return (&x)[i]; }
    uint operator[](int i) const { return (&x)[i]; }
};

struct vec3 {
    union {
        struct {
            float x, y, z;
        };
        swz2<3, 0, 1> xy;
        swz2<3, 0, 2> xz;
        swz2<3, 1, 2> yz;
        swz2<3, 1, 0> yx;
        swz2<3, 2, 0> zx;
        swz2<3, 2, 1> zy;
        swz3<3, 0, 1, 2> xyz;
        swz3<3, 2, 1, 0> zyx;
        swz3<3, 1, 2, 0> yzx;
        swz3<3, 2, 0, 1> zxy;
    };
    vec3() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }
    vec3(float a, float b, float c) {
        x = a;
        y = b;
        z = c;
    }
    explicit vec3(float s) {
        x = s;
        y = s;
        z = s;
    }
    vec3(vec2 v, float c) {
        x = v.x;
        y = v.y;
        z = c;
    }
    vec3(float a, vec2 v) {
        x = a;
        y = v.x;
        z = v.y;
    }
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

struct vec4 {
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
    vec4() {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        w = 0.0f;
    }
    vec4(float a, float b, float c, float d_) {
        x = a;
        y = b;
        z = c;
        w = d_;
    }
    explicit vec4(float s) {
        x = s;
        y = s;
        z = s;
        w = s;
    }
    vec4(vec3 v, float d_) {
        x = v.x;
        y = v.y;
        z = v.z;
        w = d_;
    }
    float &operator[](int i) { return (&x)[i]; }
    float operator[](int i) const { return (&x)[i]; }
};

template <int N, int A, int B> inline swz2<N, A, B>::operator vec2() const {
    return vec2(d[A], d[B]);
}
template <int N, int A, int B, int C> inline swz3<N, A, B, C>::operator vec3() const {
    return vec3(d[A], d[B], d[C]);
}

// ---------------------------------------------------------------------------
// Arithmetic operators (component-wise, scalar broadcast) — GLSL §5.9
// ---------------------------------------------------------------------------
#define ALICE_V2_BINOP(OP)                                                                         \
    inline vec2 operator OP(vec2 a, vec2 b) { return vec2(a.x OP b.x, a.y OP b.y); }               \
    inline vec2 operator OP(vec2 a, float s) { return vec2(a.x OP s, a.y OP s); }                  \
    inline vec2 operator OP(float s, vec2 b) { return vec2(s OP b.x, s OP b.y); }
#define ALICE_V3_BINOP(OP)                                                                         \
    inline vec3 operator OP(vec3 a, vec3 b) { return vec3(a.x OP b.x, a.y OP b.y, a.z OP b.z); }   \
    inline vec3 operator OP(vec3 a, float s) { return vec3(a.x OP s, a.y OP s, a.z OP s); }        \
    inline vec3 operator OP(float s, vec3 b) { return vec3(s OP b.x, s OP b.y, s OP b.z); }
#define ALICE_V4_BINOP(OP)                                                                         \
    inline vec4 operator OP(vec4 a, vec4 b) {                                                      \
        return vec4(a.x OP b.x, a.y OP b.y, a.z OP b.z, a.w OP b.w);                               \
    }                                                                                              \
    inline vec4 operator OP(vec4 a, float s) { return vec4(a.x OP s, a.y OP s, a.z OP s, a.w OP s); } \
    inline vec4 operator OP(float s, vec4 b) { return vec4(s OP b.x, s OP b.y, s OP b.z, s OP b.w); }

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

inline vec2 operator-(vec2 a) { return vec2(-a.x, -a.y); }
inline vec3 operator-(vec3 a) { return vec3(-a.x, -a.y, -a.z); }
inline vec4 operator-(vec4 a) { return vec4(-a.x, -a.y, -a.z, -a.w); }

#define ALICE_ASSIGN_OP(T, OP)                                                                     \
    inline T &operator OP##=(T & a, T b) {                                                         \
        a = a OP b;                                                                                \
        return a;                                                                                  \
    }                                                                                              \
    inline T &operator OP##=(T & a, float s) {                                                     \
        a = a OP s;                                                                                \
        return a;                                                                                  \
    }
ALICE_ASSIGN_OP(vec2, +)
ALICE_ASSIGN_OP(vec2, -)
ALICE_ASSIGN_OP(vec2, *)
ALICE_ASSIGN_OP(vec2, /)
ALICE_ASSIGN_OP(vec3, +)
ALICE_ASSIGN_OP(vec3, -)
ALICE_ASSIGN_OP(vec3, *)
ALICE_ASSIGN_OP(vec3, /)
ALICE_ASSIGN_OP(vec4, +)
ALICE_ASSIGN_OP(vec4, -)
ALICE_ASSIGN_OP(vec4, *)
ALICE_ASSIGN_OP(vec4, /)
#undef ALICE_ASSIGN_OP

// ---------------------------------------------------------------------------
// Scalar built-ins — GLSL §8.1 (angle / trigonometry), §8.2 (exponential), §8.3 (common)
// ---------------------------------------------------------------------------
inline float abs(float v) { return __builtin_fabsf(v); }
inline int abs(int v) { return v < 0 ? -v : v; }
inline float sqrt(float v) { return __builtin_sqrtf(v); }
inline float inversesqrt(float v) { return 1.0f / __builtin_sqrtf(v); }
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
// §8.1: atan(y, x) is the two-argument arc tangent
inline float atan(float y, float x) { return __builtin_atan2f(y, x); }
inline float log(float v) { return __builtin_logf(v); }
inline float log2(float v) { return __builtin_log2f(v); }
inline float exp(float v) { return __builtin_expf(v); }
inline float exp2(float v) { return __builtin_exp2f(v); }
inline float pow(float a, float b) { return __builtin_powf(a, b); }
// §8.3: fract(x) = x − floor(x); mod(x, y) = x − y · floor(x / y)
inline float fract(float v) { return v - __builtin_floorf(v); }
inline float mod(float a, float b) { return a - b * __builtin_floorf(a / b); }
inline float sign(float v) { return v < 0.0f ? -1.0f : (v > 0.0f ? 1.0f : 0.0f); }
// §8.3: min(x, y) = y < x ? y : x; max(x, y) = x < y ? y : x
inline float min(float a, float b) { return b < a ? b : a; }
inline float max(float a, float b) { return a < b ? b : a; }
inline int min(int a, int b) { return b < a ? b : a; }
inline int max(int a, int b) { return a < b ? b : a; }
inline uint min(uint a, uint b) { return b < a ? b : a; }
inline uint max(uint a, uint b) { return a < b ? b : a; }
// §8.3: clamp(x, lo, hi) = min(max(x, lo), hi)
inline float clamp(float v, float lo, float hi) { return min(max(v, lo), hi); }
inline int clamp(int v, int lo, int hi) { return min(max(v, lo), hi); }
// §8.3: mix(x, y, a) = x · (1 − a) + y · a
inline float mix(float a, float b, float t) { return a * (1.0f - t) + b * t; }
// §8.3: step(edge, x) = x < edge ? 0 : 1
inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
// §8.3: smoothstep, t = clamp((x − e0) / (e1 − e0), 0, 1), t² (3 − 2t)
inline float smoothstep(float e0, float e1, float x) {
    float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
inline bool isnan(float v) { return __builtin_isnan(v); }
inline bool isinf(float v) { return __builtin_isinf(v); }

// §8.4: the bit reinterpretations
inline uint floatBitsToUint(float v) {
    uint out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline int floatBitsToInt(float v) {
    int out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline float uintBitsToFloat(uint v) {
    float out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline float intBitsToFloat(int v) {
    float out;
    __builtin_memcpy(&out, &v, 4);
    return out;
}
inline uvec3 floatBitsToUint(vec3 v) {
    return uvec3(floatBitsToUint(v.x), floatBitsToUint(v.y), floatBitsToUint(v.z));
}
inline ivec3 floatBitsToInt(vec3 v) {
    return ivec3(floatBitsToInt(v.x), floatBitsToInt(v.y), floatBitsToInt(v.z));
}

// ---------------------------------------------------------------------------
// Component-wise vector built-ins
// ---------------------------------------------------------------------------
#define ALICE_V_UNARY(NAME)                                                                        \
    inline vec2 NAME(vec2 v) { return vec2(NAME(v.x), NAME(v.y)); }                                \
    inline vec3 NAME(vec3 v) { return vec3(NAME(v.x), NAME(v.y), NAME(v.z)); }                     \
    inline vec4 NAME(vec4 v) { return vec4(NAME(v.x), NAME(v.y), NAME(v.z), NAME(v.w)); }
ALICE_V_UNARY(abs)
ALICE_V_UNARY(sqrt)
ALICE_V_UNARY(inversesqrt)
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
ALICE_V_UNARY(fract)
#undef ALICE_V_UNARY

#define ALICE_V_BINARY(NAME)                                                                       \
    inline vec2 NAME(vec2 a, vec2 b) { return vec2(NAME(a.x, b.x), NAME(a.y, b.y)); }              \
    inline vec2 NAME(vec2 a, float s) { return vec2(NAME(a.x, s), NAME(a.y, s)); }                 \
    inline vec3 NAME(vec3 a, vec3 b) { return vec3(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z)); } \
    inline vec3 NAME(vec3 a, float s) { return vec3(NAME(a.x, s), NAME(a.y, s), NAME(a.z, s)); }   \
    inline vec4 NAME(vec4 a, vec4 b) {                                                             \
        return vec4(NAME(a.x, b.x), NAME(a.y, b.y), NAME(a.z, b.z), NAME(a.w, b.w));               \
    }                                                                                              \
    inline vec4 NAME(vec4 a, float s) { return vec4(NAME(a.x, s), NAME(a.y, s), NAME(a.z, s), NAME(a.w, s)); }
ALICE_V_BINARY(min)
ALICE_V_BINARY(max)
ALICE_V_BINARY(pow)
ALICE_V_BINARY(mod)
ALICE_V_BINARY(atan)
#undef ALICE_V_BINARY

// step(edge, x) with a scalar edge broadcasts the edge (§8.3)
inline vec2 step(vec2 e, vec2 x) { return vec2(step(e.x, x.x), step(e.y, x.y)); }
inline vec3 step(vec3 e, vec3 x) { return vec3(step(e.x, x.x), step(e.y, x.y), step(e.z, x.z)); }
inline vec2 step(float e, vec2 x) { return vec2(step(e, x.x), step(e, x.y)); }
inline vec3 step(float e, vec3 x) { return vec3(step(e, x.x), step(e, x.y), step(e, x.z)); }

inline vec2 clamp(vec2 v, vec2 lo, vec2 hi) { return vec2(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y)); }
inline vec2 clamp(vec2 v, float lo, float hi) { return vec2(clamp(v.x, lo, hi), clamp(v.y, lo, hi)); }
inline vec3 clamp(vec3 v, vec3 lo, vec3 hi) {
    return vec3(clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z));
}
inline vec3 clamp(vec3 v, float lo, float hi) {
    return vec3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

inline vec2 mix(vec2 a, vec2 b, float t) { return a * (1.0f - t) + b * t; }
inline vec2 mix(vec2 a, vec2 b, vec2 t) { return a * (1.0f - t) + b * t; }
inline vec3 mix(vec3 a, vec3 b, float t) { return a * (1.0f - t) + b * t; }
inline vec3 mix(vec3 a, vec3 b, vec3 t) { return a * (1.0f - t) + b * t; }
inline vec4 mix(vec4 a, vec4 b, float t) { return a * (1.0f - t) + b * t; }

// ---------------------------------------------------------------------------
// Geometric built-ins — GLSL §8.5
// ---------------------------------------------------------------------------
inline float dot(vec2 a, vec2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float dot(vec4 a, vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
inline float length(vec2 v) { return sqrt(dot(v, v)); }
inline float length(vec3 v) { return sqrt(dot(v, v)); }
inline float length(vec4 v) { return sqrt(dot(v, v)); }
inline float distance(vec3 a, vec3 b) { return length(a - b); }
inline vec2 normalize(vec2 v) { return v / length(v); }
inline vec3 normalize(vec3 v) { return v / length(v); }
inline vec4 normalize(vec4 v) { return v / length(v); }
inline vec3 cross(vec3 a, vec3 b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

#endif // ALICE_SDF_GLSL_CPU_SHIM_H
