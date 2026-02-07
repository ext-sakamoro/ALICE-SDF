"""LLM Service - Claude/Gemini API abstraction for SDF generation."""

import difflib
import json
import logging
import re
import time
from typing import AsyncIterator

from server.prompts.system_prompt import SYSTEM_PROMPT
from server.prompts.examples import format_few_shot
from server import config

logger = logging.getLogger(__name__)

# ── Lazy Singleton Clients (connection reuse) ─────────────────
_claude_client = None
_gemini_client = None

def _get_claude_client():
    """Get or create singleton AsyncAnthropic client."""
    global _claude_client
    if _claude_client is None:
        import anthropic
        _claude_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        logger.info("Claude client initialized (singleton)")
    return _claude_client

def _get_gemini_client():
    """Get or create singleton genai.Client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GOOGLE_API_KEY)
        logger.info("Gemini client initialized (singleton)")
    return _gemini_client

# Boolean operations that require both "a" and "b" fields
BOOLEAN_OPS = {"Union", "Intersection", "Subtraction",
               "SmoothUnion", "SmoothIntersection", "SmoothSubtraction"}

# Transforms/modifiers that require a "child" field
CHILD_OPS = {"Translate", "Rotate", "Scale", "ScaleNonUniform",
             "Twist", "Bend", "RepeatInfinite", "RepeatFinite",
             "Noise", "Round", "Onion", "Elongate", "Mirror",
             "Revolution", "Extrude", "Taper", "Displacement",
             "PolarRepeat", "WithMaterial"}

# All 53 known primitives (must match Rust types.rs)
KNOWN_PRIMITIVES = {
    "Sphere", "Box3d", "Cylinder", "Torus", "Plane", "Capsule", "Cone",
    "Ellipsoid", "RoundedCone", "Pyramid", "Octahedron", "HexPrism",
    "Link", "Triangle", "Bezier",
    # Extended Geometric
    "RoundedBox", "CappedCone", "CappedTorus", "RoundedCylinder",
    "TriangularPrism", "CutSphere", "CutHollowSphere", "DeathStar",
    "SolidAngle", "Rhombus", "Horseshoe", "Vesica", "InfiniteCylinder",
    "InfiniteCone", "Gyroid", "Heart",
    # 3D Native
    "Tube", "Barrel", "Diamond", "ChamferedCube", "SchwarzP",
    "Superellipsoid", "RoundedX",
    # 2D→3D Prisms
    "Pie", "Trapezoid", "Parallelogram", "Tunnel", "UnevenCapsule",
    "Egg", "ArcShape", "Moon", "CrossShape", "BlobbyCross",
    "ParabolaSegment", "RegularPolygon", "StarPolygon",
    # Complex 3D
    "Stairs", "Helix",
}

ALL_NODE_TYPES = KNOWN_PRIMITIVES | BOOLEAN_OPS | CHILD_OPS

MAX_RETRIES = 2


def _suggest_fix(error: str, sdf_json: str) -> str:
    """Generate a specific fix suggestion based on error pattern."""
    suggestions = []

    # Missing 'a' or 'b' in boolean ops
    if "missing" in error and ("'a'" in error or "'b'" in error):
        suggestions.append(
            "Boolean ops (Union, SmoothUnion, Intersection, Subtraction, etc.) "
            "require BOTH 'a' and 'b' child nodes. Add the missing field with a valid SDF node."
        )

    # Missing 'child' in transform/modifier
    if "missing" in error and "'child'" in error:
        suggestions.append(
            "Transform/modifier ops (Translate, Rotate, Scale, Twist, Noise, etc.) "
            "require a 'child' field containing the SDF node to operate on."
        )

    # Unknown variant (typo in primitive name)
    if "unknown variant" in error.lower():
        match = re.search(r"unknown variant `(\w+)`", error, re.IGNORECASE)
        if match:
            bad_name = match.group(1)
            close = difflib.get_close_matches(bad_name, ALL_NODE_TYPES, n=3, cutoff=0.6)
            if close:
                suggestions.append(
                    f"Unknown node type '{bad_name}'. Did you mean: {', '.join(close)}?"
                )
            else:
                suggestions.append(
                    f"Unknown node type '{bad_name}'. Check the primitives list for valid type names."
                )

    # NaN or infinity
    if "nan" in error.lower() or "infinity" in error.lower() or "inf" in error.lower():
        suggestions.append(
            "Numeric value is NaN or infinity. All values must be finite numbers. "
            "Reduce large scale values and ensure no division by zero."
        )

    return " ".join(suggestions) if suggestions else ""


# ── LLM Response Cache (LRU with TTL + maxsize) ──────────────

import hashlib
from collections import OrderedDict

_cache: OrderedDict[str, dict] = OrderedDict()
CACHE_TTL = config.CACHE_TTL_SECONDS
CACHE_MAX_SIZE = 256  # Max cached entries to prevent unbounded memory growth


def _cache_key(prompt: str, provider: str, model: str) -> str:
    """Generate a cache key from prompt + provider + model."""
    raw = f"{prompt}|{provider}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_evict_expired() -> int:
    """Remove all expired entries. Returns count removed."""
    now = time.time()
    expired = [k for k, v in _cache.items() if now - v["timestamp"] > CACHE_TTL]
    for k in expired:
        del _cache[k]
    return len(expired)


def _cache_get(key: str) -> dict | None:
    """Get cached entry if exists and not expired. Promotes to MRU on hit."""
    entry = _cache.get(key)
    if entry is None:
        return None
    if time.time() - entry["timestamp"] > CACHE_TTL:
        del _cache[key]
        return None
    _cache.move_to_end(key)  # LRU: move to most-recently-used
    return entry


def _cache_set(key: str, sdf_json: str, metadata: dict) -> None:
    """Store result in cache with LRU eviction."""
    _cache[key] = {
        "sdf_json": sdf_json,
        "metadata": metadata,
        "timestamp": time.time(),
    }
    _cache.move_to_end(key)
    # Evict oldest entries if over max size
    while len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)


def cache_clear() -> int:
    """Clear all cached entries. Returns number of entries cleared."""
    count = len(_cache)
    _cache.clear()
    return count


def _build_messages(prompt: str) -> list[dict]:
    """Build message list with few-shot examples."""
    return [
        {"role": "user", "content": prompt},
    ]


def _build_retry_messages(prompt: str, error: str, bad_json: str) -> list[dict]:
    """Build messages for retry with error feedback."""
    # Truncate bad_json to avoid token waste
    snippet = bad_json[:500] + "..." if len(bad_json) > 500 else bad_json
    fix_hint = _suggest_fix(error, bad_json)
    fix_line = f"\n\nSpecific fix: {fix_hint}" if fix_hint else ""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": snippet},
        {"role": "user", "content": (
            f"The JSON you produced has a structural error:\n{error}\n\n"
            "Please fix the error and output the corrected JSON. "
            "Remember: Boolean ops (Union, SmoothUnion, etc.) need BOTH \"a\" and \"b\" fields. "
            f"Keep node count under 20. Output ONLY valid JSON.{fix_line}"
        )},
    ]


def _repair_json(text: str) -> str:
    """Repair incomplete JSON by appending missing closing braces/brackets.

    LLMs sometimes lose track of nesting depth in deeply nested SDF trees.
    """
    # Count unmatched braces/brackets (ignoring those inside strings)
    in_string = False
    escape = False
    brace_depth = 0
    bracket_depth = 0

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1

    if brace_depth > 0 or bracket_depth > 0:
        text += "]" * bracket_depth + "}" * brace_depth

    return text


def _validate_sdf_structure(obj: dict) -> list[str]:
    """Validate SDF JSON structure and return list of errors found.

    Checks that Boolean ops have 'a' and 'b', transforms have 'child', etc.
    """
    errors = []

    def _check_node(node, path="root"):
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            if key in BOOLEAN_OPS:
                if not isinstance(val, dict):
                    errors.append(f"{path}.{key}: expected object, got {type(val).__name__}")
                    continue
                if "a" not in val:
                    errors.append(f"{path}.{key}: missing required field 'a'")
                if "b" not in val:
                    errors.append(f"{path}.{key}: missing required field 'b'")
                if "a" in val:
                    _check_node(val["a"], f"{path}.{key}.a")
                if "b" in val:
                    _check_node(val["b"], f"{path}.{key}.b")
            elif key in CHILD_OPS:
                if not isinstance(val, dict):
                    errors.append(f"{path}.{key}: expected object, got {type(val).__name__}")
                    continue
                if "child" not in val:
                    errors.append(f"{path}.{key}: missing required field 'child'")
                if "child" in val:
                    _check_node(val["child"], f"{path}.{key}.child")

    if "root" in obj:
        _check_node(obj["root"])
    else:
        errors.append("missing 'root' field")

    return errors


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find the JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")

    # Find matching closing brace
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Brace mismatch — try to repair by appending missing closers
    remainder = text[start:]
    repaired = _repair_json(remainder)
    # Validate repaired JSON is at least parseable
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        raise ValueError(
            f"Incomplete JSON object in LLM response "
            f"(depth={depth}, tried repair but failed)"
        )


async def generate_sdf_claude(
    prompt: str, model: str = "haiku",
    retry_messages: list[dict] | None = None,
) -> tuple[str, float]:
    """Generate SDF JSON using Claude API.

    Returns (sdf_json, elapsed_seconds).
    """
    model_id = config.CLAUDE_MODELS.get(model, model)
    client = _get_claude_client()

    system = SYSTEM_PROMPT + "\n\n## Examples\n\n" + format_few_shot()
    messages = retry_messages or _build_messages(prompt)

    t0 = time.perf_counter()
    response = await client.messages.create(
        model=model_id,
        max_tokens=8192,
        system=system,
        messages=messages,
    )
    elapsed = time.perf_counter() - t0

    text = response.content[0].text

    # Try direct JSON parse first (structured output), fallback to extraction
    try:
        json.loads(text)
        sdf_json = text
    except (json.JSONDecodeError, TypeError):
        sdf_json = _extract_json(text)
        json.loads(sdf_json)

    return sdf_json, elapsed


async def generate_sdf_gemini(
    prompt: str, model: str = "flash",
    retry_messages: list[dict] | None = None,
) -> tuple[str, float]:
    """Generate SDF JSON using Gemini API.

    Returns (sdf_json, elapsed_seconds).
    """
    model_id = config.GEMINI_MODELS.get(model, model)
    client = _get_gemini_client()

    system = SYSTEM_PROMPT + "\n\n## Examples\n\n" + format_few_shot()

    if retry_messages:
        # Build multi-turn prompt for Gemini
        full_prompt = system + "\n\n"
        for msg in retry_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "Assistant:"
    else:
        full_prompt = system + "\n\nUser: " + prompt + "\nAssistant:"

    t0 = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=model_id,
        contents=full_prompt,
        config={
            "max_output_tokens": 65536,
            "response_mime_type": "application/json",
        },
    )
    elapsed = time.perf_counter() - t0

    text = response.text

    # Try direct JSON parse first (structured output), fallback to extraction
    try:
        json.loads(text)
        sdf_json = text
    except (json.JSONDecodeError, TypeError):
        sdf_json = _extract_json(text)
        json.loads(sdf_json)

    return sdf_json, elapsed


async def generate_sdf(
    prompt: str,
    provider: str = "claude",
    model: str | None = None,
) -> tuple[str, dict]:
    """Generate SDF JSON from text prompt with retry on structural errors.

    Validates with both Python structure checks and Rust serde.
    Returns (sdf_json, metadata).
    """
    import asyncio
    from server.services import sdf_service

    if provider == "claude":
        model = model or "haiku"
        gen_fn = generate_sdf_claude
    elif provider == "gemini":
        model = model or "flash"
        gen_fn = generate_sdf_gemini
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Cache check
    key = _cache_key(prompt, provider, model)
    cached = _cache_get(key)
    if cached is not None:
        logger.info(f"Cache hit for prompt: {prompt[:50]}...")
        meta = {**cached["metadata"], "cache_hit": True}
        return cached["sdf_json"], meta

    total_elapsed = 0.0
    last_error = None
    bad_json = ""
    retries_used = 0

    for attempt in range(1 + MAX_RETRIES):
        try:
            if attempt == 0:
                sdf_json, elapsed = await gen_fn(prompt, model)
            else:
                # Retry with error feedback
                logger.info(f"Retry {attempt}/{MAX_RETRIES}: {last_error}")
                retry_msgs = _build_retry_messages(prompt, last_error, bad_json)
                sdf_json, elapsed = await gen_fn(prompt, model, retry_messages=retry_msgs)

            total_elapsed += elapsed

            # Step 1: Python-side structure validation
            parsed = json.loads(sdf_json)
            structure_errors = _validate_sdf_structure(parsed)
            if structure_errors:
                bad_json = sdf_json
                last_error = "; ".join(structure_errors)
                retries_used = attempt + 1
                if attempt < MAX_RETRIES:
                    logger.warning(f"Structure errors (attempt {attempt+1}): {last_error}")
                    continue
                else:
                    logger.warning(f"Structure errors persist after {MAX_RETRIES} retries")

            # Step 2: Rust serde validation (catches field-level errors)
            valid, node_count, rust_err = sdf_service.validate_sdf_json(sdf_json)
            if not valid:
                bad_json = sdf_json
                last_error = rust_err or "Unknown Rust validation error"
                retries_used = attempt + 1
                if attempt < MAX_RETRIES:
                    logger.warning(f"Rust validation failed (attempt {attempt+1}): {last_error}")
                    continue
                else:
                    logger.warning(f"Rust validation failed after {MAX_RETRIES} retries: {last_error}")

            metadata = {
                "provider": provider,
                "model": model,
                "llm_time_s": round(total_elapsed, 3),
                "retries": retries_used,
                "cache_hit": False,
            }
            _cache_set(key, sdf_json, metadata)
            return sdf_json, metadata

        except Exception as e:
            err_str = str(e)
            # Detect rate limiting and sleep before retry
            if "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                import re as _re
                delay_match = _re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, _re.IGNORECASE)
                wait_sec = float(delay_match.group(1)) if delay_match else 30.0
                if attempt < MAX_RETRIES:
                    logger.info(f"Rate limited, waiting {wait_sec:.0f}s before retry...")
                    await asyncio.sleep(wait_sec)
                    retries_used = attempt + 1
                    last_error = "rate limited"
                    continue
                raise

            bad_json = sdf_json if 'sdf_json' in dir() else ""
            last_error = err_str
            retries_used = attempt + 1
            if attempt < MAX_RETRIES:
                logger.warning(f"Generation failed (attempt {attempt+1}): {e}")
                continue
            raise

    raise RuntimeError(f"Generation failed after {MAX_RETRIES} retries: {last_error}")


async def stream_sdf_claude(
    prompt: str, model: str = "haiku"
) -> AsyncIterator[str]:
    """Stream SDF JSON tokens from Claude API."""
    model_id = config.CLAUDE_MODELS.get(model, model)
    client = _get_claude_client()

    system = SYSTEM_PROMPT + "\n\n## Examples\n\n" + format_few_shot()

    async with client.messages.stream(
        model=model_id,
        max_tokens=8192,
        system=system,
        messages=_build_messages(prompt),
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def stream_sdf_gemini(
    prompt: str, model: str = "flash"
) -> AsyncIterator[str]:
    """Stream SDF JSON tokens from Gemini API."""
    model_id = config.GEMINI_MODELS.get(model, model)
    client = _get_gemini_client()

    system = SYSTEM_PROMPT + "\n\n## Examples\n\n" + format_few_shot()
    full_prompt = system + "\n\nUser: " + prompt + "\nAssistant:"

    async for chunk in await client.aio.models.generate_content_stream(
        model=model_id,
        contents=full_prompt,
    ):
        if chunk.text:
            yield chunk.text
