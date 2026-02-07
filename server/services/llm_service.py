"""LLM Service - Claude/Gemini API abstraction for SDF generation."""

import json
import logging
import re
import time
from typing import AsyncIterator

from server.prompts.system_prompt import SYSTEM_PROMPT
from server.prompts.examples import format_few_shot
from server import config

logger = logging.getLogger(__name__)

# Boolean operations that require both "a" and "b" fields
BOOLEAN_OPS = {"Union", "Intersection", "Subtraction",
               "SmoothUnion", "SmoothIntersection", "SmoothSubtraction"}

# Transforms/modifiers that require a "child" field
CHILD_OPS = {"Translate", "Rotate", "Scale", "ScaleNonUniform",
             "Twist", "Bend", "RepeatInfinite", "RepeatFinite",
             "Noise", "Round", "Onion", "Elongate", "Mirror",
             "Revolution", "Extrude", "Taper", "Displacement",
             "PolarRepeat", "WithMaterial"}

MAX_RETRIES = 2


def _build_messages(prompt: str) -> list[dict]:
    """Build message list with few-shot examples."""
    return [
        {"role": "user", "content": prompt},
    ]


def _build_retry_messages(prompt: str, error: str, bad_json: str) -> list[dict]:
    """Build messages for retry with error feedback."""
    # Truncate bad_json to avoid token waste
    snippet = bad_json[:500] + "..." if len(bad_json) > 500 else bad_json
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": snippet},
        {"role": "user", "content": (
            f"The JSON you produced has a structural error:\n{error}\n\n"
            "Please fix the error and output the corrected JSON. "
            "Remember: Boolean ops (Union, SmoothUnion, etc.) need BOTH \"a\" and \"b\" fields. "
            "Keep node count under 20. Output ONLY valid JSON."
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
    import anthropic

    model_id = config.CLAUDE_MODELS.get(model, model)
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

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
    sdf_json = _extract_json(text)

    # Validate it's parseable JSON
    json.loads(sdf_json)

    return sdf_json, elapsed


async def generate_sdf_gemini(
    prompt: str, model: str = "flash",
    retry_messages: list[dict] | None = None,
) -> tuple[str, float]:
    """Generate SDF JSON using Gemini API.

    Returns (sdf_json, elapsed_seconds).
    """
    from google import genai

    model_id = config.GEMINI_MODELS.get(model, model)
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

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
        config={"max_output_tokens": 65536},
    )
    elapsed = time.perf_counter() - t0

    text = response.text
    sdf_json = _extract_json(text)

    # Validate it's parseable JSON
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
            }
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
    import anthropic

    model_id = config.CLAUDE_MODELS.get(model, model)
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)

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
    from google import genai

    model_id = config.GEMINI_MODELS.get(model, model)
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    system = SYSTEM_PROMPT + "\n\n## Examples\n\n" + format_few_shot()
    full_prompt = system + "\n\nUser: " + prompt + "\nAssistant:"

    async for chunk in await client.aio.models.generate_content_stream(
        model=model_id,
        contents=full_prompt,
    ):
        if chunk.text:
            yield chunk.text
