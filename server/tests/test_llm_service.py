"""Tests for LLM service (unit tests, no API calls)."""

import json
import pytest
from server.services.llm_service import _extract_json, _repair_json, _validate_sdf_structure


class TestExtractJson:
    def test_raw_json(self):
        raw = '{"version":"0.1.0","root":{"Sphere":{"radius":1.0}},"metadata":null}'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["root"]["Sphere"]["radius"] == 1.0

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"version":"0.1.0","root":{"Sphere":{"radius":1.0}},"metadata":null}\n```'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert "Sphere" in parsed["root"]

    def test_json_with_surrounding_text(self):
        raw = 'Here is the SDF:\n{"version":"0.1.0","root":{"Sphere":{"radius":1.0}},"metadata":null}\nDone!'
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert "Sphere" in parsed["root"]

    def test_nested_braces(self):
        raw = json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {"Sphere": {"radius": 1.0}},
                    "b": {"Box3d": {"half_extents": [1.0, 1.0, 1.0]}},
                }
            },
            "metadata": None,
        })
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert "Union" in parsed["root"]

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON"):
            _extract_json("This has no JSON at all")

    def test_whitespace_handling(self):
        raw = '  \n  {"version":"0.1.0","root":{"Sphere":{"radius":2.0}},"metadata":null}  \n  '
        result = _extract_json(raw)
        parsed = json.loads(result)
        assert parsed["root"]["Sphere"]["radius"] == 2.0


class TestRepairJson:
    def test_complete_json_unchanged(self):
        raw = '{"a":1,"b":2}'
        assert _repair_json(raw) == raw

    def test_missing_closing_brace(self):
        raw = '{"a":{"b":1}'
        repaired = _repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["a"]["b"] == 1

    def test_missing_closing_bracket_and_brace(self):
        raw = '{"a":[1,2,3'
        repaired = _repair_json(raw)
        parsed = json.loads(repaired)
        assert parsed["a"] == [1, 2, 3]

    def test_string_braces_ignored(self):
        raw = '{"a":"{{{"}'
        assert _repair_json(raw) == raw


class TestValidateSdfStructure:
    def test_valid_sphere(self):
        obj = {"root": {"Sphere": {"radius": 1.0}}}
        errors = _validate_sdf_structure(obj)
        assert errors == []

    def test_valid_union(self):
        obj = {
            "root": {
                "Union": {
                    "a": {"Sphere": {"radius": 1.0}},
                    "b": {"Box3d": {"half_extents": [1, 1, 1]}},
                }
            }
        }
        errors = _validate_sdf_structure(obj)
        assert errors == []

    def test_missing_b_in_union(self):
        obj = {
            "root": {
                "Union": {
                    "a": {"Sphere": {"radius": 1.0}},
                }
            }
        }
        errors = _validate_sdf_structure(obj)
        assert len(errors) == 1
        assert "missing required field 'b'" in errors[0]

    def test_missing_a_in_smooth_union(self):
        obj = {
            "root": {
                "SmoothUnion": {
                    "b": {"Sphere": {"radius": 1.0}},
                    "k": 0.2,
                }
            }
        }
        errors = _validate_sdf_structure(obj)
        assert len(errors) == 1
        assert "missing required field 'a'" in errors[0]

    def test_missing_child_in_translate(self):
        obj = {
            "root": {
                "Translate": {
                    "offset": [1, 0, 0],
                }
            }
        }
        errors = _validate_sdf_structure(obj)
        assert len(errors) == 1
        assert "missing required field 'child'" in errors[0]

    def test_nested_errors_detected(self):
        obj = {
            "root": {
                "Union": {
                    "a": {
                        "Translate": {
                            "offset": [0, 1, 0],
                            # missing "child"
                        }
                    },
                    "b": {
                        "Subtraction": {
                            "a": {"Sphere": {"radius": 1.0}},
                            # missing "b"
                        }
                    },
                }
            }
        }
        errors = _validate_sdf_structure(obj)
        assert len(errors) == 2

    def test_missing_root(self):
        obj = {"version": "0.1.0"}
        errors = _validate_sdf_structure(obj)
        assert len(errors) == 1
        assert "missing 'root'" in errors[0]
