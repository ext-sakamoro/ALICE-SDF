"""Few-shot examples for LLM SDF generation."""

import json

EXAMPLES = [
    {
        "prompt": "A simple sphere on a flat ground",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "Translate": {
                            "child": {"Sphere": {"radius": 1.0}},
                            "offset": [0.0, 1.0, 0.0]
                        }
                    },
                    "b": {
                        "Plane": {"normal": [0.0, 1.0, 0.0], "distance": 0.0}
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A snowman",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "SmoothUnion": {
                    "a": {
                        "SmoothUnion": {
                            "a": {
                                "Translate": {
                                    "child": {"Sphere": {"radius": 1.2}},
                                    "offset": [0.0, 1.2, 0.0]
                                }
                            },
                            "b": {
                                "Translate": {
                                    "child": {"Sphere": {"radius": 0.9}},
                                    "offset": [0.0, 3.0, 0.0]
                                }
                            },
                            "k": 0.2
                        }
                    },
                    "b": {
                        "Translate": {
                            "child": {"Sphere": {"radius": 0.6}},
                            "offset": [0.0, 4.3, 0.0]
                        }
                    },
                    "k": 0.15
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A castle tower",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "Union": {
                            "a": {
                                "Cylinder": {"radius": 1.5, "half_height": 2.5}
                            },
                            "b": {
                                "Translate": {
                                    "child": {"Cone": {"radius": 2.0, "half_height": 1.5}},
                                    "offset": [0.0, 4.0, 0.0]
                                }
                            }
                        }
                    },
                    "b": {
                        "Translate": {
                            "child": {
                                "Subtraction": {
                                    "a": {
                                        "PolarRepeat": {
                                            "child": {
                                                "Translate": {
                                                    "child": {"Box3d": {"half_extents": [0.2, 0.3, 0.2]}},
                                                    "offset": [1.5, 2.5, 0.0]
                                                }
                                            },
                                            "count": 8
                                        }
                                    },
                                    "b": {"Sphere": {"radius": 0.01}}
                                }
                            },
                            "offset": [0.0, 0.0, 0.0]
                        }
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "Alien mushroom forest",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "RepeatFinite": {
                            "child": {
                                "SmoothUnion": {
                                    "a": {
                                        "Translate": {
                                            "child": {
                                                "Scale": {
                                                    "child": {"Torus": {"major_radius": 1.0, "minor_radius": 0.4}},
                                                    "factor": 0.8
                                                }
                                            },
                                            "offset": [0.0, 2.0, 0.0]
                                        }
                                    },
                                    "b": {
                                        "Capsule": {
                                            "point_a": [0.0, 0.0, 0.0],
                                            "point_b": [0.0, 2.0, 0.0],
                                            "radius": 0.2
                                        }
                                    },
                                    "k": 0.3
                                }
                            },
                            "count": [3, 1, 3],
                            "spacing": [3.0, 0.0, 3.0]
                        }
                    },
                    "b": {
                        "Noise": {
                            "child": {"Plane": {"normal": [0.0, 1.0, 0.0], "distance": 0.0}},
                            "amplitude": 0.3,
                            "frequency": 1.5,
                            "seed": 42
                        }
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A twisted pillar with a floating orb",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "Twist": {
                            "child": {
                                "Round": {
                                    "child": {"Box3d": {"half_extents": [0.6, 2.5, 0.6]}},
                                    "radius": 0.1
                                }
                            },
                            "strength": 0.5
                        }
                    },
                    "b": {
                        "Translate": {
                            "child": {
                                "Onion": {
                                    "child": {"Sphere": {"radius": 0.8}},
                                    "thickness": 0.05
                                }
                            },
                            "offset": [0.0, 3.5, 0.0]
                        }
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A mechanical gear",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Subtraction": {
                    "a": {
                        "SmoothUnion": {
                            "a": {
                                "Extrude": {
                                    "child": {"Torus": {"major_radius": 2.0, "minor_radius": 0.5}},
                                    "half_height": 0.3
                                }
                            },
                            "b": {
                                "PolarRepeat": {
                                    "child": {
                                        "Translate": {
                                            "child": {
                                                "Round": {
                                                    "child": {"Box3d": {"half_extents": [0.3, 0.3, 0.35]}},
                                                    "radius": 0.05
                                                }
                                            },
                                            "offset": [2.0, 0.0, 0.0]
                                        }
                                    },
                                    "count": 12
                                }
                            },
                            "k": 0.1
                        }
                    },
                    "b": {
                        "Cylinder": {"radius": 0.4, "half_height": 0.5}
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A spiral staircase",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "Twist": {
                            "child": {
                                "Stairs": {
                                    "step_width": 0.8,
                                    "step_height": 0.25,
                                    "n_steps": 10.0,
                                    "half_depth": 1.2
                                }
                            },
                            "strength": 0.3
                        }
                    },
                    "b": {
                        "Cylinder": {"radius": 0.15, "half_height": 2.5}
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A treasure chest",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "Union": {
                    "a": {
                        "Translate": {
                            "child": {
                                "RoundedBox": {
                                    "half_extents": [1.2, 0.6, 0.8],
                                    "round_radius": 0.05
                                }
                            },
                            "offset": [0.0, 0.6, 0.0]
                        }
                    },
                    "b": {
                        "Translate": {
                            "child": {
                                "Barrel": {
                                    "radius": 1.2,
                                    "half_height": 0.4,
                                    "bulge": 0.15
                                }
                            },
                            "offset": [0.0, 1.5, 0.0]
                        }
                    }
                }
            },
            "metadata": None
        })
    },
    {
        "prompt": "A decorative star badge",
        "sdf_json": json.dumps({
            "version": "0.1.0",
            "root": {
                "SmoothUnion": {
                    "a": {
                        "StarPolygon": {
                            "radius": 1.5,
                            "n_points": 5.0,
                            "m": 0.5,
                            "half_height": 0.15
                        }
                    },
                    "b": {
                        "Translate": {
                            "child": {
                                "Diamond": {
                                    "radius": 0.4,
                                    "half_height": 0.3
                                }
                            },
                            "offset": [0.0, 0.3, 0.0]
                        }
                    },
                    "k": 0.1
                }
            },
            "metadata": None
        })
    },
]


def format_few_shot() -> str:
    """Format examples as few-shot prompt text."""
    parts = []
    for ex in EXAMPLES:
        parts.append(f"User: {ex['prompt']}\nAssistant: {ex['sdf_json']}")
    return "\n\n".join(parts)
