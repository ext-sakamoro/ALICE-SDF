//! ALICE-Law G8: the whole shader text this crate emits carries its semantics identifier, and the
//! readers that compile shader text check it
//!
//! A whole shader (one that a GPU API or an offline compiler takes as is) starts with a block of
//! `//` comment lines, one of which is `// alice:semantics_id=<64 hex>` ([`provenance::KEY`],
//! [`Provenance::sdf`]). GLSL starts with its `#version` line instead (GLSL ES takes nothing
//! before it, not even a comment: glslang 16.6.0 says "#version: statement must appear first in
//! es-profile shader; before comments or newlines"), so there the block follows that line.
//! [`provenance::from_bytes`] reads the identifier from that block only.
//!
//! The four readers (`GpuEvaluator::from_wgsl` / `from_wgsl_async` / `from_glsl_compute` and
//! `MslShader::from_wgsl`) return `Checked<Self>`: this crate's text is `Current`, text without
//! the key anywhere is `Unpinned`, text under another identifier is refused, and the key outside
//! the leading block (below it, or before `#version`) is invalid: a tag that is present is never
//! read as absent.
//! Each has a `*_pinned` twin that also refuses `Unpinned`.
//!
//! Oracle: the identifier is the constant `SDF_SEMANTICS_ID` (`Provenance::sdf`), and the
//! expected verdict of each case follows from the rule above; no expected value is produced by
//! the reader under test.
//!
//! Author: Moroya Sakamoto

mod common;

#[cfg(feature = "gpu")]
use alice_sdf::provenance::StoredSemantics;
use alice_sdf::provenance::{self, Provenance, KEY};

/// The tag line this crate's emitters write
fn tag_line(p: &Provenance) -> String {
    format!("// {KEY}={}", p.hex())
}

/// A GLSL version line: `#version <digits>` with an optional lower-case profile word
#[cfg(any(feature = "gpu", feature = "glsl", feature = "hlsl"))]
fn is_version_line(line: &str) -> bool {
    let Some(rest) = line.strip_prefix("#version ") else {
        return false;
    };
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    let rest = &rest[digits..];
    digits > 0
        && (rest.is_empty()
            || rest
                .strip_prefix(' ')
                .is_some_and(|w| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase())))
}

/// The number of lines before the leading block (1 for a GLSL `#version` line)
#[cfg(any(feature = "gpu", feature = "glsl", feature = "hlsl"))]
fn header_lines(text: &str) -> usize {
    usize::from(text.lines().next().is_some_and(is_version_line))
}

/// The leading block of `//` lines (after a first `#version` line)
#[cfg(any(feature = "gpu", feature = "glsl", feature = "hlsl"))]
fn leading_block(text: &str) -> Vec<&str> {
    text.lines()
        .skip(header_lines(text))
        .take_while(|l| l.starts_with("//"))
        .collect()
}

/// `text` with `line` inserted at the start of its leading block
#[cfg(feature = "gpu")]
fn insert_in_block(text: &str, line: &str) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    lines.insert(header_lines(text), line);
    lines.join("\n") + "\n"
}

/// `text` carries this crate's identifier in its leading block, and the stored-form reader finds it
#[cfg(any(feature = "gpu", feature = "glsl", feature = "hlsl"))]
fn assert_whole_tagged(what: &str, text: &str) {
    let tag = tag_line(&Provenance::sdf());
    let block = leading_block(text);
    assert!(
        block.contains(&tag.as_str()),
        "{what}: the leading `//` block {block:?} does not hold `{tag}`"
    );
    assert_eq!(
        text.lines().nth(header_lines(text)),
        Some(tag.as_str()),
        "{what}: the emitter writes the identifier on line 1, or on line 2 after `#version`"
    );
    assert_eq!(
        provenance::from_bytes(text.as_bytes()).ok(),
        Some(Provenance::sdf()),
        "{what}: provenance::from_bytes does not read the identifier"
    );
}

/// `text` without the tag lines of its leading block (as another program would write it)
#[cfg(any(feature = "gpu", feature = "glsl"))]
fn untagged(text: &str) -> String {
    let key = format!("// {KEY}=");
    let skip = header_lines(text);
    let mut out = String::new();
    let mut in_block = true;
    for (i, line) in text.lines().enumerate() {
        if i >= skip && in_block && !line.starts_with("//") {
            in_block = false;
        }
        if i >= skip && in_block && line.starts_with(&key) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The cases every reader is checked against, built from this crate's own whole text
#[cfg(feature = "gpu")]
struct Cases {
    own: String,
    untagged: String,
    other: String,
    tag_after_block: String,
    tag_after_first_code_line: String,
    two_ids: String,
    same_id_twice: String,
}

#[cfg(feature = "gpu")]
fn cases(own: &str) -> Cases {
    let bare = untagged(own);
    let other_id = Provenance::new([0x5a; 32]);
    let sdf = tag_line(&Provenance::sdf());
    // the first line that is neither the `#version` line nor a comment: a tag below it is in the
    // body, not in the block
    let first_code = bare
        .lines()
        .enumerate()
        .skip(header_lines(&bare))
        .find(|(_, l)| !l.starts_with("//"))
        .map(|(i, _)| i)
        .expect("a shader has code");
    let mut lines: Vec<&str> = bare.lines().collect();
    lines.insert(first_code + 1, &sdf);
    Cases {
        own: own.to_string(),
        untagged: bare.clone(),
        other: insert_in_block(&bare, &tag_line(&other_id)),
        tag_after_block: format!("{bare}\n{sdf}\n"),
        tag_after_first_code_line: lines.join("\n") + "\n",
        two_ids: insert_in_block(&insert_in_block(&bare, &tag_line(&other_id)), &sdf),
        same_id_twice: insert_in_block(own, &sdf),
    }
}

// ------------------------------------------------------------------------------------- emitters

#[cfg(feature = "gpu")]
#[test]
fn whole_wgsl_carries_the_identifier() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    for (name, node) in common::corpus::corpus() {
        for mode in [TranspileMode::Hardcoded, TranspileMode::Dynamic] {
            let s = WgslShader::transpile(&node, mode);
            assert_whole_tagged(
                &format!("WGSL compute {name} {mode:?}"),
                &s.to_compute_shader(),
            );
            assert_whole_tagged(
                &format!("WGSL compute with normals {name} {mode:?}"),
                &s.to_compute_shader_with_normals(),
            );
            #[cfg(feature = "volume")]
            assert_whole_tagged(
                &format!("WGSL volume {name} {mode:?}"),
                &s.to_volume_shader(),
            );
        }
    }
}

#[cfg(feature = "glsl")]
#[test]
fn whole_glsl_carries_the_identifier() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode, RenderConfig};
    for (name, node) in common::corpus::corpus() {
        for mode in [GlslTranspileMode::Hardcoded, GlslTranspileMode::Dynamic] {
            let s = GlslShader::transpile(&node, mode);
            assert_whole_tagged(
                &format!("GLSL compute {name} {mode:?}"),
                &s.to_compute_shader(),
            );
            assert_whole_tagged(
                &format!("GLSL Vulkan compute {name} {mode:?}"),
                &s.to_vulkan_compute_shader(),
            );
            assert_whole_tagged(
                &format!("GLSL fragment {name} {mode:?}"),
                &s.to_fragment_shader(),
            );
            assert_whole_tagged(
                &format!("GLSL fragment full {name} {mode:?}"),
                &s.to_fragment_shader_full(&RenderConfig::default()),
            );
        }
    }
    let s = GlslShader::transpile(&common::corpus::corpus()[0].1, GlslTranspileMode::Hardcoded);
    assert_whole_tagged(
        "GLSL build_full_shader",
        &alice_sdf::compiled::glsl::render_pipeline::build_full_shader(
            &s.source,
            &RenderConfig::default(),
        ),
    );
}

/// The whole GLSL starts with `#version`. A GLSL version line selects the shader-text branch of
/// the stored-form reader, so the tagged GLSL is read and untagged GLSL is
/// `Missing { format: "shader text" }`
#[cfg(feature = "glsl")]
#[test]
fn whole_glsl_is_read_by_the_shader_text_branch() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let s = GlslShader::transpile(&common::corpus::corpus()[0].1, GlslTranspileMode::Hardcoded);
    let whole = s.to_compute_shader();
    assert!(whole.starts_with("#version"));
    assert!(matches!(
        provenance::from_bytes(untagged(&whole).as_bytes()),
        Err(provenance::ProvenanceError::Missing {
            format: "shader text"
        })
    ));
    assert_eq!(
        provenance::from_bytes(whole.as_bytes()).ok(),
        Some(Provenance::sdf())
    );
}

/// Only `#version <digits>` with an optional profile word is a GLSL header: an OBJ whose comment
/// merely starts like one is read as an OBJ. A text that starts with a real version line is
/// shader text, unless its leading `#` lines hold the key the way an OBJ stores it
#[test]
fn obj_comments_are_not_a_glsl_header() {
    let obj_body = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    for first in [
        "#version",
        "#version ",
        "#versions 1",
        "#version x",
        "#version 450ES",
        "#version 450 es extra",
        "#version 450 Es",
        "#VERSION 450",
    ] {
        let text = format!("{first}\n{obj_body}");
        assert!(
            matches!(
                provenance::from_bytes(text.as_bytes()),
                Err(provenance::ProvenanceError::Missing { format: "OBJ" })
            ),
            "{first:?}: {:?}",
            provenance::from_bytes(text.as_bytes())
        );
    }
    // an OBJ whose first comment is `#version 2` keeps its OBJ-style tag
    let obj_tag = format!("# {KEY}={}", Provenance::sdf().hex());
    let obj = format!("#version 2\n{obj_tag}\n{obj_body}");
    assert_eq!(
        provenance::from_bytes(obj.as_bytes()).ok(),
        Some(Provenance::sdf())
    );

    let sdf = tag_line(&Provenance::sdf());
    // the reader's tolerance: white space after `#` still makes a version line
    for first in [
        "# version 2",
        "#version 2",
        "#version 450",
        "#version 300 es",
        "#version 460 core",
        "#version 330\r",
    ] {
        for bare in [
            format!("{first}\nvoid main() {{}}\n"),
            format!("{first}\n// a comment\nvoid main() {{}}\n"),
        ] {
            assert!(
                matches!(
                    provenance::from_bytes(bare.as_bytes()),
                    Err(provenance::ProvenanceError::Missing {
                        format: "shader text"
                    })
                ),
                "{bare:?}"
            );
        }
        let tagged = format!("{first}\n{sdf}\nvoid main() {{}}\n");
        assert_eq!(
            provenance::from_bytes(tagged.as_bytes()).ok(),
            Some(Provenance::sdf()),
            "{first:?}"
        );
    }
}

/// A present tag outside the leading block of GLSL text is unreadable to the stored-form reader,
/// as it is invalid to the readers that compile the text: one verdict for the same bytes, never
/// read as unstamped
#[test]
fn a_glsl_tag_outside_the_block_is_unreadable() {
    let other = tag_line(&Provenance::new([0x5a; 32]));
    for text in [
        format!("#version 450\n#extension GL_X : enable\n{other}\nvoid main() {{}}\n"),
        format!("#version 450\n\n{other}\nvoid main() {{}}\n"),
        format!("#version 450\nvoid main() {{}}\n{other}\n"),
    ] {
        assert!(
            matches!(
                provenance::from_bytes(text.as_bytes()),
                Err(provenance::ProvenanceError::Unreadable(_))
            ),
            "{text:?}: {:?}",
            provenance::from_bytes(text.as_bytes())
        );
        #[cfg(feature = "gpu")]
        assert!(
            matches!(
                alice_sdf::compiled::GpuEvaluator::from_wgsl(&text),
                Err(alice_sdf::compiled::GpuError::InvalidFormat(_))
            ),
            "{text:?}"
        );
    }
}

/// The Nanite material form (a title line, the tag on line 2; B3) stays this crate's under the
/// rule: the writer's own test is `nanite_hlsl_material_carries_the_identifier` in
/// tests/test_provenance.rs, and here the same header in front of WGSL is `Current` for a reader
#[cfg(feature = "msl")]
#[test]
fn the_nanite_header_form_is_read_as_this_crates() {
    use alice_sdf::compiled::msl::MslShader;
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let wgsl = WgslShader::transpile(&common::corpus::corpus()[0].1, TranspileMode::Hardcoded)
        .to_compute_shader();
    let nanite_form = format!(
        "// ALICE-SDF Nanite Material Function\n{}\n// Auto-generated procedural material for UE5 Nanite\n{}",
        tag_line(&Provenance::sdf()),
        untagged(&wgsl)
    );
    assert_eq!(
        provenance::from_bytes(nanite_form.as_bytes()).ok(),
        Some(Provenance::sdf())
    );
    assert_eq!(
        MslShader::from_wgsl(&nanite_form).unwrap().semantics(),
        StoredSemantics::Current
    );
}

#[cfg(feature = "hlsl")]
#[test]
fn whole_hlsl_carries_the_identifier() {
    use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
    for (name, node) in common::corpus::corpus() {
        for mode in [HlslTranspileMode::Hardcoded, HlslTranspileMode::Dynamic] {
            let s = HlslShader::transpile(&node, mode);
            assert_whole_tagged(
                &format!("HLSL compute {name} {mode:?}"),
                &s.to_compute_shader(),
            );
        }
    }
}

#[cfg(feature = "msl")]
#[test]
fn whole_msl_carries_the_identifier() {
    use alice_sdf::compiled::msl::MslShader;
    use alice_sdf::compiled::TranspileMode;
    for (name, node) in common::corpus::corpus() {
        for mode in [TranspileMode::Hardcoded, TranspileMode::Dynamic] {
            let s = MslShader::transpile(&node, mode).unwrap();
            assert_whole_tagged(&format!("MSL {name} {mode:?}"), &s.source);
        }
    }
}

// --------------------------------------------------------------------------------------- readers

/// `MslShader::from_wgsl` (naga only, no GPU)
#[cfg(feature = "msl")]
#[test]
fn msl_from_wgsl_checks_the_identifier() {
    use alice_sdf::compiled::msl::{MslError, MslShader};
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let wgsl = WgslShader::transpile(&common::corpus::corpus()[0].1, TranspileMode::Hardcoded)
        .to_compute_shader();
    let c = cases(&wgsl);

    let ok = |t: &str| MslShader::from_wgsl(t).map(|s| s.semantics());
    assert_eq!(ok(&c.own).unwrap(), StoredSemantics::Current);
    assert_eq!(ok(&c.same_id_twice).unwrap(), StoredSemantics::Current);
    assert_eq!(ok(&c.untagged).unwrap(), StoredSemantics::Unpinned);
    for (what, text) in [
        ("tag after the block", &c.tag_after_block),
        (
            "tag after the first code line",
            &c.tag_after_first_code_line,
        ),
    ] {
        assert!(
            matches!(MslShader::from_wgsl(text), Err(MslError::InvalidFormat(_))),
            "{what}: {:?}",
            MslShader::from_wgsl(text).map(|c| c.semantics())
        );
    }
    match MslShader::from_wgsl(&c.other) {
        Err(MslError::SemanticsMismatch { stored, expected }) => {
            assert_eq!(stored, Provenance::new([0x5a; 32]));
            assert_eq!(expected, Provenance::sdf());
        }
        other => panic!("another identifier: {other:?}"),
    }
    assert!(matches!(
        MslShader::from_wgsl(&c.two_ids),
        Err(MslError::InvalidFormat(_))
    ));

    // the MSL carries the identifier of the WGSL it came from, none for another program's WGSL
    let (msl, _) = MslShader::from_wgsl(&c.own).unwrap().into_parts();
    assert_whole_tagged("MSL of this crate's WGSL", &msl.source);
    let (msl, _) = MslShader::from_wgsl(&c.untagged).unwrap().into_parts();
    assert!(
        !msl.source.contains(KEY),
        "MSL of unstamped WGSL claims an identifier:\n{}",
        msl.source
    );

    assert!(MslShader::from_wgsl_pinned(&c.own).is_ok());
    assert!(matches!(
        MslShader::from_wgsl_pinned(&c.untagged),
        Err(MslError::Unpinned)
    ));
    assert!(matches!(
        MslShader::from_wgsl_pinned(&c.other),
        Err(MslError::SemanticsMismatch { .. })
    ));
}

/// The refusals of the GPU readers. They come from the leading block alone, so every case here
/// runs without a GPU (the unit test `verdict_runs_before_any_device_request` in gpu_eval.rs
/// counts the device requests)
#[cfg(feature = "gpu")]
#[test]
fn gpu_readers_refuse_without_a_gpu() {
    use alice_sdf::compiled::{GpuError, GpuEvaluator, TranspileMode, WgslShader};
    let wgsl = WgslShader::transpile(&common::corpus::corpus()[0].1, TranspileMode::Hardcoded)
        .to_compute_shader();
    let c = cases(&wgsl);
    let mismatch = |r: Result<_, GpuError>| match r {
        Err(GpuError::SemanticsMismatch { stored, expected }) => {
            assert_eq!(stored, Provenance::new([0x5a; 32]));
            assert_eq!(expected, Provenance::sdf());
        }
        Err(e) => panic!("another identifier: {e}"),
        Ok(()) => panic!("another identifier was accepted"),
    };
    mismatch(GpuEvaluator::from_wgsl(&c.other).map(|_| ()));
    mismatch(GpuEvaluator::from_wgsl_pinned(&c.other).map(|_| ()));
    mismatch(pollster::block_on(GpuEvaluator::from_wgsl_async(&c.other)).map(|_| ()));
    mismatch(pollster::block_on(GpuEvaluator::from_wgsl_async_pinned(&c.other)).map(|_| ()));
    for text in [&c.two_ids, &c.tag_after_block, &c.tag_after_first_code_line] {
        assert!(matches!(
            GpuEvaluator::from_wgsl(text),
            Err(GpuError::InvalidFormat(_))
        ));
        assert!(matches!(
            pollster::block_on(GpuEvaluator::from_wgsl_async(text)),
            Err(GpuError::InvalidFormat(_))
        ));
    }
    assert!(matches!(
        GpuEvaluator::from_wgsl_pinned(&c.untagged),
        Err(GpuError::Unpinned)
    ));
    assert!(matches!(
        pollster::block_on(GpuEvaluator::from_wgsl_async_pinned(&c.untagged)),
        Err(GpuError::Unpinned)
    ));

    #[cfg(feature = "glsl")]
    {
        use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
        let glsl =
            GlslShader::transpile(&common::corpus::corpus()[0].1, GlslTranspileMode::Hardcoded)
                .to_compute_shader();
        let g = cases(&glsl);
        mismatch(GpuEvaluator::from_glsl_compute(&g.other).map(|_| ()));
        mismatch(GpuEvaluator::from_glsl_compute_pinned(&g.other).map(|_| ()));
        // the tag on line 3, after `#version` and a code line, is outside the block
        for text in [&g.two_ids, &g.tag_after_block, &g.tag_after_first_code_line] {
            assert!(matches!(
                GpuEvaluator::from_glsl_compute(text),
                Err(GpuError::InvalidFormat(_))
            ));
        }
        // the reader's tolerance of the version line: another identifier after it is refused,
        // never read as absent
        for prefix in [
            "#version 450  core",
            "\n#version 300 es",
            "\u{feff}#version 450",
            " # version 450 ",
        ] {
            let text = format!(
                "{prefix}\n{}\nvoid main() {{}}\n",
                tag_line(&Provenance::new([0x5a; 32]))
            );
            assert!(
                matches!(
                    GpuEvaluator::from_glsl_compute(&text),
                    Err(GpuError::SemanticsMismatch { .. } | GpuError::InvalidFormat(_))
                ),
                "{prefix:?}"
            );
        }
        assert!(matches!(
            GpuEvaluator::from_glsl_compute_pinned(&g.untagged),
            Err(GpuError::Unpinned)
        ));
    }
}

/// The accepted cases of the GPU readers build an evaluator, so they need an adapter (skipped
/// without one unless `ALICE_SDF_REQUIRE_GPU` is set, as the other GPU oracles)
#[cfg(feature = "gpu")]
#[test]
fn gpu_readers_mark_what_they_accept() {
    use alice_sdf::compiled::{GpuError, GpuEvaluator, TranspileMode, WgslShader};
    fn require<T>(what: &str, r: Result<T, GpuError>) -> Option<T> {
        match r {
            Ok(v) => Some(v),
            Err(e @ (GpuError::NoAdapter | GpuError::DeviceCreation(_))) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
                );
                eprintln!("{what}: skipped ({e})");
                None
            }
            Err(e) => panic!("{what}: {e}"),
        }
    }
    let wgsl = WgslShader::transpile(&common::corpus::corpus()[0].1, TranspileMode::Hardcoded)
        .to_compute_shader();
    let c = cases(&wgsl);
    for (what, text, want) in [
        ("own", &c.own, StoredSemantics::Current),
        ("same id twice", &c.same_id_twice, StoredSemantics::Current),
        ("untagged", &c.untagged, StoredSemantics::Unpinned),
    ] {
        if let Some(e) = require(what, GpuEvaluator::from_wgsl(text)) {
            assert_eq!(e.semantics(), want, "from_wgsl {what}");
        }
        if let Some(e) = require(
            what,
            pollster::block_on(GpuEvaluator::from_wgsl_async(text)),
        ) {
            assert_eq!(e.semantics(), want, "from_wgsl_async {what}");
        }
    }
    require("pinned own", GpuEvaluator::from_wgsl_pinned(&c.own));

    // The GLSL the GPU front end takes is the Vulkan-style wrapper of the oracles in
    // test_gpu_eval_api_oracle.rs (`to_compute_shader` is OpenGL GLSL), with the tag on line 2
    // after `#version` as the GLSL emitters write it
    #[cfg(feature = "glsl")]
    {
        use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
        let library =
            GlslShader::transpile(&common::corpus::corpus()[0].1, GlslTranspileMode::Hardcoded);
        let wrapped = format!(
            r"#version 450
layout(local_size_x = 256) in;
struct InputPoint {{ float x; float y; float z; float pad; }};
struct OutputDistance {{ float distance; float pad1; float pad2; float pad3; }};
layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ InputPoint input_points[]; }};
layout(std430, set = 0, binding = 1) buffer OutputDistances {{ OutputDistance output_distances[]; }};
layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};
{}
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) {{ return; }}
    InputPoint pt = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));
}}
",
            library.get_eval_function()
        );
        let glsl = insert_in_block(&wrapped, &tag_line(&Provenance::sdf()));
        let g = cases(&glsl);
        for (what, text, want) in [
            ("glsl own", &g.own, StoredSemantics::Current),
            ("glsl untagged", &g.untagged, StoredSemantics::Unpinned),
            (
                "glsl same id twice",
                &g.same_id_twice,
                StoredSemantics::Current,
            ),
        ] {
            if let Some(e) = require(what, GpuEvaluator::from_glsl_compute(text)) {
                assert_eq!(e.semantics(), want, "from_glsl_compute {what}");
            }
        }
        require(
            "glsl pinned own",
            GpuEvaluator::from_glsl_compute_pinned(&g.own),
        );
    }
}

// -------------------------------------------------------------------------------- classification

/// Every public text emitter of `src/` is a whole shader (tagged, tested above), a fragment that
/// is pasted into a host file (no tag; nothing in this crate reads it back), or not shader text
/// (with the reason). A new `pub fn` returning a `String`, or a new `pub` `String` field under
/// `src/compiled/`, fails here until it is placed
const WHOLE: &[(&str, &str)] = &[
    ("src/compiled/wgsl/transpiler.rs", "to_compute_shader"),
    (
        "src/compiled/wgsl/transpiler.rs",
        "to_compute_shader_with_normals",
    ),
    ("src/compiled/wgsl/transpiler.rs", "to_volume_shader"),
    ("src/compiled/glsl/transpiler.rs", "to_compute_shader"),
    (
        "src/compiled/glsl/transpiler.rs",
        "to_vulkan_compute_shader",
    ),
    ("src/compiled/glsl/transpiler.rs", "to_fragment_shader"),
    ("src/compiled/glsl/transpiler.rs", "to_fragment_shader_full"),
    ("src/compiled/glsl/render_pipeline.rs", "build_full_shader"),
    ("src/compiled/hlsl/transpiler.rs", "to_compute_shader"),
    ("src/compiled/msl/transpiler.rs", "source"),
];

const FRAGMENT: &str = "fragment: pasted into a host file or graph, not compiled by this crate";
const INTERNAL: &str =
    "internal: whole text this crate compiles itself, never read back through the four readers";
const NOT_SHADER: &str = "not shader text";

const OTHER: &[(&str, &str, &str)] = &[
    ("src/compiled/wgsl/transpiler.rs", "source", FRAGMENT),
    (
        "src/compiled/wgsl/transpiler.rs",
        "transpile_material",
        FRAGMENT,
    ),
    ("src/compiled/glsl/transpiler.rs", "source", FRAGMENT),
    (
        "src/compiled/glsl/transpiler.rs",
        "to_unity_custom_function",
        FRAGMENT,
    ),
    (
        "src/compiled/glsl/transpiler.rs",
        "export_unity_shader_graph",
        FRAGMENT,
    ),
    ("src/compiled/hlsl/transpiler.rs", "source", FRAGMENT),
    (
        "src/compiled/hlsl/transpiler.rs",
        "to_ue5_custom_node",
        FRAGMENT,
    ),
    (
        "src/compiled/hlsl/transpiler.rs",
        "to_unity_custom_function",
        FRAGMENT,
    ),
    (
        "src/compiled/hlsl/transpiler.rs",
        "export_unity_shader_graph",
        FRAGMENT,
    ),
    (
        "src/compiled/hlsl/transpiler.rs",
        "export_ue5_material_function",
        FRAGMENT,
    ),
    ("src/compiled/blinkscript/transpiler.rs", "source", FRAGMENT),
    (
        "src/compiled/blinkscript/transpiler.rs",
        "to_kernel",
        FRAGMENT,
    ),
    ("src/compiled/msl/transpiler.rs", "entry_point", NOT_SHADER),
    (
        "src/compiled/msl/transpiler.rs",
        "wgsl_entry_point",
        NOT_SHADER,
    ),
    (
        "src/compiled/rust/transpiler.rs",
        "det_math_path",
        NOT_SHADER,
    ),
    ("src/compiled/transpiler_common.rs", "globals", FRAGMENT),
    (
        "src/compiled/transpiler_common.rs",
        "transpile_node_inner",
        FRAGMENT,
    ),
    (
        "src/io/json.rs",
        "to_json_string_with_provenance",
        NOT_SHADER,
    ),
    ("src/texture/shader.rs", "generate_shader", FRAGMENT),
    ("src/compiled/msl/transpiler.rs", "wgsl_source", INTERNAL),
    ("src/compiled/instanced.rs", "to_instanced_wgsl", FRAGMENT),
    (
        "src/compiled/transpiler_common.rs",
        "generate_shader",
        FRAGMENT,
    ),
    ("src/compiled/transpiler_common.rs", "next_var", NOT_SHADER),
    ("src/compiled/transpiler_common.rs", "param", NOT_SHADER),
    (
        "src/compiled/transpiler_common.rs",
        "transpile_node",
        FRAGMENT,
    ),
    ("src/compiled/transpiler_common.rs", "lit", NOT_SHADER),
    ("src/compiled/wgsl/gpu_eval.rs", "device_info", NOT_SHADER),
    (
        "src/mesh/gpu_mc_shaders.rs",
        "generate_classify_shader",
        INTERNAL,
    ),
    (
        "src/mesh/gpu_mc_shaders.rs",
        "generate_prefix_sum_propagate_shader",
        INTERNAL,
    ),
    (
        "src/mesh/gpu_mc_shaders.rs",
        "generate_prefix_sum_scan_shader",
        INTERNAL,
    ),
    (
        "src/mesh/gpu_mc_shaders.rs",
        "generate_sdf_grid_shader",
        INTERNAL,
    ),
    (
        "src/mesh/gpu_mc_shaders.rs",
        "generate_vertex_shader",
        INTERNAL,
    ),
    (
        "src/npr/compiled_color.rs",
        "emit_wgsl_bytecode_evaluator",
        FRAGMENT,
    ),
    ("src/npr/scene_composer.rs", "build", INTERNAL),
    ("src/npr/shader_glue.rs", "full_helpers_for", FRAGMENT),
    ("src/io/json.rs", "to_json_string", NOT_SHADER),
    ("src/io/mod.rs", "get_info", NOT_SHADER),
    ("src/llm_schema.rs", "schema_summary", NOT_SHADER),
    ("src/provenance.rs", "hex", NOT_SHADER),
    ("src/provenance.rs", "tagged", NOT_SHADER),
    ("src/wasm.rs", "alice_sdf_version", NOT_SHADER),
];

fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// `pub fn` returning `String` / `Result<String, _>`, and (under `src/compiled/`) `pub` fields of
/// type `String`, as (file, name)
fn text_emitters(root: &std::path::Path) -> Vec<(String, String)> {
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    let mut out = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if rel.starts_with("src/bin/") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let mut rest = text.as_str();
        while let Some(i) = rest.find("pub fn ").or_else(|| rest.find("pub async fn ")) {
            let after = &rest[i..];
            let after = &after[after.find("fn ").unwrap() + 3..];
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let sig_end = after.find(['{', ';']).unwrap_or(after.len());
            let ret = after[..sig_end].rsplit("->").next().unwrap_or("").trim();
            if after[..sig_end].contains("->")
                && (ret == "String" || ret.starts_with("Result<String"))
            {
                out.push((rel.clone(), name));
            }
            rest = &after[sig_end.min(after.len())..];
        }
        if rel.starts_with("src/compiled/") {
            for line in text.lines() {
                let t = line.trim();
                if let Some(field) = t.strip_prefix("pub ") {
                    if let Some((name, ty)) = field.split_once(':') {
                        if ty.trim() == "String,"
                            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                        {
                            out.push((rel.clone(), name.to_string()));
                        }
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn every_text_emitter_is_whole_fragment_or_not_shader() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let found = text_emitters(root);
    assert!(
        found.len() >= WHOLE.len(),
        "{} text emitters enumerated under src/ (fewer than the {} whole shaders)",
        found.len(),
        WHOLE.len()
    );
    let mut problems = Vec::new();
    for (file, name) in &found {
        let whole = WHOLE.iter().any(|(f, n)| f == file && n == name);
        let other = OTHER.iter().any(|(f, n, _)| f == file && n == name);
        if !whole && !other {
            problems.push(format!(
                "{file}: `{name}` returns text and is not placed: add it to WHOLE in \
                 tests/test_shader_text_provenance.rs (and tag its output, with a test above) \
                 or to OTHER with the reason"
            ));
        }
        if whole && other {
            problems.push(format!("{file}: `{name}` is both whole and other"));
        }
    }
    for (file, name) in WHOLE
        .iter()
        .copied()
        .chain(OTHER.iter().map(|(f, n, _)| (*f, *n)))
    {
        if !found.iter().any(|(f, n)| f == file && n == name) {
            problems.push(format!("{file}: `{name}` is listed but no longer found"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
