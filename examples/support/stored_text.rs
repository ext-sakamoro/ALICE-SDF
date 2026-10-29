//! The two ways the examples write emitted text to a file (ALICE-Law G8)
//!
//! - [`write_whole_text`]: a whole shader (one a GPU API or an offline compiler takes as is) or the
//!   generated Rust module of a scene. The text carries this crate's semantics identifier in its
//!   leading `//` block (the transpilers and `RustSource` write it); text without it, or under
//!   another identifier, is refused before anything is written.
//! - [`write_fragment_text`]: a fragment pasted into a host file or graph (a function body, a
//!   material function, an include) or a list a host build reads. A fragment carries no identifier;
//!   text that does carry one is refused, so a whole shader is never written as a fragment.
//!
//! Both write only when the file's contents change and return whether they did.
//!
//! Author: Moroya Sakamoto

// ALLOW-DEAD: each example includes this module and calls one or both of its writers
#![allow(dead_code)]

use alice_sdf::provenance::{self, Provenance, KEY, MEANING_KEY};
use std::fs;
use std::io;
use std::path::Path;

/// The line a whole text built by hand puts first in its leading `//` block
pub fn tag_line() -> String {
    format!(
        "// {}\n",
        Provenance::sdf().tagged_for(provenance::StoredForm::ShaderText)
    )
}

/// Writes the whole shader or generated Rust `text` to `path` when it differs from the file
///
/// # Errors
///
/// When `text` does not carry this crate's semantics identifier in its leading `//` block (no
/// identifier, another one, or one outside the block), and the errors of the write.
pub fn write_whole_text(path: &Path, text: &str) -> io::Result<bool> {
    match provenance::from_bytes(text.as_bytes()) {
        Ok(p)
            if matches!(
                p.judge(&Provenance::sdf(), &[]),
                Ok(provenance::StoredSemantics::Current)
            ) => {}
        Ok(p) => {
            return Err(io::Error::other(format!(
                "{}: whole text under {} (this build writes {})",
                path.display(),
                p.hex(),
                Provenance::sdf()
                    .named_for(provenance::StoredForm::ShaderText)
                    .1
            )))
        }
        Err(e) => {
            return Err(io::Error::other(format!(
                "{}: whole text without this build's identifier: {e}",
                path.display()
            )))
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::read_to_string(path).ok().as_deref() == Some(text) {
        return Ok(false);
    }
    fs::write(path, text)?;
    Ok(true)
}

/// Writes the fragment `text` to `path` when it differs from the file
///
/// # Errors
///
/// When `text` carries the identifier key (a whole text written as a fragment), and the errors
/// of the write.
pub fn write_fragment_text(path: &Path, text: &str) -> io::Result<bool> {
    for key in [MEANING_KEY, KEY] {
        if text.contains(key) {
            return Err(io::Error::other(format!(
                "{}: a fragment carries no {key} (this text is a whole one)",
                path.display()
            )));
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    if fs::read_to_string(path).ok().as_deref() == Some(text) {
        return Ok(false);
    }
    fs::write(path, text)?;
    Ok(true)
}
