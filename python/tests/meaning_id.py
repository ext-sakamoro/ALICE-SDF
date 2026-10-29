"""The meaning identifier of a stored form, computed independently of the Rust code.

`alice_sdf::provenance::meaning_id` is `alice_zip::law_id::LawIdHasher` v2 over the kind
`alice-sdf/stored-form-meaning`, the semantics identifier, the form's name (a str value) and its
layout version (a u32 value). This file re-implements that with the standard library only, from the
layout ALICE-Zip documents (scripts/law_id_v2_reference.py there): SHA-256 over the domain, the
kind and the semantics identifier, each length a big-endian u64, then each value with a one-byte
type tag (0x05 str: length and UTF-8 bytes, 0x02 u32: 4 bytes big-endian).

Author: Moroya Sakamoto
"""

from __future__ import annotations

import hashlib
import struct

DOMAIN = b"alice-law-id/v2"
KIND = b"alice-sdf/stored-form-meaning"


def meaning_id(form: str, semantics_hex: str, version: int = 1) -> str:
    semantics = bytes.fromhex(semantics_hex)
    if len(semantics) != 32:
        raise ValueError("the semantics identifier is not 32 bytes")
    h = hashlib.sha256()
    h.update(struct.pack(">Q", len(DOMAIN)) + DOMAIN)
    h.update(struct.pack(">Q", len(KIND)) + KIND)
    h.update(semantics)
    name = form.encode()
    h.update(b"\x05" + struct.pack(">Q", len(name)) + name)
    h.update(b"\x02" + struct.pack(">I", version))
    return h.hexdigest()
