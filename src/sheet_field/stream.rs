//! Streamed sheet fields: march a field that does not fit in memory, one batch of sheets at a time, with the same
//! hits as when every sheet is in memory.
//!
//! On disk a field is the ASHF v1 layout (little-endian): `b"ASHF"`, `u32` version 1, `u32` rule (1 = cell-v1,
//! 0 = none), `u32` sheet count; per sheet: `u32` level count; per level: `f32` h, `f32` lo[3], `u32` nb[3], `u32`
//! n_index, `i32` index[n_index], `u32` channel count; per channel: `u32` n_rows, `i32` rows[n_rows], `u32` n_kept,
//! `f32` scale[n_kept], `f32` offset[n_kept], `u8` q[n_kept * [`super::BRICK_NODES`]]. ASHF v2 is the same with one
//! `u32` nodes-per-axis after `n_kept` in every channel, and `q[n_kept * nodes^3]` (an attribute channel on a
//! coarse grid, [`super::COARSE_NODES`]). [`write_ashf`] writes v1 when every channel is on 9 nodes (the bytes
//! of a field without coarse channels do not change) and v2 otherwise,
//! [`AshfIndex::scan`] finds where each sheet lies (it reads the counts and seeks past the arrays), and
//! [`AshfIndex::read_sheets`] reads a contiguous range of sheets.
//!
//! Two splits, kept apart on purpose:
//! - **chunks** ([`chunk_split`]): contiguous sheets whose channel bytes fit one GPU storage binding. A chunk is
//!   marched as one [`SheetField`]. The march steps by the union's lower bound, so the chunk split decides where
//!   the samples fall and with them the last bits of `t`: the split depends only on the field and the cap, never
//!   on the memory that happens to be free.
//! - **batches** ([`host_batches`]): runs of whole chunks whose sheets fit the host memory budget. Only one batch
//!   is in memory at a time. The batch split changes nothing in the result (each chunk is marched the same way
//!   whichever batch holds it), so the budget can come from the memory measured at run time
//!   ([`available_memory`]).
//!
//! Merge ([`march_streamed`]): per ray, the hit with the smallest `t` over the chunks; on equal `t` the earlier
//! chunk (lower sheet numbers) keeps it, as one chunk keeps its lowest sheet. A ray that is undecided in any chunk
//! is undecided (the missing chunk could hold the first hit); a miss needs a miss in every chunk. Hit sheets are
//! renumbered to the field's sheet numbers.

use super::{
    build, Channel, Level, MarchResult, Sheet, SheetField, SheetFieldError, ValidRule, NODES,
};
use glam::Vec3;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::ops::Range;

/// The 4 bytes an ASHF file starts with.
pub const ASHF_MAGIC: [u8; 4] = *b"ASHF";
/// The ASHF version written for a field whose channels are all on 9 nodes per axis.
pub const ASHF_VERSION: u32 = 1;

/// The ASHF version written when a channel is on a coarse grid (a `u32` nodes count per channel).
pub const ASHF_VERSION_COARSE: u32 = 2;

/// The ASHF version written when a level holds a band ([`super::BAND`] channels).
///
/// It is v2 plus, per level before its channels, `u32` n_band and `u32` band[n_band] (the mask), and for a band channel a `u32` byte count of
/// `q` after its nodes count (its rows hold different node counts; their starts are rebuilt on reading).
pub const ASHF_VERSION_BAND: u32 = 3;

/// Errors from reading, splitting or marching a streamed field.
#[derive(Debug)]
pub enum StreamError {
    /// Reading or writing the file failed.
    Io(io::Error),
    /// The file is not a valid ASHF v1 field.
    Format(String),
    /// A sheet's data is inconsistent.
    Field(SheetFieldError),
    /// A sheet or chunk does not fit the cap or budget it was given (refused instead of overrunning it).
    Budget(String),
    /// The march callback failed.
    March(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "ASHF i/o: {e}"),
            Self::Format(s) => write!(f, "not a valid ASHF v1 field: {s}"),
            Self::Field(e) => write!(f, "{e}"),
            Self::Budget(s) => write!(f, "over budget: {s}"),
            Self::March(s) => write!(f, "march: {s}"),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<io::Error> for StreamError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<SheetFieldError> for StreamError {
    fn from(e: SheetFieldError) -> Self {
        Self::Field(e)
    }
}

const fn rule_code(rule: ValidRule) -> u32 {
    match rule {
        ValidRule::CellV1 => 1,
        ValidRule::None => 0,
    }
}

fn rule_of(code: u32) -> Result<ValidRule, StreamError> {
    match code {
        1 => Ok(ValidRule::CellV1),
        0 => Ok(ValidRule::None),
        x => Err(StreamError::Format(format!("unknown rule code {x}"))),
    }
}

fn len_u32(n: usize, what: &str) -> Result<u32, StreamError> {
    u32::try_from(n).map_err(|_| StreamError::Format(format!("{what} count {n} does not fit u32")))
}

/// Write `field` in the ASHF layout (v1, or v2 when a channel is on a coarse grid); returns the bytes written.
pub fn write_ashf<W: Write>(w: &mut W, field: &SheetField) -> Result<u64, StreamError> {
    let mut n = 0u64;
    let mut put = |w: &mut W, b: &[u8]| -> io::Result<()> {
        n += b.len() as u64;
        w.write_all(b)
    };
    let levels = || field.sheets.iter().flat_map(|s| &s.levels);
    let banded = levels().any(|l| !l.band.is_empty());
    let coarse = banded
        || levels()
            .flat_map(|l| &l.channels)
            .any(|c| c.nodes != NODES as u32);
    let version = if banded {
        ASHF_VERSION_BAND
    } else if coarse {
        ASHF_VERSION_COARSE
    } else {
        ASHF_VERSION
    };
    put(w, &ASHF_MAGIC)?;
    put(w, &version.to_le_bytes())?;
    put(w, &rule_code(field.rule).to_le_bytes())?;
    put(w, &len_u32(field.sheets.len(), "sheet")?.to_le_bytes())?;
    for sh in &field.sheets {
        put(w, &len_u32(sh.levels.len(), "level")?.to_le_bytes())?;
        for lv in &sh.levels {
            put(w, &lv.h.to_le_bytes())?;
            for x in lv.lo.to_array() {
                put(w, &x.to_le_bytes())?;
            }
            for x in lv.nb {
                put(w, &x.to_le_bytes())?;
            }
            put(w, &len_u32(lv.index.len(), "index")?.to_le_bytes())?;
            put(w, &i32s_le(&lv.index))?;
            if banded {
                put(w, &len_u32(lv.band.len(), "band word")?.to_le_bytes())?;
                put(
                    w,
                    &lv.band
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
            }
            put(w, &len_u32(lv.channels.len(), "channel")?.to_le_bytes())?;
            for c in &lv.channels {
                put(w, &len_u32(c.rows.len(), "row")?.to_le_bytes())?;
                put(w, &i32s_le(&c.rows))?;
                put(w, &len_u32(c.scale.len(), "kept row")?.to_le_bytes())?;
                if coarse {
                    put(w, &c.nodes.to_le_bytes())?;
                }
                if banded && c.nodes == super::BAND {
                    put(w, &len_u32(c.q.len(), "band byte")?.to_le_bytes())?;
                }
                put(w, &f32s_le(&c.scale))?;
                put(w, &f32s_le(&c.offset))?;
                put(w, &c.q)?;
            }
        }
    }
    Ok(n)
}

fn i32s_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, StreamError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, StreamError> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn read_words<R: Read>(r: &mut R, n: usize) -> Result<Vec<[u8; 4]>, StreamError> {
    let mut b = vec![0u8; n * 4];
    r.read_exact(&mut b)?;
    Ok((0..n)
        .map(|i| [b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
        .collect())
}

fn read_i32s<R: Read>(r: &mut R, n: usize) -> Result<Vec<i32>, StreamError> {
    Ok(read_words(r, n)?
        .into_iter()
        .map(i32::from_le_bytes)
        .collect())
}

fn read_f32s<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>, StreamError> {
    Ok(read_words(r, n)?
        .into_iter()
        .map(f32::from_le_bytes)
        .collect())
}

/// Where one sheet lies in an ASHF file and what it costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SheetSpan {
    /// Byte offset of the sheet's level count.
    pub offset: u64,
    /// Bytes of the sheet in the file.
    pub len: u64,
    /// Bytes of quantized channel data (`q`), what one GPU storage binding holds.
    pub channel_bytes: u64,
    /// Bytes of the sheet's arrays once read ([`sheet_payload_bytes`] of the read sheet).
    pub host_bytes: u64,
}

/// Bytes of a sheet's arrays in memory: brick index, rows, scale, offset (4 bytes each) and `q`.
pub fn sheet_payload_bytes(sh: &Sheet) -> u64 {
    sh.levels
        .iter()
        .map(|lv| {
            4 * (lv.index.len() + lv.band.len()) as u64
                + lv.channels
                    .iter()
                    .map(|c| {
                        4 * (c.rows.len() + c.scale.len() + c.offset.len() + c.starts.len()) as u64
                            + c.q.len() as u64
                    })
                    .sum::<u64>()
        })
        .sum()
}

/// The byte where each kept row of a band channel starts: rows are packed in row order, each holding the nodes
/// of its brick's band mask.
fn band_starts(rows: &[i32], nk: usize, band: &[u32]) -> Result<Vec<u32>, StreamError> {
    let mut count = vec![u32::MAX; nk];
    for (k, &row) in rows.iter().enumerate() {
        if row >= 0 {
            let m = band
                .get(k * super::BAND_WORDS..(k + 1) * super::BAND_WORDS)
                .ok_or_else(|| StreamError::Format(format!("band mask too short for brick {k}")))?;
            let r = row as usize;
            if r >= nk {
                return Err(StreamError::Format(format!("row {r} of {nk}")));
            }
            count[r] = m.iter().map(|x| x.count_ones()).sum();
        }
    }
    let mut starts = Vec::with_capacity(nk);
    let mut at = 0u32;
    for (r, c) in count.into_iter().enumerate() {
        if c == u32::MAX {
            return Err(StreamError::Format(format!("band row {r} has no brick")));
        }
        starts.push(at);
        at += c;
    }
    Ok(starts)
}

/// The sheets of an ASHF file, located without reading their arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AshfIndex {
    /// The field's validity rule.
    pub rule: ValidRule,
    /// One span per sheet, in file order.
    pub spans: Vec<SheetSpan>,
    /// The file's ASHF version (1, 2 or 3).
    pub version: u32,
}

/// A nodes-per-axis count read from a v2 / v3 file: [`NODES`], one of [`super::COARSE_NODES`], or (v3 only)
/// [`super::BAND`].
fn nodes_of(n: u32, version: u32) -> Result<u32, StreamError> {
    if n == NODES as u32
        || super::COARSE_NODES.contains(&n)
        || (n == super::BAND && version == ASHF_VERSION_BAND)
    {
        Ok(n)
    } else {
        Err(StreamError::Format(format!("{n} nodes per brick axis")))
    }
}

impl AshfIndex {
    /// Read the header and every count, seeking past the arrays. Refuses a wrong magic or version, an unknown
    /// rule, a brick index that does not match its grid, a level without channels, and trailing bytes.
    pub fn scan<R: Read + Seek>(r: &mut R) -> Result<Self, StreamError> {
        let end = r.seek(SeekFrom::End(0))?;
        r.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != ASHF_MAGIC {
            return Err(StreamError::Format(format!("magic {magic:?}")));
        }
        let version = read_u32(r)?;
        if ![ASHF_VERSION, ASHF_VERSION_COARSE, ASHF_VERSION_BAND].contains(&version) {
            return Err(StreamError::Format(format!("version {version}")));
        }
        let rule = rule_of(read_u32(r)?)?;
        let ns = read_u32(r)? as usize;
        let mut spans = Vec::with_capacity(ns);
        let mut pos = 16u64;
        let skip = |r: &mut R, pos: &mut u64, n: u64| -> Result<(), StreamError> {
            *pos += n;
            if *pos > end {
                return Err(StreamError::Format(format!(
                    "array runs past the end ({} > {end})",
                    *pos
                )));
            }
            r.seek(SeekFrom::Start(*pos))?;
            Ok(())
        };
        for k in 0..ns {
            let offset = pos;
            let (mut channel_bytes, mut host_bytes) = (0u64, 0u64);
            let nl = read_u32(r)?;
            pos += 4;
            for _ in 0..nl {
                let _h = read_f32(r)?;
                for _ in 0..3 {
                    read_f32(r)?;
                }
                let nb = [read_u32(r)?, read_u32(r)?, read_u32(r)?];
                let ni = read_u32(r)?;
                pos += 32;
                let cells = u64::from(nb[0]) * u64::from(nb[1]) * u64::from(nb[2]);
                if u64::from(ni) != cells {
                    return Err(StreamError::Format(format!(
                        "sheet {k}: index of {ni} for a {nb:?} grid"
                    )));
                }
                skip(r, &mut pos, 4 * u64::from(ni))?;
                host_bytes += 4 * u64::from(ni);
                if version == ASHF_VERSION_BAND {
                    let nw = read_u32(r)?;
                    pos += 4;
                    skip(r, &mut pos, 4 * u64::from(nw))?;
                    host_bytes += 4 * u64::from(nw);
                }
                let nch = read_u32(r)?;
                pos += 4;
                if nch == 0 {
                    return Err(StreamError::Format(format!(
                        "sheet {k}: a level without channels"
                    )));
                }
                for _ in 0..nch {
                    let nr = read_u32(r)?;
                    pos += 4;
                    skip(r, &mut pos, 4 * u64::from(nr))?;
                    let nk = read_u32(r)?;
                    pos += 4;
                    let nodes = if version >= ASHF_VERSION_COARSE {
                        pos += 4;
                        nodes_of(read_u32(r)?, version)?
                    } else {
                        NODES as u32
                    };
                    let (q, starts) = if nodes == super::BAND {
                        pos += 4;
                        (u64::from(read_u32(r)?), 4 * u64::from(nk))
                    } else {
                        (u64::from(nk) * u64::from(nodes).pow(3), 0)
                    };
                    skip(r, &mut pos, 8 * u64::from(nk) + q)?;
                    channel_bytes += q;
                    host_bytes += 4 * (u64::from(nr) + 2 * u64::from(nk)) + q + starts;
                }
            }
            spans.push(SheetSpan {
                offset,
                len: pos - offset,
                channel_bytes,
                host_bytes,
            });
        }
        if pos != end {
            return Err(StreamError::Format(format!("{} trailing bytes", end - pos)));
        }
        Ok(Self {
            rule,
            spans,
            version,
        })
    }

    /// Read sheets `range` (file order) into memory.
    pub fn read_sheets<R: Read + Seek>(
        &self,
        r: &mut R,
        range: Range<usize>,
    ) -> Result<Vec<Sheet>, StreamError> {
        if range.end > self.spans.len() || range.start > range.end {
            return Err(StreamError::Format(format!(
                "sheets {range:?} of {}",
                self.spans.len()
            )));
        }
        let mut out = Vec::with_capacity(range.len());
        let Some(first) = self.spans.get(range.start) else {
            return Ok(out);
        };
        r.seek(SeekFrom::Start(first.offset))?;
        let mut rd = io::BufReader::with_capacity(1 << 20, r);
        for k in range {
            let nl = read_u32(&mut rd)? as usize;
            let mut levels = Vec::with_capacity(nl);
            for _ in 0..nl {
                let h = read_f32(&mut rd)?;
                let lo = Vec3::new(read_f32(&mut rd)?, read_f32(&mut rd)?, read_f32(&mut rd)?);
                let nb = [read_u32(&mut rd)?, read_u32(&mut rd)?, read_u32(&mut rd)?];
                let ni = read_u32(&mut rd)? as usize;
                let index = read_i32s(&mut rd, ni)?;
                let band = if self.version == ASHF_VERSION_BAND {
                    let nw = read_u32(&mut rd)? as usize;
                    let mut b = vec![0u8; 4 * nw];
                    rd.read_exact(&mut b)?;
                    b.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect()
                } else {
                    Vec::new()
                };
                let nch = read_u32(&mut rd)? as usize;
                let mut channels = Vec::with_capacity(nch);
                for _ in 0..nch {
                    let nr = read_u32(&mut rd)? as usize;
                    let rows = read_i32s(&mut rd, nr)?;
                    let nk = read_u32(&mut rd)? as usize;
                    let nodes = if self.version >= ASHF_VERSION_COARSE {
                        nodes_of(read_u32(&mut rd)?, self.version)?
                    } else {
                        NODES as u32
                    };
                    let qlen = if nodes == super::BAND {
                        read_u32(&mut rd)? as usize
                    } else {
                        nk * (nodes as usize).pow(3)
                    };
                    let scale = read_f32s(&mut rd, nk)?;
                    let offset = read_f32s(&mut rd, nk)?;
                    let mut q = vec![0u8; qlen];
                    rd.read_exact(&mut q)?;
                    let starts = if nodes == super::BAND {
                        band_starts(&rows, nk, &band)?
                    } else {
                        Vec::new()
                    };
                    channels.push(Channel {
                        rows,
                        q,
                        nodes,
                        starts,
                        scale,
                        offset,
                    });
                }
                levels.push(Level {
                    h,
                    lo,
                    nb,
                    index,
                    channels,
                    band,
                });
            }
            let sh = build::sheet(levels)?;
            let got = sheet_payload_bytes(&sh);
            if got != self.spans[k].host_bytes {
                return Err(StreamError::Format(format!(
                    "sheet {k}: {got} bytes read, the index says {}",
                    self.spans[k].host_bytes
                )));
            }
            out.push(sh);
        }
        Ok(out)
    }
}

/// Split the sheets into contiguous chunks whose channel bytes stay within `cap`.
///
/// A sheet larger than `cap` ends up in a chunk of its own (nothing fits beside it), up to `single_max` (the device's storage binding limit). Depends only on the field and the
/// two caps, never on free memory. Refuses a sheet over `single_max`.
pub fn chunk_split(
    index: &AshfIndex,
    cap: u64,
    single_max: u64,
) -> Result<Vec<Range<usize>>, StreamError> {
    let mut out = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for (k, s) in index.spans.iter().enumerate() {
        if s.channel_bytes > single_max {
            return Err(StreamError::Budget(format!(
                "sheet {k} alone has {} channel bytes, over the device's storage binding limit {single_max}",
                s.channel_bytes
            )));
        }
        if bytes + s.channel_bytes > cap && k > start {
            out.push(start..k);
            start = k;
            bytes = 0;
        }
        bytes += s.channel_bytes;
    }
    if start < index.spans.len() {
        out.push(start..index.spans.len());
    }
    Ok(out)
}

/// Group whole chunks into batches whose sheets' host bytes stay within `budget`; returns ranges over `chunks`.
/// Refuses a chunk that alone exceeds `budget` (rather than overrun the memory it was given).
pub fn host_batches(
    index: &AshfIndex,
    chunks: &[Range<usize>],
    budget: u64,
) -> Result<Vec<Range<usize>>, StreamError> {
    let cost = |c: &Range<usize>| {
        index.spans[c.clone()]
            .iter()
            .map(|s| s.host_bytes)
            .sum::<u64>()
    };
    let mut out = Vec::new();
    let (mut start, mut bytes) = (0usize, 0u64);
    for (i, c) in chunks.iter().enumerate() {
        let b = cost(c);
        if b > budget {
            return Err(StreamError::Budget(format!(
                "chunk {i} (sheets {c:?}) needs {b} host bytes, over the budget {budget}"
            )));
        }
        if bytes + b > budget && i > start {
            out.push(start..i);
            start = i;
            bytes = 0;
        }
        bytes += b;
    }
    if start < chunks.len() {
        out.push(start..chunks.len());
    }
    Ok(out)
}

/// The memory the system says can be allocated without swapping (Linux `MemAvailable`), in bytes. `None` where
/// the system does not report it (the caller then passes an explicit budget).
pub fn available_memory() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = s
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    kb.checked_mul(1024)
}

/// What a streamed march did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    /// Batches read.
    pub batches: usize,
    /// Chunks marched.
    pub chunks: usize,
    /// The largest host bytes of the sheets in memory at one time (measured on the read sheets).
    pub peak_host_bytes: u64,
}

/// Merge one chunk's results into the running ones (see the module docs): smaller `t` wins, equal `t` keeps the
/// earlier chunk, undecided anywhere is undecided, miss needs a miss everywhere. `base` renumbers hit sheets.
fn merge(acc: &mut [Option<MarchResult>], part: &[MarchResult], base: usize) {
    for (a, p) in acc.iter_mut().zip(part) {
        let p = match *p {
            MarchResult::Hit(mut h) => {
                h.sheet += base;
                MarchResult::Hit(h)
            }
            other => other,
        };
        *a = Some(match (*a, p) {
            (None, p) => p,
            (Some(MarchResult::Undecided), _) | (_, MarchResult::Undecided) => {
                MarchResult::Undecided
            }
            (Some(MarchResult::Hit(x)), MarchResult::Hit(y)) => {
                MarchResult::Hit(if y.t < x.t { y } else { x })
            }
            (Some(MarchResult::Hit(x)), MarchResult::Miss)
            | (Some(MarchResult::Miss), MarchResult::Hit(x)) => MarchResult::Hit(x),
            (Some(MarchResult::Miss), MarchResult::Miss) => MarchResult::Miss,
        });
    }
}

/// March `n_rays` rays through the field in `index`, reading one batch of chunks at a time from `r`.
///
/// `march` is called once per chunk with that chunk's field and must return one result per ray (CPU or GPU; the caller
/// owns the rays). Returns the merged results (sheets numbered as in the file) and what was read.
pub fn march_streamed<R, F>(
    index: &AshfIndex,
    r: &mut R,
    chunks: &[Range<usize>],
    batches: &[Range<usize>],
    n_rays: usize,
    march: F,
) -> Result<(Vec<MarchResult>, StreamStats), StreamError>
where
    R: Read + Seek,
    F: FnMut(&SheetField) -> Result<Vec<MarchResult>, String>,
{
    march_streamed_kink(index, r, None, chunks, batches, n_rays, march)
}

/// [`march_streamed`] with the field's kink cells ([`super::kink`]).
///
/// Each batch's sheets get their kink cells from `kink` (sheets numbered as in the file) before they are marched. The
/// peak host bytes count the sheets only.
pub fn march_streamed_kink<R, F>(
    index: &AshfIndex,
    r: &mut R,
    kink: Option<&super::kink::KinkFile>,
    chunks: &[Range<usize>],
    batches: &[Range<usize>],
    n_rays: usize,
    mut march: F,
) -> Result<(Vec<MarchResult>, StreamStats), StreamError>
where
    R: Read + Seek,
    F: FnMut(&SheetField) -> Result<Vec<MarchResult>, String>,
{
    if let Some(k) = kink {
        if k.len() != index.spans.len() {
            return Err(StreamError::Format(format!(
                "the kink file describes {} sheets, the field has {}",
                k.len(),
                index.spans.len()
            )));
        }
    }
    let covered: usize = batches.iter().map(|b| b.len()).sum();
    if covered != chunks.len() || batches.windows(2).any(|w| w[0].end != w[1].start) {
        return Err(StreamError::Format(format!(
            "batches {batches:?} do not tile the {} chunks",
            chunks.len()
        )));
    }
    let mut acc: Vec<Option<MarchResult>> = vec![None; n_rays];
    let mut stats = StreamStats {
        batches: 0,
        chunks: 0,
        peak_host_bytes: 0,
    };
    for b in batches {
        let cs = &chunks[b.clone()];
        let (Some(first), Some(last)) = (cs.first(), cs.last()) else {
            continue;
        };
        let mut sheets = index.read_sheets(r, first.start..last.end)?;
        if let Some(k) = kink {
            k.attach(&mut sheets, first.start)
                .map_err(|e| StreamError::Format(e.to_string()))?;
        }
        let held: u64 = sheets.iter().map(sheet_payload_bytes).sum();
        stats.peak_host_bytes = stats.peak_host_bytes.max(held);
        stats.batches += 1;
        // split the batch's sheets back into its chunks, last chunk first so each split_off is cheap
        let mut parts: Vec<(usize, Vec<Sheet>)> = Vec::with_capacity(cs.len());
        for c in cs.iter().rev() {
            parts.push((c.start, sheets.split_off(c.start - first.start)));
        }
        for (base, ss) in parts.into_iter().rev() {
            let field = SheetField::new(ss, index.rule);
            let part = march(&field).map_err(StreamError::March)?;
            if part.len() != n_rays {
                return Err(StreamError::March(format!(
                    "{} results for {n_rays} rays",
                    part.len()
                )));
            }
            merge(&mut acc, &part, base);
            stats.chunks += 1;
        }
    }
    let out = acc
        .into_iter()
        .map(|a| a.ok_or_else(|| StreamError::Format("a field with no sheets".into())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((out, stats))
}
