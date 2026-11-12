//! A GGUF file mapped read-only and private, through `mapfile`.

use std::collections::HashMap;
use std::path::Path;

use mapfile::Mapped;

use crate::{Error, ErrorKind, GgmlType, Gguf, Result};

/// Options for [`GgufFile::open_with`].
///
/// The default maps the file without prefaulting it. Parsing the header
/// faults only the header pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapOptions {
    /// Prefault the whole file inside `open_with`.
    ///
    /// The process resident set then grows by about the file size. Which
    /// targets honour it is in [`mapfile::Mapped::open`].
    pub populate: bool,
}

/// A GGUF file mapped read-only and private.
///
/// `open` parses the header once and keeps a tensor table: name, ggml type,
/// and absolute byte range. [`tensor_bytes`](Self::tensor_bytes) and
/// [`tensor_blocks`](Self::tensor_blocks) use that table. [`parsed`](Self::parsed)
/// walks the header again when a caller wants metadata views. Those views
/// borrow this file. Nothing in the struct pretends the borrows are `'static`.
pub struct GgufFile {
    map: Mapped,
    tensors: HashMap<String, TensorLoc>,
}

/// Where one tensor's bytes sit in the mapping.
///
/// Owned, so the table does not borrow the map. The names are copied out of
/// the header at `open`. The weight bytes stay in the mapping.
#[derive(Clone, Copy)]
struct TensorLoc {
    ggml_type: GgmlType,
    start: u64,
    len: u64,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("len", &self.map.len())
            .finish()
    }
}

impl GgufFile {
    /// Maps `path` read-only and private, then parses the header from the
    /// mapping.
    ///
    /// # File contract
    ///
    /// Nothing may truncate or write the file while this value is alive. A
    /// fault on a page that no longer exists delivers `SIGBUS` and kills the
    /// process. A write, by this process or another, is undefined behaviour
    /// for the slices this type returns. `MAP_PRIVATE` with `PROT_READ` does
    /// not take a private copy of the bytes.
    ///
    /// This is a safe function all the same. The obligation is on whoever
    /// controls the file system, and no code at a call site can check or
    /// discharge it, so an `unsafe` block there would prove nothing. See
    /// [`mapfile::Mapped`], which holds the one `unsafe` block, and
    /// ADR 0004.
    ///
    /// # Errors
    ///
    /// Returns an [`ErrorKind::Io`] when the file cannot be opened or mapped,
    /// a parse [`Error`] when the header is invalid, and
    /// [`ErrorKind::UnsupportedTensorTypes`] when the file contains a tensor
    /// type Scribe cannot load yet. That error lists every such type, in the
    /// order the tensors appear.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with(path, MapOptions::default())
    }

    /// [`open`](Self::open) with [`MapOptions`].
    ///
    /// The file contract of [`open`](Self::open) applies.
    pub fn open_with(path: impl AsRef<Path>, options: MapOptions) -> Result<Self> {
        let path = path.as_ref();
        let map = Mapped::open(path, options.populate).map_err(|err| io_err(path, err))?;
        let parsed = Gguf::parse(&map)?;
        let unsupported = parsed.unsupported_types();
        if !unsupported.is_empty() {
            return Err(Error {
                offset: 0,
                kind: ErrorKind::UnsupportedTensorTypes { types: unsupported },
            });
        }
        let tensors = tensor_index(&parsed);
        Ok(GgufFile { map, tensors })
    }

    /// The mapped bytes.
    ///
    /// [`Gguf::parse`](crate::Gguf::parse) of this slice borrows `self`. Parse
    /// once and keep the [`Gguf`](crate::Gguf) when metadata is needed more
    /// than once. [`parsed`](Self::parsed) walks the header on every call.
    pub fn bytes(&self) -> &[u8] {
        &self.map
    }

    /// Parses the header from the mapping.
    ///
    /// Each call walks the header again. Tensor lookup does not. The borrows
    /// in the returned value are tied to `self`, and the walk does not copy
    /// tensor bytes. After [`open`](Self::open) those pages are already resident.
    pub fn parsed(&self) -> Result<Gguf<'_>> {
        Gguf::parse(&self.map)
    }

    /// Bytes of the named tensor. The slice points into the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let loc = self.loc(name)?;
        self.bytes_at(loc, name)
    }

    /// Blocks of the named tensor.
    ///
    /// Fails when the tensor is missing, when its ggml type is not `B`, or
    /// when its byte length is not a multiple of the block size.
    pub fn tensor_blocks<B: kernels::Block>(&self, name: &str) -> Result<&[B]> {
        let loc = self.loc(name)?;
        let bytes = self.bytes_at(loc, name)?;
        crate::view_blocks(name, loc.ggml_type, bytes)
    }

    /// Bytes of `output.weight`, or of `token_embd.weight` when the file has
    /// no `output.weight`.
    ///
    /// The fallback is the tied-embedding case (Qwen3-0.6B). The returned
    /// slice is the tensor's own bytes, not a copy.
    pub fn output_weight(&self) -> Result<&[u8]> {
        let name = if self.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.tensor_bytes(name)
    }

    fn loc(&self, name: &str) -> Result<TensorLoc> {
        self.tensors.get(name).copied().ok_or_else(|| Error {
            offset: 0,
            kind: ErrorKind::UnknownTensor {
                name: name.to_owned(),
            },
        })
    }

    fn bytes_at(&self, loc: TensorLoc, name: &str) -> Result<&[u8]> {
        crate::slice_range(&self.map, loc.start, loc.len).ok_or_else(|| Error {
            offset: loc.start,
            kind: ErrorKind::TensorDataOutOfBounds {
                name: name.to_owned(),
            },
        })
    }

    /// `madvise(MADV_WILLNEED)` over the whole mapping.
    ///
    /// This asks the kernel to read the file in. It is not used by [`open`](Self::open).
    pub fn advise_willneed(&self) -> Result<()> {
        self.map
            .advise_willneed()
            .map_err(|err| io_msg("madvise(WILLNEED)", err))
    }

    /// `madvise(MADV_POPULATE_READ)` over the whole mapping.
    ///
    /// This prefaults every page. Which targets honour it is in
    /// [`mapfile::Mapped::advise_populate_read`].
    pub fn advise_populate_read(&self) -> Result<()> {
        self.map
            .advise_populate_read()
            .map_err(|err| io_msg("madvise(POPULATE_READ)", err))
    }

    /// `madvise(MADV_HUGEPAGE)` over the whole mapping.
    ///
    /// A kernel that rejects the advice is ignored. Huge pages are
    /// best-effort. Which targets honour it is in
    /// [`mapfile::Mapped::advise_hugepage`].
    pub fn advise_hugepage(&self) {
        self.map.advise_hugepage();
    }

    /// `mlock` the whole mapping.
    ///
    /// Fails when the process is not allowed to lock that many bytes.
    pub fn lock(&self) -> Result<()> {
        self.map.lock().map_err(|err| io_msg("mlock", err))
    }
}

#[expect(
    clippy::expect_used,
    reason = "parse already rejected an offset that overflows the file"
)]
fn tensor_index(parsed: &Gguf<'_>) -> HashMap<String, TensorLoc> {
    let base = parsed.tensor_data_offset();
    parsed
        .tensors()
        .iter()
        .map(|tensor| {
            // `parse` already rejected an offset that overflows the file.
            let start = base
                .checked_add(tensor.offset)
                .expect("in-range tensor offset");
            (
                tensor.name.to_owned(),
                TensorLoc {
                    ggml_type: tensor.ggml_type,
                    start,
                    len: tensor.size,
                },
            )
        })
        .collect()
}

fn io_err(path: &Path, err: std::io::Error) -> Error {
    io_msg(&format!("opening {}", path.display()), err)
}

fn io_msg(what: &str, err: std::io::Error) -> Error {
    Error {
        offset: 0,
        kind: ErrorKind::Io {
            message: format!("{what}: {err}"),
        },
    }
}
