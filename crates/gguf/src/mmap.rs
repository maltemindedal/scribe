//! Read-only private mapping of a GGUF file.
//!
//! `unsafe` is allowed in this module only. The rest of the crate denies it.
#![allow(unsafe_code)]

use std::fs::File;
use std::mem::transmute;
use std::path::Path;

use memmap2::{Advice, Mmap, MmapOptions};

use crate::{Error, ErrorKind, Gguf, Result};

/// Options for [`GgufFile::open_with`].
///
/// The default maps the file without prefaulting it. Parsing the header
/// faults only the header pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapOptions {
    /// Pass `MAP_POPULATE` on Linux.
    ///
    /// This prefaults the whole file. The process resident set then grows by
    /// about the file size. The flag is ignored on other operating systems.
    pub populate: bool,
}

/// A GGUF file mapped read-only and private.
///
/// The parsed header borrows the mapping. Moving a `GgufFile` does not move
/// the mapped pages, and the header is dropped before the mapping.
pub struct GgufFile {
    inner: Gguf<'static>,
    map: Mmap,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("len", &self.map.len())
            .field("tensors", &self.inner.tensors().len())
            .finish()
    }
}

impl GgufFile {
    /// Maps `path` read-only and private, then parses the header from the
    /// mapping.
    ///
    /// # Safety
    ///
    /// The file must not be truncated or otherwise shortened for as long as
    /// this value is alive. A fault on a page that no longer exists delivers
    /// `SIGBUS` and kills the process. A write from another process can also
    /// tear a read. `memmap2` documents the same constraint on `Mmap`.
    ///
    /// # Errors
    ///
    /// Returns an [`ErrorKind::Io`] when the file cannot be opened or mapped,
    /// a parse [`Error`] when the header is invalid, and
    /// [`ErrorKind::UnsupportedTensorTypes`] when the file contains a tensor
    /// type Scribe cannot load yet. That error lists every such type, in the
    /// order the tensors appear.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        unsafe { Self::open_with(path, MapOptions::default()) }
    }

    /// [`open`](Self::open) with [`MapOptions`].
    ///
    /// # Safety
    ///
    /// Same as [`open`](Self::open).
    pub unsafe fn open_with(path: impl AsRef<Path>, options: MapOptions) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|err| io_err(path, err))?;
        let mut opts = MmapOptions::new();
        if options.populate {
            opts.populate();
        }
        // `MmapOptions::map` is `MAP_SHARED`. This is `PROT_READ` and `MAP_PRIVATE`.
        // SAFETY: forwarded to the caller of `open_with`. See that Safety section.
        let map = unsafe { opts.map_copy_read_only(&file) }.map_err(|err| io_err(path, err))?;
        let parsed = Gguf::parse(&map)?;
        let unsupported = parsed.unsupported_types();
        if !unsupported.is_empty() {
            return Err(Error {
                offset: 0,
                kind: ErrorKind::UnsupportedTensorTypes { types: unsupported },
            });
        }
        // SAFETY: `parsed` borrows `map`. `inner` is declared first, so it is
        // dropped first. Moving `Mmap` copies the mapping address. It does not
        // relocate the pages.
        let inner = unsafe { transmute::<Gguf<'_>, Gguf<'static>>(parsed) };
        Ok(GgufFile { inner, map })
    }

    /// The parsed header. Keys and string values borrow the mapping.
    pub fn parsed(&self) -> &Gguf<'_> {
        &self.inner
    }

    /// Bytes of the named tensor. The slice points into the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        self.inner.tensor_bytes(&self.map, name)
    }

    /// Blocks of the named tensor.
    ///
    /// Fails when the tensor is missing, when its ggml type is not `B`, or
    /// when its byte length is not a multiple of the block size.
    pub fn tensor_blocks<B: kernels::Block>(&self, name: &str) -> Result<&[B]> {
        self.inner.tensor_blocks(&self.map, name)
    }

    /// Bytes of `output.weight`, or of `token_embd.weight` when the file has
    /// no `output.weight`.
    ///
    /// The fallback is the tied-embedding case (Qwen3-0.6B). The returned
    /// slice is the tensor's own bytes, not a copy.
    pub fn output_weight(&self) -> Result<&[u8]> {
        self.inner.output_weight(&self.map)
    }

    /// `madvise(MADV_WILLNEED)` over the whole mapping.
    ///
    /// This asks the kernel to read the file in. It is not used by [`open`](Self::open).
    pub fn advise_willneed(&self) -> Result<()> {
        self.map
            .advise(Advice::WillNeed)
            .map_err(|err| io_msg("madvise(WILLNEED)", err))
    }

    /// `madvise(MADV_POPULATE_READ)` over the whole mapping, on Linux.
    ///
    /// This prefaults every page. On other operating systems it does nothing
    /// and returns `Ok(())`.
    pub fn advise_populate_read(&self) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.map
                .advise(Advice::PopulateRead)
                .map_err(|err| io_msg("madvise(POPULATE_READ)", err))?;
        }
        Ok(())
    }

    /// `madvise(MADV_HUGEPAGE)` over the whole mapping, on Linux.
    ///
    /// A kernel that rejects the advice is ignored. Other operating systems
    /// do nothing. Huge pages are best-effort.
    pub fn advise_hugepage(&self) {
        #[cfg(target_os = "linux")]
        {
            let _ = self.map.advise(Advice::HugePage);
        }
    }

    /// `mlock` the whole mapping.
    ///
    /// Fails when the process is not allowed to lock that many bytes.
    pub fn lock(&self) -> Result<()> {
        self.map.lock().map_err(|err| io_msg("mlock", err))
    }
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
