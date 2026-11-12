//! Read-only private mapping of a file.
//!
//! This crate and `kernels` are the only crates allowed `unsafe`. The one
//! `unsafe` block here is the `mmap` call in [`Mapped::open`]. The crate
//! depends on nothing else in the workspace, so it cannot name a type that
//! borrows the mapping, and a crate that can name one cannot write `unsafe`.
//!
//! It builds only for Unix targets. `memmap2` 0.9.11 provides `Advice`,
//! `advise` and `lock` only under `cfg(unix)`.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;

use memmap2::{Advice, Mmap, MmapOptions};

/// A file mapped `PROT_READ` and `MAP_PRIVATE`. Derefs to its bytes.
///
/// # File contract
///
/// The bytes are only sound while nothing truncates or writes the file. A
/// fault on a page cut off by truncation delivers `SIGBUS` and kills the
/// process. A write, by this process or another, changes bytes behind a live
/// `&[u8]`, which is undefined behaviour. `MAP_PRIVATE` with `PROT_READ`
/// does not take a private copy.
///
/// [`open`](Self::open) is a safe function all the same. The obligation is on
/// whoever controls the file system, and no code at a call site can check or
/// discharge it. An `unsafe fn` would make every caller write an `unsafe`
/// block that proves nothing, and would keep every calling crate from
/// forbidding `unsafe`. Don't map a file that something else may rewrite in
/// place. Replace model files by writing a new file and renaming it.
pub struct Mapped(Mmap);

impl Mapped {
    /// Maps `path` read-only and private.
    ///
    /// With `populate`, passes `MAP_POPULATE` on Linux and Android, which
    /// prefaults the whole file. Other Unix targets ignore it.
    ///
    /// The [file contract](Self#file-contract) applies for as long as the
    /// returned value is alive.
    ///
    /// # Errors
    ///
    /// Returns the `io::Error` of the failed `open` or `mmap`.
    pub fn open(path: &Path, populate: bool) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut opts = MmapOptions::new();
        if populate {
            opts.populate();
        }
        // `MmapOptions::map` is `MAP_SHARED`. This is `PROT_READ` and `MAP_PRIVATE`.
        // SAFETY: nothing here can prove that the file stays unchanged. The
        // file contract on `Mapped` passes that requirement to whoever
        // controls the file. Nothing else is required: the mapping is
        // read-only, and `Mapped` owns it for as long as its bytes are borrowed.
        let map = unsafe { opts.map_copy_read_only(&file) }?;
        Ok(Mapped(map))
    }

    /// `madvise(MADV_WILLNEED)` over the whole mapping.
    pub fn advise_willneed(&self) -> io::Result<()> {
        self.0.advise(Advice::WillNeed)
    }

    /// `madvise(MADV_POPULATE_READ)` over the whole mapping, on Linux. Other
    /// Unix targets do nothing and return `Ok(())`.
    pub fn advise_populate_read(&self) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        self.0.advise(Advice::PopulateRead)?;
        Ok(())
    }

    /// `madvise(MADV_HUGEPAGE)` over the whole mapping, on Linux. Best-effort:
    /// a rejection is ignored, and other Unix targets do nothing.
    pub fn advise_hugepage(&self) {
        #[cfg(target_os = "linux")]
        {
            let _ = self.0.advise(Advice::HugePage);
        }
    }

    /// `mlock` the whole mapping.
    pub fn lock(&self) -> io::Result<()> {
        self.0.lock()
    }
}

impl Deref for Mapped {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for Mapped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapped")
            .field("len", &self.0.len())
            .finish()
    }
}
