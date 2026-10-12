//! Download-and-verify logic for `cargo xtask models fetch`.

use std::fs::{self, File};
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::manifest::Model;
use crate::sha256;

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The file was already there and valid; nothing was downloaded.
    Present,
    Downloaded,
}

/// Fetches `url` into `dest`. Abstracted so tests need no network.
pub type Downloader<'a> = &'a dyn Fn(&str, &Path) -> Result<(), String>;

/// Downloads with `curl`, which follows redirects and fails on HTTP errors.
///
/// Every failure is retried, not only the ones curl deems transient; a retry
/// truncates the output file first, and `ensure` verifies the result anyway.
/// A progress bar is shown when stderr is a terminal, and nothing but errors
/// otherwise, so CI logs stay short. Needs curl 7.71 or later.
pub fn curl(url: &str, dest: &Path) -> Result<(), String> {
    let progress = if std::io::stderr().is_terminal() {
        "--progress-bar"
    } else {
        "--silent"
    };
    let status = Command::new("curl")
        .args([
            "--fail",
            progress,
            "--show-error",
            "--location",
            "--connect-timeout",
            "30",
            "--retry",
            "3",
            "--retry-all-errors",
            "--output",
        ])
        .arg(dest)
        .arg(url)
        .status()
        .map_err(|e| format!("cannot run `curl` (is it installed?): {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("curl failed with {status} for {url}"))
    }
}

/// Checks the size, then the SHA-256, of `path` against the manifest entry.
pub fn verify(path: &Path, model: &Model) -> Result<(), String> {
    let size = fs::metadata(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .len();
    if size != model.size {
        return Err(format!(
            "size mismatch for {}: expected {} bytes, found {size}",
            path.display(),
            model.size
        ));
    }
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let actual =
        sha256::hex_digest(file).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if actual != model.sha256 {
        return Err(format!(
            "checksum mismatch for {}: expected sha256 {}, found {actual}",
            path.display(),
            model.sha256
        ));
    }
    Ok(())
}

/// Makes sure `dir/<dest>` exists and matches the manifest.
///
/// A valid file is left alone. An invalid existing file is never overwritten:
/// it fails with a message telling the user to delete it. A fresh download
/// goes to `<dest>.part` and is verified before it is moved into place; if the
/// download fails or does not verify, the `.part` file is removed, so nothing
/// is left behind under either name.
pub fn ensure(dir: &Path, model: &Model, download: Downloader) -> Result<Outcome, String> {
    let target = dir.join(&model.dest);
    if target.exists() {
        verify(&target, model).map_err(|e| {
            format!("{e}\nThe file is corrupt or from another revision; delete it and re-run.")
        })?;
        return Ok(Outcome::Present);
    }
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut part: PathBuf = target.clone().into_os_string().into();
    part.as_mut_os_string().push(".part");
    let _ = fs::remove_file(&part);
    let fetched = download(&model.url(), &part).and_then(|()| {
        verify(&part, model)
            .map_err(|e| format!("download of {} failed verification: {e}", model.name))
    });
    if let Err(e) = fetched {
        let _ = fs::remove_file(&part);
        return Err(e);
    }
    fs::rename(&part, &target)
        .map_err(|e| format!("cannot move {} into place: {e}", part.display()))?;
    Ok(Outcome::Downloaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Set;
    use std::cell::Cell;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("xtask-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn model() -> Model {
        Model {
            name: "abc".into(),
            repo: "o/r".into(),
            file: "abc.bin".into(),
            dest: "abc.bin".into(),
            revision: "0".repeat(40),
            // sha256("abc")
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            size: 3,
            license: "MIT".into(),
            role: "test".into(),
            set: Set::Ci,
        }
    }

    #[test]
    fn downloads_then_is_a_no_op() {
        let dir = scratch("noop");
        let calls = Cell::new(0);
        let good = |_: &str, p: &Path| {
            calls.set(calls.get() + 1);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(p, b"abc").map_err(|e| e.to_string())
        };
        assert_eq!(ensure(&dir, &model(), &good), Ok(Outcome::Downloaded));
        assert_eq!(ensure(&dir, &model(), &good), Ok(Outcome::Present));
        assert_eq!(calls.get(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupted_download_fails_and_leaves_nothing() {
        let dir = scratch("bad-download");
        // Right size, wrong content: only the checksum can catch it.
        let bad = |_: &str, p: &Path| fs::write(p, b"abd").map_err(|e| e.to_string());
        let err = ensure(&dir, &model(), &bad).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        assert!(!dir.join("abc.bin").exists());
        assert!(!dir.join("abc.bin.part").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn failed_download_propagates_and_leaves_nothing() {
        let dir = scratch("failed-download");
        // A transfer that dies halfway: some bytes written, then an error.
        let failing = |_: &str, p: &Path| -> Result<(), String> {
            fs::write(p, b"a").unwrap();
            Err("connection reset".into())
        };
        assert_eq!(
            ensure(&dir, &model(), &failing),
            Err("connection reset".into())
        );
        assert!(!dir.join("abc.bin").exists());
        assert!(!dir.join("abc.bin.part").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn corrupted_existing_file_fails_with_a_clear_message() {
        let dir = scratch("bad-existing");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("abc.bin"), b"abd").unwrap();
        let never = |_: &str, _: &Path| -> Result<(), String> { panic!("must not download") };
        let err = ensure(&dir, &model(), &never).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        assert!(err.contains("delete it"), "{err}");
        assert_eq!(fs::read(dir.join("abc.bin")).unwrap(), b"abd");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn truncated_file_reports_size() {
        let dir = scratch("truncated");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("abc.bin"), b"ab").unwrap();
        let never = |_: &str, _: &Path| -> Result<(), String> { panic!("must not download") };
        let err = ensure(&dir, &model(), &never).unwrap_err();
        assert!(err.contains("size mismatch"), "{err}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
