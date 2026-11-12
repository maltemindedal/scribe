//! `Mapped::open` against real files in the temp directory.

use std::path::PathBuf;

use mapfile::Mapped;

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("scribe-mapfile-{tag}-{}", std::process::id()))
}

#[test]
fn open_maps_the_bytes_of_the_file() {
    let path = temp_path("bytes");
    let bytes: Vec<u8> = (0..=255).cycle().take(10_000).collect();
    std::fs::write(&path, &bytes).unwrap();
    for populate in [false, true] {
        let mapped = Mapped::open(&path, populate).unwrap();
        assert_eq!(&mapped[..], &bytes[..]);
        mapped.advise_willneed().unwrap();
        mapped.advise_populate_read().unwrap();
        mapped.advise_hugepage();
        assert_eq!(format!("{mapped:?}"), "Mapped { len: 10000 }");
    }
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn open_of_a_missing_file_is_not_found() {
    let err = Mapped::open(&temp_path("missing"), false).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}
