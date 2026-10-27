//! Resident set of `GgufFile::open` on the real Qwen3-0.6B-Q8_0 file.
//!
//! Ignored until the model is on disk. Run with `cargo test -p gguf -- --ignored`.

use std::path::PathBuf;
use std::process::Command;

use gguf::{GgufFile, MapOptions};

fn model_path(file: &str) -> PathBuf {
    let dir = std::env::var_os("SCRIBE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    dir.join(file)
}

fn proc_kib(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(key) {
            return parts.next()?.parse().ok();
        }
    }
    None
}

fn vmrss_kib() -> u64 {
    if let Some(kib) = proc_kib("VmRSS:") {
        return kib;
    }
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap_or_else(|err| panic!("ps: {err}"));
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim()
        .parse()
        .unwrap_or_else(|_| panic!("ps rss {:?} status {}", text.trim(), out.status))
}

fn show_kib(kib: Option<u64>) -> String {
    kib.map(|n| n.to_string())
        .unwrap_or_else(|| "na".to_owned())
}

#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf (see #5)"]
fn open_qwen3_does_not_resident_the_file() {
    let path = model_path("Qwen3-0.6B-Q8_0.gguf");
    let before = vmrss_kib();
    let rssfile_before = proc_kib("RssFile:");
    let file_len = std::fs::metadata(&path)
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
        .len();

    // verify.sh sets this so the same assertion fails when open prefaults.
    // On Linux and Android that is MAP_POPULATE inside open_with. The flag is
    // zero on other Unix targets, so those fault the mapping open
    // returned.
    let eager = std::env::var_os("SCRIBE_EAGER_READ").is_some();
    let file = unsafe { GgufFile::open_with(&path, MapOptions { populate: eager }) }.unwrap();
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    if eager {
        std::hint::black_box(fault_mapped_pages(file.bytes()));
    }
    let after_open = vmrss_kib();
    let rssfile_open = proc_kib("RssFile:");

    assert!(file.parsed().unwrap().tensor("output.weight").is_none());
    let emb = file.tensor_bytes("token_embd.weight").unwrap();
    let out = file.output_weight().unwrap();
    assert!(std::ptr::eq(emb.as_ptr(), out.as_ptr()));
    assert_eq!(emb.len(), out.len());

    let norm = file.tensor_bytes("blk.0.attn_q_norm.weight").unwrap();
    let sum: u64 = norm.iter().map(|byte| u64::from(*byte)).sum();
    std::hint::black_box(sum);

    let after_touch = vmrss_kib();
    let rssfile_touch = proc_kib("RssFile:");
    let header = file.parsed().unwrap().tensor_data_offset();
    let file_kib = file_len.div_ceil(1024);
    let header_kib = header.div_ceil(1024);
    let limit_kib = header_kib + 32 * 1024;

    println!(
        "rss vmrss_before_kib={before} vmrss_after_open_kib={after_open} vmrss_after_touch_kib={after_touch} rssfile_before_kib={} rssfile_after_open_kib={} rssfile_after_touch_kib={} file_kib={file_kib} header_kib={header_kib}",
        show_kib(rssfile_before),
        show_kib(rssfile_open),
        show_kib(rssfile_touch),
    );
    #[cfg(target_os = "linux")]
    {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mapped = smaps
            .lines()
            .find(|line| line.contains("Qwen3-0.6B-Q8_0.gguf"))
            .unwrap_or_else(|| panic!("model mapping missing from smaps"));
        let perms = mapped.split_whitespace().nth(1).unwrap();
        assert_eq!(perms, "r--p", "{mapped}");
    }

    let grew = after_open.saturating_sub(before);
    assert!(
        grew < limit_kib,
        "open grew VmRSS by {grew} KiB, limit {limit_kib} KiB (header {header_kib} KiB plus 32 MiB), file {file_kib} KiB"
    );
    let grew_touch = after_touch.saturating_sub(before);
    assert!(
        grew_touch < limit_kib,
        "touching blk.0.attn_q_norm.weight grew VmRSS by {grew_touch} KiB, limit {limit_kib} KiB, file {file_kib} KiB"
    );
    if let (Some(start), Some(open)) = (rssfile_before, rssfile_open) {
        let file_grew = open.saturating_sub(start);
        assert!(
            file_grew < limit_kib,
            "open grew RssFile by {file_grew} KiB, limit {limit_kib} KiB, file {file_kib} KiB"
        );
    }
}

/// Touch one byte per page of the mapping `open` returned.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn fault_mapped_pages(bytes: &[u8]) -> u64 {
    let mut n = 0u64;
    let mut i = 0usize;
    while i < bytes.len() {
        n = n.wrapping_add(u64::from(bytes[i]));
        i = i.saturating_add(4096);
    }
    n
}
