//! Checks against the real Qwen3-0.6B-Q8_0 file. Ignored until the model
//! downloader (#5) puts it in `models/`; run with `cargo test -- --ignored`.

use std::path::PathBuf;

use gguf::{GgmlType, Gguf};

fn model_path(file: &str) -> PathBuf {
    let dir = std::env::var_os("SCRIBE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    dir.join(file)
}

#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf (see #5)"]
fn parses_qwen3_0_6b_q8_0() {
    let path = model_path("Qwen3-0.6B-Q8_0.gguf");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.version(), 3);
    assert_eq!(g.tensors().len(), 310);
    assert_eq!(g.kvs().len(), 28);
    assert_eq!(
        g.get("general.architecture").unwrap().as_str(),
        Some("qwen3")
    );

    let tokens = g.get("tokenizer.ggml.tokens").unwrap().as_array().unwrap();
    assert_eq!(tokens.len(), 151_936);
    assert_eq!(tokens.strings().unwrap().count(), 151_936);

    let dims = |name: &str| g.tensor(name).unwrap().dims().to_vec();
    assert_eq!(dims("blk.0.attn_q.weight"), [1024, 2048]);
    assert_eq!(dims("blk.0.attn_k.weight"), [1024, 1024]);
    assert_eq!(dims("blk.0.attn_v.weight"), [1024, 1024]);
    assert_eq!(dims("blk.0.attn_output.weight"), [2048, 1024]);
    assert_eq!(dims("blk.0.attn_q_norm.weight"), [128]);
    assert_eq!(dims("blk.0.attn_k_norm.weight"), [128]);
    assert_eq!(
        g.tensor("blk.0.attn_q.weight").unwrap().ggml_type,
        GgmlType::Q8_0
    );

    // The last tensor ends exactly at the end of the file.
    let last = g.tensors().last().unwrap();
    assert_eq!(
        g.tensor_data_offset() + last.offset + last.size,
        bytes.len() as u64
    );
}
