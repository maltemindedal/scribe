//! Checks against the real Qwen3-0.6B-Q8_0 file. Ignored until the model
//! downloader (#5) puts it in `models/`; run with `cargo test -- --ignored`.

use std::path::PathBuf;

use gguf::{GgmlType, Gguf, GgufFile};

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

#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf (see #5)"]
fn tensor_bytes_over_every_qwen3_tensor_is_a_lookup() {
    let path = model_path("Qwen3-0.6B-Q8_0.gguf");
    let file =
        unsafe { GgufFile::open(&path) }.unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let model = Gguf::parse(file.bytes()).unwrap();
    let names: Vec<(String, u64)> = model
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.to_owned(), tensor.size))
        .collect();
    drop(model);
    let once = std::time::Instant::now();
    let again = file.parsed().unwrap();
    let parsed_once = once.elapsed();
    assert_eq!(again.tensors().len(), names.len());
    drop(again);
    println!("parsed_once_us={}", parsed_once.as_micros());
    assert_eq!(names.len(), 310);
    let expect: u64 = names.iter().map(|(_, size)| size).sum();

    let started = std::time::Instant::now();
    let mut got = 0u64;
    for (name, _) in &names {
        let slice = file.tensor_bytes(name).unwrap();
        got += slice.len() as u64;
        std::hint::black_box(slice.as_ptr());
    }
    let elapsed = started.elapsed();
    println!(
        "tensor_bytes_all count={} bytes={got} elapsed_us={}",
        names.len(),
        elapsed.as_micros()
    );

    assert_eq!(
        got, expect,
        "tensor_bytes lengths must match the tensor table"
    );
    assert_eq!(
        file.tensor_bytes("blk.0.attn_q.weight").unwrap().len(),
        2_228_224
    );
    assert!(
        elapsed.as_millis() < 500,
        "tensor_bytes over 310 tensors took {elapsed:?}; a header walk per tensor is about 2s in release and about 6s in debug"
    );
}
