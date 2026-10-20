use std::path::PathBuf;

use gguf::card::{json_from_witness, to_json};
use gguf::{Gguf, GgufFile};

const CI_GGUF: &[&str] = &[
    "stories260K.gguf",
    "stories15M-q8_0.gguf",
    "SmolLM2-135M-Instruct-Q8_0.gguf",
    "Qwen3-0.6B-Q8_0.gguf",
];

const TEXT: &str = "\"\\\nü";

const CARD: &str = r#"{
  "version": 3,
  "alignment": 32,
  "tensor_data_offset": "256",
  "metadata": [
    {
      "key": "general.alignment",
      "type": "u32",
      "value": 32
    },
    {
      "key": "neg0",
      "type": "f32",
      "value": "80000000"
    },
    {
      "key": "nan",
      "type": "f32",
      "value": "7fc00001"
    },
    {
      "key": "big",
      "type": "u64",
      "value": "9007199254740993"
    },
    {
      "key": "neg",
      "type": "i64",
      "value": "-42"
    },
    {
      "key": "text",
      "type": "string",
      "value": "\"\\\nü"
    },
    {
      "key": "arr",
      "type": "array",
      "elem": "u32",
      "len": "6",
      "head": [
        1,
        2,
        3,
        4
      ],
      "digest": "1872e72089559482"
    }
  ],
  "tensors": [
    {
      "name": "w",
      "dims": [
        "1"
      ],
      "type": "f32",
      "offset": "0"
    }
  ]
}
"#;

#[test]
fn synthetic_gguf_matches_card_literal() {
    assert_eq!(
        TEXT.as_bytes(),
        [0x22, 0x5c, 0x0a, 0xc3, 0xbc],
        "fixture string is quote, backslash, newline, and raw UTF-8 ü"
    );
    let bytes = synthetic_gguf();
    let model = Gguf::parse(&bytes).unwrap();
    assert_eq!(model.tensor_data_offset(), 256);
    assert_eq!(to_json(&model), CARD);
}

#[test]
fn committed_fuzz_seed_is_the_synthetic_file() {
    let seed = include_bytes!("../../../fuzz/corpus/parse/minimal");
    assert_eq!(seed, synthetic_gguf().as_slice());
}

#[test]
fn synthetic_witness_matches_same_literal() {
    let json = json_from_witness(&synthetic_witness()).unwrap();
    assert_eq!(json, CARD);
}

#[test]
fn truncated_witness_is_an_error() {
    assert!(json_from_witness(b"").is_err());
    assert!(json_from_witness(b"SWR").is_err());
    assert!(json_from_witness(b"SWR1").is_err());
    assert!(json_from_witness(b"NOPE").is_err());
}

#[test]
fn ci_models_toml_names_the_four_gguf_dests() {
    let text = include_str!("../../xtask/models.toml");
    for dest in [
        "stories260K.gguf",
        "stories15M-q8_0.gguf",
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "Qwen3-0.6B-Q8_0.gguf",
    ] {
        assert!(
            dest_block_is_ci(text, dest),
            "{dest} is missing from a set = \"ci\" block"
        );
    }
}

#[test]
#[ignore = "needs the CI models under models/ or $SCRIBE_MODELS"]
fn card_matches_committed_goldens() {
    let goldens = [
        ("stories260K.gguf", include_str!("golden/stories260K.json")),
        (
            "stories15M-q8_0.gguf",
            include_str!("golden/stories15M-q8_0.json"),
        ),
        (
            "SmolLM2-135M-Instruct-Q8_0.gguf",
            include_str!("golden/SmolLM2-135M-Instruct-Q8_0.json"),
        ),
        (
            "Qwen3-0.6B-Q8_0.gguf",
            include_str!("golden/Qwen3-0.6B-Q8_0.json"),
        ),
    ];
    for (file, golden) in goldens {
        let path = model_path(file);
        let mapped = unsafe { GgufFile::open(&path) }
            .unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        let model =
            Gguf::parse(mapped.bytes()).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
        assert_eq!(to_json(&model), golden, "{file}");
    }
}

#[test]
#[ignore = "needs the CI models under models/ or $SCRIBE_MODELS"]
fn seeded_header_mutations_do_not_panic() {
    for file in CI_GGUF {
        let (header, info_at) = header_and_tensor_info_start(file);
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for it in 0..3000 {
            let mut v = header[..(rnd() as usize % header.len()).max(1)].to_vec();
            if it % 2 == 0 {
                v = header.to_vec();
                let info_len = v.len() - info_at;
                for _ in 0..(1 + rnd() % 8) {
                    let i = if rnd() % 2 == 0 {
                        info_at + (rnd() as usize % info_len)
                    } else {
                        rnd() as usize % v.len().min(4096)
                    };
                    v[i] = rnd() as u8;
                }
            }
            let _ = std::hint::black_box(Gguf::parse(std::hint::black_box(&v)));
        }
    }
}

#[test]
#[ignore = "needs the CI models under models/ or $SCRIBE_MODELS"]
fn stories260k_every_header_prefix_is_an_error() {
    let path = model_path("stories260K.gguf");
    let mapped =
        unsafe { GgufFile::open(&path) }.unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let model =
        Gguf::parse(mapped.bytes()).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let end = usize::try_from(model.tensor_data_offset()).unwrap();
    assert!(
        end <= 64 * 1024,
        "header is {end} bytes, above the 64 KiB ceiling"
    );
    let header = mapped.bytes()[..end].to_vec();
    drop(model);
    drop(mapped);
    for len in 0..=header.len() {
        assert!(Gguf::parse(&header[..len]).is_err(), "prefix {len} parsed");
    }
}

fn model_path(file: &str) -> PathBuf {
    let dir = std::env::var_os("SCRIBE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    dir.join(file)
}

fn header_and_tensor_info_start(file: &str) -> (Vec<u8>, usize) {
    let path = model_path(file);
    let mapped =
        unsafe { GgufFile::open(&path) }.unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let model =
        Gguf::parse(mapped.bytes()).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let end = usize::try_from(model.tensor_data_offset()).unwrap();
    let info_at = tensor_info_start(mapped.bytes(), &model);
    assert!(
        info_at > 4096,
        "{file} tensor info starts at {info_at}, inside the old 4096-byte window"
    );
    assert!(info_at < end, "{file} tensor info is empty");
    let header = mapped.bytes()[..end].to_vec();
    drop(model);
    drop(mapped);
    (header, info_at)
}

fn tensor_info_start(file: &[u8], model: &Gguf) -> usize {
    let mut blob = Vec::new();
    for tensor in model.tensors() {
        push_str(&mut blob, tensor.name.as_bytes());
        push_u32(&mut blob, tensor.dims().len() as u32);
        for dim in tensor.dims() {
            push_u64(&mut blob, *dim);
        }
        push_u32(&mut blob, tensor.ggml_type.id());
        push_u64(&mut blob, tensor.offset);
    }
    let end = usize::try_from(model.tensor_data_offset()).unwrap();
    file[..end]
        .windows(blob.len())
        .rposition(|window| window == blob.as_slice())
        .unwrap_or_else(|| panic!("tensor-info table is not in the header"))
}

fn dest_block_is_ci(text: &str, dest: &str) -> bool {
    let needle = format!("dest = \"{dest}\"");
    text.split("[[model]]")
        .any(|block| block.contains(&needle) && block.contains("set = \"ci\""))
}

fn synthetic_gguf() -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    push_u32(&mut out, 3);
    push_u64(&mut out, 1);
    push_u64(&mut out, 7);
    kv(&mut out, "general.alignment", 4, &32u32.to_le_bytes());
    kv(&mut out, "neg0", 6, &0x8000_0000u32.to_le_bytes());
    kv(&mut out, "nan", 6, &0x7fc0_0001u32.to_le_bytes());
    kv(&mut out, "big", 10, &9_007_199_254_740_993u64.to_le_bytes());
    kv(&mut out, "neg", 11, &(-42i64).to_le_bytes());
    let mut text = Vec::new();
    push_str(&mut text, TEXT.as_bytes());
    kv(&mut out, "text", 8, &text);
    let mut arr = Vec::new();
    push_u32(&mut arr, 4);
    push_u64(&mut arr, 6);
    for n in 1u32..=6 {
        push_u32(&mut arr, n);
    }
    kv(&mut out, "arr", 9, &arr);
    push_str(&mut out, b"w");
    push_u32(&mut out, 1);
    push_u64(&mut out, 1);
    push_u32(&mut out, 0);
    push_u64(&mut out, 0);
    let header = out.len();
    let padded = header.next_multiple_of(32);
    out.resize(padded, 0);
    out.extend_from_slice(&[0, 0, 0, 0]);
    assert_eq!(
        (header, padded),
        (255, 256),
        "one alignment byte sits between the tensor info and the blob"
    );
    out
}

fn synthetic_witness() -> Vec<u8> {
    let mut out = b"SWR1".to_vec();
    push_u64(&mut out, 3);
    push_u64(&mut out, 32);
    push_u64(&mut out, 256);
    push_u64(&mut out, 7);
    witness_meta(&mut out, "general.alignment", 4, &32u32.to_le_bytes());
    witness_meta(&mut out, "neg0", 6, &0x8000_0000u32.to_le_bytes());
    witness_meta(&mut out, "nan", 6, &0x7fc0_0001u32.to_le_bytes());
    witness_meta(&mut out, "big", 10, &9_007_199_254_740_993u64.to_le_bytes());
    witness_meta(&mut out, "neg", 11, &(-42i64).to_le_bytes());
    let mut text = Vec::new();
    push_str(&mut text, TEXT.as_bytes());
    witness_meta(&mut out, "text", 8, &text);
    let mut arr = Vec::new();
    push_u32(&mut arr, 4);
    push_u64(&mut arr, 6);
    push_u64(&mut arr, 6);
    for n in 1u32..=6 {
        push_u32(&mut arr, n);
    }
    push_u64(&mut arr, 0x1872_e720_8955_9482);
    witness_meta(&mut out, "arr", 9, &arr);
    push_u64(&mut out, 1);
    push_str(&mut out, b"w");
    push_u64(&mut out, 1);
    push_u64(&mut out, 1);
    push_u32(&mut out, 0);
    push_u64(&mut out, 0);
    out
}

fn kv(out: &mut Vec<u8>, key: &str, type_id: u32, value: &[u8]) {
    push_str(out, key.as_bytes());
    push_u32(out, type_id);
    out.extend_from_slice(value);
}

fn witness_meta(out: &mut Vec<u8>, key: &str, type_id: u32, payload: &[u8]) {
    push_str(out, key.as_bytes());
    push_u32(out, type_id);
    out.extend_from_slice(payload);
}

fn push_str(out: &mut Vec<u8>, bytes: &[u8]) {
    push_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
