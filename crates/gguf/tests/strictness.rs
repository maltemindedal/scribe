//! Hand-built tiny GGUF files, one per strictness rule. Each test starts from
//! a file that parses and changes exactly one thing, so it proves the rule
//! is what rejects it.

use gguf::{ErrorKind, GgmlType, Gguf, GgufFile, MapOptions, Value, ValueType};
use kernels::{Block, BlockQ4_0, BlockQ8_0, BlockQ8K};

/// Minimal GGUF writer with hooks for producing invalid files.
#[derive(Clone)]
struct Builder {
    version: u32,
    kvs: Vec<Vec<u8>>,
    tensors: Vec<Tensor>,
    alignment: u64,
    /// Overrides the header counts when set.
    n_kv: Option<u64>,
    n_tensors: Option<u64>,
}

#[derive(Clone)]
struct Tensor {
    name: Vec<u8>,
    dims: Vec<u64>,
    type_id: u32,
    /// `None` = the correct packed offset.
    offset: Option<u64>,
    data_len: u64,
}

fn s(bytes: &[u8]) -> Vec<u8> {
    let mut out = (bytes.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(bytes);
    out
}

fn kv(key: &str, type_id: u32, value: &[u8]) -> Vec<u8> {
    let mut out = s(key.as_bytes());
    out.extend_from_slice(&type_id.to_le_bytes());
    out.extend_from_slice(value);
    out
}

fn array(elem: u32, len: u64, data: &[u8]) -> Vec<u8> {
    let mut out = elem.to_le_bytes().to_vec();
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(data);
    out
}

impl Builder {
    fn new() -> Self {
        Builder {
            version: 3,
            kvs: vec![],
            tensors: vec![],
            alignment: 32,
            n_kv: None,
            n_tensors: None,
        }
    }

    /// A small valid file exercising every value type and two tensors.
    fn valid() -> Self {
        let mut b = Self::new();
        b.kvs.push(kv("general.architecture", 8, &s(b"test")));
        b.kvs.push(kv("u8", 0, &[7]));
        b.kvs.push(kv("i8", 1, &[0xff]));
        b.kvs.push(kv("u16", 2, &500u16.to_le_bytes()));
        b.kvs.push(kv("i16", 3, &(-500i16).to_le_bytes()));
        b.kvs.push(kv("u32", 4, &70_000u32.to_le_bytes()));
        b.kvs.push(kv("i32", 5, &(-70_000i32).to_le_bytes()));
        b.kvs.push(kv("f32", 6, &1.5f32.to_le_bytes()));
        b.kvs.push(kv("bool", 7, &[1]));
        b.kvs.push(kv("u64", 10, &(1u64 << 40).to_le_bytes()));
        b.kvs.push(kv("i64", 11, &(-1i64).to_le_bytes()));
        b.kvs.push(kv("f64", 12, &2.25f64.to_le_bytes()));
        let mut strings = s(b"hello");
        strings.extend(s("wörld".as_bytes()));
        strings.extend(s(b""));
        b.kvs.push(kv("tokens", 9, &array(8, 3, &strings)));
        let ints: Vec<u8> = [1i32, -2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        b.kvs.push(kv("ints", 9, &array(5, 3, &ints)));
        b.kvs.push(kv("empty", 9, &array(6, 0, &[])));
        // 64 x 2 Q8_0 = 4 blocks of 34 bytes = 136 bytes, padded to 160.
        b.tensors.push(Tensor::new("blk.0.w", &[64, 2], 8));
        // 3 f32 = 12 bytes, ends the file.
        b.tensors.push(Tensor::new("norm", &[3], 0));
        b
    }

    fn build(&self) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend_from_slice(&self.version.to_le_bytes());
        let n_tensors = self.n_tensors.unwrap_or(self.tensors.len() as u64);
        out.extend_from_slice(&n_tensors.to_le_bytes());
        out.extend_from_slice(&self.n_kv.unwrap_or(self.kvs.len() as u64).to_le_bytes());
        for kv in &self.kvs {
            out.extend_from_slice(kv);
        }
        let mut offset = 0u64;
        for t in &self.tensors {
            out.extend(s(&t.name));
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.type_id.to_le_bytes());
            out.extend_from_slice(&t.offset.unwrap_or(offset).to_le_bytes());
            offset += t.data_len.next_multiple_of(self.alignment);
        }
        if !self.tensors.is_empty() {
            out.resize(out.len().next_multiple_of(self.alignment as usize), 0);
        }
        let data_start = out.len();
        for (i, t) in self.tensors.iter().enumerate() {
            out.extend((0..t.data_len).map(|j| (i as u64 + j) as u8));
            if i + 1 < self.tensors.len() {
                let pad = out.len() - data_start;
                out.resize(
                    data_start + pad.next_multiple_of(self.alignment as usize),
                    0,
                );
            }
        }
        out
    }

    fn kind(&self) -> ErrorKind {
        Gguf::parse(&self.build())
            .expect_err("file should be rejected")
            .kind
    }
}

impl Tensor {
    fn new(name: &str, dims: &[u64], type_id: u32) -> Self {
        let ty = GgmlType::from_id(type_id);
        let n: u64 = dims.iter().product();
        let data_len = ty.map_or(0, |t| n / t.block_size() * t.type_size());
        Tensor {
            name: name.as_bytes().to_vec(),
            dims: dims.to_vec(),
            type_id,
            offset: None,
            data_len,
        }
    }
}

#[test]
fn valid_file_parses() {
    let bytes = Builder::valid().build();
    let g = Gguf::parse(&bytes).unwrap();

    assert_eq!(g.version(), 3);
    assert_eq!(g.kvs().len(), 15);
    assert_eq!(g.alignment(), 32);
    assert_eq!(g.get("general.architecture"), Some(Value::String("test")));
    assert_eq!(g.get("u8"), Some(Value::U8(7)));
    assert_eq!(g.get("i8"), Some(Value::I8(-1)));
    assert_eq!(g.get("u16"), Some(Value::U16(500)));
    assert_eq!(g.get("i16"), Some(Value::I16(-500)));
    assert_eq!(g.get("u32").and_then(|v| v.as_u64()), Some(70_000));
    assert_eq!(g.get("i32"), Some(Value::I32(-70_000)));
    assert_eq!(g.get("f32").and_then(|v| v.as_f32()), Some(1.5));
    assert_eq!(g.get("bool").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(g.get("u64"), Some(Value::U64(1 << 40)));
    assert_eq!(g.get("i64"), Some(Value::I64(-1)));
    assert_eq!(g.get("f64"), Some(Value::F64(2.25)));
    assert_eq!(g.get("missing"), None);

    let tokens = g.get("tokens").unwrap().as_array().unwrap();
    assert_eq!(tokens.elem_type(), ValueType::String);
    assert_eq!(
        tokens.strings().unwrap().collect::<Vec<_>>(),
        ["hello", "wörld", ""]
    );
    assert!(
        tokens.i32s().is_none(),
        "typed iterators check the element type"
    );
    let ints = g.get("ints").unwrap().as_array().unwrap();
    assert_eq!(ints.i32s().unwrap().collect::<Vec<_>>(), [1, -2, 3]);
    assert!(g.get("empty").unwrap().as_array().unwrap().is_empty());

    let [w, norm] = g.tensors() else { panic!() };
    assert_eq!(
        (w.name, w.dims(), w.ggml_type),
        ("blk.0.w", &[64, 2][..], GgmlType::Q8_0)
    );
    assert_eq!((w.offset, w.size, w.n_elements()), (0, 136, 128));
    assert_eq!(
        (norm.name, norm.dims(), norm.ggml_type),
        ("norm", &[3][..], GgmlType::F32)
    );
    assert_eq!((norm.offset, norm.size), (160, 12));
    assert_eq!(g.tensor_data_offset() % 32, 0);
    assert_eq!(g.tensor_data_offset() + 160 + 12, bytes.len() as u64);
}

#[test]
fn empty_file_with_no_tensors_parses() {
    let bytes = Builder::new().build();
    let g = Gguf::parse(&bytes).unwrap();
    assert!(g.kvs().is_empty() && g.tensors().is_empty());
}

#[test]
fn version_2_parses() {
    let mut b = Builder::valid();
    b.version = 2;
    assert_eq!(Gguf::parse(&b.build()).unwrap().version(), 2);
}

#[test]
fn every_truncation_is_an_error_not_a_panic() {
    let bytes = Builder::valid().build();
    for len in 0..bytes.len() {
        assert!(
            Gguf::parse(&bytes[..len]).is_err(),
            "prefix of {len} bytes parsed"
        );
    }
}

#[test]
fn truncation_error_names_the_field() {
    let bytes = Builder::valid().build();
    let err = Gguf::parse(&bytes[..10]).unwrap_err();
    assert_eq!(err.offset, 8);
    assert!(matches!(
        err.kind,
        ErrorKind::Truncated {
            field: "header.tensor_count",
            ..
        }
    ));
    assert!(err.to_string().contains("header.tensor_count"), "{err}");
}

#[test]
fn bad_magic() {
    let mut bytes = Builder::valid().build();
    bytes[0] = b'X';
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err().kind,
        ErrorKind::BadMagic(*b"XGUF")
    );
}

#[test]
fn big_endian_file_is_detected() {
    let mut bytes = Builder::valid().build();
    bytes[4..8].copy_from_slice(&3u32.to_be_bytes());
    assert_eq!(Gguf::parse(&bytes).unwrap_err().kind, ErrorKind::BigEndian);
}

#[test]
fn unsupported_versions() {
    for v in [0, 1, 4] {
        let mut b = Builder::valid();
        b.version = v;
        assert_eq!(b.kind(), ErrorKind::UnsupportedVersion(v));
    }
}

#[test]
fn alignment_must_be_a_u32_power_of_two() {
    for bad in [0u32, 3, 48] {
        let mut b = Builder::valid();
        b.kvs.push(kv("general.alignment", 4, &bad.to_le_bytes()));
        assert_eq!(b.kind(), ErrorKind::BadAlignment(bad));
    }
    let mut b = Builder::valid();
    b.kvs
        .push(kv("general.alignment", 10, &64u64.to_le_bytes()));
    assert_eq!(b.kind(), ErrorKind::AlignmentWrongType);
}

#[test]
fn custom_alignment_is_honoured() {
    let mut b = Builder::valid();
    b.alignment = 64;
    b.kvs.push(kv("general.alignment", 4, &64u32.to_le_bytes()));
    let bytes = b.build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.alignment(), 64);
    assert_eq!(g.tensor_data_offset() % 64, 0);
    assert_eq!(g.tensors()[1].offset, 192);
}

#[test]
fn tensor_name_must_be_shorter_than_64_bytes() {
    let mut b = Builder::valid();
    b.tensors[1].name = vec![b'n'; 63];
    assert!(Gguf::parse(&b.build()).is_ok());
    b.tensors[1].name = vec![b'n'; 64];
    assert!(matches!(b.kind(), ErrorKind::TensorNameTooLong { .. }));
}

#[test]
fn duplicate_key() {
    let mut b = Builder::valid();
    b.kvs.push(kv("u8", 0, &[1]));
    assert_eq!(b.kind(), ErrorKind::DuplicateKey("u8".into()));
}

#[test]
fn empty_key() {
    let mut b = Builder::valid();
    b.kvs.push(kv("", 0, &[1]));
    assert_eq!(b.kind(), ErrorKind::EmptyKey);
}

#[test]
fn duplicate_tensor_name() {
    let mut b = Builder::valid();
    b.tensors[1].name = b"blk.0.w".to_vec();
    assert_eq!(b.kind(), ErrorKind::DuplicateTensorName("blk.0.w".into()));
}

#[test]
fn gap_between_tensors() {
    let mut b = Builder::valid();
    b.tensors[1].offset = Some(192);
    assert_eq!(
        b.kind(),
        ErrorKind::TensorOffsetMismatch {
            name: "norm".into(),
            offset: 192,
            expected: 160
        }
    );
}

#[test]
fn unpadded_offset_is_a_gap_too() {
    let mut b = Builder::valid();
    b.tensors[1].offset = Some(136);
    assert!(matches!(
        b.kind(),
        ErrorKind::TensorOffsetMismatch { expected: 160, .. }
    ));
}

#[test]
fn tensors_out_of_info_order() {
    let mut b = Builder::valid();
    b.tensors[0].offset = Some(32);
    assert!(matches!(
        b.kind(),
        ErrorKind::TensorOffsetMismatch {
            offset: 32,
            expected: 0,
            ..
        }
    ));
}

#[test]
fn tensor_data_past_end_of_file() {
    let mut b = Builder::valid();
    let mut bytes = b.build();
    bytes.pop();
    assert_eq!(
        Gguf::parse(&bytes).unwrap_err().kind,
        ErrorKind::TensorDataOutOfBounds {
            name: "norm".into()
        }
    );
    // A shape claiming far more data than the file holds.
    b.tensors[1].dims = vec![1 << 40];
    assert_eq!(
        b.kind(),
        ErrorKind::TensorDataOutOfBounds {
            name: "norm".into()
        }
    );
}

#[test]
fn nested_array() {
    let mut b = Builder::valid();
    b.kvs
        .push(kv("nested", 9, &array(9, 1, &array(0, 1, &[1]))));
    assert_eq!(
        b.kind(),
        ErrorKind::NestedArray {
            key: "nested".into()
        }
    );
}

#[test]
fn invalid_value_type() {
    let mut b = Builder::valid();
    b.kvs.push(kv("weird", 13, &[0]));
    assert_eq!(
        b.kind(),
        ErrorKind::InvalidValueType {
            key: "weird".into(),
            type_id: 13
        }
    );
    let mut b = Builder::valid();
    b.kvs.push(kv("weird", 9, &array(99, 0, &[])));
    assert_eq!(
        b.kind(),
        ErrorKind::InvalidValueType {
            key: "weird".into(),
            type_id: 99
        }
    );
}

#[test]
fn huge_counts_fail_before_allocating() {
    let mut b = Builder::valid();
    b.n_kv = Some(u64::MAX);
    assert!(matches!(
        b.kind(),
        ErrorKind::CountTooLarge {
            field: "header.kv_count",
            ..
        }
    ));

    let mut b = Builder::valid();
    b.n_tensors = Some(u64::MAX / 2);
    assert!(matches!(
        b.kind(),
        ErrorKind::CountTooLarge {
            field: "header.tensor_count",
            ..
        }
    ));

    let mut b = Builder::valid();
    b.kvs.push(kv("big", 9, &array(4, 1 << 31, &[])));
    assert!(matches!(b.kind(), ErrorKind::ArrayTooLong { len, .. } if len == 1 << 31));

    // Within the 2^30 cap but far more strings than bytes remain.
    let mut b = Builder::valid();
    b.kvs.push(kv("big", 9, &array(8, 1 << 30, &[])));
    assert!(matches!(
        b.kind(),
        ErrorKind::CountTooLarge {
            field: "kv.array_len",
            ..
        }
    ));

    let mut b = Builder::valid();
    b.kvs.push(kv("big", 9, &array(10, 1 << 30, &[])));
    assert!(matches!(
        b.kind(),
        ErrorKind::Truncated {
            field: "kv.array_data",
            ..
        }
    ));

    let mut b = Builder::valid();
    let mut v = ((1u64 << 30) + 1).to_le_bytes().to_vec();
    v.push(b'x');
    b.kvs.push(kv("big", 8, &v));
    assert!(matches!(b.kind(), ErrorKind::StringTooLong { .. }));
}

#[test]
fn invalid_utf8() {
    let mut b = Builder::valid();
    b.kvs.push(kv("bad", 8, &s(&[b'a', 0xff])));
    assert_eq!(b.kind(), ErrorKind::InvalidUtf8 { field: "kv.value" });

    let mut b = Builder::valid();
    b.tensors[1].name = vec![0xc3];
    assert_eq!(
        b.kind(),
        ErrorKind::InvalidUtf8 {
            field: "tensor.name"
        }
    );
}

#[test]
fn bool_must_be_zero_or_one() {
    let mut b = Builder::valid();
    b.kvs.push(kv("b", 7, &[2]));
    assert_eq!(
        b.kind(),
        ErrorKind::InvalidBool {
            key: "b".into(),
            byte: 2
        }
    );
    let mut b = Builder::valid();
    b.kvs.push(kv("bs", 9, &array(7, 3, &[0, 1, 5])));
    assert_eq!(
        b.kind(),
        ErrorKind::InvalidBool {
            key: "bs".into(),
            byte: 5
        }
    );
}

#[test]
fn too_many_dims() {
    let mut b = Builder::valid();
    b.tensors[1].dims = vec![1, 1, 1, 1, 3];
    assert!(matches!(b.kind(), ErrorKind::TooManyDims { n_dims: 5, .. }));
}

#[test]
fn dimension_must_fit_in_i64() {
    let mut b = Builder::valid();
    b.tensors[1].dims = vec![1 << 63];
    assert!(matches!(b.kind(), ErrorKind::DimOutOfRange { .. }));
}

#[test]
fn shape_overflow() {
    let mut b = Builder::valid();
    b.tensors[1].dims = vec![1 << 32, 1 << 32];
    assert!(matches!(b.kind(), ErrorKind::ShapeOverflow { .. }));
}

#[test]
fn unknown_ggml_types_are_errors() {
    for id in [4, 31, 43, u32::MAX] {
        let mut b = Builder::valid();
        b.tensors[1].type_id = id;
        assert!(
            matches!(b.kind(), ErrorKind::UnknownGgmlType { type_id, .. } if type_id == id),
            "id {id}"
        );
    }
}

#[test]
fn newest_ggml_types_are_known() {
    for (id, ty) in [
        (40, GgmlType::Nvfp4),
        (41, GgmlType::Q1_0),
        (42, GgmlType::Q2_0),
    ] {
        let mut b = Builder::valid();
        b.tensors[1] = Tensor::new("norm", &[128], id);
        let bytes = b.build();
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.tensors()[1].ggml_type, ty);
    }
}

#[test]
fn row_must_be_a_whole_number_of_blocks() {
    let mut b = Builder::valid();
    b.tensors[0].dims = vec![33, 2];
    assert_eq!(
        b.kind(),
        ErrorKind::RowNotBlockAligned {
            name: "blk.0.w".into(),
            ne0: 33,
            block_size: 32
        }
    );
}

#[test]
fn zero_sized_tensor_is_allowed() {
    let mut b = Builder::valid();
    b.tensors.insert(1, Tensor::new("empty", &[0, 4], 0));
    let bytes = b.build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!((g.tensors()[1].size, g.tensors()[2].offset), (0, 160));
}

#[test]
fn header_truncations_report_truncated_or_count() {
    let bytes = Builder::valid().build();
    let data_start = Gguf::parse(&bytes).unwrap().tensor_data_offset() as usize;
    for len in 0..data_start {
        let kind = Gguf::parse(&bytes[..len]).unwrap_err().kind;
        assert!(
            matches!(
                kind,
                ErrorKind::Truncated { .. } | ErrorKind::CountTooLarge { .. }
            ),
            "prefix of {len} bytes gave {kind:?}"
        );
    }
}

#[test]
fn error_offsets_point_at_the_offending_field() {
    let offset = |b: &Builder| Gguf::parse(&b.build()).unwrap_err().offset;

    let mut b = Builder::valid();
    b.n_tensors = Some(u64::MAX / 2);
    assert_eq!(offset(&b), 8);
    b.n_tensors = None;
    b.n_kv = Some(u64::MAX);
    assert_eq!(offset(&b), 16);

    // general.alignment is the first pair, right after the 24-byte header.
    let mut b = Builder::valid();
    b.kvs
        .insert(0, kv("general.alignment", 4, &3u32.to_le_bytes()));
    assert_eq!(offset(&b), 24);

    // The second tensor's offset field is the last 8 bytes of the info table.
    let mut b = Builder::valid();
    let info_end = 24
        + b.kvs.iter().map(Vec::len).sum::<usize>()
        + b.tensors
            .iter()
            .map(|t| 8 + t.name.len() + 4 + 8 * t.dims.len() + 4 + 8)
            .sum::<usize>();
    b.tensors[1].offset = Some(192);
    assert_eq!(offset(&b), info_end as u64 - 8);
    let b = Builder::valid();
    let mut bytes = b.build();
    bytes.pop();
    assert_eq!(Gguf::parse(&bytes).unwrap_err().offset, info_end as u64 - 8);
}

#[test]
fn tensor_bytes_are_that_tensors_span() {
    let bytes = Builder::valid().build();
    let g = Gguf::parse(&bytes).unwrap();
    let w = g.tensor_bytes(&bytes, "blk.0.w").unwrap();
    let norm = g.tensor_bytes(&bytes, "norm").unwrap();
    assert_eq!(w.len(), 136);
    assert_eq!(w[0], 0);
    assert_eq!(w[135], 135);
    assert_eq!(norm.len(), 12);
    assert_ne!(w.as_ptr(), norm.as_ptr());
    assert!(matches!(
        g.tensor_bytes(&bytes, "missing").unwrap_err().kind,
        ErrorKind::UnknownTensor { name } if name == "missing"
    ));
    assert!(matches!(
        g.tensor_bytes(&bytes[..10], "blk.0.w").unwrap_err().kind,
        ErrorKind::TensorDataOutOfBounds { name } if name == "blk.0.w"
    ));
}

#[test]
fn tensor_blocks_rejects_a_type_mismatch() {
    let bytes = Builder::valid().build();
    let g = Gguf::parse(&bytes).unwrap();
    let blocks = g.tensor_blocks::<BlockQ8_0>(&bytes, "blk.0.w").unwrap();
    assert_eq!(blocks.len(), 4);
    assert_eq!(blocks[0].d, [0, 1]);
    assert_eq!(
        blocks.as_ptr() as *const u8,
        g.tensor_bytes(&bytes, "blk.0.w").unwrap().as_ptr()
    );
    assert_eq!(
        g.tensor_blocks::<BlockQ8_0>(&bytes, "norm")
            .unwrap_err()
            .kind,
        ErrorKind::TensorTypeMismatch {
            name: "norm".into(),
            expected: GgmlType::Q8_0,
            actual: GgmlType::F32,
        }
    );
    assert_eq!(
        g.tensor_blocks::<BlockQ4_0>(&bytes, "blk.0.w")
            .unwrap_err()
            .kind,
        ErrorKind::TensorTypeMismatch {
            name: "blk.0.w".into(),
            expected: GgmlType::Q4_0,
            actual: GgmlType::Q8_0,
        }
    );
}

#[test]
fn output_weight_falls_back_to_token_embd_without_copying() {
    let mut tied = Builder::new();
    tied.tensors
        .push(Tensor::new("token_embd.weight", &[32], 8));
    let bytes = tied.build();
    let g = Gguf::parse(&bytes).unwrap();
    let emb = g.tensor_bytes(&bytes, "token_embd.weight").unwrap();
    let out = g.output_weight(&bytes).unwrap();
    assert!(std::ptr::eq(emb.as_ptr(), out.as_ptr()));
    assert_eq!(emb.len(), out.len());

    let mut untied = Builder::new();
    untied.tensors.push(Tensor::new("output.weight", &[32], 8));
    untied
        .tensors
        .push(Tensor::new("token_embd.weight", &[32], 8));
    let bytes = untied.build();
    let g = Gguf::parse(&bytes).unwrap();
    let out = g.output_weight(&bytes).unwrap();
    let emb = g.tensor_bytes(&bytes, "token_embd.weight").unwrap();
    assert_eq!(out[0], 0);
    assert_eq!(emb[0], 1);
    assert!(!std::ptr::eq(out.as_ptr(), emb.as_ptr()));

    let bytes = Builder::valid().build();
    let g = Gguf::parse(&bytes).unwrap();
    assert!(matches!(
        g.output_weight(&bytes).unwrap_err().kind,
        ErrorKind::UnknownTensor { name } if name == "token_embd.weight"
    ));
}

#[test]
fn unsupported_types_lists_every_kind_and_open_rejects_them() {
    let mut b = Builder::new();
    b.tensors.push(Tensor::new("a", &[256], 16));
    b.tensors.push(Tensor::new("b", &[32], 39));
    b.tensors.push(Tensor::new("c", &[256], 16));
    let bytes = b.build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(
        g.unsupported_types(),
        vec![GgmlType::Iq2Xxs, GgmlType::Mxfp4]
    );
    let path = std::env::temp_dir().join(format!("scribe-gguf-bad-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let err = unsafe { GgufFile::open(&path) }.unwrap_err();
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        err.kind,
        ErrorKind::UnsupportedTensorTypes {
            types: vec![GgmlType::Iq2Xxs, GgmlType::Mxfp4],
        }
    );
    assert!(err.to_string().contains("iq2_xxs"));
    assert!(err.to_string().contains("mxfp4"));
}

#[test]
fn open_maps_a_supported_file_and_views_share_the_mapping() {
    let bytes = Builder::valid().build();
    let path = std::env::temp_dir().join(format!("scribe-gguf-ok-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let file = unsafe { GgufFile::open(&path) }.unwrap();
    let file = file;
    assert!(file.parsed().unwrap().unsupported_types().is_empty());
    #[cfg(target_os = "linux")]
    {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let needle = path.file_name().unwrap().to_str().unwrap();
        let mapped = smaps
            .lines()
            .find(|line| line.contains(needle))
            .unwrap_or_else(|| panic!("mapping for {needle} missing from smaps"));
        assert_eq!(mapped.split_whitespace().nth(1), Some("r--p"), "{mapped}");
    }
    let w = file.tensor_bytes("blk.0.w").unwrap();
    let blocks = file.tensor_blocks::<BlockQ8_0>("blk.0.w").unwrap();
    assert_eq!(blocks.len(), 4);
    assert_eq!(blocks.as_ptr() as *const u8, w.as_ptr());
    assert_eq!(
        file.tensor_blocks::<BlockQ4_0>("blk.0.w").unwrap_err().kind,
        ErrorKind::TensorTypeMismatch {
            name: "blk.0.w".into(),
            expected: GgmlType::Q4_0,
            actual: GgmlType::Q8_0,
        }
    );
    file.advise_willneed().unwrap();
    file.advise_hugepage();
    assert!(file.advise_populate_read().is_ok());
    let populated = unsafe { GgufFile::open_with(&path, MapOptions { populate: true }) }.unwrap();
    assert_eq!(populated.tensor_bytes("norm").unwrap().len(), 12);
    drop(file);
    drop(populated);
    std::fs::remove_file(&path).unwrap();
    let missing = unsafe { GgufFile::open(path).unwrap_err() };
    assert!(matches!(missing.kind, ErrorKind::Io { .. }));
}

#[test]
fn an_unknown_block_id_is_an_error() {
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct NotAGgmlBlock(u8);
    // SAFETY: one byte, alignment 1, every pattern valid. The id is not a ggml type.
    unsafe impl Block for NotAGgmlBlock {
        const ID: u32 = 999;
    }
    let bytes = Builder::valid().build();
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(
        g.tensor_blocks::<NotAGgmlBlock>(&bytes, "blk.0.w")
            .unwrap_err()
            .kind,
        ErrorKind::UnknownBlockId { id: 999 }
    );
}

#[test]
fn block_ids_and_sizes_match_the_ggml_table() {
    use kernels::{
        BlockQ2K, BlockQ3K, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    };
    let pairs = [
        (BlockQ4_0::ID, BlockQ4_0::BYTES, GgmlType::Q4_0),
        (BlockQ4_1::ID, BlockQ4_1::BYTES, GgmlType::Q4_1),
        (BlockQ5_0::ID, BlockQ5_0::BYTES, GgmlType::Q5_0),
        (BlockQ5_1::ID, BlockQ5_1::BYTES, GgmlType::Q5_1),
        (BlockQ8_0::ID, BlockQ8_0::BYTES, GgmlType::Q8_0),
        (BlockQ2K::ID, BlockQ2K::BYTES, GgmlType::Q2K),
        (BlockQ3K::ID, BlockQ3K::BYTES, GgmlType::Q3K),
        (BlockQ4K::ID, BlockQ4K::BYTES, GgmlType::Q4K),
        (BlockQ5K::ID, BlockQ5K::BYTES, GgmlType::Q5K),
        (BlockQ6K::ID, BlockQ6K::BYTES, GgmlType::Q6K),
        (BlockQ8K::ID, BlockQ8K::BYTES, GgmlType::Q8K),
    ];
    for (id, bytes, ty) in pairs {
        assert_eq!(id, ty.id(), "{ty}");
        assert_eq!(bytes as u64, ty.type_size(), "{ty}");
    }
}
