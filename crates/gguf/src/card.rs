use std::fmt;

use crate::{Array, GgmlType, Gguf, Value, ValueType};

const HEAD: usize = 4;
const WITNESS_MAGIC: &[u8; 4] = b"SWR1";
const MIN_META_RECORD: usize = 8 + 4 + 1;
const MIN_TENSOR_RECORD: usize = 8 + 8 + 4 + 8;

/// JSON header card for `model`.
///
/// The text is not the `scribe info` command.
pub fn to_json(model: &Gguf<'_>) -> String {
    write_json(&project(parts_from_gguf(model)))
}

/// JSON header card for an `SWR1` witness.
///
/// The same writer as [`to_json`]. A short or malformed buffer is an error.
pub fn json_from_witness(bytes: &[u8]) -> Result<String, WitnessError> {
    Ok(write_json(&project(parts_from_witness(bytes)?)))
}

/// A witness blob [`json_from_witness`] rejected.
#[derive(Debug)]
pub struct WitnessError {
    reason: &'static str,
}

impl WitnessError {
    fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl fmt::Display for WitnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid witness: {}", self.reason)
    }
}

impl std::error::Error for WitnessError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarKind {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Scalar {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32Bits(u32),
    F64Bits(u64),
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug)]
enum MetaBody {
    Scalar(Scalar),
    Array {
        elem: ScalarKind,
        len: u64,
        head: Vec<Scalar>,
        digest: u64,
    },
}

#[derive(Clone, Debug)]
struct Meta {
    key: String,
    body: MetaBody,
}

#[derive(Clone, Debug)]
struct TensorCard {
    name: String,
    dims: Vec<u64>,
    ggml_type: GgmlType,
    offset: u64,
}

#[derive(Clone, Debug)]
struct Card {
    version: u32,
    alignment: u32,
    tensor_data_offset: u64,
    metadata: Vec<Meta>,
    tensors: Vec<TensorCard>,
}

fn head_len(len: u64) -> usize {
    usize::try_from(len).unwrap_or(HEAD).min(HEAD)
}

fn project(mut card: Card) -> Card {
    for meta in &mut card.metadata {
        if let MetaBody::Array {
            elem, len, head, ..
        } = &mut meta.body
        {
            let keep = head_len(*len);
            head.truncate(keep);
            assert_eq!(head.len(), keep, "array head shorter than the card allows");
            assert!(
                head.iter().all(|scalar| scalar_kind(scalar) == *elem),
                "array head element kind"
            );
        }
    }
    assert!(card.tensors.iter().all(|tensor| tensor.dims.len() <= 4));
    card
}

fn parts_from_gguf(model: &Gguf<'_>) -> Card {
    let metadata = model
        .kvs()
        .iter()
        .map(|(key, value)| Meta {
            key: (*key).to_owned(),
            body: match *value {
                Value::Array(array) => MetaBody::Array {
                    elem: kind_of(array.elem_type()),
                    len: array.len(),
                    head: array_head(array),
                    digest: array_digest(array),
                },
                other => MetaBody::Scalar(scalar_from_value(other)),
            },
        })
        .collect();
    let tensors = model
        .tensors()
        .iter()
        .map(|tensor| TensorCard {
            name: tensor.name.to_owned(),
            dims: tensor.dims().to_vec(),
            ggml_type: tensor.ggml_type,
            offset: tensor.offset,
        })
        .collect();
    Card {
        version: model.version(),
        alignment: model.alignment(),
        tensor_data_offset: model.tensor_data_offset(),
        metadata,
        tensors,
    }
}

fn scalar_from_value(value: Value<'_>) -> Scalar {
    match value {
        Value::U8(v) => Scalar::U8(v),
        Value::I8(v) => Scalar::I8(v),
        Value::U16(v) => Scalar::U16(v),
        Value::I16(v) => Scalar::I16(v),
        Value::U32(v) => Scalar::U32(v),
        Value::I32(v) => Scalar::I32(v),
        Value::U64(v) => Scalar::U64(v),
        Value::I64(v) => Scalar::I64(v),
        Value::F32(v) => Scalar::F32Bits(v.to_bits()),
        Value::F64(v) => Scalar::F64Bits(v.to_bits()),
        Value::Bool(v) => Scalar::Bool(v),
        Value::String(v) => Scalar::String(v.to_owned()),
        Value::Array(_) => unreachable!("nested arrays cannot be stored"),
    }
}

fn array_head(array: Array<'_>) -> Vec<Scalar> {
    let n = head_len(array.len());
    match array.elem_type() {
        ValueType::U8 => array.u8s().unwrap().take(n).map(Scalar::U8).collect(),
        ValueType::I8 => array.i8s().unwrap().take(n).map(Scalar::I8).collect(),
        ValueType::U16 => array.u16s().unwrap().take(n).map(Scalar::U16).collect(),
        ValueType::I16 => array.i16s().unwrap().take(n).map(Scalar::I16).collect(),
        ValueType::U32 => array.u32s().unwrap().take(n).map(Scalar::U32).collect(),
        ValueType::I32 => array.i32s().unwrap().take(n).map(Scalar::I32).collect(),
        ValueType::U64 => array.u64s().unwrap().take(n).map(Scalar::U64).collect(),
        ValueType::I64 => array.i64s().unwrap().take(n).map(Scalar::I64).collect(),
        ValueType::F32 => array
            .f32s()
            .unwrap()
            .take(n)
            .map(|v| Scalar::F32Bits(v.to_bits()))
            .collect(),
        ValueType::F64 => array
            .f64s()
            .unwrap()
            .take(n)
            .map(|v| Scalar::F64Bits(v.to_bits()))
            .collect(),
        ValueType::Bool => array.bools().unwrap().take(n).map(Scalar::Bool).collect(),
        ValueType::String => array
            .strings()
            .unwrap()
            .take(n)
            .map(|s| Scalar::String(s.to_owned()))
            .collect(),
        ValueType::Array => unreachable!("nested arrays cannot be stored"),
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(FNV_OFFSET)
    }

    fn push(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn array_digest(array: Array<'_>) -> u64 {
    let mut fnv = Fnv::new();
    match array.elem_type() {
        ValueType::U8 => {
            for value in array.u8s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::I8 => {
            for value in array.i8s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::U16 => {
            for value in array.u16s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::I16 => {
            for value in array.i16s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::U32 => {
            for value in array.u32s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::I32 => {
            for value in array.i32s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::U64 => {
            for value in array.u64s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::I64 => {
            for value in array.i64s().unwrap() {
                fnv.push(&value.to_le_bytes());
            }
        }
        ValueType::F32 => {
            for value in array.f32s().unwrap() {
                fnv.push(&value.to_bits().to_le_bytes());
            }
        }
        ValueType::F64 => {
            for value in array.f64s().unwrap() {
                fnv.push(&value.to_bits().to_le_bytes());
            }
        }
        ValueType::Bool => {
            for value in array.bools().unwrap() {
                fnv.push(&[u8::from(value)]);
            }
        }
        ValueType::String => {
            for value in array.strings().unwrap() {
                fnv.push(&(value.len() as u64).to_le_bytes());
                fnv.push(value.as_bytes());
            }
        }
        ValueType::Array => unreachable!("nested arrays cannot be stored"),
    }
    fnv.finish()
}

fn kind_of(ty: ValueType) -> ScalarKind {
    match ty {
        ValueType::U8 => ScalarKind::U8,
        ValueType::I8 => ScalarKind::I8,
        ValueType::U16 => ScalarKind::U16,
        ValueType::I16 => ScalarKind::I16,
        ValueType::U32 => ScalarKind::U32,
        ValueType::I32 => ScalarKind::I32,
        ValueType::U64 => ScalarKind::U64,
        ValueType::I64 => ScalarKind::I64,
        ValueType::F32 => ScalarKind::F32,
        ValueType::F64 => ScalarKind::F64,
        ValueType::Bool => ScalarKind::Bool,
        ValueType::String => ScalarKind::String,
        ValueType::Array => unreachable!("nested arrays cannot be stored"),
    }
}

fn scalar_kind(scalar: &Scalar) -> ScalarKind {
    match scalar {
        Scalar::U8(_) => ScalarKind::U8,
        Scalar::I8(_) => ScalarKind::I8,
        Scalar::U16(_) => ScalarKind::U16,
        Scalar::I16(_) => ScalarKind::I16,
        Scalar::U32(_) => ScalarKind::U32,
        Scalar::I32(_) => ScalarKind::I32,
        Scalar::U64(_) => ScalarKind::U64,
        Scalar::I64(_) => ScalarKind::I64,
        Scalar::F32Bits(_) => ScalarKind::F32,
        Scalar::F64Bits(_) => ScalarKind::F64,
        Scalar::Bool(_) => ScalarKind::Bool,
        Scalar::String(_) => ScalarKind::String,
    }
}

fn kind_ty(kind: ScalarKind) -> ValueType {
    match kind {
        ScalarKind::U8 => ValueType::U8,
        ScalarKind::I8 => ValueType::I8,
        ScalarKind::U16 => ValueType::U16,
        ScalarKind::I16 => ValueType::I16,
        ScalarKind::U32 => ValueType::U32,
        ScalarKind::I32 => ValueType::I32,
        ScalarKind::U64 => ValueType::U64,
        ScalarKind::I64 => ValueType::I64,
        ScalarKind::F32 => ValueType::F32,
        ScalarKind::F64 => ValueType::F64,
        ScalarKind::Bool => ValueType::Bool,
        ScalarKind::String => ValueType::String,
    }
}

fn scalar_ty(scalar: &Scalar) -> ValueType {
    kind_ty(scalar_kind(scalar))
}

fn write_json(card: &Card) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    write_key(&mut out, 1, "version");
    push_dec(&mut out, card.version);
    end_field(&mut out, true);
    write_key(&mut out, 1, "alignment");
    push_dec(&mut out, card.alignment);
    end_field(&mut out, true);
    write_key(&mut out, 1, "tensor_data_offset");
    push_quoted_dec(&mut out, card.tensor_data_offset);
    end_field(&mut out, true);
    write_key(&mut out, 1, "metadata");
    write_array(&mut out, 1, card.metadata.len(), |out, i| {
        write_meta(out, &card.metadata[i]);
    });
    end_field(&mut out, true);
    write_key(&mut out, 1, "tensors");
    write_array(&mut out, 1, card.tensors.len(), |out, i| {
        write_tensor(out, &card.tensors[i]);
    });
    end_field(&mut out, false);
    out.push_str("}\n");
    out
}

fn write_meta(out: &mut String, meta: &Meta) {
    out.push_str("{\n");
    write_key(out, 3, "key");
    push_json_string(out, &meta.key);
    end_field(out, true);
    match &meta.body {
        MetaBody::Scalar(scalar) => {
            write_key(out, 3, "type");
            push_json_string(out, scalar_ty(scalar).name());
            end_field(out, true);
            write_key(out, 3, "value");
            write_scalar(out, scalar);
            end_field(out, false);
        }
        MetaBody::Array {
            elem,
            len,
            head,
            digest,
        } => {
            write_key(out, 3, "type");
            push_json_string(out, ValueType::Array.name());
            end_field(out, true);
            write_key(out, 3, "elem");
            push_json_string(out, kind_ty(*elem).name());
            end_field(out, true);
            write_key(out, 3, "len");
            push_quoted_dec(out, *len);
            end_field(out, true);
            write_key(out, 3, "head");
            write_array(out, 3, head.len(), |out, i| write_scalar(out, &head[i]));
            end_field(out, true);
            write_key(out, 3, "digest");
            push_json_string(out, &format!("{digest:016x}"));
            end_field(out, false);
        }
    }
    indent(out, 2);
    out.push('}');
}

fn write_tensor(out: &mut String, tensor: &TensorCard) {
    out.push_str("{\n");
    write_key(out, 3, "name");
    push_json_string(out, &tensor.name);
    end_field(out, true);
    write_key(out, 3, "dims");
    write_array(out, 3, tensor.dims.len(), |out, i| {
        push_quoted_dec(out, tensor.dims[i]);
    });
    end_field(out, true);
    write_key(out, 3, "type");
    push_json_string(out, tensor.ggml_type.name());
    end_field(out, true);
    write_key(out, 3, "offset");
    push_quoted_dec(out, tensor.offset);
    end_field(out, false);
    indent(out, 2);
    out.push('}');
}

fn write_array(
    out: &mut String,
    level: usize,
    len: usize,
    mut item: impl FnMut(&mut String, usize),
) {
    if len == 0 {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for i in 0..len {
        indent(out, level + 1);
        item(out, i);
        if i + 1 != len {
            out.push(',');
        }
        out.push('\n');
    }
    indent(out, level);
    out.push(']');
}

fn write_scalar(out: &mut String, scalar: &Scalar) {
    match *scalar {
        Scalar::U8(v) => push_dec(out, v),
        Scalar::I8(v) => push_dec(out, v),
        Scalar::U16(v) => push_dec(out, v),
        Scalar::I16(v) => push_dec(out, v),
        Scalar::U32(v) => push_dec(out, v),
        Scalar::I32(v) => push_dec(out, v),
        Scalar::U64(v) => push_quoted_dec(out, v),
        Scalar::I64(v) => push_quoted_dec(out, v),
        Scalar::F32Bits(bits) => push_json_string(out, &format!("{bits:08x}")),
        Scalar::F64Bits(bits) => push_json_string(out, &format!("{bits:016x}")),
        Scalar::Bool(true) => out.push_str("true"),
        Scalar::Bool(false) => out.push_str("false"),
        Scalar::String(ref text) => push_json_string(out, text),
    }
}

fn push_dec(out: &mut String, value: impl fmt::Display) {
    use std::fmt::Write;
    write!(out, "{value}").unwrap();
}

fn push_quoted_dec(out: &mut String, value: impl fmt::Display) {
    out.push('"');
    push_dec(out, value);
    out.push('"');
}

fn push_json_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if u32::from(ch) < 0x20 => {
                use std::fmt::Write;
                write!(out, "\\u00{:02x}", u32::from(ch)).unwrap();
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

fn write_key(out: &mut String, level: usize, key: &str) {
    indent(out, level);
    push_json_string(out, key);
    out.push_str(": ");
}

fn end_field(out: &mut String, comma: bool) {
    if comma {
        out.push(',');
    }
    out.push('\n');
}

fn indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

struct WitnessCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> WitnessCursor<'a> {
    fn rest(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WitnessError> {
        if self.rest() < n {
            return Err(WitnessError::new("truncated"));
        }
        let bytes = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], WitnessError> {
        let bytes = self.take(N)?;
        Ok(bytes.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32, WitnessError> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Result<u64, WitnessError> {
        self.array().map(u64::from_le_bytes)
    }

    fn fits(&self, count: u64, min_each: usize) -> Result<(), WitnessError> {
        let max = (self.rest() / min_each) as u64;
        if count > max {
            Err(WitnessError::new("truncated"))
        } else {
            Ok(())
        }
    }

    fn utf8(&mut self) -> Result<String, WitnessError> {
        let len = self.u64()?;
        let n = usize::try_from(len).map_err(|_| WitnessError::new("truncated"))?;
        if n > self.rest() {
            return Err(WitnessError::new("truncated"));
        }
        let bytes = self.take(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| WitnessError::new("invalid utf-8"))
    }
}

fn parts_from_witness(bytes: &[u8]) -> Result<Card, WitnessError> {
    let mut cur = WitnessCursor {
        data: bytes,
        pos: 0,
    };
    let magic = cur.take(4)?;
    if magic != WITNESS_MAGIC {
        return Err(WitnessError::new("bad magic"));
    }
    let version =
        u32::try_from(cur.u64()?).map_err(|_| WitnessError::new("version does not fit in u32"))?;
    let alignment = u32::try_from(cur.u64()?)
        .map_err(|_| WitnessError::new("alignment does not fit in u32"))?;
    let tensor_data_offset = cur.u64()?;
    let meta_count = cur.u64()?;
    cur.fits(meta_count, MIN_META_RECORD)?;
    let mut metadata = Vec::new();
    for _ in 0..meta_count {
        let key = cur.utf8()?;
        let type_id = cur.u32()?;
        metadata.push(Meta {
            key,
            body: read_meta_body(&mut cur, type_id)?,
        });
    }
    let tensor_count = cur.u64()?;
    cur.fits(tensor_count, MIN_TENSOR_RECORD)?;
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        tensors.push(read_tensor(&mut cur)?);
    }
    if cur.pos != cur.data.len() {
        return Err(WitnessError::new("trailing bytes"));
    }
    Ok(Card {
        version,
        alignment,
        tensor_data_offset,
        metadata,
        tensors,
    })
}

fn read_meta_body(cur: &mut WitnessCursor<'_>, type_id: u32) -> Result<MetaBody, WitnessError> {
    let ty = ValueType::from_id(type_id).ok_or(WitnessError::new("unknown value type"))?;
    if ty == ValueType::Array {
        read_array(cur)
    } else {
        Ok(MetaBody::Scalar(read_scalar(cur, ty)?))
    }
}

fn read_array(cur: &mut WitnessCursor<'_>) -> Result<MetaBody, WitnessError> {
    let elem_id = cur.u32()?;
    let elem_ty = ValueType::from_id(elem_id).ok_or(WitnessError::new("unknown value type"))?;
    if elem_ty == ValueType::Array {
        return Err(WitnessError::new("nested array"));
    }
    let len = cur.u64()?;
    let n_sent = cur.u64()?;
    if n_sent != len {
        return Err(WitnessError::new("array length mismatch"));
    }
    let min_each = elem_ty.fixed_size().unwrap_or(8);
    cur.fits(n_sent, min_each)?;
    let keep = head_len(len);
    let mut head = Vec::with_capacity(keep);
    for i in 0..n_sent {
        let scalar = read_scalar(cur, elem_ty)?;
        if i < keep as u64 {
            head.push(scalar);
        }
    }
    let digest = cur.u64()?;
    Ok(MetaBody::Array {
        elem: kind_of(elem_ty),
        len,
        head,
        digest,
    })
}

fn read_scalar(cur: &mut WitnessCursor<'_>, ty: ValueType) -> Result<Scalar, WitnessError> {
    Ok(match ty {
        ValueType::U8 => Scalar::U8(u8::from_le_bytes(cur.array()?)),
        ValueType::I8 => Scalar::I8(i8::from_le_bytes(cur.array()?)),
        ValueType::U16 => Scalar::U16(u16::from_le_bytes(cur.array()?)),
        ValueType::I16 => Scalar::I16(i16::from_le_bytes(cur.array()?)),
        ValueType::U32 => Scalar::U32(cur.u32()?),
        ValueType::I32 => Scalar::I32(i32::from_le_bytes(cur.array()?)),
        ValueType::F32 => Scalar::F32Bits(cur.u32()?),
        ValueType::Bool => {
            let [byte] = cur.array()?;
            if byte > 1 {
                return Err(WitnessError::new("bool is not 0 or 1"));
            }
            Scalar::Bool(byte == 1)
        }
        ValueType::String => Scalar::String(cur.utf8()?),
        ValueType::U64 => Scalar::U64(cur.u64()?),
        ValueType::I64 => Scalar::I64(i64::from_le_bytes(cur.array()?)),
        ValueType::F64 => Scalar::F64Bits(cur.u64()?),
        ValueType::Array => return Err(WitnessError::new("nested array")),
    })
}

fn read_tensor(cur: &mut WitnessCursor<'_>) -> Result<TensorCard, WitnessError> {
    let name = cur.utf8()?;
    let n_dims = cur.u64()?;
    if n_dims > 4 {
        return Err(WitnessError::new("too many dimensions"));
    }
    cur.fits(n_dims, 8)?;
    let mut dims = Vec::with_capacity(usize::try_from(n_dims).unwrap());
    for _ in 0..n_dims {
        dims.push(cur.u64()?);
    }
    let type_id = cur.u32()?;
    let ggml_type = GgmlType::from_id(type_id).ok_or(WitnessError::new("unknown ggml type"))?;
    let offset = cur.u64()?;
    Ok(TensorCard {
        name,
        dims,
        ggml_type,
        offset,
    })
}
