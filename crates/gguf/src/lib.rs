//! GGUF file parsing, mmap and tensor views.
//!
//! `unsafe` is denied here and allowed only in the mmap module.
//!
//! [`Gguf::parse`] reads a GGUF v2/v3 header, metadata and tensor-info table
//! from any byte slice, treating the input as hostile. It is at least as
//! strict as llama.cpp's `gguf.cpp`: every count is checked against the
//! remaining bytes before anything is allocated, all offset arithmetic is
//! checked, and tensor data must be laid out in info order with no gaps.
#![deny(unsafe_code)]

mod error;
mod ggml_type;
mod value;

use std::collections::HashSet;

pub use error::{Error, ErrorKind, Result};
pub use ggml_type::GgmlType;
pub use value::{Array, Strings, Value, ValueType};

const MAGIC: [u8; 4] = *b"GGUF";
/// llama.cpp's `GGUF_MAX_STRING_LENGTH`.
const MAX_STRING_LEN: u64 = 1 << 30;
/// llama.cpp's `GGUF_MAX_ARRAY_ELEMENTS`.
const MAX_ARRAY_LEN: u64 = 1 << 30;
/// ggml's `GGML_MAX_NAME`: names must be strictly shorter.
const MAX_NAME_LEN: usize = 64;
const MAX_DIMS: u32 = 4;
pub const DEFAULT_ALIGNMENT: u32 = 32;
pub const ALIGNMENT_KEY: &str = "general.alignment";

/// Smallest possible encoded KV pair: key length, a 1-byte key, type, 1-byte value.
const MIN_KV_BYTES: usize = 8 + 1 + 4 + 1;
/// Smallest possible tensor info: name length, n_dims, type, offset.
const MIN_TENSOR_INFO_BYTES: usize = 8 + 4 + 4 + 8;

/// A parsed GGUF file borrowing from its bytes.
#[derive(Debug)]
pub struct Gguf<'a> {
    version: u32,
    kvs: Vec<(&'a str, Value<'a>)>,
    tensors: Vec<TensorInfo<'a>>,
    alignment: u32,
    tensor_data_offset: u64,
}

/// One entry of the tensor-info table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    n_dims: u32,
    ne: [u64; MAX_DIMS as usize],
    pub ggml_type: GgmlType,
    /// Offset of the tensor's data relative to [`Gguf::tensor_data_offset`].
    pub offset: u64,
    /// Size of the tensor's data in bytes.
    pub size: u64,
}

impl TensorInfo<'_> {
    /// Dimensions in ggml order: `dims()[0]` is the contiguous dimension, so a
    /// PyTorch `[out, in]` weight appears as `[in, out]`.
    pub fn dims(&self) -> &[u64] {
        &self.ne[..self.n_dims as usize]
    }

    pub fn n_elements(&self) -> u64 {
        // Checked at parse time not to overflow.
        self.ne.iter().product()
    }
}

impl<'a> Gguf<'a> {
    /// Parses the header, metadata and tensor-info table of a GGUF file, and
    /// checks that every tensor's data lies inside `data`.
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        Parser { data, pos: 0 }.parse()
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// Looks up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<Value<'a>> {
        self.kvs.iter().find(|(k, _)| *k == key).map(|&(_, v)| v)
    }

    /// All metadata pairs in file order.
    pub fn kvs(&self) -> &[(&'a str, Value<'a>)] {
        &self.kvs
    }

    pub fn tensors(&self) -> &[TensorInfo<'a>] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Absolute byte offset where the tensor data blob starts.
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }
}

struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, kind: ErrorKind) -> Error {
        Error {
            offset: self.pos as u64,
            kind,
        }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: u64, field: &'static str) -> Result<&'a [u8]> {
        let available = self.remaining() as u64;
        if n > available {
            return Err(self.err(ErrorKind::Truncated {
                field,
                needed: n - available,
            }));
        }
        let n = n as usize; // n <= remaining, so it fits.
        let bytes = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(bytes)
    }

    fn array<const N: usize>(&mut self, field: &'static str) -> Result<[u8; N]> {
        let bytes = self.take(N as u64, field)?;
        Ok(bytes.try_into().expect("take returned N bytes"))
    }

    fn u32(&mut self, field: &'static str) -> Result<u32> {
        self.array(field).map(u32::from_le_bytes)
    }

    fn u64(&mut self, field: &'static str) -> Result<u64> {
        self.array(field).map(u64::from_le_bytes)
    }

    /// Reads a length-prefixed UTF-8 string.
    fn string(&mut self, field: &'static str) -> Result<&'a str> {
        let start = self.pos;
        let len = self.u64(field)?;
        if len > MAX_STRING_LEN {
            self.pos = start;
            return Err(self.err(ErrorKind::StringTooLong { field, len }));
        }
        let bytes = self.take(len, field)?;
        std::str::from_utf8(bytes).map_err(|e| Error {
            offset: (self.pos - bytes.len() + e.valid_up_to()) as u64,
            kind: ErrorKind::InvalidUtf8 { field },
        })
    }

    /// Fails if `count` items of at least `min_bytes` each cannot fit in the
    /// remaining input. Called before every count-driven allocation.
    fn check_count(&self, count: u64, min_bytes: usize, field: &'static str) -> Result<usize> {
        let max = (self.remaining() / min_bytes) as u64;
        if count > max {
            return Err(self.err(ErrorKind::CountTooLarge { field, count }));
        }
        Ok(count as usize)
    }

    fn parse(mut self) -> Result<Gguf<'a>> {
        let magic: [u8; 4] = self.array("header.magic")?;
        if magic != MAGIC {
            self.pos = 0;
            return Err(self.err(ErrorKind::BadMagic(magic)));
        }
        let version = self.u32("header.version")?;
        if version & 0xFFFF == 0 && version != 0 {
            self.pos -= 4;
            return Err(self.err(ErrorKind::BigEndian));
        }
        if !(2..=3).contains(&version) {
            self.pos -= 4;
            return Err(self.err(ErrorKind::UnsupportedVersion(version)));
        }
        let n_tensors = self.u64("header.tensor_count")?;
        let n_kv = self.u64("header.kv_count")?;

        let kvs = self.parse_kvs(n_kv)?;
        let alignment = match kvs.iter().find(|(k, _)| *k == ALIGNMENT_KEY) {
            None => DEFAULT_ALIGNMENT,
            Some((_, Value::U32(a))) if a.is_power_of_two() => *a,
            Some((_, Value::U32(a))) => return Err(self.err(ErrorKind::BadAlignment(*a))),
            Some(_) => return Err(self.err(ErrorKind::AlignmentWrongType)),
        };

        let tensors = self.parse_tensor_infos(n_tensors)?;

        // The blob starts at the next multiple of the alignment (only when
        // there are tensors, as in llama.cpp).
        let mut tensor_data_offset = self.pos as u64;
        if !tensors.is_empty() {
            tensor_data_offset = tensor_data_offset.next_multiple_of(alignment.into());
            if tensor_data_offset > self.data.len() as u64 {
                return Err(self.err(ErrorKind::Truncated {
                    field: "data.padding",
                    needed: tensor_data_offset - self.data.len() as u64,
                }));
            }
        }

        self.check_tensor_layout(&tensors, alignment, tensor_data_offset)?;

        Ok(Gguf {
            version,
            kvs,
            tensors,
            alignment,
            tensor_data_offset,
        })
    }

    fn parse_kvs(&mut self, n_kv: u64) -> Result<Vec<(&'a str, Value<'a>)>> {
        let n = self.check_count(n_kv, MIN_KV_BYTES, "header.kv_count")?;
        let mut kvs = Vec::with_capacity(n);
        let mut seen = HashSet::with_capacity(n);
        for _ in 0..n {
            let key_pos = self.pos;
            let key = self.string("kv.key")?;
            if key.is_empty() {
                self.pos = key_pos;
                return Err(self.err(ErrorKind::EmptyKey));
            }
            if !seen.insert(key) {
                self.pos = key_pos;
                return Err(self.err(ErrorKind::DuplicateKey(key.to_owned())));
            }
            let ty = self.value_type(key)?;
            let value = if ty == ValueType::Array {
                Value::Array(self.array_value(key)?)
            } else {
                self.scalar(key, ty)?
            };
            kvs.push((key, value));
        }
        Ok(kvs)
    }

    fn value_type(&mut self, key: &str) -> Result<ValueType> {
        let id = self.u32("kv.type")?;
        ValueType::from_id(id).ok_or_else(|| Error {
            offset: self.pos as u64 - 4,
            kind: ErrorKind::InvalidValueType {
                key: key.to_owned(),
                type_id: id,
            },
        })
    }

    fn scalar(&mut self, key: &str, ty: ValueType) -> Result<Value<'a>> {
        use ValueType as T;
        let field = "kv.value";
        Ok(match ty {
            T::U8 => Value::U8(u8::from_le_bytes(self.array(field)?)),
            T::I8 => Value::I8(i8::from_le_bytes(self.array(field)?)),
            T::U16 => Value::U16(u16::from_le_bytes(self.array(field)?)),
            T::I16 => Value::I16(i16::from_le_bytes(self.array(field)?)),
            T::U32 => Value::U32(u32::from_le_bytes(self.array(field)?)),
            T::I32 => Value::I32(i32::from_le_bytes(self.array(field)?)),
            T::F32 => Value::F32(f32::from_le_bytes(self.array(field)?)),
            T::U64 => Value::U64(u64::from_le_bytes(self.array(field)?)),
            T::I64 => Value::I64(i64::from_le_bytes(self.array(field)?)),
            T::F64 => Value::F64(f64::from_le_bytes(self.array(field)?)),
            T::Bool => {
                let [b] = self.array(field)?;
                self.check_bool(key, b, self.pos - 1)?;
                Value::Bool(b == 1)
            }
            T::String => Value::String(self.string(field)?),
            T::Array => unreachable!("arrays are handled by array_value"),
        })
    }

    fn check_bool(&self, key: &str, byte: u8, at: usize) -> Result<()> {
        if byte > 1 {
            return Err(Error {
                offset: at as u64,
                kind: ErrorKind::InvalidBool {
                    key: key.to_owned(),
                    byte,
                },
            });
        }
        Ok(())
    }

    fn array_value(&mut self, key: &str) -> Result<Array<'a>> {
        let elem = self.value_type(key)?;
        if elem == ValueType::Array {
            self.pos -= 4;
            return Err(self.err(ErrorKind::NestedArray {
                key: key.to_owned(),
            }));
        }
        let len = self.u64("kv.array_len")?;
        if len > MAX_ARRAY_LEN {
            self.pos -= 8;
            return Err(self.err(ErrorKind::ArrayTooLong {
                key: key.to_owned(),
                len,
            }));
        }
        let start = self.pos;
        match elem.fixed_size() {
            Some(width) => {
                // len <= 2^30 and width <= 8, so this cannot overflow.
                let bytes = self.take(len * width as u64, "kv.array_data")?;
                if elem == ValueType::Bool {
                    for (i, &b) in bytes.iter().enumerate() {
                        self.check_bool(key, b, start + i)?;
                    }
                }
            }
            None => {
                // Every string needs at least its 8-byte length prefix.
                self.check_count(len, 8, "kv.array_len")?;
                for _ in 0..len {
                    self.string("kv.array_string")?;
                }
            }
        }
        Ok(Array {
            elem,
            len,
            bytes: &self.data[start..self.pos],
        })
    }

    fn parse_tensor_infos(&mut self, n_tensors: u64) -> Result<Vec<TensorInfo<'a>>> {
        let n = self.check_count(n_tensors, MIN_TENSOR_INFO_BYTES, "header.tensor_count")?;
        let mut tensors = Vec::with_capacity(n);
        let mut seen = HashSet::with_capacity(n);
        for _ in 0..n {
            let name_pos = self.pos;
            let name = self.string("tensor.name")?;
            if name.len() >= MAX_NAME_LEN {
                self.pos = name_pos;
                return Err(self.err(ErrorKind::TensorNameTooLong {
                    name: name.to_owned(),
                }));
            }
            if !seen.insert(name) {
                self.pos = name_pos;
                return Err(self.err(ErrorKind::DuplicateTensorName(name.to_owned())));
            }

            let n_dims = self.u32("tensor.n_dims")?;
            if n_dims > MAX_DIMS {
                self.pos -= 4;
                return Err(self.err(ErrorKind::TooManyDims {
                    name: name.to_owned(),
                    n_dims,
                }));
            }
            let mut ne = [1u64; MAX_DIMS as usize];
            for d in ne.iter_mut().take(n_dims as usize) {
                *d = self.u64("tensor.dims")?;
                if *d > i64::MAX as u64 {
                    self.pos -= 8;
                    return Err(self.err(ErrorKind::DimOutOfRange {
                        name: name.to_owned(),
                        dim: *d,
                    }));
                }
            }
            let shape_overflow = || ErrorKind::ShapeOverflow {
                name: name.to_owned(),
            };
            let n_elements = ne
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .filter(|&n| n <= i64::MAX as u64)
                .ok_or_else(|| self.err(shape_overflow()))?;

            let type_id = self.u32("tensor.type")?;
            let ggml_type = GgmlType::from_id(type_id).ok_or_else(|| Error {
                offset: self.pos as u64 - 4,
                kind: ErrorKind::UnknownGgmlType {
                    name: name.to_owned(),
                    type_id,
                },
            })?;
            let block_size = ggml_type.block_size();
            if ne[0] % block_size != 0 {
                self.pos -= 4;
                return Err(self.err(ErrorKind::RowNotBlockAligned {
                    name: name.to_owned(),
                    ne0: ne[0],
                    block_size,
                }));
            }
            let size = (n_elements / block_size)
                .checked_mul(ggml_type.type_size())
                .ok_or_else(|| self.err(shape_overflow()))?;

            let offset = self.u64("tensor.offset")?;
            tensors.push(TensorInfo {
                name,
                n_dims,
                ne,
                ggml_type,
                offset,
                size,
            });
        }
        Ok(tensors)
    }

    /// Checks that tensors are packed in info order, each starting at the
    /// previous end rounded up to the alignment, and that all data is in bounds.
    fn check_tensor_layout(
        &self,
        tensors: &[TensorInfo<'a>],
        alignment: u32,
        data_offset: u64,
    ) -> Result<()> {
        let file_len = self.data.len() as u64;
        let mut expected = 0u64;
        for t in tensors.iter() {
            if t.offset != expected {
                return Err(self.err(ErrorKind::TensorOffsetMismatch {
                    name: t.name.to_owned(),
                    offset: t.offset,
                    expected,
                }));
            }
            let end = data_offset
                .checked_add(t.offset)
                .and_then(|start| start.checked_add(t.size));
            if end.is_none_or(|end| end > file_len) {
                return Err(self.err(ErrorKind::TensorDataOutOfBounds {
                    name: t.name.to_owned(),
                }));
            }
            expected = t
                .size
                .checked_next_multiple_of(alignment.into())
                .and_then(|padded| expected.checked_add(padded))
                .ok_or_else(|| {
                    self.err(ErrorKind::ShapeOverflow {
                        name: t.name.to_owned(),
                    })
                })?;
        }
        Ok(())
    }
}
