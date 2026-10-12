use std::fmt;

/// A parse failure, with the byte offset where it was detected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Byte offset in the input where the problem was found.
    pub offset: u64,
    /// What went wrong.
    pub kind: ErrorKind,
}

/// What went wrong. Field names are dotted paths such as `"kv.key"` or
/// `"tensor.dims"`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The input ended while reading `field`.
    Truncated {
        field: &'static str,
        needed: u64,
    },
    BadMagic([u8; 4]),
    /// The version reads as a huge number, which is what a big-endian file
    /// looks like on a little-endian reader.
    BigEndian,
    UnsupportedVersion(u32),
    /// A count is larger than the remaining bytes could possibly hold.
    CountTooLarge {
        field: &'static str,
        count: u64,
    },
    /// A string is longer than the 1 GiB cap llama.cpp uses.
    StringTooLong {
        field: &'static str,
        len: u64,
    },
    /// An array has more elements than the 1 GiB cap llama.cpp uses.
    ArrayTooLong {
        key: String,
        len: u64,
    },
    InvalidUtf8 {
        field: &'static str,
    },
    EmptyKey,
    DuplicateKey(String),
    InvalidValueType {
        key: String,
        type_id: u32,
    },
    NestedArray {
        key: String,
    },
    /// A bool was neither 0 nor 1.
    InvalidBool {
        key: String,
        byte: u8,
    },
    /// `general.alignment` is not a `u32`.
    AlignmentWrongType,
    /// `general.alignment` is zero or not a power of two.
    BadAlignment(u32),
    /// Tensor names must be shorter than 64 bytes (ggml's `GGML_MAX_NAME`).
    TensorNameTooLong {
        name: String,
    },
    DuplicateTensorName(String),
    TooManyDims {
        name: String,
        n_dims: u32,
    },
    /// A dimension does not fit in an `i64`, as ggml requires.
    DimOutOfRange {
        name: String,
        dim: u64,
    },
    /// The element count or byte size overflows.
    ShapeOverflow {
        name: String,
    },
    UnknownGgmlType {
        name: String,
        type_id: u32,
    },
    /// `ne[0]` is not a multiple of the type's block size.
    RowNotBlockAligned {
        name: String,
        ne0: u64,
        block_size: u64,
    },
    /// Tensor data must be laid out in info order, each tensor starting at
    /// the previous one's end rounded up to the alignment.
    TensorOffsetMismatch {
        name: String,
        offset: u64,
        expected: u64,
    },
    /// The tensor's bytes run past the end of the input.
    TensorDataOutOfBounds {
        name: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid GGUF at byte {}: ", self.offset)?;
        use ErrorKind::*;
        match &self.kind {
            Truncated { field, needed } => {
                write!(
                    f,
                    "file ends while reading {field} ({needed} more bytes needed)"
                )
            }
            BadMagic(m) => write!(f, "bad magic {m:02x?}, expected \"GGUF\""),
            BigEndian => f.write_str("big-endian GGUF files are not supported"),
            UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v} (supported: 2, 3)"),
            CountTooLarge { field, count } => {
                write!(
                    f,
                    "{field} count {count} exceeds what the remaining bytes can hold"
                )
            }
            StringTooLong { field, len } => write!(f, "{field} string length {len} exceeds 1 GiB"),
            ArrayTooLong { key, len } => write!(f, "array '{key}' length {len} exceeds 2^30"),
            InvalidUtf8 { field } => write!(f, "{field} is not valid UTF-8"),
            EmptyKey => f.write_str("empty metadata key"),
            DuplicateKey(k) => write!(f, "duplicate metadata key '{k}'"),
            InvalidValueType { key, type_id } => {
                write!(f, "key '{key}' has invalid value type {type_id}")
            }
            NestedArray { key } => write!(f, "key '{key}' is a nested array"),
            InvalidBool { key, byte } => write!(f, "key '{key}' has bool byte {byte}, not 0 or 1"),
            AlignmentWrongType => f.write_str("general.alignment is not a u32"),
            BadAlignment(a) => write!(f, "alignment {a} is not a power of two"),
            TensorNameTooLong { name } => {
                write!(f, "tensor name '{name}' is {} bytes, max is 63", name.len())
            }
            DuplicateTensorName(n) => write!(f, "duplicate tensor name '{n}'"),
            TooManyDims { name, n_dims } => {
                write!(f, "tensor '{name}' has {n_dims} dimensions, max is 4")
            }
            DimOutOfRange { name, dim } => {
                write!(f, "tensor '{name}' dimension {dim} does not fit in i64")
            }
            ShapeOverflow { name } => write!(f, "tensor '{name}' size overflows"),
            UnknownGgmlType { name, type_id } => {
                write!(f, "tensor '{name}' has unknown ggml type {type_id}")
            }
            RowNotBlockAligned {
                name,
                ne0,
                block_size,
            } => write!(
                f,
                "tensor '{name}' has {ne0} elements per row, not a multiple of block size {block_size}"
            ),
            TensorOffsetMismatch {
                name,
                offset,
                expected,
            } => {
                write!(
                    f,
                    "tensor '{name}' has offset {offset}, expected {expected}"
                )
            }
            TensorDataOutOfBounds { name } => {
                write!(f, "tensor '{name}' data runs past the end of the file")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Result type for GGUF parsing.
pub type Result<T> = std::result::Result<T, Error>;
