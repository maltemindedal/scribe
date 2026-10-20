//! Metadata values. Scalars are decoded; arrays stay as byte ranges into the
//! file so a 150k-entry vocabulary costs nothing until it is iterated.

/// A GGUF metadata value type id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    /// Looks up a GGUF value type id. Unknown ids return `None`.
    pub fn from_id(id: u32) -> Option<Self> {
        use ValueType::*;
        Some(match id {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }

    /// Card spelling for this type, such as `"u8"` or `"array"`.
    pub fn name(self) -> &'static str {
        match self {
            ValueType::U8 => "u8",
            ValueType::I8 => "i8",
            ValueType::U16 => "u16",
            ValueType::I16 => "i16",
            ValueType::U32 => "u32",
            ValueType::I32 => "i32",
            ValueType::U64 => "u64",
            ValueType::I64 => "i64",
            ValueType::F32 => "f32",
            ValueType::F64 => "f64",
            ValueType::Bool => "bool",
            ValueType::String => "string",
            ValueType::Array => "array",
        }
    }

    /// Byte width of a fixed-size value, `None` for strings and arrays.
    pub(crate) fn fixed_size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            U8 | I8 | Bool => Some(1),
            U16 | I16 => Some(2),
            U32 | I32 | F32 => Some(4),
            U64 | I64 | F64 => Some(8),
            String | Array => None,
        }
    }
}

/// A metadata value borrowed from the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'a str),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Array<'a>),
}

impl<'a> Value<'a> {
    /// The string, if this is a `String`.
    pub fn as_str(&self) -> Option<&'a str> {
        match *self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The array, if this is an `Array`.
    pub fn as_array(&self) -> Option<Array<'a>> {
        match *self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Any unsigned or non-negative signed integer, widened to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value, if this is an `F32`.
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            _ => None,
        }
    }

    /// The value, if this is a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }
}

/// An array value: element type, length and the raw bytes holding it.
/// Contents were validated at parse time (lengths, UTF-8, bool bytes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Array<'a> {
    pub(crate) elem: ValueType,
    pub(crate) len: u64,
    pub(crate) bytes: &'a [u8],
}

macro_rules! typed_iter {
    ($($(#[$doc:meta])* $fn:ident, $variant:ident, $t:ty;)*) => {$(
        $(#[$doc])*
        #[doc = concat!("Iterates a `", stringify!($t), "` array, or `None` if this is not one.")]
        pub fn $fn(&self) -> Option<impl ExactSizeIterator<Item = $t> + 'a> {
            const N: usize = std::mem::size_of::<$t>();
            (self.elem == ValueType::$variant).then(|| {
                self.bytes
                    .chunks_exact(N)
                    .map(|c| <$t>::from_le_bytes(c.try_into().expect("chunk has N bytes")))
            })
        }
    )*};
}

impl<'a> Array<'a> {
    /// The type of every element.
    pub fn elem_type(&self) -> ValueType {
        self.elem
    }

    /// Number of elements.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The raw little-endian bytes of the array.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// Iterates a string array, or `None` if this is not one.
    pub fn strings(&self) -> Option<Strings<'a>> {
        (self.elem == ValueType::String).then_some(Strings {
            rest: self.bytes,
            remaining: self.len,
        })
    }

    typed_iter! {
        u8s, U8, u8;
        i8s, I8, i8;
        u16s, U16, u16;
        i16s, I16, i16;
        u32s, U32, u32;
        i32s, I32, i32;
        f32s, F32, f32;
        u64s, U64, u64;
        i64s, I64, i64;
        f64s, F64, f64;
    }

    /// Iterates a bool array, or `None` if this is not one.
    pub fn bools(&self) -> Option<impl ExactSizeIterator<Item = bool> + 'a> {
        (self.elem == ValueType::Bool).then(|| self.bytes.iter().map(|&b| b != 0))
    }
}

/// Iterator over the strings of a string array.
#[derive(Clone, Debug)]
pub struct Strings<'a> {
    rest: &'a [u8],
    remaining: u64,
}

impl<'a> Iterator for Strings<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.remaining == 0 {
            return None;
        }
        // The parser already checked every length and the UTF-8 of every
        // string, so these steps cannot fail on an array it produced.
        let (len, rest) = self.rest.split_first_chunk::<8>()?;
        let len = usize::try_from(u64::from_le_bytes(*len)).ok()?;
        let (s, rest) = rest.split_at_checked(len)?;
        self.rest = rest;
        self.remaining -= 1;
        std::str::from_utf8(s).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        (n, Some(n))
    }
}

impl ExactSizeIterator for Strings<'_> {}
