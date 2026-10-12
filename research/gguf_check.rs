//! std-only GGUF parser + dequantizers + raw-libc mmap. Test harness for research notes.
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------- errors
#[derive(Debug)]
pub enum Error { Eof, Overflow, BadMagic, BadVersion(u32), Invalid(&'static str), Io(String) }
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{:?}", self) } }
type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------- limits (mirror ggml/src/gguf.cpp)
const MAX_STRING: u64 = 1 << 30;
const MAX_ARRAY: u64 = 1 << 30;
const MAX_KEY: u64 = 65535;
const MAX_NAME: usize = 64; // reference rejects name.len() >= 64
const MAX_DIMS: usize = 4;

// ---------------------------------------------------------------- ggml types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType { F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, BF16 }
impl GgmlType {
    pub fn from_u32(v: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match v { 0=>F32, 1=>F16, 2=>Q4_0, 3=>Q4_1, 6=>Q5_0, 7=>Q5_1, 8=>Q8_0, 9=>Q8_1,
            10=>Q2K, 11=>Q3K, 12=>Q4K, 13=>Q5K, 14=>Q6K, 15=>Q8K, 30=>BF16,
            _ => return Err(Error::Invalid("unsupported ggml type")) })
    }
    /// (elements per block, bytes per block)
    pub fn block(self) -> (u64, u64) {
        use GgmlType::*;
        match self { F32=>(1,4), F16=>(1,2), BF16=>(1,2), Q4_0=>(32,18), Q4_1=>(32,20), Q5_0=>(32,22), Q5_1=>(32,24),
            Q8_0=>(32,34), Q8_1=>(32,36), Q2K=>(256,84), Q3K=>(256,110), Q4K=>(256,144), Q5K=>(256,176),
            Q6K=>(256,210), Q8K=>(256,292) }
    }
}

// ---------------------------------------------------------------- reader
struct Reader<'a> { buf: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Overflow)?;
        let s = self.buf.get(self.pos..end).ok_or(Error::Eof)?;
        self.pos = end; Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N]> { Ok(self.take(N)?.try_into().unwrap()) }
    fn u8(&mut self) -> Result<u8> { Ok(self.arr::<1>()?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.arr()?)) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.arr()?)) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.arr()?)) }
    fn remaining(&self) -> usize { self.buf.len() - self.pos }
    fn string(&mut self, max: u64) -> Result<&'a str> {
        let len = self.u64()?;
        if len > max { return Err(Error::Invalid("string too long")); }
        let len = usize::try_from(len).map_err(|_| Error::Overflow)?;
        let b = self.take(len)?;
        std::str::from_utf8(b).map_err(|_| Error::Invalid("string is not utf-8"))
    }
}

// ---------------------------------------------------------------- metadata values
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VType { U8, I8, U16, I16, U32, I32, F32, Bool, Str, Arr, U64, I64, F64 }
impl VType {
    fn from_u32(v: u32) -> Result<Self> { use VType::*;
        Ok(match v { 0=>U8,1=>I8,2=>U16,3=>I16,4=>U32,5=>I32,6=>F32,7=>Bool,8=>Str,9=>Arr,10=>U64,11=>I64,12=>F64,
            _ => return Err(Error::Invalid("bad metadata value type")) }) }
    fn fixed_size(self) -> Option<usize> { use VType::*;
        match self { U8|I8|Bool=>Some(1), U16|I16=>Some(2), U32|I32|F32=>Some(4), U64|I64|F64=>Some(8), Str|Arr=>None } }
}
#[derive(Clone, Debug)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool), U64(u64), I64(i64), F64(f64),
    Str(String),
    /// Arrays stay zero-copy: remember the byte range inside the file; decode on demand.
    Arr { elem: VType, len: usize, start: usize, end: usize },
}
fn read_scalar(r: &mut Reader, t: VType) -> Result<Value> {
    Ok(match t {
        VType::U8 => Value::U8(r.u8()?), VType::I8 => Value::I8(r.u8()? as i8),
        VType::U16 => Value::U16(r.u16()?), VType::I16 => Value::I16(r.u16()? as i16),
        VType::U32 => Value::U32(r.u32()?), VType::I32 => Value::I32(r.u32()? as i32),
        VType::F32 => Value::F32(f32::from_bits(r.u32()?)),
        VType::Bool => match r.u8()? { 0 => Value::Bool(false), 1 => Value::Bool(true), _ => return Err(Error::Invalid("bool not 0/1")) },
        VType::U64 => Value::U64(r.u64()?), VType::I64 => Value::I64(r.u64()? as i64), VType::F64 => Value::F64(f64::from_bits(r.u64()?)),
        VType::Str => Value::Str(r.string(MAX_STRING)?.to_owned()),
        VType::Arr => unreachable!(),
    })
}
fn read_value(r: &mut Reader, t: VType) -> Result<Value> {
    if t != VType::Arr { return read_scalar(r, t); }
    let et = VType::from_u32(r.u32()?)?;
    if et == VType::Arr { return Err(Error::Invalid("nested arrays rejected (as in reference gguf.cpp)")); }
    let n = r.u64()?;
    if n > MAX_ARRAY { return Err(Error::Invalid("array too long")); }
    let n = usize::try_from(n).map_err(|_| Error::Overflow)?;
    let start = r.pos;
    match et.fixed_size() {
        Some(sz) => { let bytes = n.checked_mul(sz).ok_or(Error::Overflow)?; r.take(bytes)?; }
        None => { // strings: each needs >= 8 bytes of length prefix; reject before looping
            if n.checked_mul(8).ok_or(Error::Overflow)? > r.remaining() { return Err(Error::Eof); }
            for _ in 0..n { r.string(MAX_STRING)?; }
        }
    }
    Ok(Value::Arr { elem: et, len: n, start, end: r.pos })
}

// ---------------------------------------------------------------- tensor info
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: usize,
    pub ne: [u64; 4],        // ggml order: ne[0] = contiguous (fastest) dim = "in_features" for a linear weight
    pub ty: GgmlType,
    pub offset: u64,         // relative to data section start (as stored)
    pub n_bytes: u64,
}
impl TensorInfo {
    pub fn row_bytes(&self) -> u64 { let (b, s) = self.ty.block(); self.ne[0] / b * s }
    pub fn n_rows(&self) -> u64 { self.ne[1] * self.ne[2] * self.ne[3] }
}

pub struct Gguf {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub index: HashMap<String, usize>,
    pub data_start: u64,     // absolute file offset of tensor data section
}

fn align_up(x: u64, a: u64) -> Result<u64> { Ok(x.checked_add(a - 1).ok_or(Error::Overflow)? / a * a) }

impl Gguf {
    pub fn parse(file: &[u8]) -> Result<Gguf> {
        let mut r = Reader { buf: file, pos: 0 };
        if r.take(4)? != b"GGUF" { return Err(Error::BadMagic); }
        let version = r.u32()?;
        if version & 0xFFFF == 0 { return Err(Error::Invalid("looks like big-endian file")); }
        if version != 2 && version != 3 { return Err(Error::BadVersion(version)); }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        // cheap plausibility check before allocating anything (min sizes: kv = 8+4+1, tensor = 8+4+4+8)
        let need = n_kv.checked_mul(13).and_then(|a| n_tensors.checked_mul(24).and_then(|b| a.checked_add(b))).ok_or(Error::Overflow)?;
        if need > r.remaining() as u64 { return Err(Error::Invalid("counts exceed file size")); }
        let (n_tensors, n_kv) = (n_tensors as usize, n_kv as usize);

        let mut metadata = HashMap::with_capacity(n_kv.min(4096));
        for _ in 0..n_kv {
            let key = r.string(MAX_KEY)?;
            if key.is_empty() { return Err(Error::Invalid("empty key")); }
            let t = VType::from_u32(r.u32()?)?;
            let v = read_value(&mut r, t)?;
            if metadata.insert(key.to_owned(), v).is_some() { return Err(Error::Invalid("duplicate key")); }
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(Value::U32(a)) => *a as u64,
            Some(_) => return Err(Error::Invalid("general.alignment must be u32")),
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > (1 << 30) { return Err(Error::Invalid("bad alignment")); }

        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16));
        let mut index = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string(MAX_NAME as u64)?;       // name.len() > 64 -> error
            if name.len() >= MAX_NAME { return Err(Error::Invalid("tensor name too long")); }
            let n_dims = r.u32()? as usize;
            if n_dims > MAX_DIMS { return Err(Error::Invalid("too many dims")); }
            let mut ne = [1u64; 4];
            for d in ne.iter_mut().take(n_dims) {
                *d = r.u64()?;
                if *d > i64::MAX as u64 { return Err(Error::Invalid("dim > i64::MAX")); }
            }
            let ty = GgmlType::from_u32(r.u32()?)?;
            let offset = r.u64()?;
            let (blk, tsize) = ty.block();
            if ne[0] % blk != 0 { return Err(Error::Invalid("ne[0] not multiple of block size")); }
            let mut n_el: u64 = 1;
            for d in ne { n_el = n_el.checked_mul(d).ok_or(Error::Overflow)?; }
            let n_bytes = (n_el / blk).checked_mul(tsize).ok_or(Error::Overflow)?;
            if offset % alignment != 0 { return Err(Error::Invalid("tensor offset not aligned")); }
            if index.insert(name.to_owned(), tensors.len()).is_some() { return Err(Error::Invalid("duplicate tensor name")); }
            tensors.push(TensorInfo { name: name.to_owned(), n_dims, ne, ty, offset, n_bytes });
        }
        // alignment is relative to the start of the file (reference uses gr.start() == 0 for plain files)
        let data_start = align_up(r.pos as u64, alignment)?;
        Ok(Gguf { version, alignment, metadata, tensors, index, data_start })
    }

    /// Bounds-checked tensor payload. Fails (no panic) if the file is truncated.
    pub fn tensor_bytes<'a>(&self, file: &'a [u8], t: &TensorInfo) -> Result<&'a [u8]> {
        let start = self.data_start.checked_add(t.offset).ok_or(Error::Overflow)?;
        let end = start.checked_add(t.n_bytes).ok_or(Error::Overflow)?;
        let (s, e) = (usize::try_from(start).map_err(|_| Error::Overflow)?, usize::try_from(end).map_err(|_| Error::Overflow)?);
        file.get(s..e).ok_or(Error::Eof)
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> { self.index.get(name).map(|&i| &self.tensors[i]) }
    pub fn get_u64(&self, k: &str) -> Option<u64> { match self.metadata.get(k)? {
        Value::U32(v) => Some(*v as u64), Value::U64(v) => Some(*v), Value::I32(v) if *v >= 0 => Some(*v as u64),
        Value::I64(v) if *v >= 0 => Some(*v as u64), Value::U16(v) => Some(*v as u64), Value::U8(v) => Some(*v as u64), _ => None } }
    pub fn get_f32(&self, k: &str) -> Option<f32> { match self.metadata.get(k)? { Value::F32(v) => Some(*v), Value::F64(v) => Some(*v as f32), _ => None } }
    pub fn get_str(&self, k: &str) -> Option<&str> { match self.metadata.get(k)? { Value::Str(s) => Some(s), _ => None } }
    pub fn string_array<'a>(&self, file: &'a [u8], k: &str) -> Result<Vec<&'a str>> {
        match self.metadata.get(k) {
            Some(Value::Arr { elem: VType::Str, len, start, end }) => {
                let mut r = Reader { buf: file.get(*start..*end).ok_or(Error::Eof)?, pos: 0 };
                (0..*len).map(|_| r.string(MAX_STRING)).collect()
            }
            _ => Err(Error::Invalid("not a string array")),
        }
    }
}

// ---------------------------------------------------------------- align-1 block structs (no padding, every bit pattern valid)
/// SAFETY contract: repr(C), align 1, no padding, only u8/i8/[u8;N]/[i8;N] fields.
pub unsafe trait Block: Copy { }
macro_rules! block { ($name:ident, $size:expr, { $($f:ident : $t:ty),* }) => {
    #[repr(C)] #[derive(Clone, Copy)] pub struct $name { $(pub $f: $t),* }
    const _: () = assert!(std::mem::size_of::<$name>() == $size && std::mem::align_of::<$name>() == 1);
    unsafe impl Block for $name {}
}}
block!(BlockQ4_0, 18,  { d: [u8;2], qs: [u8;16] });
block!(BlockQ4_1, 20,  { d: [u8;2], m: [u8;2], qs: [u8;16] });
block!(BlockQ5_0, 22,  { d: [u8;2], qh: [u8;4], qs: [u8;16] });
block!(BlockQ5_1, 24,  { d: [u8;2], m: [u8;2], qh: [u8;4], qs: [u8;16] });
block!(BlockQ8_0, 34,  { d: [u8;2], qs: [i8;32] });
block!(BlockQ2K,  84,  { scales: [u8;16], qs: [u8;64], d: [u8;2], dmin: [u8;2] });
block!(BlockQ3K,  110, { hmask: [u8;32], qs: [u8;64], scales: [u8;12], d: [u8;2] });
block!(BlockQ4K,  144, { d: [u8;2], dmin: [u8;2], scales: [u8;12], qs: [u8;128] });
block!(BlockQ5K,  176, { d: [u8;2], dmin: [u8;2], scales: [u8;12], qh: [u8;32], qs: [u8;128] });
block!(BlockQ6K,  210, { ql: [u8;128], qh: [u8;64], scales: [i8;16], d: [u8;2] });
block!(BlockQ8K,  292, { d: [u8;4], qs: [i8;256], bsums: [u8;32] });

/// Reinterpret bytes as blocks. Sound because T has align 1, no padding, all bit patterns valid.
pub fn cast_blocks<T: Block>(bytes: &[u8]) -> Result<&[T]> {
    const { assert!(std::mem::align_of::<T>() == 1) }
    let sz = std::mem::size_of::<T>();
    if bytes.len() % sz != 0 { return Err(Error::Invalid("byte length not a multiple of block size")); }
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / sz) })
}

// ---------------------------------------------------------------- scalar conversions
#[inline] pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32; let exp = ((h >> 10) & 0x1f) as u32; let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 { sign << 31 } else {
            let (mut e, mut m) = (113u32, man);          // subnormal: normalise
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f { (sign << 31) | (0xff << 23) | (man << 13) }
    else { (sign << 31) | ((exp + 112) << 23) | (man << 13) };
    f32::from_bits(bits)
}
#[inline] fn h(b: [u8; 2]) -> f32 { f16_to_f32(u16::from_le_bytes(b)) }
#[inline] pub fn bf16_to_f32(b: u16) -> f32 { f32::from_bits((b as u32) << 16) }

// ---------------------------------------------------------------- 6-bit scale/min unpack for Q4_K / Q5_K (ggml get_scale_min_k4)
#[inline] fn scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

// ---------------------------------------------------------------- dequantize one row (src = row bytes, dst = ne0 floats)
pub fn dequantize_row(ty: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let (blk, tsize) = ty.block();
    if dst.len() as u64 % blk != 0 || src.len() as u64 != dst.len() as u64 / blk * tsize { return Err(Error::Invalid("row size mismatch")); }
    match ty {
        GgmlType::F32 => for (o, c) in dst.iter_mut().zip(src.chunks_exact(4)) { *o = f32::from_le_bytes(c.try_into().unwrap()) },
        GgmlType::F16 => for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) { *o = f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())) },
        GgmlType::BF16 => for (o, c) in dst.iter_mut().zip(src.chunks_exact(2)) { *o = bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())) },
        GgmlType::Q4_0 => for (b, y) in cast_blocks::<BlockQ4_0>(src)?.iter().zip(dst.chunks_exact_mut(32)) {
            let d = h(b.d);
            for j in 0..16 { y[j] = ((b.qs[j] & 0xF) as i32 - 8) as f32 * d; y[j + 16] = ((b.qs[j] >> 4) as i32 - 8) as f32 * d; }
        },
        GgmlType::Q4_1 => for (b, y) in cast_blocks::<BlockQ4_1>(src)?.iter().zip(dst.chunks_exact_mut(32)) {
            let (d, m) = (h(b.d), h(b.m));
            for j in 0..16 { y[j] = (b.qs[j] & 0xF) as f32 * d + m; y[j + 16] = (b.qs[j] >> 4) as f32 * d + m; }
        },
        GgmlType::Q5_0 => for (b, y) in cast_blocks::<BlockQ5_0>(src)?.iter().zip(dst.chunks_exact_mut(32)) {
            let d = h(b.d); let qh = u32::from_le_bytes(b.qh);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10; let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = (((b.qs[j] & 0xF) as u32 | xh0) as i32 - 16) as f32 * d;
                y[j + 16] = (((b.qs[j] >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
            }
        },
        GgmlType::Q5_1 => for (b, y) in cast_blocks::<BlockQ5_1>(src)?.iter().zip(dst.chunks_exact_mut(32)) {
            let (d, m) = (h(b.d), h(b.m)); let qh = u32::from_le_bytes(b.qh);
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10; let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = ((b.qs[j] & 0xF) as u32 | xh0) as f32 * d + m;
                y[j + 16] = ((b.qs[j] >> 4) as u32 | xh1) as f32 * d + m;
            }
        },
        GgmlType::Q8_0 => for (b, y) in cast_blocks::<BlockQ8_0>(src)?.iter().zip(dst.chunks_exact_mut(32)) {
            let d = h(b.d); for j in 0..32 { y[j] = b.qs[j] as f32 * d; }
        },
        GgmlType::Q2K => for (b, y) in cast_blocks::<BlockQ2K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let (d, dmin) = (h(b.d), h(b.dmin));
            let (mut is, mut o) = (0usize, 0usize);
            for n in 0..2 { let q = &b.qs[n * 32..n * 32 + 32]; let mut shift = 0;
                for _ in 0..4 {
                    for half in 0..2 {
                        let sc = b.scales[is]; is += 1;
                        let (dl, ml) = (d * (sc & 0xF) as f32, dmin * (sc >> 4) as f32);
                        for l in 0..16 { y[o] = dl * ((q[half * 16 + l] >> shift) & 3) as f32 - ml; o += 1; }
                    }
                    shift += 2;
                } }
        },
        GgmlType::Q3K => for (b, y) in cast_blocks::<BlockQ3K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let d_all = h(b.d);
            let (k1, k2) = (0x03030303u32, 0x0f0f0f0fu32);
            let w = |i: usize| u32::from_le_bytes(b.scales[i * 4..i * 4 + 4].try_into().unwrap());
            let (a0, a1, tmp) = (w(0), w(1), w(2));
            let aux = [ (a0 & k2) | (((tmp >> 0) & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4),
                        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4) ];
            let mut sc = [0u8; 16]; for i in 0..4 { sc[i * 4..i * 4 + 4].copy_from_slice(&aux[i].to_le_bytes()); }
            let (mut is, mut o, mut m) = (0usize, 0usize, 1u8);
            for n in 0..2 { let q = &b.qs[n * 32..n * 32 + 32]; let mut shift = 0;
                for _ in 0..4 {
                    for half in 0..2 {
                        let dl = d_all * (sc[is] as i32 - 32) as f32; is += 1;
                        for l in 0..16 {
                            let lo = ((q[half * 16 + l] >> shift) & 3) as i32;
                            let hi = if b.hmask[half * 16 + l] & m != 0 { 0 } else { 4 };
                            y[o] = dl * (lo - hi) as f32; o += 1;
                        }
                    }
                    shift += 2; m <<= 1;
                } }
        },
        GgmlType::Q4K => for (b, y) in cast_blocks::<BlockQ4K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let (d, dmin) = (h(b.d), h(b.dmin));
            for g in 0..4 {
                let q = &b.qs[g * 32..g * 32 + 32];
                let (s1, m1) = scale_min_k4(2 * g, &b.scales); let (s2, m2) = scale_min_k4(2 * g + 1, &b.scales);
                let (d1, mm1, d2, mm2) = (d * s1 as f32, dmin * m1 as f32, d * s2 as f32, dmin * m2 as f32);
                for l in 0..32 { y[g * 64 + l] = d1 * (q[l] & 0xF) as f32 - mm1; y[g * 64 + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2; }
            }
        },
        GgmlType::Q5K => for (b, y) in cast_blocks::<BlockQ5K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let (d, dmin) = (h(b.d), h(b.dmin));
            for g in 0..4 {
                let ql = &b.qs[g * 32..g * 32 + 32];
                let (s1, m1) = scale_min_k4(2 * g, &b.scales); let (s2, m2) = scale_min_k4(2 * g + 1, &b.scales);
                let (d1, mm1, d2, mm2) = (d * s1 as f32, dmin * m1 as f32, d * s2 as f32, dmin * m2 as f32);
                let (u1, u2) = (1u8 << (2 * g), 2u8 << (2 * g));
                for l in 0..32 {
                    let h1 = if b.qh[l] & u1 != 0 { 16 } else { 0 }; let h2 = if b.qh[l] & u2 != 0 { 16 } else { 0 };
                    y[g * 64 + l] = d1 * ((ql[l] & 0xF) as i32 + h1) as f32 - mm1;
                    y[g * 64 + 32 + l] = d2 * ((ql[l] >> 4) as i32 + h2) as f32 - mm2;
                }
            }
        },
        GgmlType::Q6K => for (b, y) in cast_blocks::<BlockQ6K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let d = h(b.d);
            for n in 0..2 {
                let (ql, qh, sc, yy) = (&b.ql[n * 64..n * 64 + 64], &b.qh[n * 32..n * 32 + 32], &b.scales[n * 8..n * 8 + 8], &mut y[n * 128..n * 128 + 128]);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    yy[l] = d * sc[is] as f32 * q1 as f32; yy[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    yy[l + 64] = d * sc[is + 4] as f32 * q3 as f32; yy[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        },
        GgmlType::Q8K => for (b, y) in cast_blocks::<BlockQ8K>(src)?.iter().zip(dst.chunks_exact_mut(256)) {
            let d = f32::from_le_bytes(b.d); for j in 0..256 { y[j] = d * b.qs[j] as f32; }
        },
        GgmlType::Q8_1 => return Err(Error::Invalid("Q8_1 is activation-only")),
    }
    Ok(())
}

// ---------------------------------------------------------------- raw libc mmap (no crates). Linux/macOS 64-bit.
#[cfg(unix)]
pub mod mm {
    use std::ffi::c_void;
    use std::os::fd::AsRawFd;
    extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        fn mlock(addr: *const c_void, len: usize) -> i32;
        fn sysconf(name: i32) -> i64;
    }
    const PROT_READ: i32 = 1; const MAP_PRIVATE: i32 = 2;
    #[cfg(target_os = "linux")] const SC_PAGESIZE: i32 = 30;
    #[cfg(target_os = "macos")] const SC_PAGESIZE: i32 = 29;
    #[cfg(target_os = "linux")] pub const MAP_POPULATE: i32 = 0x8000;
    #[derive(Clone, Copy)] pub enum Advice { Random = 1, Sequential = 2, WillNeed = 3, #[cfg(target_os = "linux")] HugePage = 14 }
    pub struct Mmap { ptr: *mut u8, len: usize }
    unsafe impl Send for Mmap {} unsafe impl Sync for Mmap {}
    impl Mmap {
        /// SAFETY (caller): file must not be truncated/modified while mapped (else SIGBUS / torn reads).
        pub unsafe fn open(path: &str, flags_extra: i32) -> std::io::Result<Mmap> {
            let f = std::fs::File::open(path)?;
            let len = f.metadata()?.len() as usize;
            if len == 0 { return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty file")); }
            let p = mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE | flags_extra, f.as_raw_fd(), 0);
            if p as usize == usize::MAX { return Err(std::io::Error::last_os_error()); }
            Ok(Mmap { ptr: p as *mut u8, len })     // fd may be closed now; mapping stays valid
        }
        pub fn as_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
        pub fn page_size() -> usize { unsafe { sysconf(SC_PAGESIZE) as usize } }
        /// range is rounded outward to page boundaries; errors are returned, hints may be ignored by the kernel
        pub fn advise(&self, a: Advice, off: usize, len: usize) -> std::io::Result<()> {
            let ps = Self::page_size(); let start = off & !(ps - 1);
            let end = (off.saturating_add(len).saturating_add(ps - 1) & !(ps - 1)).min(self.len.div_ceil(ps) * ps);
            if start >= end { return Ok(()); }
            if unsafe { madvise(self.ptr.add(start) as *mut c_void, end - start, a as i32) } != 0 { return Err(std::io::Error::last_os_error()); }
            Ok(())
        }
        pub fn lock(&self) -> std::io::Result<()> { if unsafe { mlock(self.ptr as *const c_void, self.len) } != 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) } }
    }
    impl Drop for Mmap { fn drop(&mut self) { unsafe { munmap(self.ptr as *mut c_void, self.len); } } }
}

// ---------------------------------------------------------------- test driver
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("f16") => { // exhaustive f16 -> f32 check against numpy table
            let t = std::fs::read(&args[2]).unwrap(); let mut bad = 0;
            for i in 0..65536u32 { let want = f32::from_le_bytes(t[i as usize*4..i as usize*4+4].try_into().unwrap());
                let got = f16_to_f32(i as u16);
                if !(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())) { bad += 1; if bad < 5 { println!("f16 mismatch {i:#06x} got {got:?} want {want:?}"); } } }
            println!("f16_to_f32 exhaustive: {} mismatches of 65536", bad);
        }
        Some("dequant") => { // args: samples dir
            let dir = &args[2]; let man = std::fs::read_to_string(format!("{dir}/manifest.tsv")).unwrap();
            let mut all_ok = true;
            for line in man.lines() {
                let f: Vec<&str> = line.split('\t').collect(); let (id, ty, ne0, nrows) = (f[0], f[1].parse::<u32>().unwrap(), f[2].parse::<usize>().unwrap(), f[3].parse::<usize>().unwrap());
                let ty = GgmlType::from_u32(ty).unwrap();
                let raw = std::fs::read(format!("{dir}/{id}.bin")).unwrap(); let rf = std::fs::read(format!("{dir}/{id}.ref.f32")).unwrap();
                let (b, s) = ty.block(); let rb = ne0 / b as usize * s as usize;
                let mut out = vec![0f32; ne0]; let (mut maxd, mut maxabs) = (0f32, 0f32); let mut n = 0usize;
                for r in 0..nrows {
                    dequantize_row(ty, &raw[r * rb..(r + 1) * rb], &mut out).unwrap();
                    for (i, v) in out.iter().enumerate() {
                        let want = f32::from_le_bytes(rf[(r * ne0 + i) * 4..(r * ne0 + i) * 4 + 4].try_into().unwrap());
                        maxd = maxd.max((v - want).abs()); maxabs = maxabs.max(want.abs()); n += 1;
                    }
                }
                let ok = maxd <= 1e-6 * maxabs.max(1.0); all_ok &= ok;
                println!("{} {:8?} n={} max|diff|={:.3e} max|ref|={:.4} {}", if ok {"PASS"} else {"FAIL"}, ty, n, maxd, maxabs, id);
            }
            println!("ALL {}", if all_ok { "PASS" } else { "FAIL" });
        }
        Some("parse") => { // args: file [--mmap]
            let path = &args[2];
            let mm = unsafe { mm::Mmap::open(path, 0) }.unwrap();
            let _ = mm.advise(mm::Advice::Random, 0, 0);
            let file = mm.as_slice();
            println!("mmap base alignment: {} (ptr % 4096 = {}), len = {}", file.as_ptr() as usize % 64, file.as_ptr() as usize % 4096, file.len());
            let g = Gguf::parse(file).unwrap();
            println!("version={} alignment={} data_start={} n_tensors={} n_kv={}", g.version, g.alignment, g.data_start, g.tensors.len(), g.metadata.len());
            let arch = g.get_str("general.architecture").unwrap().to_owned();
            for k in ["block_count","context_length","embedding_length","feed_forward_length","attention.head_count","attention.head_count_kv","attention.key_length","rope.dimension_count","vocab_size"] {
                println!("  {arch}.{k} = {:?}", g.get_u64(&format!("{arch}.{k}"))); }
            for k in ["rope.freq_base","attention.layer_norm_rms_epsilon"] { println!("  {arch}.{k} = {:?}", g.get_f32(&format!("{arch}.{k}"))); }
            let toks = g.string_array(file, "tokenizer.ggml.tokens").unwrap(); println!("  tokens: {} first={:?}", toks.len(), &toks[..3]);
            let mut ok = 0; let mut trunc = 0;
            for t in &g.tensors { match g.tensor_bytes(file, t) { Ok(b) => { ok += 1; assert_eq!(b.len() as u64, t.n_bytes); } Err(Error::Eof) => trunc += 1, Err(e) => panic!("{e}") } }
            println!("  tensors fully inside file: {ok}, beyond truncated file (clean Err(Eof)): {trunc}");
            // fuzz: mutate/truncate header bytes, must never panic or hang
            let hdr = &file[..(g.data_start as usize).min(file.len())];
            let mut seed = 0x1234_5678_9abc_def0u64; let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
            let (mut parsed, mut rejected) = (0, 0);
            for it in 0..3000 {
                let mut v = hdr[..(rnd() as usize % hdr.len()).max(1)].to_vec();
                if it % 2 == 0 { v = hdr.to_vec(); for _ in 0..(1 + rnd() % 8) { let i = rnd() as usize % v.len().min(4096); v[i] = rnd() as u8; } }
                match Gguf::parse(&v) { Ok(_) => parsed += 1, Err(_) => rejected += 1 }
            }
            println!("  fuzz: 3000 mutated/truncated headers -> {parsed} parsed, {rejected} rejected, no panic");
        }
        _ => eprintln!("usage: gguf_check f16 <table> | dequant <dir> | parse <file>"),
    }
}
