//! The ggml tensor type table.
//!
//! Ids, block sizes and byte sizes are copied from `ggml.h`, `ggml.c`
//! (`type_traits`) and the `static_assert`s in `ggml-common.h`, not from
//! `gguf.md`, which lags behind. Ids 4, 5, 31 to 33 and 36 to 38 were
//! removed from ggml and are rejected like any other unknown id.

/// Every ggml type, one row each: `Variant = (id, name, elements per block,
/// bytes per block, scribe loads it)`. The enum and the table come from this
/// one list, so a type cannot exist without its row.
macro_rules! ggml_types {
    ($($ty:ident = ($id:expr, $name:expr, $bs:expr, $ts:expr, $loads:expr),)*) => {
        /// A ggml tensor element type.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum GgmlType { $($ty,)* }

        /// Rows are in enum order, so `TABLE[ty as usize]` is `ty`'s row.
        const TABLE: &[Row] = &[$(Row {
            ty: GgmlType::$ty,
            id: $id,
            name: $name,
            block_size: $bs,
            type_size: $ts,
            scribe_loads: $loads,
        },)*];
    };
    ($($rest:tt)*) => {
        compile_error!(
            "ggml_types! rows must look like `Variant = (id, \"name\", elements per block, \
             bytes per block, scribe loads it),`"
        );
    };
}

struct Row {
    ty: GgmlType,
    id: u32,
    name: &'static str,
    block_size: u64,
    type_size: u64,
    scribe_loads: bool,
}

ggml_types! {
    F32 = (0, "f32", 1, 4, true),
    F16 = (1, "f16", 1, 2, true),
    Q4_0 = (2, "q4_0", 32, 18, true),
    Q4_1 = (3, "q4_1", 32, 20, true),
    Q5_0 = (6, "q5_0", 32, 22, true),
    Q5_1 = (7, "q5_1", 32, 24, true),
    Q8_0 = (8, "q8_0", 32, 34, true),
    Q8_1 = (9, "q8_1", 32, 36, false),
    Q2K = (10, "q2_K", 256, 84, true),
    Q3K = (11, "q3_K", 256, 110, true),
    Q4K = (12, "q4_K", 256, 144, true),
    Q5K = (13, "q5_K", 256, 176, true),
    Q6K = (14, "q6_K", 256, 210, true),
    Q8K = (15, "q8_K", 256, 292, true),
    Iq2Xxs = (16, "iq2_xxs", 256, 66, false),
    Iq2Xs = (17, "iq2_xs", 256, 74, false),
    Iq3Xxs = (18, "iq3_xxs", 256, 98, false),
    Iq1S = (19, "iq1_s", 256, 50, false),
    Iq4Nl = (20, "iq4_nl", 32, 18, false),
    Iq3S = (21, "iq3_s", 256, 110, false),
    Iq2S = (22, "iq2_s", 256, 82, false),
    Iq4Xs = (23, "iq4_xs", 256, 136, false),
    I8 = (24, "i8", 1, 1, true),
    I16 = (25, "i16", 1, 2, true),
    I32 = (26, "i32", 1, 4, true),
    I64 = (27, "i64", 1, 8, true),
    F64 = (28, "f64", 1, 8, true),
    Iq1M = (29, "iq1_m", 256, 56, false),
    Bf16 = (30, "bf16", 1, 2, true),
    Tq1_0 = (34, "tq1_0", 256, 54, false),
    Tq2_0 = (35, "tq2_0", 256, 66, false),
    Mxfp4 = (39, "mxfp4", 32, 17, false),
    Nvfp4 = (40, "nvfp4", 64, 36, false),
    Q1_0 = (41, "q1_0", 128, 18, false),
    Q2_0 = (42, "q2_0", 64, 18, false),
}

impl GgmlType {
    /// Looks up a ggml type id. Unknown and removed ids return `None`.
    pub fn from_id(id: u32) -> Option<Self> {
        TABLE.iter().find(|row| row.id == id).map(|row| row.ty)
    }

    fn row(self) -> &'static Row {
        &TABLE[self as usize]
    }

    /// The numeric id stored in GGUF files.
    pub fn id(self) -> u32 {
        self.row().id
    }

    /// The name ggml uses, e.g. `"q8_0"`.
    pub fn name(self) -> &'static str {
        self.row().name
    }

    /// Elements per block (1 for plain types such as f32).
    pub fn block_size(self) -> u64 {
        self.row().block_size
    }

    /// Whether a Scribe load can hand out tensors of this type. Quantized
    /// types are the blocks in `kernels`; plain numeric types have no block
    /// struct, and their bytes come from `Gguf::tensor_bytes`.
    pub(crate) fn scribe_loads(self) -> bool {
        self.row().scribe_loads
    }

    /// Bytes per block.
    pub fn type_size(self) -> u64 {
        self.row().type_size
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_in_enum_order_with_unique_ids_and_names() {
        for (index, row) in TABLE.iter().enumerate() {
            assert_eq!(row.ty as usize, index, "{}", row.name);
            assert_eq!(GgmlType::from_id(row.id), Some(row.ty));
            assert_eq!(row.ty.id(), row.id);
            assert_eq!(
                TABLE.iter().filter(|r| r.id == row.id).count(),
                1,
                "id {}",
                row.id
            );
            assert_eq!(
                TABLE.iter().filter(|r| r.name == row.name).count(),
                1,
                "{}",
                row.name
            );
        }
    }

    #[test]
    fn removed_and_out_of_range_ids_are_unknown() {
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 43, u32::MAX] {
            assert_eq!(GgmlType::from_id(id), None, "id {id}");
        }
    }

    #[test]
    fn spot_check_sizes_against_ggml_common() {
        assert_eq!(
            (GgmlType::Q8_0.block_size(), GgmlType::Q8_0.type_size()),
            (32, 34)
        );
        assert_eq!(
            (GgmlType::Q4K.block_size(), GgmlType::Q4K.type_size()),
            (256, 144)
        );
        assert_eq!(
            (GgmlType::Q6K.block_size(), GgmlType::Q6K.type_size()),
            (256, 210)
        );
        assert_eq!(
            (GgmlType::Nvfp4.block_size(), GgmlType::Nvfp4.type_size()),
            (64, 36)
        );
        assert_eq!(
            (GgmlType::Q1_0.block_size(), GgmlType::Q1_0.type_size()),
            (128, 18)
        );
        assert_eq!(
            (GgmlType::Q2_0.block_size(), GgmlType::Q2_0.type_size()),
            (64, 18)
        );
    }
}
