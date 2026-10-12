//! The ggml tensor type table.
//!
//! Ids, block sizes and byte sizes are copied from `ggml.h`, `ggml.c`
//! (`type_traits`) and the `static_assert`s in `ggml-common.h`, not from
//! `gguf.md`, which lags behind. Ids 4, 5 and 31 to 38 were removed from
//! ggml and are rejected like any other unknown id.

/// A ggml tensor element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
    Q2_0,
}

/// `(type, id, name, elements per block, bytes per block)`.
const TABLE: &[(GgmlType, u32, &str, u64, u64)] = {
    use GgmlType::*;
    &[
        (F32, 0, "f32", 1, 4),
        (F16, 1, "f16", 1, 2),
        (Q4_0, 2, "q4_0", 32, 18),
        (Q4_1, 3, "q4_1", 32, 20),
        (Q5_0, 6, "q5_0", 32, 22),
        (Q5_1, 7, "q5_1", 32, 24),
        (Q8_0, 8, "q8_0", 32, 34),
        (Q8_1, 9, "q8_1", 32, 36),
        (Q2K, 10, "q2_K", 256, 84),
        (Q3K, 11, "q3_K", 256, 110),
        (Q4K, 12, "q4_K", 256, 144),
        (Q5K, 13, "q5_K", 256, 176),
        (Q6K, 14, "q6_K", 256, 210),
        (Q8K, 15, "q8_K", 256, 292),
        (Iq2Xxs, 16, "iq2_xxs", 256, 66),
        (Iq2Xs, 17, "iq2_xs", 256, 74),
        (Iq3Xxs, 18, "iq3_xxs", 256, 98),
        (Iq1S, 19, "iq1_s", 256, 50),
        (Iq4Nl, 20, "iq4_nl", 32, 18),
        (Iq3S, 21, "iq3_s", 256, 110),
        (Iq2S, 22, "iq2_s", 256, 82),
        (Iq4Xs, 23, "iq4_xs", 256, 136),
        (I8, 24, "i8", 1, 1),
        (I16, 25, "i16", 1, 2),
        (I32, 26, "i32", 1, 4),
        (I64, 27, "i64", 1, 8),
        (F64, 28, "f64", 1, 8),
        (Iq1M, 29, "iq1_m", 256, 56),
        (Bf16, 30, "bf16", 1, 2),
        (Tq1_0, 34, "tq1_0", 256, 54),
        (Tq2_0, 35, "tq2_0", 256, 66),
        (Mxfp4, 39, "mxfp4", 32, 17),
        (Nvfp4, 40, "nvfp4", 64, 36),
        (Q1_0, 41, "q1_0", 128, 18),
        (Q2_0, 42, "q2_0", 64, 18),
    ]
};

impl GgmlType {
    /// Looks up a ggml type id. Unknown and removed ids return `None`.
    pub fn from_id(id: u32) -> Option<Self> {
        TABLE.iter().find(|row| row.1 == id).map(|row| row.0)
    }

    fn row(self) -> &'static (GgmlType, u32, &'static str, u64, u64) {
        TABLE
            .iter()
            .find(|row| row.0 == self)
            .expect("every GgmlType has a row; checked by ids_round_trip_and_are_unique")
    }

    /// The numeric id stored in GGUF files.
    pub fn id(self) -> u32 {
        self.row().1
    }

    /// The name ggml uses, e.g. `"q8_0"`.
    pub fn name(self) -> &'static str {
        self.row().2
    }

    /// Elements per block (1 for plain types such as f32).
    pub fn block_size(self) -> u64 {
        self.row().3
    }

    /// Bytes per block.
    pub fn type_size(self) -> u64 {
        self.row().4
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
    fn ids_round_trip_and_are_unique() {
        for &(ty, id, ..) in TABLE {
            assert_eq!(GgmlType::from_id(id), Some(ty));
            assert_eq!(ty.id(), id);
            assert_eq!(TABLE.iter().filter(|r| r.0 == ty).count(), 1, "{ty}");
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
