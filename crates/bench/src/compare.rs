//! Elementwise tensor comparison: max absolute difference and NMSE, with the
//! bands from llama.cpp's `check-nmse.py`.

/// Comparison of a test tensor against a reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub max_abs: f64,
    /// `sum((test - ref)^2) / sum(ref^2)`. Zero when identical; infinite when
    /// the reference is all zeros and the test is not.
    pub nmse: f64,
}

pub fn compare(reference: &[f32], test: &[f32]) -> Stats {
    assert_eq!(reference.len(), test.len(), "compare: lengths differ");
    let (mut err, mut norm, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for (&r, &t) in reference.iter().zip(test) {
        let d = f64::from(t) - f64::from(r);
        err += d * d;
        norm += f64::from(r) * f64::from(r);
        // f64::max drops NaN, which would hide a NaN in either input.
        max_abs = if d.is_nan() || max_abs.is_nan() {
            f64::NAN
        } else {
            max_abs.max(d.abs())
        };
    }
    // A NaN NMSE lands in `Band::Bad`: every band test is a `<`.
    let nmse = if err == 0.0 { 0.0 } else { err / norm };
    Stats { max_abs, nmse }
}

/// The `check-nmse.py` band an NMSE falls in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    Identical,
    Excellent,
    VeryGood,
    Good,
    Bad,
}

impl Band {
    pub fn of(nmse: f64) -> Band {
        if nmse < 1e-6 {
            Band::Identical
        } else if nmse < 1e-4 {
            Band::Excellent
        } else if nmse < 1e-3 {
            Band::VeryGood
        } else if nmse < 1e-2 {
            Band::Good
        } else {
            Band::Bad
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Band::Identical => "essentially identical (< 1e-6)",
            Band::Excellent => "excellent (< 1e-4)",
            Band::VeryGood => "very good (< 1e-3)",
            Band::Good => {
                "good (< 1e-2); 1e-3 to 1e-2 usually means a slightly wrong epsilon, theta or scale"
            }
            Band::Bad => "bad (>= 1e-2); usually a layout bug",
        }
    }
}
