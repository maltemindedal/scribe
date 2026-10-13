//! KL divergence between two sets of logits, as llama.cpp's
//! `llama-perplexity --kl-divergence` reports it: per position
//! `KL(P_ref || P_test)` over the softmax of each row, then the mean, plus
//! the share of positions whose arg-max token agrees.

/// Summary over all positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KldStats {
    pub positions: usize,
    pub mean_kld: f64,
    pub max_kld: f64,
    /// Percentage of positions where both arg-max tokens are the same.
    pub same_top_pct: f64,
}

/// A logit that is NaN or infinite. KLD over such a row is undefined, so the
/// comparison refuses it rather than report a number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonFinite {
    pub in_reference: bool,
    pub index: usize,
    pub value: f32,
}

impl std::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let which = if self.in_reference {
            "reference"
        } else {
            "test"
        };
        write!(f, "kld: {which} logit {} is {}", self.index, self.value)
    }
}

impl std::error::Error for NonFinite {}

/// `reference` and `test` hold `positions` rows of `n_vocab` logits each.
/// Every logit must be finite.
pub fn kld(reference: &[f32], test: &[f32], n_vocab: usize) -> Result<KldStats, NonFinite> {
    assert!(n_vocab > 0, "kld: n_vocab is zero");
    assert_eq!(reference.len(), test.len(), "kld: lengths differ");
    assert_eq!(
        reference.len() % n_vocab,
        0,
        "kld: length is not a multiple of n_vocab"
    );
    for (in_reference, values) in [(true, reference), (false, test)] {
        if let Some(index) = values.iter().position(|v| !v.is_finite()) {
            return Err(NonFinite {
                in_reference,
                index,
                value: values[index],
            });
        }
    }
    let positions = reference.len() / n_vocab;
    let (mut sum, mut max, mut same) = (0.0f64, 0.0f64, 0usize);
    for (r, t) in reference
        .chunks_exact(n_vocab)
        .zip(test.chunks_exact(n_vocab))
    {
        let lr = log_softmax(r);
        let lt = log_softmax(t);
        let k: f64 = lr.iter().zip(&lt).map(|(a, b)| a.exp() * (a - b)).sum();
        // Rounding can give a tiny negative value for identical rows. The
        // inputs are finite, so `k` is too.
        let k = if k < 0.0 { 0.0 } else { k };
        sum += k;
        max = max.max(k);
        if argmax(r) == argmax(t) {
            same += 1;
        }
    }
    Ok(KldStats {
        positions,
        mean_kld: if positions == 0 {
            0.0
        } else {
            sum / positions as f64
        },
        max_kld: max,
        same_top_pct: if positions == 0 {
            100.0
        } else {
            100.0 * same as f64 / positions as f64
        },
    })
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let lse = max
        + row
            .iter()
            .map(|&v| (f64::from(v) - max).exp())
            .sum::<f64>()
            .ln();
    row.iter().map(|&v| f64::from(v) - lse).collect()
}

/// The first index of the largest value by `f32::total_cmp`, so a NaN is
/// ordered rather than skipped (`kld` rejects NaN before it gets here). On a
/// near-tie the winner can flip under tiny noise, which is why same-top is a
/// percentage and not a hard requirement.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best
}

/// A deterministic relative perturbation, `v * (1 + rel * s)` with `s` a
/// pseudo-random sign per element. The comparison self-tests use it to show
/// the checks can fail.
pub fn perturb(values: &[f32], rel: f32) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    values
        .iter()
        .map(|&v| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            v * (1.0 + rel * sign)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::argmax;

    #[test]
    fn argmax_does_not_skip_nan() {
        // `>` is false against NaN, so the old loop returned 0 for both rows.
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 5.0]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1, "first of a tie");
    }
}
