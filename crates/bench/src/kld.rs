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

/// `reference` and `test` hold `positions` rows of `n_vocab` logits each.
pub fn kld(reference: &[f32], test: &[f32], n_vocab: usize) -> KldStats {
    assert!(n_vocab > 0, "kld: n_vocab is zero");
    assert_eq!(reference.len(), test.len(), "kld: lengths differ");
    assert_eq!(
        reference.len() % n_vocab,
        0,
        "kld: length is not a multiple of n_vocab"
    );
    let positions = reference.len() / n_vocab;
    let (mut sum, mut max, mut same) = (0.0f64, 0.0f64, 0usize);
    for (r, t) in reference
        .chunks_exact(n_vocab)
        .zip(test.chunks_exact(n_vocab))
    {
        let lr = log_softmax(r);
        let lt = log_softmax(t);
        let k: f64 = lr.iter().zip(&lt).map(|(a, b)| a.exp() * (a - b)).sum();
        // Rounding can give a tiny negative value for identical rows.
        let k = k.max(0.0);
        sum += k;
        max = max.max(k);
        if argmax(r) == argmax(t) {
            same += 1;
        }
    }
    KldStats {
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
    }
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

fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
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
