//! The comparison checks are able to fail: identical inputs give zero,
//! perturbed inputs do not.

use bench::compare::{Band, compare};
use bench::dump::{self, Dump, Tensor};
use bench::kld::{kld, perturb};

fn logits(n_vocab: usize, positions: usize) -> Vec<f32> {
    (0..n_vocab * positions)
        .map(|i| ((i * 7919) % 1013) as f32 / 101.3 - 5.0)
        .collect()
}

#[test]
fn kld_of_logits_against_themselves_is_zero() {
    let l = logits(512, 9);
    let k = kld(&l, &l, 512).unwrap();
    assert_eq!(k.positions, 9);
    assert_eq!(k.mean_kld, 0.0);
    assert_eq!(k.same_top_pct, 100.0);
}

#[test]
fn kld_of_perturbed_logits_is_positive() {
    let l = logits(512, 9);
    let k = kld(&l, &perturb(&l, 1e-2), 512).unwrap();
    assert!(k.mean_kld > 0.0, "mean_kld {}", k.mean_kld);
}

#[test]
fn kld_matches_a_hand_computed_value() {
    // P = softmax([0, ln 3]) = [0.25, 0.75]; Q = [0.5, 0.5].
    // KL = 0.25 ln 0.5 + 0.75 ln 1.5.
    let p = [0.0, 3f32.ln()];
    let q = [0.0, 0.0];
    let expected = 0.25 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
    let k = kld(&p, &q, 2).unwrap();
    assert!(
        (k.mean_kld - expected).abs() < 1e-7,
        "{} vs {expected}",
        k.mean_kld
    );
}

#[test]
fn same_top_counts_positions_whose_argmax_moves() {
    let r = [1.0, 0.0, 0.0, 1.0];
    let t = [1.0, 0.0, 1.0, 0.0];
    assert_eq!(kld(&r, &t, 2).unwrap().same_top_pct, 50.0);
}

#[test]
fn nmse_puts_a_1e_3_relative_perturbation_in_a_non_identical_band() {
    let v = logits(1000, 1);
    assert_eq!(compare(&v, &v).nmse, 0.0);
    let s = compare(&v, &perturb(&v, 1e-3));
    assert!((s.nmse - 1e-6).abs() < 1e-7, "nmse {}", s.nmse);
    assert_eq!(Band::of(s.nmse), Band::Excellent);
    assert_eq!(Band::of(compare(&v, &perturb(&v, 0.2)).nmse), Band::Bad);
}

#[test]
fn a_written_dump_reads_back_identically() {
    let dir = std::env::temp_dir().join(format!("scribe-bench-dump-{}", std::process::id()));
    let t = Tensor {
        ne: [3, 2, 1, 1],
        data: vec![1.0, -2.5, 3.25, 0.0, f32::MIN_POSITIVE, 7.0],
    };
    dump::write(&dir, &[("l_out-0", &t)]).unwrap();
    let d = Dump::open(&dir).unwrap();
    assert_eq!(d.read("l_out-0").unwrap(), t);
    assert!(d.read("missing").is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn index_rejects_paths_outside_the_dump() {
    assert!(dump::parse_index("x\tf32\t1,1,1,1\t../etc/passwd\n").is_err());
    assert!(dump::parse_index("x\tf32\t1,1,1\tx.f32\n").is_err());
}

/// Uses the llama.cpp dumps that `tools/llama.cpp/run.sh` writes.
#[test]
#[ignore = "needs models/reference/llama.cpp dumps (tools/llama.cpp/run.sh)"]
fn llama_cpp_logits_have_zero_kld_with_themselves_and_not_with_a_perturbed_copy() {
    let root = std::env::var_os("SCRIBE_MODELS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
        });
    for model in ["stories15M-q8_0", "Qwen3-0.6B-Q8_0"] {
        let d = Dump::open(root.join("reference/llama.cpp").join(model).join("17")).unwrap();
        let l = d.read("logits").unwrap();
        assert_eq!(
            kld(&l.data, &l.data, l.ne[0]).unwrap().mean_kld,
            0.0,
            "{model}"
        );
        assert!(
            kld(&l.data, &perturb(&l.data, 1e-3), l.ne[0])
                .unwrap()
                .mean_kld
                > 0.0,
            "{model}"
        );
    }
}

#[test]
fn kld_rejects_non_finite_logits() {
    let good = [0.0f32, 1.0, 2.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let b = [0.0, bad, 2.0];
        assert!(bench::kld::kld(&good, &b, 3).is_err(), "test {bad}");
        assert!(bench::kld::kld(&b, &good, 3).is_err(), "reference {bad}");
    }
    assert!(bench::kld::kld(&good, &good, 3).is_ok());
}

#[test]
fn compare_rejects_non_finite_values() {
    assert!(compare(&[1.0, 2.0], &[1.0, f32::NAN]).max_abs.is_nan());
    assert_eq!(
        Band::of(compare(&[1.0, 2.0], &[1.0, f32::NAN]).nmse),
        Band::Bad
    );
}
