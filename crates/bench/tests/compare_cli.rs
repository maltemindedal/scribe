//! The `compare` binary fails when the test dump is incomplete or holds
//! non-finite values, instead of reporting success.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use bench::dump::{self, Tensor};

fn dir(tag: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!(
        "scribe-compare-cli-{}-{tag}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn logits(values: Vec<f32>) -> Tensor {
    Tensor {
        ne: [3, values.len() / 3, 1, 1],
        data: values,
    }
}

fn act() -> Tensor {
    Tensor {
        ne: [2, 1, 1, 1],
        data: vec![1.0, -1.0],
    }
}

fn run(reference: &Path, test: &Path) -> i32 {
    Command::new(env!("CARGO_BIN_EXE_compare"))
        .args([reference.as_os_str(), test.as_os_str()])
        .args(["--max-kld", "0", "--min-same-top", "100"])
        .status()
        .unwrap()
        .code()
        .unwrap()
}

fn reference() -> PathBuf {
    let r = dir("ref");
    dump::write(
        &r,
        &[
            ("l_out-0", &act()),
            ("logits", &logits(vec![0.0, 1.0, 2.0])),
        ],
    )
    .unwrap();
    r
}

#[test]
fn identical_dumps_pass() {
    let r = reference();
    assert_eq!(run(&r, &r), 0);
}

#[test]
fn a_test_dump_missing_an_activation_fails() {
    let (r, t) = (reference(), dir("no-act"));
    dump::write(&t, &[("logits", &logits(vec![0.0, 1.0, 2.0]))]).unwrap();
    assert_eq!(run(&r, &t), 1);
}

#[test]
fn a_test_dump_missing_logits_fails() {
    let (r, t) = (reference(), dir("no-logits"));
    dump::write(&t, &[("l_out-0", &act())]).unwrap();
    assert_eq!(run(&r, &t), 1);
}

#[test]
fn an_empty_test_dump_fails() {
    let (r, t) = (reference(), dir("empty-test"));
    dump::write(&t, &[]).unwrap();
    assert_eq!(run(&r, &t), 1);
}

#[test]
fn an_empty_reference_fails() {
    let r = dir("empty-ref");
    dump::write(&r, &[]).unwrap();
    assert_eq!(run(&r, &r), 1);
}

#[test]
fn nan_or_infinite_test_logits_fail() {
    let r = reference();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let t = dir("nonfinite");
        dump::write(
            &t,
            &[
                ("l_out-0", &act()),
                ("logits", &logits(vec![0.0, bad, 2.0])),
            ],
        )
        .unwrap();
        assert_eq!(run(&r, &t), 1, "logit {bad}");
    }
}

#[test]
fn negative_infinity_only_in_the_reference_fails() {
    let (r, t) = (dir("neg-inf-ref"), dir("neg-inf-test"));
    dump::write(
        &r,
        &[
            ("l_out-0", &act()),
            ("logits", &logits(vec![f32::NEG_INFINITY, 1.0, 2.0])),
        ],
    )
    .unwrap();
    dump::write(
        &t,
        &[
            ("l_out-0", &act()),
            ("logits", &logits(vec![0.0, 1.0, 2.0])),
        ],
    )
    .unwrap();
    assert_eq!(run(&r, &t), 1);
}

#[test]
fn a_nan_activation_fails() {
    let (r, t) = (reference(), dir("nan-act"));
    let bad = Tensor {
        ne: [2, 1, 1, 1],
        data: vec![1.0, f32::NAN],
    };
    dump::write(
        &t,
        &[("l_out-0", &bad), ("logits", &logits(vec![0.0, 1.0, 2.0]))],
    )
    .unwrap();
    assert_eq!(run(&r, &t), 1);
}
