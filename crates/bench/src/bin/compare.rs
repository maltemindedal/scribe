//! Compare a test dump against a reference dump.
//!
//! Usage: compare <REF_DIR> <TEST_DIR> [--perturb REL] [--max-kld X] [--max-max-kld X]
//!                [--min-same-top PCT]
//!
//! Prints NMSE and its band for every tensor in both dumps, then mean KLD and
//! same-top-token % for `logits`. `--perturb REL` compares the reference with
//! a perturbed copy of itself (TEST_DIR is ignored and may be `-`), which is
//! how the self-test shows the check can fail. Exits 1 when a tensor is in the
//! `bad` band (NaN included), a reference tensor or the logits are missing
//! from the test dump, the reference is empty, a logit is not finite, or a
//! `--max-kld` (mean), `--max-max-kld` (worst position) or `--min-same-top`
//! bound is broken. Same-top can flip on near-tied logits, so keep its bound
//! below 100 when comparing different implementations.
#![forbid(unsafe_code)]

use std::process::ExitCode;

use bench::compare::{Band, compare};
use bench::dump::{Dump, Tensor};
use bench::kld::{kld, perturb};

const USAGE: &str = "usage: compare <REF_DIR> <TEST_DIR> [--perturb REL] [--max-kld X] [--max-max-kld X] [--min-same-top PCT]";

struct Args {
    reference: String,
    test: String,
    perturb: Option<f32>,
    max_kld: Option<f64>,
    max_max_kld: Option<f64>,
    min_same_top: Option<f64>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let mut pos = Vec::new();
    let (mut perturb, mut max_kld, mut max_max_kld, mut min_same_top) = (None, None, None, None);
    while let Some(a) = it.next() {
        let mut num = |flag: &str| -> Result<f64, String> {
            it.next()
                .and_then(|v| v.parse().ok())
                .ok_or(format!("{flag} needs a number"))
        };
        match a.as_str() {
            "--perturb" => perturb = Some(num("--perturb")? as f32),
            "--max-kld" => max_kld = Some(num("--max-kld")?),
            "--max-max-kld" => max_max_kld = Some(num("--max-max-kld")?),
            "--min-same-top" => min_same_top = Some(num("--min-same-top")?),
            "-h" | "--help" => return Err(USAGE.to_string()),
            _ if a.starts_with("--") => return Err(format!("unknown option {a}\n{USAGE}")),
            _ => pos.push(a),
        }
    }
    let [reference, test] = <[String; 2]>::try_from(pos).map_err(|_| USAGE.to_string())?;
    Ok(Args {
        reference,
        test,
        perturb,
        max_kld,
        max_max_kld,
        min_same_top,
    })
}

fn run(args: &Args) -> Result<bool, Box<dyn std::error::Error>> {
    let reference = Dump::open(&args.reference)?;
    let test = if args.perturb.is_some() {
        None
    } else {
        Some(Dump::open(&args.test)?)
    };
    let load = |name: &str, r: &Tensor| -> Result<Option<Tensor>, Box<dyn std::error::Error>> {
        match (&test, args.perturb) {
            (_, Some(rel)) => Ok(Some(Tensor {
                ne: r.ne,
                data: perturb(&r.data, rel),
            })),
            (Some(t), None) if t.entry(name).is_some() => Ok(Some(t.read(name)?)),
            _ => Ok(None),
        }
    };

    let mut ok = true;
    if reference.entries.is_empty() {
        println!("reference dump {} is empty", args.reference);
        ok = false;
    }
    println!("{:<20} {:>12} {:>12}  band", "tensor", "max_abs", "nmse");
    for entry in reference.entries.iter().filter(|e| e.name != "logits") {
        let r = reference.read(&entry.name)?;
        let Some(t) = load(&entry.name, &r)? else {
            println!("{:<20} missing from the test dump", entry.name);
            ok = false;
            continue;
        };
        if t.ne != r.ne {
            println!(
                "{:<20} shape {:?} != reference {:?}",
                entry.name, t.ne, r.ne
            );
            ok = false;
            continue;
        }
        let s = compare(&r.data, &t.data);
        let band = Band::of(s.nmse);
        ok &= band != Band::Bad;
        println!(
            "{:<20} {:>12.3e} {:>12.3e}  {}",
            entry.name,
            s.max_abs,
            s.nmse,
            band.describe()
        );
    }

    if reference.entry("logits").is_some() {
        let r = reference.read("logits")?;
        match load("logits", &r)? {
            Some(t) if t.ne == r.ne => {
                let k = match kld(&r.data, &t.data, r.ne[0]) {
                    Ok(k) => k,
                    Err(e) => {
                        println!("logits: {e}");
                        return Ok(false);
                    }
                };
                println!(
                    "logits: positions={} mean_kld={:.6e} max_kld={:.6e} same_top={:.2}%",
                    k.positions, k.mean_kld, k.max_kld, k.same_top_pct
                );
                if let Some(max) = args.max_kld {
                    let pass = k.mean_kld <= max;
                    println!(
                        "check mean_kld <= {max:e}: {}",
                        if pass { "PASS" } else { "FAIL" }
                    );
                    ok &= pass;
                }
                if let Some(max) = args.max_max_kld {
                    let pass = k.max_kld <= max;
                    println!(
                        "check max_kld <= {max:e}: {}",
                        if pass { "PASS" } else { "FAIL" }
                    );
                    ok &= pass;
                }
                if let Some(min) = args.min_same_top {
                    let pass = k.same_top_pct >= min;
                    println!(
                        "check same_top >= {min}%: {}",
                        if pass { "PASS" } else { "FAIL" }
                    );
                    ok &= pass;
                }
            }
            Some(t) => {
                println!("logits: shape {:?} != reference {:?}", t.ne, r.ne);
                ok = false;
            }
            None => {
                println!("logits: missing from the test dump");
                ok = false;
            }
        }
    } else if args.max_kld.is_some() || args.max_max_kld.is_some() || args.min_same_top.is_some() {
        println!("logits: missing from the reference dump, so the KLD bounds cannot be checked");
        ok = false;
    }
    Ok(ok)
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
