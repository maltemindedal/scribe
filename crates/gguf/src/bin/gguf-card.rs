#![forbid(unsafe_code)]

use std::io::{Read, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next();
    let path = args.next();
    if args.next().is_some() {
        return Err(usage().into());
    }
    match (cmd.as_deref(), path) {
        (Some("accept"), None) => accept(),
        (Some("dump"), Some(path)) => dump(&path),
        (Some("header"), Some(path)) => header(&path),
        _ => Err(usage().into()),
    }
}

fn usage() -> &'static str {
    "usage: gguf-card dump <file> | gguf-card header <file> | gguf-card accept"
}

fn dump(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let model = gguf::Gguf::parse(&bytes)?;
    let json = gguf::card::to_json(&model);
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(json.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

fn accept() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes)?;
    let json = gguf::card::json_from_witness(&bytes)?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(json.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

fn header(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let model = gguf::Gguf::parse(&bytes)?;
    let end = usize::try_from(model.tensor_data_offset())?;
    let header = bytes
        .get(..end)
        .ok_or("tensor data offset past end of file")?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(header)?;
    stdout.flush()?;
    Ok(())
}
