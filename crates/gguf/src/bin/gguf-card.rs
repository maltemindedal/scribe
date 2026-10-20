//! `gguf-card dump` prints header-card JSON.
//! `gguf-card accept` reads an SWR1 witness on stdin and prints the same JSON.
//! `gguf-card header` writes the header bytes, up to the tensor blob.
#![deny(unsafe_code)]

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
    let file = map_file(path)?;
    let model = gguf::Gguf::parse(file.bytes())?;
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
    let file = map_file(path)?;
    let model = gguf::Gguf::parse(file.bytes())?;
    let end = usize::try_from(model.tensor_data_offset())?;
    let bytes = file
        .bytes()
        .get(..end)
        .ok_or("tensor data offset past end of file")?;
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(bytes)?;
    stdout.flush()?;
    Ok(())
}

#[allow(unsafe_code)]
fn map_file(path: &str) -> Result<gguf::GgufFile, Box<dyn std::error::Error>> {
    // `open` is unsafe because a later truncation of this path is SIGBUS.
    // This process only reads the map, then drops it.
    let file = unsafe { gguf::GgufFile::open(path) }?;
    Ok(file)
}
