//! The reference dump format: a directory of raw little-endian `f32` files
//! plus an `index.tsv` naming them.
//!
//! Each index line is `name<TAB>ggml_type<TAB>ne0,ne1,ne2,ne3<TAB>file`,
//! with dimensions in ggml order (`ne0` varies fastest). `ggml_type` records
//! the type the tensor had inside llama.cpp; the file always holds `f32`.
//! Logits are the tensor `logits` with `ne0 = n_vocab`, `ne1 = n_tokens`.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// One line of `index.tsv`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub ggml_type: String,
    pub ne: [usize; 4],
    pub file: String,
}

impl Entry {
    pub fn len(&self) -> usize {
        self.ne.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A tensor read from a dump.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub ne: [usize; 4],
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum DumpError {
    Io(PathBuf, std::io::Error),
    Index {
        line: usize,
        reason: String,
    },
    Missing(String),
    Size {
        name: String,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for DumpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DumpError::Io(p, e) => write!(f, "dump: {}: {e}", p.display()),
            DumpError::Index { line, reason } => write!(f, "dump: index.tsv line {line}: {reason}"),
            DumpError::Missing(name) => write!(f, "dump: no tensor named `{name}`"),
            DumpError::Size {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "dump: `{name}` has {actual} bytes, its shape needs {expected}"
                )
            }
        }
    }
}

impl std::error::Error for DumpError {}

/// An opened dump directory.
#[derive(Debug, Clone)]
pub struct Dump {
    pub dir: PathBuf,
    pub entries: Vec<Entry>,
}

impl Dump {
    pub fn open(dir: impl AsRef<Path>) -> Result<Dump, DumpError> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join("index.tsv");
        let text = fs::read_to_string(&path).map_err(|e| DumpError::Io(path, e))?;
        let entries = parse_index(&text)?;
        Ok(Dump { dir, entries })
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn read(&self, name: &str) -> Result<Tensor, DumpError> {
        let entry = self
            .entry(name)
            .ok_or_else(|| DumpError::Missing(name.to_string()))?;
        let path = self.dir.join(&entry.file);
        let bytes = fs::read(&path).map_err(|e| DumpError::Io(path, e))?;
        let expected = entry.len() * 4;
        if bytes.len() != expected {
            return Err(DumpError::Size {
                name: name.to_string(),
                expected,
                actual: bytes.len(),
            });
        }
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(Tensor { ne: entry.ne, data })
    }
}

pub fn parse_index(text: &str) -> Result<Vec<Entry>, DumpError> {
    let mut entries = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let bad = |reason: &str| DumpError::Index {
            line: i + 1,
            reason: reason.to_string(),
        };
        let fields: Vec<&str> = line.split('\t').collect();
        let [name, ggml_type, dims, file] = fields[..] else {
            return Err(bad("expected 4 tab-separated fields"));
        };
        let dims: Vec<usize> = dims
            .split(',')
            .map(|d| d.parse().map_err(|_| bad("dimension is not a number")))
            .collect::<Result<_, _>>()?;
        let ne: [usize; 4] = dims.try_into().map_err(|_| bad("expected 4 dimensions"))?;
        if file.contains('/') || file.contains("..") {
            return Err(bad("file must be a plain name inside the dump directory"));
        }
        entries.push(Entry {
            name: name.to_string(),
            ggml_type: ggml_type.to_string(),
            ne,
            file: file.to_string(),
        });
    }
    Ok(entries)
}

/// Writes `tensors` as a dump in `dir`, in the format [`Dump::open`] reads.
pub fn write(dir: impl AsRef<Path>, tensors: &[(&str, &Tensor)]) -> std::io::Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let mut index = String::new();
    for (name, t) in tensors {
        let file = format!("{name}.f32");
        let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(dir.join(&file), bytes)?;
        let [a, b, c, d] = t.ne;
        index.push_str(&format!("{name}\tf32\t{a},{b},{c},{d}\t{file}\n"));
    }
    fs::write(dir.join("index.tsv"), index)
}
