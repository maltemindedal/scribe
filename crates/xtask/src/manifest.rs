//! Parser for `models.toml`: a deliberately tiny TOML subset (`[[model]]`
//! tables holding `key = "string"` and `key = integer` lines), so the xtask
//! needs no dependencies.

use std::collections::BTreeMap;

pub const MANIFEST: &str = include_str!("../models.toml");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Set {
    Ci,
    All,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Model {
    pub name: String,
    pub repo: String,
    pub file: String,
    pub dest: String,
    pub revision: String,
    pub sha256: String,
    pub size: u64,
    pub license: String,
    pub role: String,
    pub set: Set,
}

impl Model {
    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, self.file
        )
    }
}

/// Models belonging to `set`; `All` includes every model, `Ci` only the CI ones.
pub fn select(models: &[Model], set: Set) -> Vec<&Model> {
    models
        .iter()
        .filter(|m| set == Set::All || m.set == Set::Ci)
        .collect()
}

pub fn parse(text: &str) -> Result<Vec<Model>, String> {
    let mut tables: Vec<(usize, BTreeMap<String, Value>)> = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[model]]" {
            tables.push((line_no, BTreeMap::new()));
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {line_no}: expected `key = value`"))?;
        let table = &mut tables
            .last_mut()
            .ok_or_else(|| format!("line {line_no}: key before the first [[model]]"))?
            .1;
        let value = value.trim();
        let value = if let Some(s) = value.strip_prefix('"') {
            Value::Str(
                s.strip_suffix('"')
                    .ok_or_else(|| format!("line {line_no}: unterminated string"))?
                    .to_string(),
            )
        } else {
            Value::Int(
                value
                    .parse()
                    .map_err(|_| format!("line {line_no}: expected a string or integer"))?,
            )
        };
        if table.insert(key.trim().to_string(), value).is_some() {
            return Err(format!("line {line_no}: duplicate key `{}`", key.trim()));
        }
    }
    tables.into_iter().map(|(l, t)| build(l, t)).collect()
}

#[derive(Debug)]
enum Value {
    Str(String),
    Int(u64),
}

fn build(line: usize, mut t: BTreeMap<String, Value>) -> Result<Model, String> {
    let mut string = |key: &str| match t.remove(key) {
        Some(Value::Str(s)) => Ok(s),
        Some(Value::Int(_)) => Err(format!(
            "[[model]] at line {line}: `{key}` must be a string"
        )),
        None => Err(format!("[[model]] at line {line}: missing `{key}`")),
    };
    let name = string("name")?;
    let repo = string("repo")?;
    let file = string("file")?;
    let dest = string("dest")?;
    let revision = string("revision")?;
    let sha256 = string("sha256")?;
    let license = string("license")?;
    let role = string("role")?;
    let set = match string("set")?.as_str() {
        "ci" => Set::Ci,
        "all" => Set::All,
        other => return Err(format!("{name}: unknown set `{other}`")),
    };
    let size = match t.remove("size") {
        Some(Value::Int(n)) => n,
        _ => return Err(format!("{name}: `size` must be an integer")),
    };
    if let Some(extra) = t.keys().next() {
        return Err(format!("{name}: unknown key `{extra}`"));
    }
    let hex = sha256.len() == 64
        && sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if !hex {
        return Err(format!("{name}: `sha256` must be 64 lowercase hex digits"));
    }
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{name}: `revision` must be a 40-digit commit hash"));
    }
    if dest.contains('/') || dest.contains('\\') || dest.starts_with('.') {
        return Err(format!("{name}: `dest` must be a plain file name"));
    }
    Ok(Model {
        name,
        repo,
        file,
        dest,
        revision,
        sha256,
        size,
        license,
        role,
        set,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_manifest_parses_with_expected_sets() {
        let models = parse(MANIFEST).unwrap();
        assert_eq!(models.len(), 6);
        let ci: Vec<_> = select(&models, Set::Ci)
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(
            ci,
            [
                "stories260K",
                "stories15M-q8_0",
                "SmolLM2-135M-Instruct-Q8_0",
                "Qwen3-0.6B-Q8_0"
            ]
        );
        assert_eq!(select(&models, Set::All).len(), 6);
        assert!(models[3].url().ends_with(
            "Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
        ));
    }

    fn one(replace: (&str, &str)) -> Result<Vec<Model>, String> {
        let first = MANIFEST.split("[[model]]").nth(1).unwrap();
        parse(&format!("[[model]]{}", first.replace(replace.0, replace.1)))
    }

    #[test]
    fn rejects_a_malformed_hash() {
        assert!(one(("047bf464", "047bf46")).is_err());
    }

    #[test]
    fn rejects_unknown_keys_and_missing_keys() {
        assert!(one(("size =", "sizes =")).is_err());
        assert!(one(("set = \"ci\"", "extra = 1\nset = \"ci\"")).is_err());
    }

    #[test]
    fn rejects_a_path_traversing_dest() {
        assert!(one(("dest = \"stories260K.gguf\"", "dest = \"../x\"")).is_err());
    }
}
