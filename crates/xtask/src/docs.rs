//! `cargo xtask docs-check [root]`: prose that restates a fact it does not
//! own, or that disagrees with the code, fails. The same check runs as a unit
//! test, so `cargo test --workspace` enforces it with no CI change.
//!
//! Three rules, std only:
//!
//! 1. Single owner. Each [`Fact`] names the one file that may state where its
//!    terms apply (a term and a platform word in one paragraph). Any other
//!    file links to the owner or says nothing platform-specific.
//! 2. Dependencies. A clause "`A` depends on `B`" must match the
//!    `[dependencies]` of `crates/*/Cargo.toml`, or `Cargo.lock` for a package
//!    outside the workspace. `AGENTS.md` and the architecture overview must
//!    also list every dependency of each crate they discuss. A claim that no
//!    third-party crate exists is false once one does. Architecture decision
//!    records are exempt: they record what was true when they were written.
//! 3. Ignored tests. A count of ignored tests in prose equals the number of
//!    `#[ignore` attributes under `crates/`.
//!
//! Prose means Markdown outside code fences, `//` comments in `.rs` files and
//! `#` comments in `.sh` files.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// A fact that one file owns. Adding a fact is one entry here.
struct Fact {
    owner: &'static str,
    terms: &'static [&'static str],
}

/// The `mmap` flag and `madvise` facts. They are owned by the `mapfile`
/// rustdoc because the `cfg` that decides them sits beside it.
const FACTS: &[Fact] = &[Fact {
    owner: "crates/mapfile/src/lib.rs",
    terms: &[
        "MAP_POPULATE",
        "populate: true",
        "MADV_POPULATE_READ",
        "MADV_HUGEPAGE",
    ],
}];

/// A paragraph with one of a fact's terms and one of these words states where
/// the term applies.
const PLATFORM_WORDS: &[&str] = &[
    "Linux",
    "Android",
    "Unix",
    "macOS",
    "Windows",
    "operating system",
];

/// Files whose dependency sentences must also list every dependency of each
/// crate they discuss.
const COMPLETE_FILES: &[&str] = &["AGENTS.md", "docs/architecture/overview.md"];

const SKIP_DIRS: &[&str] = &["target", "research", "models"];
const HISTORICAL_DIR: &str = "docs/architecture/decisions/";
const READ_EXTENSIONS: &[&str] = &["md", "rs", "sh", "toml", "lock"];

pub struct SourceFile {
    path: String,
    text: String,
}

pub fn run(root: &Path) -> Result<(), String> {
    let findings = check(&load(root)?);
    if findings.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} docs finding(s)\n{}",
        findings.len(),
        findings.join("\n")
    ))
}

fn load(root: &Path) -> Result<Vec<SourceFile>, String> {
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn walk(dir: &Path, root: &Path, out: &mut Vec<SourceFile>) -> Result<(), String> {
    let io = |p: &Path, e: std::io::Error| format!("{}: {e}", p.display());
    for entry in fs::read_dir(dir).map_err(|e| io(dir, e))? {
        let path = entry.map_err(|e| io(dir, e))?.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if path.is_dir() {
            let hidden = name.starts_with('.') && name != ".agents";
            if !hidden && !SKIP_DIRS.contains(&name.as_str()) {
                walk(&path, root, out)?;
            }
            continue;
        }
        let wanted = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| READ_EXTENSIONS.contains(&e));
        if !wanted {
            continue;
        }
        let text = fs::read_to_string(&path).map_err(|e| io(&path, e))?;
        let rel = path.strip_prefix(root).unwrap_or(&path);
        out.push(SourceFile {
            path: rel.to_string_lossy().into_owned(),
            text,
        });
    }
    Ok(())
}

struct Paragraph {
    line: usize,
    text: String,
}

/// The prose of one file, a paragraph at a time. A blank line, a table row, a
/// list item, a heading, or a line that is not prose ends the paragraph.
fn paragraphs(path: &str, text: &str) -> Vec<Paragraph> {
    let mut out = Vec::new();
    let mut current: Option<Paragraph> = None;
    let mut in_fence = false;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let prose = if path.ends_with(".md") {
            if line.starts_with("```") {
                in_fence = !in_fence;
                None
            } else {
                (!in_fence).then_some(line)
            }
        } else if path.ends_with(".rs") {
            line.strip_prefix("//")
                .map(|s| s.trim_start_matches(['/', '!']).trim())
        } else if path.ends_with(".sh") {
            line.strip_prefix('#').map(str::trim)
        } else {
            None
        };
        let starts_block = prose.is_none_or(|p| {
            p.is_empty() || ["|", "- ", "* ", "#"].iter().any(|b| p.starts_with(b))
        });
        if starts_block {
            out.extend(current.take());
        }
        let Some(p) = prose.filter(|p| !p.is_empty()) else {
            continue;
        };
        match &mut current {
            Some(par) => {
                par.text.push(' ');
                par.text.push_str(p);
            }
            None => {
                current = Some(Paragraph {
                    line: i + 1,
                    text: p.to_string(),
                });
            }
        }
    }
    out.extend(current);
    out
}

/// Backtick-quoted spans of a sentence.
fn ticked(s: &str) -> Vec<&str> {
    s.split('`').skip(1).step_by(2).collect()
}

/// Direct dependencies per package. Workspace crates come from their
/// `[dependencies]`; packages outside the workspace come from `Cargo.lock`.
struct DepGraph {
    members: BTreeSet<String>,
    deps: BTreeMap<String, BTreeSet<String>>,
    /// Dev- and build-dependencies of workspace crates.
    tooling: BTreeMap<String, BTreeSet<String>>,
}

impl DepGraph {
    fn new(files: &[SourceFile]) -> Self {
        let mut graph = DepGraph {
            members: BTreeSet::new(),
            deps: BTreeMap::new(),
            tooling: BTreeMap::new(),
        };
        for f in files {
            if f.path.starts_with("crates/") && f.path.ends_with("/Cargo.toml") {
                let (name, deps, tooling) = manifest_deps(&f.path, &f.text);
                graph.members.insert(name.clone());
                graph.deps.insert(name.clone(), deps);
                graph.tooling.insert(name, tooling);
            }
        }
        if let Some(lock) = files.iter().find(|f| f.path == "Cargo.lock") {
            graph.add_lock(&lock.text);
        }
        graph
    }

    fn add_lock(&mut self, lock: &str) {
        let (mut name, mut in_deps) = (String::new(), false);
        for l in lock.lines().map(str::trim) {
            if let Some(rest) = l.strip_prefix("name = ") {
                name = rest.trim_matches('"').to_string();
                in_deps = false;
                if !self.members.contains(&name) {
                    self.deps.entry(name.clone()).or_default();
                }
            } else if l.starts_with("dependencies = [") {
                in_deps = true;
            } else if l == "]" {
                in_deps = false;
            } else if in_deps && !self.members.contains(&name) {
                let dep = l.trim_matches([',', '"']).split(' ').next().unwrap_or("");
                self.deps
                    .entry(name.clone())
                    .or_default()
                    .insert(dep.to_string());
            }
        }
    }

    /// Every package a workspace crate reaches that is not a workspace crate.
    fn third_party(&self) -> BTreeSet<&str> {
        let mut seen = BTreeSet::new();
        let mut todo: Vec<&str> = self.members.iter().map(String::as_str).collect();
        while let Some(name) = todo.pop() {
            let direct = self.deps.get(name).into_iter().flatten();
            let tooling = self.tooling.get(name).into_iter().flatten();
            for dep in direct.chain(tooling) {
                if !self.members.contains(dep) && seen.insert(dep.as_str()) {
                    todo.push(dep);
                }
            }
        }
        seen
    }
}

/// Package name, normal dependencies, and dev- and build-dependencies from one
/// `Cargo.toml`.
fn manifest_deps(path: &str, text: &str) -> (String, BTreeSet<String>, BTreeSet<String>) {
    let (mut section, mut name) = (None, path.to_string());
    let (mut deps, mut tooling) = (BTreeSet::new(), BTreeSet::new());
    let mut in_package = false;
    for l in text.lines().map(str::trim) {
        if let Some(header) = l.strip_prefix('[') {
            let header = header.trim_end_matches(']');
            in_package = header == "package";
            section = dep_table(header);
            if let Some((kind, Some(dep))) = &section {
                kind.set(&mut deps, &mut tooling).insert(dep.to_string());
            }
        } else if in_package && l.starts_with("name") {
            name = l.split('"').nth(1).unwrap_or(path).to_string();
        } else if let Some((kind, None)) = &section
            && !l.is_empty()
            && !l.starts_with('#')
            && let Some((key, _)) = l.split_once('=')
        {
            let key = key.split('.').next().unwrap_or(key);
            kind.set(&mut deps, &mut tooling)
                .insert(key.trim().to_string());
        }
    }
    (name, deps, tooling)
}

#[derive(Clone, Copy)]
enum DepKind {
    Normal,
    Tooling,
}

impl DepKind {
    fn set<'a>(
        self,
        normal: &'a mut BTreeSet<String>,
        tooling: &'a mut BTreeSet<String>,
    ) -> &'a mut BTreeSet<String> {
        match self {
            DepKind::Normal => normal,
            DepKind::Tooling => tooling,
        }
    }
}

/// A dependency table header, with or without a `target.<cfg>.` prefix: the
/// kind of table, and the dependency name for `[dependencies.name]`.
fn dep_table(header: &str) -> Option<(DepKind, Option<&str>)> {
    for (word, kind) in [
        ("dependencies", DepKind::Normal),
        ("dev-dependencies", DepKind::Tooling),
        ("build-dependencies", DepKind::Tooling),
    ] {
        if header == word {
            return Some((kind, None));
        }
        if let Some(rest) = header.strip_prefix(word).and_then(|r| r.strip_prefix('.')) {
            return Some((kind, Some(rest)));
        }
        if header.starts_with("target.") {
            let needle = format!(".{word}");
            if let Some(at) = header.rfind(&needle) {
                let rest = &header[at + needle.len()..];
                match rest.strip_prefix('.') {
                    Some(dep) => return Some((kind, Some(dep))),
                    None if rest.is_empty() => return Some((kind, None)),
                    None => {}
                }
            }
        }
    }
    None
}

/// One "`subject` depends on `a`, `b`" clause.
#[derive(Debug, PartialEq)]
struct Claim {
    subject: String,
    objects: Vec<String>,
    /// The clause also says "no workspace crate".
    no_workspace: bool,
}

const DEPENDS: &str = " depends on ";

/// The dependency clauses of a paragraph. A clause needs a backticked subject
/// right before "depends on", or "which" to continue the previous object.
/// Anything else ("No other crate depends on ...") is not parsed.
fn claims(para: &str) -> Vec<Claim> {
    let marks: Vec<usize> = para.match_indices(DEPENDS).map(|(i, _)| i).collect();
    let mut out: Vec<Claim> = Vec::new();
    for (k, &at) in marks.iter().enumerate() {
        let before = &para[..at];
        let subject = if let Some(inner) = before.strip_suffix('`') {
            inner.rsplit_once('`').map(|(_, name)| name.to_string())
        } else if before.ends_with("which") {
            out.last().and_then(|c| c.objects.last().cloned())
        } else {
            None
        };
        let Some(subject) = subject else { continue };
        let next = marks.get(k + 1).copied();
        let raw = &para[at + DEPENDS.len()..next.unwrap_or(para.len())];
        let (tail, cut_at_sentence) = match raw.split_once(". ") {
            Some((t, _)) => (t, true),
            None => (raw, false),
        };
        let mut objects: Vec<String> = ticked(tail).into_iter().map(str::to_string).collect();
        let names_next_subject = !cut_at_sentence && tail.trim_end().ends_with('`');
        if next.is_some() && names_next_subject {
            objects.pop();
        }
        out.push(Claim {
            subject,
            objects,
            no_workspace: tail.contains("no workspace crate"),
        });
    }
    out.extend(used_by_clauses(para));
    out
}

/// "`a` and `b`, which `S` uses": the packages before "which" are what `S`
/// depends on.
fn used_by_clauses(para: &str) -> Vec<Claim> {
    let mut out = Vec::new();
    for (at, _) in para.match_indices("which `") {
        let rest = &para[at + "which `".len()..];
        let Some((subject, after)) = rest.split_once('`') else {
            continue;
        };
        if !after.starts_with(" uses ") {
            continue;
        }
        let sentence = para[..at].rsplit(". ").next().unwrap_or("");
        out.push(Claim {
            subject: subject.to_string(),
            objects: ticked(sentence).into_iter().map(str::to_string).collect(),
            no_workspace: false,
        });
    }
    out
}

fn number_word(w: &str) -> Option<usize> {
    const WORDS: [&str; 13] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve",
    ];
    w.parse()
        .ok()
        .or_else(|| WORDS.iter().position(|x| *x == w))
}

/// The numbers a paragraph claims as an ignored-test count: a number, then
/// "ignored" or "ignore" a few words later with no other number between. The
/// word "one" needs "test" next, so a pronoun is not a count.
fn ignored_counts(para: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for sentence in para.split(". ") {
        let words: Vec<String> = sentence
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .collect();
        for (i, n) in words
            .iter()
            .enumerate()
            .filter_map(|(i, w)| number_word(w).map(|n| (i, n)))
        {
            let after = &words[i + 1..];
            if words[i] == "one" && !after.first().is_some_and(|w| w.starts_with("test")) {
                continue;
            }
            let window = &after[..after.len().min(6)];
            let hit = window
                .iter()
                .take_while(|w| number_word(w).is_none())
                .any(|w| w == "ignored" || w == "ignore");
            if hit {
                out.push(n);
            }
        }
    }
    out
}

fn count_ignored(files: &[SourceFile]) -> usize {
    files
        .iter()
        .filter(|f| f.path.starts_with("crates/") && f.path.ends_with(".rs"))
        .flat_map(|f| f.text.lines())
        .filter(|l| l.trim_start().starts_with("#[ignore"))
        .count()
}

pub fn check(files: &[SourceFile]) -> Vec<String> {
    let mut found = Vec::new();
    for fact in FACTS {
        if !files.iter().any(|f| f.path == fact.owner) {
            found.push(format!("{}: owner file is missing", fact.owner));
        }
    }
    let graph = DepGraph::new(files);
    let third_party = graph.third_party();
    let ignored = count_ignored(files);

    for file in files {
        let path = file.path.as_str();
        let mut said: BTreeMap<String, (usize, BTreeSet<String>)> = BTreeMap::new();
        for para in paragraphs(path, &file.text) {
            let (line, text) = (para.line, para.text.as_str());
            for fact in FACTS.iter().filter(|f| f.owner != path) {
                let term = fact.terms.iter().find(|t| text.contains(**t));
                let word = PLATFORM_WORDS.iter().find(|w| text.contains(**w));
                if let (Some(term), Some(word)) = (term, word) {
                    found.push(format!(
                        "{path}:{line}: `{term}` and `{word}` in one paragraph; only {} states where it applies; state it without a platform word and link to it",
                        fact.owner
                    ));
                }
            }

            let lower = text.to_lowercase();
            if lower.contains("no third-party dependencies") && !third_party.is_empty() {
                found.push(format!(
                    "{path}:{line}: says there are no third-party dependencies; Cargo has {third_party:?}"
                ));
            }
            let exhaustive = [
                "no other third-party dependencies",
                "only third-party dependencies",
            ];
            if exhaustive.iter().any(|p| lower.contains(p)) {
                let named = ticked(text);
                let missing: Vec<_> = third_party.iter().filter(|d| !named.contains(d)).collect();
                if !missing.is_empty() {
                    found.push(format!(
                        "{path}:{line}: calls its list the only third-party dependencies but leaves out {missing:?}"
                    ));
                }
            }

            let claims = if path.starts_with(HISTORICAL_DIR) {
                Vec::new()
            } else {
                claims(text)
            };
            for claim in claims {
                let Some(actual) = graph.deps.get(&claim.subject) else {
                    continue;
                };
                let objects: BTreeSet<String> = claim
                    .objects
                    .into_iter()
                    .filter(|o| graph.deps.contains_key(o))
                    .collect();
                let extra: Vec<_> = objects.difference(actual).collect();
                if !extra.is_empty() {
                    found.push(format!(
                        "{path}:{line}: says `{}` depends on {extra:?}; Cargo has {actual:?}",
                        claim.subject
                    ));
                }
                let internal: Vec<_> = actual.intersection(&graph.members).collect();
                if claim.no_workspace && !internal.is_empty() {
                    found.push(format!(
                        "{path}:{line}: says `{}` depends on no workspace crate; Cargo has {internal:?}",
                        claim.subject
                    ));
                }
                let entry = said.entry(claim.subject).or_insert((line, BTreeSet::new()));
                entry.1.extend(objects);
            }

            for n in ignored_counts(text).into_iter().filter(|n| *n != ignored) {
                found.push(format!(
                    "{path}:{line}: says {n} ignored; the source has {ignored} `#[ignore` attributes"
                ));
            }
        }
        if COMPLETE_FILES.contains(&path) {
            for (subject, (line, objects)) in said {
                let missing: Vec<_> = graph.deps[&subject].difference(&objects).collect();
                if !missing.is_empty() {
                    found.push(format!(
                        "{path}:{line}: never says `{subject}` depends on {missing:?}, which Cargo lists"
                    ));
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(items: &[(&str, &str)]) -> Vec<SourceFile> {
        items
            .iter()
            .map(|(path, text)| SourceFile {
                path: path.to_string(),
                text: text.to_string(),
            })
            .collect()
    }

    #[test]
    fn repository_docs_match_the_code() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        if let Err(e) = run(&root) {
            panic!("{e}");
        }
    }

    const OWNER: (&str, &str) = ("crates/mapfile/src/lib.rs", "//! x\n");
    const GGUF: (&str, &str) = (
        "crates/gguf/Cargo.toml",
        "[package]\nname = \"gguf\"\n[dependencies]\nkernels = { path = \"../kernels\" }\nmapfile.workspace = true\n",
    );
    const KERNELS: (&str, &str) = (
        "crates/kernels/Cargo.toml",
        "[package]\nname = \"kernels\"\n",
    );
    const MAPFILE: (&str, &str) = (
        "crates/mapfile/Cargo.toml",
        "[package]\nname = \"mapfile\"\n[dependencies]\nmemmap2 = \"0.9\"\n",
    );
    const LOCK: (&str, &str) = (
        "Cargo.lock",
        "[[package]]\nname = \"memmap2\"\ndependencies = [\n \"libc\",\n]\n[[package]]\nname = \"libc\"\n[[package]]\nname = \"mapfile\"\ndependencies = [\n \"stale\",\n]\n",
    );

    fn tree<'a>(doc: &'a str, doc_path: &'a str) -> Vec<SourceFile> {
        files(&[OWNER, GGUF, KERNELS, MAPFILE, LOCK, (doc_path, doc)])
    }

    #[test]
    fn restating_a_fact_outside_its_owner_fails() {
        let bad = tree("On Linux, `MAP_POPULATE` prefaults.\n", "docs/a.md");
        let found = check(&bad);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("docs/a.md:1:"), "{found:?}");
    }

    #[test]
    fn a_link_to_the_owner_does_not_excuse_a_platform_word() {
        let linked =
            "`MAP_POPULATE` on Linux: see [`Mapped::open`](../crates/mapfile/src/lib.rs).\n";
        assert_eq!(check(&tree(linked, "docs/a.md")).len(), 1);
        let free = "`MAP_POPULATE` prefaults the file; see [`Mapped::open`](../crates/mapfile/src/lib.rs).\n\nLinux is supported.\n";
        assert!(check(&tree(free, "docs/a.md")).is_empty());
    }

    #[test]
    fn the_owner_may_state_its_own_fact_and_comments_are_checked() {
        let owner = (
            "crates/mapfile/src/lib.rs",
            "//! `MAP_POPULATE` on Linux.\n",
        );
        let mut tree = tree("", "docs/a.md");
        tree.retain(|f| f.path != OWNER.0);
        tree.extend(files(&[owner]));
        assert!(check(&tree).is_empty());
        let rs = (
            "crates/a/src/lib.rs",
            "/// Pass `MAP_POPULATE` on Linux.\npub fn f() {}\n",
        );
        tree.extend(files(&[rs]));
        assert_eq!(check(&tree).len(), 1);
        let sh = ("a.sh", "# populate: true is Linux only\n");
        assert_eq!(check(&files(&[OWNER, sh])).len(), 1);
    }

    #[test]
    fn a_missing_owner_file_fails() {
        let found = check(&files(&[("docs/a.md", "x\n")]));
        assert_eq!(found, ["crates/mapfile/src/lib.rs: owner file is missing"]);
    }

    #[test]
    fn dependency_sentences_must_match_cargo() {
        let ok = "`gguf` depends on `kernels` and `mapfile`. `mapfile` depends on `memmap2`, and `memmap2` depends on `libc`.\n";
        assert!(check(&tree(ok, "docs/a.md")).is_empty());
        let wrong = "`gguf` depends on `libc`.\n";
        let found = check(&tree(wrong, "docs/a.md"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("depends on [\"libc\"]"), "{found:?}");
    }

    #[test]
    fn the_which_and_no_workspace_phrasings_are_read() {
        let ok =
            "`mapfile` depends on `memmap2` (which depends on `libc`) and on no workspace crate.\n";
        assert!(check(&tree(ok, "docs/a.md")).is_empty());
        let wrong = "`gguf` depends on `mapfile` (which depends on `kernels`).\n";
        assert_eq!(check(&tree(wrong, "docs/a.md")).len(), 1);
        let internal = "`gguf` depends on no workspace crate.\n";
        assert_eq!(check(&tree(internal, "docs/a.md")).len(), 1);
    }

    #[test]
    fn a_package_list_followed_by_which_a_crate_uses_is_read() {
        let wrong = "The build downloads `memmap2` and `libc`, which `gguf` uses to map files.\n";
        let found = check(&tree(wrong, "docs/a.md"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("\"libc\""), "{found:?}");
        let right = "The build downloads `memmap2`, which `mapfile` uses to map files.\n";
        assert!(check(&tree(right, "docs/a.md")).is_empty());
    }

    #[test]
    fn complete_files_must_list_every_dependency() {
        let partial = "`gguf` depends on `mapfile`. `mapfile` depends on `memmap2`, and `memmap2` depends on `libc`.\n";
        assert!(check(&tree(partial, "docs/a.md")).is_empty());
        let found = check(&tree(partial, "AGENTS.md"));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            found[0].contains("never says `gguf` depends on [\"kernels\"]"),
            "{found:?}"
        );
        let split = format!("{partial}\n`gguf` depends on `kernels`.\n");
        assert!(check(&tree(&split, "AGENTS.md")).is_empty());
    }

    #[test]
    fn no_third_party_claim_fails_while_one_exists() {
        let found = check(&tree(
            "There are no third-party dependencies.\n",
            "docs/a.md",
        ));
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("\"memmap2\""), "{found:?}");
        let lone = ("crates/gguf/Cargo.toml", "[package]\nname = \"gguf\"\n");
        let none = files(&[OWNER, lone, ("docs/a.md", "No third-party dependencies.\n")]);
        assert!(check(&none).is_empty());
    }

    #[test]
    fn dev_and_build_dependencies_are_third_party_but_not_claimed_edges() {
        for table in [
            "[dev-dependencies]\ntempfile = \"3\"\n",
            "[build-dependencies]\ntempfile = \"3\"\n",
            "[target.'cfg(unix)'.dev-dependencies]\ntempfile = \"3\"\n",
            "[dev-dependencies.tempfile]\nversion = \"3\"\n",
        ] {
            let bench = (
                "crates/bench/Cargo.toml",
                &*format!("[package]\nname = \"bench\"\n{table}"),
            );
            let none = files(&[
                OWNER,
                bench,
                ("docs/a.md", "No third-party dependencies.\n"),
            ]);
            let found = check(&none);
            assert_eq!(found.len(), 1, "{table}: {found:?}");
            assert!(found[0].contains("tempfile"), "{found:?}");
        }
        let manifest = format!("{}[dev-dependencies]\ntempfile = \"3\"\n", GGUF.1);
        let claim = "`gguf` depends on `kernels` and `mapfile`.\n";
        let tree = files(&[
            OWNER,
            ("crates/gguf/Cargo.toml", &manifest),
            KERNELS,
            MAPFILE,
            LOCK,
            ("docs/a.md", claim),
        ]);
        assert!(check(&tree).is_empty());
    }

    #[test]
    fn a_closed_third_party_list_must_name_each_one() {
        let short = "No other third-party dependencies than `memmap2`.\n";
        assert_eq!(check(&tree(short, "docs/a.md")).len(), 1);
        let full = "No other third-party dependencies than `memmap2` and `libc`.\n";
        assert!(check(&tree(full, "docs/a.md")).is_empty());
    }

    #[test]
    fn ignored_count_must_match_source() {
        let src = ("crates/a/tests/t.rs", "#[ignore]\nfn t() {}\n");
        let doc = |text| files(&[OWNER, src, ("README.md", text)]);
        assert_eq!(check(&doc("Two tests are ignored.\n")).len(), 1);
        assert_eq!(check(&doc("with 2 ignored tests\n")).len(), 1);
        assert!(check(&doc("One test is ignored.\n")).is_empty());
        let result = "`ok. 0 passed; 0 failed; 1 ignored` and the one that is ignored\n";
        assert!(check(&doc(result)).is_empty());
        let fenced = "```\n7 tests ignored\n```\n";
        assert!(check(&doc(fenced)).is_empty());
    }

    #[test]
    fn claims_parse_the_phrasings_in_use() {
        let c = claims("`a` depends on `b` and on `c`, and `c` depends on `d`.");
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].objects, ["b", "c"]);
        assert_eq!(
            (c[1].subject.as_str(), c[1].objects.as_slice()),
            ("c", ["d".to_string()].as_slice())
        );
        assert!(claims("No other crate depends on `kernels`.").is_empty());
    }
}
