//! `cargo xtask pr-check`: checks that a pull request body names a merge
//! method that fits the pull request's commits.
//!
//! [`check`] is a pure function over the body, the commits and the added
//! lines of Markdown files. [`gather`] is the thin shell that reads those
//! from git.

use std::fmt;
use std::process::Command;

const HINT: &str =
    "Use a merge commit if unsure: it is the only method that keeps the commit hashes.";

const COMMIT_TYPES: [&str; 10] = [
    "feat", "fix", "docs", "test", "perf", "refactor", "ci", "build", "chore", "bench",
];

/// Phrases that present a merge method as a repository-wide default. There
/// is none: each pull request chooses.
const CONVENTION_PHRASES: [&str; 7] = [
    "how this repository merges",
    "how this repo merges",
    "how the repository merges",
    "how the repo merges",
    "by convention",
    "repository convention",
    "repo convention",
];

const MIN_REASON_WORDS: usize = 4;
const HASH_PREFIX_LEN: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub sha: String,
    pub subject: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Merge,
    Squash,
    Rebase,
}

impl Method {
    fn parse(word: &str) -> Option<Self> {
        match word {
            "merge" => Some(Self::Merge),
            "squash" => Some(Self::Squash),
            "rebase" => Some(Self::Rebase),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Squash => "squash",
            Self::Rebase => "rebase",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    MissingSection,
    UnknownMethod(String),
    ReasonTooShort {
        words: usize,
    },
    RepoConvention(&'static str),
    RebaseNeedsOneCommit {
        commits: usize,
    },
    RebaseSubjectFormat(String),
    MissingSquashMessage,
    SquashSubjectFormat(String),
    CitesOwnCommit {
        method: &'static str,
        prefix: String,
        in_body: bool,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSection => write!(
                f,
                "the PR body has no `## Merge method` section. Start it with merge, squash or rebase and give the reason."
            ),
            Self::UnknownMethod(word) if word.is_empty() => write!(
                f,
                "the `## Merge method` section is empty. Start it with merge, squash or rebase and give the reason."
            ),
            Self::UnknownMethod(word) => write!(
                f,
                "the `## Merge method` section must start with merge, squash or rebase, not {word:?}."
            ),
            Self::ReasonTooShort { words } => write!(
                f,
                "the `## Merge method` reason has {words} word(s); say in at least {MIN_REASON_WORDS} what this PR's commits are."
            ),
            Self::RepoConvention(phrase) => write!(
                f,
                "the `## Merge method` reason says {phrase:?}. This repository has no default method; say what this PR's commits are."
            ),
            Self::RebaseNeedsOneCommit { commits } => write!(
                f,
                "rebase needs exactly one commit and this PR has {commits}."
            ),
            Self::RebaseSubjectFormat(subject) => write!(
                f,
                "rebase keeps the commit message as is and {subject:?} is not `type(scope): subject` ({types}); use squash with your own message.",
                types = COMMIT_TYPES.join("|")
            ),
            Self::MissingSquashMessage => write!(
                f,
                "squash needs a `Squash message:` line followed by a fenced block holding the combined commit message."
            ),
            Self::SquashSubjectFormat(subject) => write!(
                f,
                "the first line of the squash message, {subject:?}, is not `type(scope): subject` ({types}).",
                types = COMMIT_TYPES.join("|")
            ),
            Self::CitesOwnCommit {
                method,
                prefix,
                in_body,
            } => write!(
                f,
                "{where_} cites {prefix}, a commit of this PR, and {method} drops that hash from main.",
                where_ = if *in_body {
                    "the PR body"
                } else {
                    "a Markdown file in the diff"
                }
            ),
        }?;
        write!(f, " {HINT}")
    }
}

/// `type(scope)!: subject`, the format of every commit since #58.
pub fn is_commit_format(subject: &str) -> bool {
    let Some((head, rest)) = subject.split_once(": ") else {
        return false;
    };
    if !rest.chars().next().is_some_and(|c| !c.is_whitespace()) {
        return false;
    }
    let head = head.strip_suffix('!').unwrap_or(head);
    let (kind, scope) = match head.split_once('(') {
        Some((kind, scope)) => (kind, Some(scope)),
        None => (head, None),
    };
    if !COMMIT_TYPES.contains(&kind) {
        return false;
    }
    scope.is_none_or(|scope| {
        scope.strip_suffix(')').is_some_and(|names| {
            !names.is_empty()
                && names
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | ',' | '-'))
        })
    })
}

/// The decision. `commits` are the PR's own non-merge commits, and
/// `added_md_lines` the lines its diff adds to Markdown files.
pub fn check(body: &str, commits: &[Commit], added_md_lines: &[String]) -> Vec<Error> {
    let body = strip_generated(body);
    let Some(section) = section(&body, "Merge method") else {
        return vec![Error::MissingSection];
    };
    let Section {
        first_word,
        reason,
        squash_subject,
    } = parse_section(&section);
    let Some(method) = Method::parse(&first_word.to_lowercase()) else {
        return vec![Error::UnknownMethod(first_word)];
    };

    let mut errors = Vec::new();
    let words = reason_words(&reason, method);
    if words < MIN_REASON_WORDS {
        errors.push(Error::ReasonTooShort { words });
    }
    let flat = reason
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if let Some(phrase) = CONVENTION_PHRASES.iter().find(|p| flat.contains(*p)) {
        errors.push(Error::RepoConvention(phrase));
    }

    match method {
        Method::Merge => {}
        Method::Rebase => {
            if let [only] = commits {
                if !is_commit_format(&only.subject) {
                    errors.push(Error::RebaseSubjectFormat(only.subject.clone()));
                }
            } else {
                errors.push(Error::RebaseNeedsOneCommit {
                    commits: commits.len(),
                });
            }
        }
        Method::Squash => match squash_subject {
            None => errors.push(Error::MissingSquashMessage),
            Some(subject) if !is_commit_format(&subject) => {
                errors.push(Error::SquashSubjectFormat(subject));
            }
            Some(_) => {}
        },
    }

    if method != Method::Merge {
        for commit in commits {
            let prefix: String = commit.sha.chars().take(HASH_PREFIX_LEN).collect();
            let cited_in = |text: &str| cites(text, &prefix);
            let in_body = cited_in(&body);
            if in_body || added_md_lines.iter().any(|line| cited_in(line)) {
                errors.push(Error::CitesOwnCommit {
                    method: method.name(),
                    prefix,
                    in_body,
                });
            }
        }
    }
    errors
}

/// Drops what tooling adds to a body: HTML comments and the Cursor Bugbot
/// summary, which names the head commit and is rewritten on every push.
fn strip_generated(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    let mut skipping_summary = false;
    while let Some(start) = rest.find("<!--") {
        let (before, after) = rest.split_at(start);
        if !skipping_summary {
            out.push_str(before);
        }
        let Some(end) = after.find("-->") else {
            rest = "";
            break;
        };
        let marker = after[4..end].trim();
        if marker == "CURSOR_SUMMARY" {
            skipping_summary = true;
        } else if marker == "/CURSOR_SUMMARY" {
            skipping_summary = false;
        }
        rest = &after[end + 3..];
    }
    if !skipping_summary {
        out.push_str(rest);
    }
    out
}

/// The lines under the level-2 heading `title`, up to the next heading of
/// level 1 or 2. Headings inside code fences do not count.
fn section(body: &str, title: &str) -> Option<String> {
    let mut lines = Vec::new();
    let mut found = false;
    let mut in_fence = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
        }
        let heading = !in_fence && (trimmed.starts_with("# ") || trimmed.starts_with("## "));
        if heading {
            if found {
                break;
            }
            found = trimmed.strip_prefix("##").is_some_and(|t| {
                t.starts_with(' ') && t.trim().trim_end_matches(':').eq_ignore_ascii_case(title)
            });
        } else if found {
            lines.push(line);
        }
    }
    found.then(|| lines.join("\n"))
}

struct Section {
    first_word: String,
    /// Prose outside code fences and the `Squash message:` label.
    reason: String,
    /// First line of the fenced block after `Squash message:`.
    squash_subject: Option<String>,
}

fn parse_section(section: &str) -> Section {
    let mut reason = String::new();
    let mut squash_subject = None;
    let mut fence: Option<bool> = None;
    let mut after_label = false;
    for line in section.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            fence = match fence {
                None => Some(after_label),
                Some(_) => None,
            };
            after_label = false;
            continue;
        }
        match fence {
            Some(is_squash) => {
                if is_squash && squash_subject.is_none() && !trimmed.is_empty() {
                    squash_subject = Some(trimmed.to_string());
                }
            }
            None if trimmed.to_lowercase().starts_with("squash message:") => after_label = true,
            None => {
                if !trimmed.is_empty() {
                    after_label = false;
                }
                reason.push_str(line);
                reason.push('\n');
            }
        }
    }
    let reason = reason.trim_start_matches(|c: char| c.is_whitespace() || "*_-+>#`~".contains(c));
    let word_end = reason
        .find(|c: char| !c.is_alphabetic())
        .unwrap_or(reason.len());
    Section {
        first_word: reason[..word_end].to_string(),
        reason: reason[word_end..].to_string(),
        squash_subject,
    }
}

/// Words in the reason, not counting the `commit` of "Merge commit".
fn reason_words(reason: &str, method: Method) -> usize {
    let mut words = reason
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .peekable();
    if method == Method::Merge
        && words
            .peek()
            .is_some_and(|w| w.eq_ignore_ascii_case("commit"))
    {
        words.next();
    }
    words.count()
}

/// Whether `text` holds a hex word of at least seven digits that starts with
/// `prefix`, so a full 40-digit hash counts and a word that merely contains
/// the digits does not.
fn cites(text: &str, prefix: &str) -> bool {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|word| word.len() >= HASH_PREFIX_LEN && word.starts_with(prefix))
}

/// Added lines of `.md` files in a unified diff taken with `-U0`.
pub fn added_md_lines(diff: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut md = false;
    let mut in_hunk = false;
    for line in diff.lines() {
        if line.starts_with("diff --git ") {
            in_hunk = false;
            md = false;
        } else if line.starts_with("@@") {
            in_hunk = true;
        } else if !in_hunk {
            if let Some(path) = line.strip_prefix("+++ ") {
                md = path.trim_end().ends_with(".md");
            }
        } else if md && let Some(added) = line.strip_prefix('+') {
            lines.push(added.to_string());
        }
    }
    lines
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run `git` (is it installed?): {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The PR's non-merge commits and the Markdown lines its diff adds.
pub fn gather(base: &str, head: &str) -> Result<(Vec<Commit>, Vec<String>), String> {
    let range = format!("{base}..{head}");
    let commits = git(&["rev-list", "--no-merges", &range])?
        .split_whitespace()
        .map(|sha| {
            let subject = git(&["log", "-1", "--format=%s", sha])?;
            Ok(Commit {
                sha: sha.to_string(),
                subject: subject.trim_end().to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let diff = git(&[
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--unified=0",
        &format!("{base}...{head}"),
    ])?;
    Ok((commits, added_md_lines(&diff)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(sha: &str, subject: &str) -> Commit {
        Commit {
            sha: sha.into(),
            subject: subject.into(),
        }
    }

    fn rendered(errors: &[Error]) -> Vec<String> {
        errors.iter().map(ToString::to_string).collect()
    }

    // Excerpts of the bodies of PRs #46 to #67 (maltemindedal/scribe).

    /// #50: the body before the section existed.
    const PRE_59_BODY: &str = "\
## What changed and why

Adds a docs/ structure, root community files and a rewritten README.

## Linked issue

Closes #49

## Checks

- [x] `cargo fmt --all --check`
";

    /// #61.
    const PR_61_BODY: &str = "\
## Linked issue

Closes #9

## Recommended merge: squash

The branch is one issue's review fixes. Squash keeps that as a single commit on main, which is how this repository merges.

## Poteto-mode steps skipped or degraded

- Interrogate was not run.
";

    /// #62, trimmed: the real body also cites `57c6c17` in its checks section,
    /// see `PR_62_CITING_LINE`.
    const PR_62_BODY_TRIMMED: &str = "\
## Linked issue

Closes #60

## Merge method

Rebase. The branch is one commit, and this body is the proof log, so a squash would replace the commit message with that log.

<!-- CURSOR_AGENT_PR_BODY_END -->
";

    /// The line of #62's body that cites its own commit.
    const PR_62_CITING_LINE: &str = "I ran them on `57c6c17` before opening this PR:";

    /// #63.
    const PR_63_BODY: &str = "\
## Scope

Documentation only. Rust source is unchanged, including the comment issues in #60.

## Merge method

Squash. The branch is one docs commit, and the subject is the change.

<!-- CURSOR_AGENT_PR_BODY_END -->
";

    /// #65.
    const PR_65_BODY: &str = "\
## What changed

- `docs/reference/gguf-parser.md` line 45. The `open_with` row names Linux and Android. Commit `c41df10`.
- `docs/reference/configuration.md` line 12. The `SCRIBE_EAGER_READ` table names Linux and Android. Commit `fbdfa2d`.

## Merge method

Merge commit. The first commit is the three sentences from #64. The second commit is the rest of the sites from the SHOULD on #63. Those two commits are worth keeping.

<!-- CURSOR_AGENT_PR_BODY_END -->
";

    const CURSOR_SUMMARY: &str = "
<!-- CURSOR_SUMMARY -->
---

> [!NOTE]
> **Low Risk**
> Reviewed by Cursor Bugbot for commit f4bd8f4dfd905753e49a6a7bbd58f5389942b783.
<!-- /CURSOR_SUMMARY -->
";

    #[test]
    fn body_without_the_section_fails() {
        let commits = [commit("57c6c17aaaa", "docs(gguf): fix comments")];
        let errors = check(PRE_59_BODY, &commits, &[]);
        assert_eq!(errors, [Error::MissingSection]);
        assert!(rendered(&errors)[0].ends_with(HINT));
    }

    #[test]
    fn a_heading_that_is_not_merge_method_does_not_count() {
        let errors = check(PR_61_BODY, &[], &[]);
        assert_eq!(errors, [Error::MissingSection]);
    }

    #[test]
    fn a_repo_wide_convention_reason_fails() {
        let body = "## Merge method\n\nSquash. One issue's review fixes, which is how this repository merges.\n\nSquash message:\n\n```\ndocs: tidy\n```\n";
        let errors = check(body, &[commit("aaaaaaa1", "docs: tidy")], &[]);
        assert_eq!(
            errors,
            [Error::RepoConvention("how this repository merges")]
        );
        for phrase in ["By convention", "repository\nconvention"] {
            let body =
                format!("## Merge method\n\nMerge. The commits are separate changes, {phrase}.\n");
            assert!(
                matches!(check(&body, &[], &[])[..], [Error::RepoConvention(_)]),
                "{phrase}"
            );
        }
    }

    #[test]
    fn rebase_of_one_formatted_commit_passes() {
        let commits = [commit(
            "57c6c17a",
            "docs(gguf): fix stale block, error, and mmap comments",
        )];
        assert_eq!(check(PR_62_BODY_TRIMMED, &commits, &[]), []);
    }

    #[test]
    fn rebase_needs_exactly_one_commit_with_a_formatted_subject() {
        let two = [
            commit("1111111a", "docs: one"),
            commit("2222222b", "docs: two"),
        ];
        assert_eq!(
            check(PR_62_BODY_TRIMMED, &two, &[]),
            [Error::RebaseNeedsOneCommit { commits: 2 }]
        );
        assert_eq!(
            check(PR_62_BODY_TRIMMED, &[], &[]),
            [Error::RebaseNeedsOneCommit { commits: 0 }]
        );
        let plain = [commit("1111111a", "Fix the comments")];
        assert_eq!(
            check(PR_62_BODY_TRIMMED, &plain, &[]),
            [Error::RebaseSubjectFormat("Fix the comments".into())]
        );
    }

    #[test]
    fn squash_without_a_squash_message_fails() {
        let commits = [commit(
            "c95caa9a",
            "docs(gguf): quote the mutation-profile test command",
        )];
        assert_eq!(
            check(PR_63_BODY, &commits, &[]),
            [Error::MissingSquashMessage]
        );
    }

    #[test]
    fn squash_message_is_checked_against_the_format() {
        let squash = |message: &str| {
            format!(
                "## Merge method\n\nSquash. The branch is one docs commit, and the subject is the change.\n\nSquash message:\n\n```text\n{message}\n```\n"
            )
        };
        let commits = [commit("c95caa9a", "wip")];
        assert_eq!(
            check(
                &squash("docs(gguf): quote the command\n\nBody."),
                &commits,
                &[]
            ),
            []
        );
        assert_eq!(
            check(&squash("Quote the command"), &commits, &[]),
            [Error::SquashSubjectFormat("Quote the command".into())]
        );
    }

    #[test]
    fn squash_fix_ups_with_a_block_pass_even_when_the_block_has_headings() {
        let body = "## Merge method\n\nSquash. Both commits are review fixes of one change.\n\nSquash message:\n\n```\nfeat(gguf): map the file\n\n# Why\n\nBecause.\n```\n\n## Next\n";
        let commits = [commit("1111111a", "wip"), commit("2222222b", "wip 2")];
        assert_eq!(check(body, &commits, &[]), []);
    }

    /// Replaces the one occurrence of `from`; a missing pattern fails the test.
    fn replace_once(text: &str, from: &str, to: &str) -> String {
        let (head, tail) = text
            .split_once(from)
            .unwrap_or_else(|| panic!("`{from}` not found"));
        format!("{head}{to}{tail}")
    }

    #[test]
    fn squash_or_rebase_that_cites_its_own_commit_in_the_body_fails() {
        let commits = [commit(
            "57c6c17aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "docs(gguf): fix comments",
        )];
        let body = replace_once(
            PR_62_BODY_TRIMMED,
            "## Merge method",
            "I ran the checks on `57c6c17`.\n\n## Merge method",
        );
        assert_eq!(
            check(&body, &commits, &[]),
            [Error::CitesOwnCommit {
                method: "rebase",
                prefix: "57c6c17".into(),
                in_body: true
            }]
        );
        let full = replace_once(
            &body,
            "`57c6c17`",
            "57c6c17AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        );
        assert_eq!(check(&full, &commits, &[]).len(), 1);
    }

    #[test]
    fn the_real_pr_62_body_fails_because_it_cites_its_rebased_commit() {
        let commits = [commit(
            "57c6c17aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "docs(gguf): fix stale block, error, and mmap comments",
        )];
        let body = format!("{PR_62_BODY_TRIMMED}\n## Checks\n\n{PR_62_CITING_LINE}\n");
        assert_eq!(
            check(&body, &commits, &[]),
            [Error::CitesOwnCommit {
                method: "rebase",
                prefix: "57c6c17".into(),
                in_body: true
            }]
        );
    }

    #[test]
    fn markup_before_the_method_word_is_skipped() {
        for body in [
            "## Merge method\n\n**Merge commit.** Two separate changes here.\n",
            "## Merge method\n\n- Merge commit, two separate changes here.\n",
            "## Merge method\n\n> _Merge_ because two separate changes.\n",
        ] {
            assert_eq!(check(body, &[], &[]), [], "{body}");
        }
    }

    #[test]
    fn as_usual_and_as_always_are_not_convention_claims() {
        let body = "## Merge method\n\nMerge. The two commits are separate changes, as usual for a stack, and as always reviewed in order.\n";
        assert_eq!(check(body, &[], &[]), []);
    }

    #[test]
    fn squash_or_rebase_that_cites_its_own_commit_in_a_doc_fails() {
        let commits = [commit("57c6c17aaaa", "docs(gguf): fix comments")];
        let doc = ["Fixed in 57c6c17 and again in deadbee.".to_string()];
        assert_eq!(
            check(PR_62_BODY_TRIMMED, &commits, &doc),
            [Error::CitesOwnCommit {
                method: "rebase",
                prefix: "57c6c17".into(),
                in_body: false
            }]
        );
        let unrelated = ["See 57c6c1 and 57c6c18 for details.".to_string()];
        assert_eq!(check(PR_62_BODY_TRIMMED, &commits, &unrelated), []);
    }

    #[test]
    fn merge_commit_may_cite_its_hashes() {
        let commits = [
            commit("c41df10aaaa", "docs(gguf): name Linux and Android"),
            commit("fbdfa2dbbbb", "docs: sweep the remaining sites"),
        ];
        let doc = ["Introduced in c41df10.".to_string()];
        assert_eq!(check(PR_65_BODY, &commits, &doc), []);
    }

    #[test]
    fn the_reason_needs_four_words_and_merge_commit_counts_as_one_word() {
        for body in [
            "## Merge method\n\nMerge commit.\n",
            "## Merge method\n\nMerge commit. Two changes.\n",
            "## Merge method\n\nMerge.\n",
        ] {
            assert!(
                matches!(check(body, &[], &[])[..], [Error::ReasonTooShort { .. }]),
                "{body}"
            );
        }
        assert_eq!(
            check(
                "## Merge method\n\nMerge. Two separate changes here.\n",
                &[],
                &[]
            ),
            []
        );
    }

    #[test]
    fn first_word_must_be_a_method() {
        let errors = check("## Merge method\n\nTBD\n", &[], &[]);
        assert_eq!(errors, [Error::UnknownMethod("TBD".into())]);
        let empty = check(
            "## Merge method\n\n<!-- merge, squash or rebase, and why -->\n",
            &[],
            &[],
        );
        assert_eq!(empty, [Error::UnknownMethod(String::new())]);
        assert_eq!(
            check(
                "## Merge method\n\nMerge commit: two separate changes here.\n",
                &[],
                &[]
            ),
            []
        );
    }

    #[test]
    fn the_bugbot_summary_is_not_a_citation() {
        let body = format!("{PR_63_BODY_WITH_BLOCK}{CURSOR_SUMMARY}");
        let commits = [commit("f4bd8f4dfd90", "docs: quote the command")];
        assert_eq!(check(&body, &commits, &[]), []);
        let cited = replace_once(
            &body,
            "<!-- CURSOR_SUMMARY -->",
            "Head is f4bd8f4.\n<!-- CURSOR_SUMMARY -->",
        );
        assert_eq!(check(&cited, &commits, &[]).len(), 1);
    }

    const PR_63_BODY_WITH_BLOCK: &str = "\
## Merge method

Squash. The branch is one docs commit, and the subject is the change.

Squash message:

```
docs(gguf): quote the mutation-profile test command
```
";

    #[test]
    fn commit_format_matches_the_conventional_shape() {
        for ok in [
            "docs(gguf): limit mmap flag notes to Unix targets (#67)",
            "feat(kernels): add align-1 ggml block types",
            "docs: match the mmap notes to the merged code (#59)",
            "fix(a,b-c_d)!: break it",
            "bench: add a case",
        ] {
            assert!(is_commit_format(ok), "{ok}");
        }
        for bad in [
            "Add CI: correctness-only matrix",
            "docs:missing space",
            "docs: ",
            "docs(): empty scope",
            "docs(gguf: unclosed",
            "docs (gguf): space",
            "Docs: capitalised",
            "feature: unknown type",
            "docs(gg uf): space in scope",
            "",
        ] {
            assert!(!is_commit_format(bad), "{bad}");
        }
    }

    #[test]
    fn added_lines_come_from_markdown_files_only() {
        let diff = "\
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 context
+see 57c6c17
+++ not a header
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1 @@
-old 57c6c17
+new 57c6c17
diff --git a/docs/x.md b/docs/x.md
deleted file mode 100644
--- a/docs/x.md
+++ /dev/null
@@ -1 +0,0 @@
-gone
";
        assert_eq!(added_md_lines(diff), ["see 57c6c17", "++ not a header"]);
    }
}
