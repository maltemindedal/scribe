# 0004: Confine mmap to mapfile, and make open a safe function

- **Status:** Accepted. Supersedes the `unsafe fn open` decision, the
  `gguf` → `memmap2` dependency and the `gguf` mmap-module exception of
  [ADR 0003](0003-mmap-and-align-1-blocks.md).
- **Source:** [issue #70](https://github.com/maltemindedal/scribe/issues/70).
  The maintainer approved a safe `open` on 2026-11-12.

## Context

[ADR 0003](0003-mmap-and-align-1-blocks.md) made `GgufFile::open` an
`unsafe fn` and allowed `unsafe` in the mmap module of `gguf`. Two things
followed from that.

`gguf` could not have `#![forbid(unsafe_code)]`, because `forbid` cannot be
relaxed for one module. It had `#![deny(unsafe_code)]`, and `deny` yields to
any `#[allow(unsafe_code)]`. Two changes got through that way:

- Commit `a4a59ca` stored the parsed header inside `GgufFile` by
  transmuting `Gguf<'_>` to `Gguf<'static>`. The mmap module already
  allowed `unsafe`, so nothing flagged it. Commit `fc39d2e` removed it.
- Commit `6ed7601` added `#[allow(unsafe_code)]` to a function in the
  `gguf-card` binary so that it could call `open`.

Every caller of `open` also had to write an `unsafe` block. The obligation
behind it is that nothing truncates or writes the file while it is mapped.
That is a fact about other processes and the file system. No code at the call
site can check it or make it true, so the nine blocks in the tests carried no
proof, and each new caller outside `kernels` and `gguf` would have needed an
exception to the `unsafe` rule.

Cargo's lint tables have two properties that matter here. Both were measured
on Rust 1.99:

- A member cannot override one key of `[workspace.lints]`. A manifest with
  `[lints] workspace = true` and its own `[lints.rust]` table fails to load.
- A member with no `[lints]` table gets none of the workspace lints, and
  Cargo prints nothing.

## Decision

- A new crate, `mapfile`, holds the mapping. It depends on `memmap2` 0.9.11
  and on no workspace crate. Its public surface is `Mapped`, which derefs to
  `&[u8]`, with `open(path, populate)`, `advise_willneed`,
  `advise_populate_read`, `advise_hugepage` and `lock`. The `mmap` call in
  `Mapped::open` is the crate's one `unsafe` block.
- `gguf` depends on `kernels` and `mapfile`, not on `memmap2`, and has
  `#![forbid(unsafe_code)]`.
- `Mapped::open`, `GgufFile::open` and `GgufFile::open_with` are safe
  functions. Their docs have a "File contract" section that states the
  obligation: truncation delivers `SIGBUS`, and a write is undefined
  behaviour for the slices handed out. The functions are safe because the
  `unsafe` keyword would ask each caller for a proof that no caller can
  give. The requirement itself is unchanged, and the safe signature is unsound
  in the strict sense: safe code plus a hostile or careless writer to the file
  can cause undefined behaviour.
- The workspace sets `unsafe_code = "forbid"` in `[workspace.lints.rust]`.
  Cargo passes it on the command line of every target of an inheriting
  member, so `#[allow(unsafe_code)]` anywhere in such a crate is error E0453.
- `kernels` and `mapfile` are the two crates allowed `unsafe`. They cannot
  inherit the workspace tables, so each repeats them with `unsafe_code` left
  out. `crates/xtask/tests/lint_policy.rs` fails when a member neither has
  exactly `[lints] workspace = true` nor is one of those two crates with
  tables equal to the workspace's, and when the workspace stops forbidding
  `unsafe_code` or gains a member outside `crates/*`.
- The `Block` with an unknown id that `gguf`'s tests need is
  `kernels::testing::UnknownIdBlock`, behind the `kernels` feature `testing`.
  `gguf` enables that feature as a dev-dependency only. A test in `gguf` can
  no longer write `unsafe impl Block`.

## Consequences

- Replaying the two commits above now fails to compile. The `a4a59ca`
  transmute is "usage of an `unsafe` block" under `forbid`, and the
  `6ed7601` attribute is E0453.
- The self-referential `GgufFile` has nowhere to live. `gguf` can name
  `Gguf` and cannot write `unsafe`. `mapfile` can write `unsafe` and cannot
  name `Gguf`, because a dependency on `gguf` would be a cycle and Cargo
  rejects it.
- A search for `unsafe` no longer finds the places that map a file. The
  "File contract" sections are the record. A caller must not map a file
  that something else rewrites in place. Replacing a model by writing a new
  file and renaming it over the old one is safe, because the mapping keeps
  the old file alive.
- A reviewer of `unsafe` reads two crates: `kernels` (the block cast and the
  `Block` impls) and `mapfile` (one block).
- Adding a lint to the workspace means adding it to `kernels` and `mapfile`
  too. The lint-policy test fails until all three agree.
- The workspace has eleven crates. `mapfile`, and so `gguf`, builds only
  for Unix targets, for the reason ADR 0003 gives.
