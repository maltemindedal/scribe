# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's
**Security** tab and choose **Report a vulnerability**
(<https://github.com/maltemindedal/scribe/security/advisories/new>). Do not
open a public issue for a suspected vulnerability.

Include the commit, your OS and architecture, and the smallest input that
reproduces the problem. For a malformed model file, attach the file or the
bytes that trigger the problem.

## Supported versions

Scribe has no tagged releases. Only the current `main` branch receives fixes.

## Scope

In scope: crashes, hangs, excessive memory allocation or out-of-bounds reads
reachable from a malformed GGUF file passed to `gguf::Gguf::parse`, which
treats its input as hostile (see the
[GGUF parser reference](docs/reference/gguf-parser.md#validation-rules)).

Scribe has no network-facing code yet.
