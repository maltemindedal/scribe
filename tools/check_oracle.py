"""Differential check of tools/tokenizer_oracle.py against llama.cpp.

Needs `pip install llama-cpp-python` (tested with 0.3.36) and a GGUF file; only the
vocabulary is loaded, so any quantisation works. Run from anywhere:

    python3 tools/check_oracle.py --gguf models/Qwen3-0.6B-Q8_0.gguf --pre qwen2
    python3 tools/check_oracle.py --gguf gemma-3-1b-it.gguf --pre spm

`--pre` selects the oracle parameters (see the tokenizer_oracle.py docstring):
qwen2, qwen35, llama3 (BPE) or spm (SentencePiece, no BOS, no dummy prefix).
Exit status is 1 if any string differs. The corpus is fixed strings plus seeded random
strings, so a run is reproducible.
"""

import argparse
import importlib.util
import random
import struct
import sys
from pathlib import Path

PRESETS = {  # --pre -> (max_digits, marks, ignore_merges)
    "qwen2": (1, False, False),
    "qwen35": (1, True, False),
    "llama3": (3, False, True),
}

FIXED = [
    "Hello, world!", "1234567890", "it's  fine\n\nok", "   leading", "trailing   ", "a\n\n\nb",
    "\n", " ", "I'm sure you'll see we've done it; they'd say so.",
    "x = [1, 2, 3]  # comment\n\tdef f(a, b):\n\t\treturn a+b\n",
    "Räksmörgås på svenska: § 3 kap. 2 § brottsbalken, åäö ÅÄÖ", "日本語のテキスト 中文 한국어",
    "emoji 😀👍🏽 🇸🇪 ́ é", "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
    "https://example.com/a/b?c=d&e=f#g", "a" * 300, "!" * 200, ".\n" * 50,
    "  　 mixed space", "0.5% 1,000,000 3.14159", "ﬁ ligature, ß, İ, ǅ, ᾈ",
    "Σίσυφος", "مرحبا بالعالم", "हिन्दी", "café vs café", "  two  spaces",
]
ALPHABET = list("abcXYZ019 \n\t\r'.,;:!?()-_/\\\"") + [
    "é", "ö", "日", "本", "😀", "́", " ", " ", "§", "Σ", "'s", "'re", " the", "ing", "tion",
]


def load_kv(path):
    """Read GGUF metadata (not tensors) with the struct module, enough to get the vocabulary."""
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))

    def rstr():
        (n,) = rd("Q")
        return f.read(n).decode("utf8", "surrogateescape")

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t in scalar:
            return rd(scalar[t])[0]
        if t == 8:
            return rstr()
        if t == 9:
            (et,) = rd("I")
            (n,) = rd("Q")
            return [rval(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")

    f.read(4)  # magic
    rd("I")  # version
    rd("Q")  # tensor count
    (nkv,) = rd("Q")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        (t,) = rd("I")
        kv[k] = rval(t)
    return kv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--pre", required=True, choices=[*PRESETS, "spm"])
    ap.add_argument("--random", type=int, default=4000, help="number of seeded random strings")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from llama_cpp import Llama

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("tokenizer_oracle", here / "tokenizer_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    kv = load_kv(args.gguf)
    tokens, types = kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.token_type"]
    llama = Llama(model_path=args.gguf, vocab_only=True, verbose=False)

    if args.pre == "spm":
        sp = oracle.Spm()
        sp.tokens, sp.types, sp.scores = tokens, types, kv["tokenizer.ggml.scores"]
        sp.t2i = {t: i for i, t in enumerate(tokens)}
        # USER_DEFINED tokens (type 4, e.g. Gemma-3's runs of spaces) are split out before SPM
        # runs, even when special tokens are not parsed. Spm has no encode(), so do it here.
        users = sorted((i for i, ty in enumerate(types) if ty == 4), key=lambda i: -len(tokens[i]))

        def encode(text):
            frags = [("t", text)]
            for sid in users:
                new = []
                for k, v in frags:
                    if k == "k":
                        new.append((k, v))
                        continue
                    rest = v
                    while (i := rest.find(tokens[sid])) >= 0:
                        if i:
                            new.append(("t", rest[:i]))
                        new.append(("k", sid))
                        rest = rest[i + len(tokens[sid]):]
                    if rest:
                        new.append(("t", rest))
                frags = new
            out = []
            for k, v in frags:
                out.append(v) if k == "k" else out.extend(sp.piece(v.replace(" ", "▁")))
            return out

        def reference(text):
            return list(llama.tokenize(text.encode("utf8"), add_bos=False, special=False)) if text else []
    else:
        md, marks, ig = PRESETS[args.pre]
        bpe = oracle.Bpe(tokens, types, kv["tokenizer.ggml.merges"], md, marks, ig)
        encode = bpe.encode

        def reference(text):
            return list(llama.tokenize(text.encode("utf8"), add_bos=False, special=True)) if text else []

    rnd = random.Random(args.seed)
    corpus = list(FIXED)
    corpus += ["".join(rnd.choice(ALPHABET) for _ in range(rnd.randint(1, 40))) for _ in range(args.random)]
    bad = [(t, encode(t), reference(t)) for t in corpus if encode(t) != reference(t)]

    print(f"{args.pre}: {len(corpus) - len(bad)}/{len(corpus)} strings match llama.cpp")
    for t, got, want in bad[:5]:
        print(f"  MISMATCH {t!r:.70}\n    oracle: {got[:12]}\n    llama : {want[:12]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
