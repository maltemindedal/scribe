"""Executable spec for Scribe's tokenizer: a Python reference of llama.cpp's tokenization.

Extracted from the research notes (tokenizer_and_models.md, sections 2 and 3). One fix was
needed: the notes' `pre_split` snippet had no `return out`, so `return out` was added. The
notes' snippet is a condensed rewrite of the scripts that produced the numbers below, so
those numbers describe the original scripts. The extracted code was re-checked against
llama-cpp-python 0.3.36 when it was imported (see research/README.md for what was run).
Status of the original scripts (2026-10-08):

* Byte-level BPE (`pre_split`, `B2U`, `Bpe`): 0 mismatches against llama.cpp (the copy bundled
  in llama-cpp-python 0.3.36, exact commit unknown) on 8,028 strings for each of Qwen3-0.6B,
  Qwen3.5-0.8B, Llama-3.2-1B and LFM2.5-1.2B, and on an exhaustive code-point sweep.
* SentencePiece (`Spm.piece`): 0 mismatches on 7,028 strings, Gemma-3 only
  (`add_space_prefix=False`). The Llama-2 style dummy-prefix branch is untested.
* `Spm` has no constructor and no `encode()`: the notes give those only as comments. `piece()`
  expects `self.t2i`, `self.scores` and text with ' ' already replaced by U+2581.
* Hugging Face `tokenizers` differs from llama.cpp on Qwen vocabularies (NFC normalisation),
  so llama.cpp, not HF, is the oracle. Unassigned code points (category Cn) count as "other".

Per-model parameters for `Bpe(tokens, types, merges, max_digits, marks, ignore_merges)`:
    qwen2:                 max_digits=1, marks=False, ignore_merges=False
    qwen35:                max_digits=1, marks=True,  ignore_merges=False
    llama-bpe/llama3/lfm2: max_digits=3, marks=False, ignore_merges=True
"""

import unicodedata, heapq
WS = {0x9,0xA,0xB,0xC,0xD,0x20,0x85,0xA0,0x1680,0x2028,0x2029,0x202F,0x205F,0x3000, *range(0x2000,0x200B)}  # = Unicode White_Space
cat0 = lambda c: unicodedata.category(c)[0]            # 'L','N','M','Z','P','S','C'

def pre_split(t, max_digits, marks):   # qwen2:(1,False)  llama3/lfm2:(3,False)  qwen35:(1,True)
    n, out, pos = len(t), [], 0
    L = lambda c: cat0(c) == 'L' or (marks and cat0(c) == 'M')
    N = lambda c: cat0(c) == 'N'
    W = lambda c: ord(c) in WS
    other = lambda c: not W(c) and not L(c) and not N(c)      # NOTE: unassigned (Cn) counts as 'other'
    while pos < n:
        c = t[pos]
        if c == "'" and pos+1 < n:                             # (?i:'s|'t|'re|'ve|'m|'ll|'d)
            n1 = t[pos+1].lower()
            if n1 in 'stmd': out.append((pos,pos+2)); pos += 2; continue
            if pos+2 < n and (n1, t[pos+2].lower()) in (('r','e'),('v','e'),('l','l')):
                out.append((pos,pos+3)); pos += 3; continue
        if c not in '\r\n' and not N(c) and (L(c) or (pos+1 < n and L(t[pos+1]))):   # [^\r\n\p{L}\p{N}]?\p{L}+
            s = pos; pos += 1
            while pos < n and L(t[pos]): pos += 1
            out.append((s,pos)); continue
        if N(c):                                               # \p{N}   |  \p{N}{1,3}
            s = pos; pos += 1
            while pos < n and pos-s < max_digits and N(t[pos]): pos += 1
            out.append((s,pos)); continue
        sr = pos+1 if c == ' ' else pos                        #  ?[^\s\p{L}\p{N}]+[\r\n]*
        if sr < n and other(t[sr]):
            s, p = pos, sr
            while p < n and other(t[p]): p += 1
            while p < n and t[p] in '\r\n': p += 1
            out.append((s,p)); pos = p; continue
        p, last_nl = pos, 0                                    # \s*[\r\n]+  |  \s+(?!\S)  |  \s+
        while p < n and W(t[p]):
            if t[p] in '\r\n': last_nl = p+1
            p += 1
        if last_nl:            out.append((pos,last_nl)); pos = last_nl; continue
        if p-pos > 1 and p<n:  out.append((pos,p-1));     pos = p-1;     continue
        if p > pos:            out.append((pos,p));       pos = p;       continue
        out.append((pos,pos+1)); pos += 1                      # no alternative matched
    return out

# byte -> GPT-2 unicode
bs = list(range(0x21,0x7F)) + list(range(0xA1,0xAD)) + list(range(0xAE,0x100)); cs = bs[:]; k = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+k); k += 1
B2U = {b: chr(c) for b, c in zip(bs, cs)}

class Bpe:                                   # tok2id: dict[str,int] over GGUF tokens; merges: GGUF list[str]
    def __init__(s, tokens, types, merges, max_digits, marks, ignore_merges):
        s.t2i = {t:i for i,t in enumerate(tokens)}; s.tokens, s.types = tokens, types
        s.md, s.marks, s.ig = max_digits, marks, ignore_merges
        s.rank = {}                          # (id_left,id_right) -> (rank, id_merged)
        for r, m in enumerate(merges):
            a, b = m.split(' ', 1)           # byte-encoded tokens never contain ' '
            ia, ib, iab = s.t2i.get(a), s.t2i.get(b), s.t2i.get(a+b)
            if None not in (ia, ib, iab): s.rank[(ia, ib)] = (r, iab)
        s.b2i = {b: s.t2i[B2U[b]] for b in range(256)}
        s.specials = sorted([i for i,ty in enumerate(types) if ty in (3,4)], key=lambda i: -len(tokens[i]))
    def word(s, w):
        raw = w.encode()
        if s.ig:
            enc = ''.join(B2U[b] for b in raw)
            if enc in s.t2i: return [s.t2i[enc]]
        ids = [s.b2i[b] for b in raw]; n = len(ids)
        prev = list(range(-1, n-1)); nxt = list(range(1, n+1)); nxt[-1] = -1; alive = [True]*n; h = []
        def push(l, r):
            if l >= 0 and r >= 0 and (ids[l], ids[r]) in s.rank:
                heapq.heappush(h, (s.rank[(ids[l], ids[r])][0], l, r, ids[l], ids[r]))
        for i in range(1, n): push(i-1, i)
        while h:
            _, l, r, il, ir = heapq.heappop(h)                 # lowest rank, then leftmost
            if not alive[l] or not alive[r] or ids[l] != il or ids[r] != ir: continue   # stale
            ids[l] = s.rank[(il, ir)][1]; alive[r] = False
            nxt[l] = nxt[r]
            if nxt[r] >= 0: prev[nxt[r]] = l
            push(prev[l], l); push(l, nxt[l])
        return [ids[i] for i in range(n) if alive[i]]
    def encode(s, text, parse_special=True):
        frags = [('t', text)]
        for sid in s.specials:                                 # longest first
            if not parse_special and s.types[sid] in (2, 3): continue   # USER_DEFINED (4) always parsed
            st, new = s.tokens[sid], []
            for k, v in frags:
                if k == 'k': new.append((k, v)); continue
                rest = v
                while (i := rest.find(st)) >= 0:
                    if i: new.append(('t', rest[:i]))
                    new.append(('k', sid)); rest = rest[i+len(st):]
                if rest: new.append(('t', rest))
            frags = new
        out = []
        for k, v in frags:
            if k == 'k': out.append(v)
            else:
                for a, b in pre_split(v, s.md, s.marks): out.extend(s.word(v[a:b]))
        return out      # BOS/EOS are added by the caller per add_bos/add_eos


class Spm:    # tokens/scores/types from GGUF; add_space_prefix from tokenizer.ggml.add_space_prefix (Gemma-3: False)
    def piece(s, text):                      # text already has ' ' -> '▁'
        syms = [[c, True] for c in text]; n = len(syms)
        prev = list(range(-1, n-1)); nxt = list(range(1, n+1)); nxt[-1] = -1; h = []
        def push(l, r):
            if l < 0 or r < 0: return
            tx = syms[l][0] + syms[r][0]; tid = s.t2i.get(tx)
            if tid is not None: heapq.heappush(h, (-s.scores[tid], l, r, len(tx.encode())))
        for i in range(1, n): push(i-1, i)
        while h:
            _, l, r, size = heapq.heappop(h)
            if not syms[l][1] or not syms[r][1]: continue
            if len(syms[l][0].encode()) + len(syms[r][0].encode()) != size: continue
            syms[l][0] += syms[r][0]; syms[r][1] = False; syms[r][0] = ''
            nxt[l] = nxt[r]
            if nxt[r] >= 0: prev[nxt[r]] = l
            push(prev[l], l); push(l, nxt[l])
        out, i = [], 0
        while i != -1:
            tid = s.t2i.get(syms[i][0])
            out += [tid] if tid is not None else [s.t2i['<0x%02X>' % b] for b in syms[i][0].encode()]
            i = nxt[i]
        return out
    # encode(): for each text fragment: t = (' ' if add_space_prefix and prev_is_special_or_start else '') + frag
    #           out += piece(t.replace(' ', '▁'))
