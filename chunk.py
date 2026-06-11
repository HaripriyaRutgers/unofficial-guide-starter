"""
chunk.py — Stage 2 of the RAG pipeline: Chunking.

Reads the .txt files in raw_docs/ (produced by ingest.py) and splits them
into retrieval-sized chunks:

  - Articles      -> sliding window: 400-500 tokens, 50-token overlap.
  - Forum sources -> one chunk per reply (sliding window if a reply > 500 tok).
  - Career maps   -> one chunk per role/section (sliding window if > 500 tok).

Token counting uses tiktoken (cl100k_base) so sizes are accurate, not
word-count approximations. Output is written to chunks.json.

Run with:  python chunk.py   (after running ingest.py)
Libraries:  tiktoken, json, os, re   (no LangChain / no chunking library)
"""

import os
import re
import json

import tiktoken

# cl100k_base is the encoding used by modern OpenAI models; it's a solid,
# widely-available tokenizer for *counting* tokens accurately. We only use
# it to measure size — the actual embedding model is all-MiniLM later.
ENC = tiktoken.get_encoding("cl100k_base")

RAW_DIR = "raw_docs"
OUTPUT_FILE = "chunks.json"

# Sliding-window parameters (in tokens), per the spec.
TARGET_MAX = 500      # never exceed ~500 tokens per chunk
TARGET_MIN = 400      # aim for at least 400 tokens on full-size windows
OVERLAP = 50          # 50-token overlap between consecutive windows

# Map raw filename stem -> (chunk "type", human-readable "source" label).
# This drives which splitting strategy each file gets.
FILE_META = {
    "reddit":                 ("forum_reply",     "reddit.com"),
    "teamblind":              ("forum_reply",     "teamblind.com"),
    "screenskills":           ("career_profile",  "screenskills.com"),
    "uchicago":               ("career_profile",  "careeradvancement.uchicago.edu"),
    "careerexplorer":         ("career_profile",  "careerexplorer.com"),
    "themuse":                ("article",         "themuse.com"),
    "collegewise":            ("article",         "go.collegewise.com"),
    "technicalwriting":       ("article",         "everythingtechnicalwriting.com"),
    "awesome_subreddits":     ("article",         "github.com"),
    "awesome_cybersecurity":  ("article",         "github.com"),
}


def count_tokens(text):
    """Return the exact tiktoken token count for a string."""
    # encode() turns text into a list of token ids; its length is the count.
    return len(ENC.encode(text))


def chunk_text(text, max_tokens=TARGET_MAX, overlap=OVERLAP):
    """Sliding-window chunker implemented from scratch.

    Splits `text` into windows of up to `max_tokens` tokens, where each
    window after the first re-includes the last `overlap` tokens of the
    previous window (so context isn't lost at boundaries).

    Returns a list of text strings. We tokenize once, then slice the token
    list and decode each slice back to text — this guarantees sizes are
    measured in real tokens, not characters or words.
    """
    # Encode the whole text to token ids a single time (efficient + accurate).
    token_ids = ENC.encode(text)
    total = len(token_ids)

    # Short input: it already fits in one window, so return it unchanged.
    if total <= max_tokens:
        return [text]

    chunks = []
    start = 0  # index into token_ids where the current window begins.
    # step is how far we advance the window start each iteration. Overlap means
    # we move forward by (max_tokens - overlap), re-reading `overlap` tokens.
    step = max_tokens - overlap
    # Guard against a pathological config (overlap >= max) that would loop forever.
    if step <= 0:
        step = max_tokens

    while start < total:
        end = start + max_tokens          # window covers [start, end)
        window_ids = token_ids[start:end]
        # decode() converts token ids back into a human-readable string.
        chunks.append(ENC.decode(window_ids).strip())
        if end >= total:
            break  # we've consumed the last tokens; stop.
        start += step
    return chunks


def split_units(raw):
    """Split a raw_docs file into pre-segmented units (forum/career maps).

    ingest.py joined forum replies / career sections with a '----' delimiter
    and prefixed each with a '[type=... source=... role_name=...]' header.
    Here we split on that delimiter and parse the header back out.

    Returns a list of (meta_dict, body_text) tuples. If the file has no
    delimiter (a plain article), returns a single unit with empty meta.
    """
    parts = [p.strip() for p in raw.split("\n----\n") if p.strip()]
    units = []
    for part in parts:
        meta = {}
        body = part
        # If this unit starts with a [ ... ] metadata header, parse + strip it.
        m = re.match(r"^\[([^\]]*)\]\s*\n?(.*)$", part, flags=re.DOTALL)
        if m:
            header, body = m.group(1), m.group(2).strip()
            # Header looks like: type=forum_reply source=reddit role_name=Foo Bar
            # Parse key=value tokens; role_name's value may contain spaces, so we
            # capture each value up to the next " key=" boundary.
            for km in re.finditer(r"(\w+)=(.*?)(?=\s+\w+=|$)", header):
                meta[km.group(1)] = km.group(2).strip()
        units.append((meta, body))
    return units


def make_chunk(chunk_id, source, ctype, text):
    """Build one output chunk dict in the spec's exact shape."""
    return {
        "chunk_id": chunk_id,
        "source": source,
        "type": ctype,
        "text": text,
        "token_count": count_tokens(text),
    }


# Minimum tokens for a standalone GitHub chunk. Anything smaller is a list
# fragment that retrieves poorly, so we merge it into a neighbor instead.
GITHUB_MIN_TOKENS = 100


def merge_small_chunks(pieces, min_tokens=GITHUB_MIN_TOKENS):
    """Merge any chunk under `min_tokens` into an adjacent chunk.

    Used for GitHub list pages, where the sliding window can leave short
    trailing fragments. We merge FORWARD: a too-small chunk is buffered and
    glued onto the next one. A too-small FINAL fragment (no next chunk) folds
    back into the previous chunk so no text is ever dropped.
    """
    if not pieces:
        return pieces
    merged = []
    buffer = ""  # accumulates text that's still below the threshold.
    for piece in pieces:
        # Glue the pending buffer onto the current piece (forward merge).
        candidate = (buffer + "\n\n" + piece).strip() if buffer else piece
        if count_tokens(candidate) < min_tokens:
            buffer = candidate  # still too small — keep accumulating.
        else:
            merged.append(candidate)
            buffer = ""
    # Leftover undersized tail: fold into the previous chunk if one exists.
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + "\n\n" + buffer).strip()
        else:
            merged.append(buffer)  # entire doc was tiny — keep it anyway.
    return merged


def split_themuse_roles(text):
    """Split a TheMuse 'jobs for CS majors' article into one chunk per role.

    Each role is a title line ("UX Researcher") immediately followed by an
    "Average salary" line. The default sliding window straddles role
    boundaries — e.g. it glued the tail of UX Researcher onto the full
    Product Manager + Data Scientist + Web Developer entries, diluting the
    semantic signal so a UX query couldn't match. Here we cut a fresh chunk at
    every role title so each chunk describes exactly ONE role.

    Returns a list of role segments, or None if the "Average salary" layout
    isn't found (so the caller can fall back to the normal sliding window).
    """
    lines = text.split("\n")
    # Boundary = the title line directly before each "Average salary" line.
    boundaries = []
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("average salary"):
            j = i - 1
            while j >= 0 and not lines[j].strip():  # skip blank lines upward.
                j -= 1
            if j >= 0:
                boundaries.append(j)
    if not boundaries:
        return None  # not the expected layout — let the caller fall back.

    segments = []
    # Intro text before the first role becomes its own segment.
    if boundaries[0] > 0:
        pre = "\n".join(lines[:boundaries[0]]).strip()
        if pre:
            segments.append(pre)
    # Each role spans from its title line up to the next role's title line.
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        seg = "\n".join(lines[start:end]).strip()
        if seg:
            segments.append(seg)
    return segments


def process_file(stem, raw):
    """Turn one raw_docs file into a list of chunk dicts.

    Dispatches on the file's type:
      - article      -> sliding window over the whole document.
      - forum_reply  -> one chunk per reply; window only if a reply > max.
      - career_profile -> one chunk per role; window only if a role > max.
    """
    # Look up this file's type/source; default to a generic article.
    ctype, source = FILE_META.get(stem, ("article", stem))
    chunks = []
    idx = 0  # running per-source chunk index for chunk_id.

    if ctype == "article":
        # TheMuse: split on role boundaries so one chunk == one job role.
        role_segs = split_themuse_roles(raw) if source == "themuse.com" else None
        if role_segs is not None:
            for seg in role_segs:
                # One chunk per role; only window a role that exceeds the max.
                if count_tokens(seg) <= TARGET_MAX:
                    chunks.append(make_chunk(f"{stem}_{idx}", source, ctype, seg))
                    idx += 1
                else:
                    for piece in chunk_text(seg):
                        chunks.append(make_chunk(f"{stem}_{idx}", source, ctype, piece))
                        idx += 1
        else:
            # Default: slide a window over the entire cleaned document.
            pieces = chunk_text(raw)
            # GitHub sources: merge sub-100-token fragments before emitting.
            if source == "github.com":
                before = len(pieces)
                pieces = merge_small_chunks(pieces)
                # Show the before/after specifically for awesome_cybersecurity.
                if stem == "awesome_cybersecurity":
                    print(f"  [{stem}] chunks before merge: {before} "
                          f"-> after merge: {len(pieces)}")
            for piece in pieces:
                chunks.append(make_chunk(f"{stem}_{idx}", source, ctype, piece))
                idx += 1
    else:
        # Forum / career-map: ingest.py already segmented this into units.
        for meta, body in split_units(raw):
            # Prefer the per-unit type from the header (e.g. forum_reply); fall
            # back to the file-level type if the header was missing.
            unit_type = meta.get("type", ctype)
            # A career_profile keeps its role_name as a prefix so the chunk reads
            # as a standalone, self-describing unit during retrieval.
            role = meta.get("role_name")
            if role and not body.startswith(role):
                body = f"{role}\n{body}"

            # If a single unit is already small enough, emit it as one chunk.
            if count_tokens(body) <= TARGET_MAX:
                chunks.append(make_chunk(f"{stem}_{idx}", source, unit_type, body))
                idx += 1
            else:
                # Oversized reply/section: fall back to the sliding window inside it.
                for piece in chunk_text(body):
                    chunks.append(make_chunk(f"{stem}_{idx}", source, unit_type, piece))
                    idx += 1
    return chunks


def main():
    # Bail early with a clear message if ingestion hasn't been run yet.
    if not os.path.isdir(RAW_DIR):
        print(f"[ERROR] '{RAW_DIR}/' not found — run `python ingest.py` first.")
        return

    all_chunks = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]  # strip ".txt" to get the source key.
        with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
            raw = f.read()
        file_chunks = process_file(stem, raw)
        print(f"  {fname}: {len(file_chunks)} chunks")
        all_chunks.extend(file_chunks)

    # Persist the full list as a JSON array (consumed by embed_and_retrieve.py).
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ---------------- VALIDATION REPORT ----------------
    print("\n" + "=" * 60)
    print("CHUNKING VALIDATION")
    print("=" * 60)

    total = len(all_chunks)
    print(f"Total chunks: {total}")

    if total == 0:
        print("[WARN] No chunks produced — check that raw_docs/ has content.")
        return

    counts = [c["token_count"] for c in all_chunks]
    print(f"Token count -> min: {min(counts)}, "
          f"max: {max(counts)}, avg: {sum(counts) / total:.1f}")

    # Print one representative chunk per type, in full, for manual review.
    print("\n--- Representative chunks (one per type) ---")
    shown_types = set()
    for c in all_chunks:
        if c["type"] not in shown_types:
            shown_types.add(c["type"])
            print(f"\n[{c['chunk_id']}] type={c['type']} "
                  f"source={c['source']} tokens={c['token_count']}")
            print(c["text"])
            print("-" * 40)

    # Flag chunks that are too small (<50) or too large (>600 tokens).
    bad = [c for c in all_chunks if c["token_count"] < 50 or c["token_count"] > 600]
    if bad:
        print(f"\n[WARN] {len(bad)} chunk(s) outside the 50-600 token range:")
        for c in bad:
            print(f"   {c['chunk_id']}: {c['token_count']} tokens")

    # Flag if the overall corpus size falls outside the 50-500 target.
    if total < 50:
        print(f"\n[WARN] Only {total} total chunks (< 50). "
              f"Consider adding sources or shrinking chunk size.")
    elif total > 500:
        print(f"\n[WARN] {total} total chunks (> 500). "
              f"Consider larger chunks or fewer sources.")
    else:
        print(f"\nTotal chunk count {total} is within the 50-500 target. OK.")


if __name__ == "__main__":
    main()
