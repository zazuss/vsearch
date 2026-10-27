---
name: vsearch
description: Use when you need semantic recall over local markdown notes instead of keyword matching — "where did I write about X" where X shares no words with the original note. Wraps the vsearch CLI (index + query) backed by any OpenAI-compatible embeddings endpoint.
---

# vsearch — local semantic recall over markdown

> 中文速览：给 agent 用的语义检索技能。用户说「上次那个关于 X 的想法」而原话不含关键词时，用 `vsearch query` 而不是 grep。需要先起一个 embedding 服务（llama.cpp 或 ollama）。本文正文英文，因为要随仓库分发给其他人的 agent。

Keyword search (grep, FTS5, `session_search`) matches literal tokens. When the user refers to something by meaning — "that thing about the duck", "the note where I figured out the memory problem" — the original text may share **zero** words with the query. That is the only case where this tool is the right one.

## Preconditions

One thing must be running: an OpenAI-compatible `/v1/embeddings` endpoint.

```bash
# llama.cpp (recommended)
llama-server -m bge-m3-Q8_0.gguf --embedding --pooling cls \
  -c 512 -b 512 -ub 512 --port 8098 -ngl 99 --no-warmup

# or ollama
ollama serve && ollama pull bge-m3   # then: export VSEARCH_EMBED_URL=http://127.0.0.1:11434/v1/embeddings
```

Confirm before querying, do not guess:

```bash
curl -sS --max-time 5 http://127.0.0.1:8098/health
vsearch stats          # is there even an index?
```

If `vsearch` is not on PATH: `export PATH="$HOME/.local/bin:$PATH"`.

## Use

```bash
vsearch query "自然语言问题" 5      # top 5 (default 5)
vsearch query "ducks" 20 0.30      # top 20, min cosine 0.30
vsearch stats                      # composition, size, freshness, per-root counts
vsearch index                      # build or update (idempotent, mtime-based)
```

Output format is `score  path:line` followed by the chunk body:

```
0.612   /home/me/notes/2026-10-27.md:41
        我把向量存成了 JSON，一个数一个数写着……
```

**Read the source file before acting on a hit.** The chunk is a window into a file; the file may contain a correction two paragraphs later. `read_file` the path, then answer.

## Configuration

| Var | Default |
|---|---|
| `VSEARCH_ROOTS` | `~/.hermes/notebook`, `~/.hermes/memories`, `~/.hermes/skills` (or `.` if none exist) |
| `VSEARCH_DB` | `~/.hermes/vsearch/index.db` |
| `VSEARCH_EMBED_URL` | `http://127.0.0.1:8098/v1/embeddings` |
| `VSEARCH_HEALTH_URL` | derived from `VSEARCH_EMBED_URL` |
| `VSEARCH_MODEL` | `bge-m3` |
| `VSEARCH_START_CMD` | *(unset)* — shell command to start the backend when the health check fails, e.g. `systemctl --user start vsearch-embed` or `ollama serve &` |
| `VSEARCH_UNIT` | `vsearch-embed` — systemd user unit tried when `VSEARCH_START_CMD` is unset |

Colon-separated roots, `~/` allowed:

```bash
export VSEARCH_ROOTS="$HOME/notes:$HOME/work/wiki"
vsearch index
```

## When NOT to use this

- **The user remembers an exact word, filename, commit, error string.** Use `rg` / `grep`. Vector search is strictly worse at exact matching and will happily rank a semantically-similar-but-wrong note first.
- **You need the authoritative current value of something.** Semantic search returns what was *written*, including what was later corrected. It is a pointer to a file, not a source of truth.
- **The index is stale.** Check `vsearch stats` for the newest file date. If notes changed today and the index says last week, run `vsearch index` first (incremental, usually under a second).
- **Nothing indexed covers the topic.** An empty-ish result set means "not written down", not "didn't happen". Say so instead of inventing.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `embedding 请求失败` / connection refused | backend down → start it (`systemctl --user start vsearch-embed` if installed as a unit) |
| `读不到索引` / no such table | `vsearch index` |
| backend starts then dies | `journalctl --user -u vsearch-embed -n 30 --no-pager`; check RAM |
| results clearly stale | `vsearch index` |
| query returns nonsense-but-similar chunks | that is the model, not a bug — re-read the source and answer from it |

## Design invariants (do not "optimise" these away)

1. **Chunking honours `§`.** Memory files separated by `§` split on it, so one line = one retrievable fact. Without this the whole file is one chunk and retrieval granularity is destroyed. Blank-line splitting is the fallback; blocks over 320 chars are hard-wrapped.
2. **Vectors are little-endian f32 BLOBs, never JSON.** Storing f32 as decimal text was 166 MB and 1.0 s per query — most of that second was `serde_json` parsing decimals. Binary BLOB: 60 MB, 0.12 s, 13k chunks. Same data. If you touch storage, keep this.
3. **Do not give an embedding model a chat-sized context.** bge-m3 is trained at 512 tokens. Handing it `-c 4096` wasted over 1.4 GB of resident RAM for zero quality gain. One flag fixes it, and the symptom is invisible — nothing runs slower, memory just sits there. Related: the server climbs to ~2.7 GB after a full index pass and never gives it back, so prefer on-demand start over always-on.
4. **Incremental indexing is mtime-based.** Any file whose mtime changed is re-embedded whole-file; unchanged files are skipped (0.01 s for 355 files). Files that disappeared are purged.
5. **Changing the embedding model requires a full rebuild** — dimensions differ, old vectors are garbage.
6. **`rusqlite` uses `features = ["bundled"]`.** First build takes ~2 min (compiles SQLite C); binary 2.7 → 4.5 MB. This is what keeps "zero system dependencies" true. Do not switch to dynamic linking.
7. **`index.db-wal` / `-shm` are live files.** Do not delete them as "temp junk" — you can lose recent transactions.
8. **Never `pkill -f "llama-server.*8098"`.** `-f` matches the full command line, which includes your own shell — it SIGTERMs itself. Use `systemctl --user stop` or kill by PID.
9. **`ureq` 2.12 `post()` wants `&str`**, not `String`.
10. **Watch what you index — it matters more than the model.** Same index, same queries, only `VSEARCH_ROOTS` changed. Five "I know I wrote this but not where" queries: with skills + notes + memory (13,210 chunks) **0/5 correct** — 13 of the top 15 hits were bare `## heading` lines from skill files, and three queries returned completely unrelated results. With notes + memory only (1,904 chunks) **4/5 landed the exact chunk at rank 1**. Cause: 302 skill files made up 86% of all chunks, and a `## heading` block scores high against almost any query while containing no answer. If recall feels noisy, narrow `VSEARCH_ROOTS` — do not swap the model.

## Rebuild

```bash
export PATH="$HOME/.cargo/bin:$PATH"
unset HTTP_PROXY HTTPS_PROXY        # stale proxy vars break cargo
cd vsearch && cargo build --release && install -m755 target/release/vsearch ~/.local/bin/
```

Four dependencies: `ureq` (sync HTTP), `serde`, `serde_json`, `rusqlite` (bundled). Deliberately no async / no tokio — smaller surface, fewer failure modes.

## Reference numbers

Measured on a Jetson Orin Nano 8GB, bge-m3 Q8_0, llama.cpp, GPU offload (`-ngl 99`). Useful as an order of magnitude, not as a promise:

| | |
|---|---|
| Full index, notes + memory + skills | 355 files / 13,265 chunks / 4m48s (bottleneck: embedding, not storage) |
| Full index, notes + memory only | 53 files / 1,904 chunks / pruning 302 skill files took 0.68s (cold full pass on 1.9k chunks is under a minute, not separately timed) |
| Incremental, nothing changed | 0.01s |
| Query end-to-end | 0.12s warm; 3.1s cold (includes starting the backend) |
| Index size | 8.7 MB (notes + memory) vs 61 MB (with skills) |
| Binary | 4.5 MB |
| Embedding server RSS | ~1.0 GB idle, ~2.7 GB after a full index (does not shrink back) |
| API cost | 0 |
