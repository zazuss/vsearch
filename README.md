# vsearch

Local semantic search over your markdown notes — one static binary, no runtime, no cloud.

`vsearch` indexes a directory of markdown files into embeddings and answers "where did I write about X?" questions by meaning rather than by keyword. It was built to give a single-user AI agent a memory it can grep semantically: daily notes, memory files, and skill documents written in plain markdown.

> 中文速览：把一堆 markdown 笔记做成本地向量索引，用一句话找回"我上次在哪儿写过这件事"。单二进制、零运行时依赖、全本地推理、不烧 API 费。中文说明见文末《中文说明》一节。

## Why not just grep / FTS?

Keyword search misses the case where you remember the *idea*, not the words. Ask `vsearch query "the servo was too big for the duck"` and it finds the note that says `20×34×26mm 档装不进，只有 18g 微型档可行` — zero shared tokens.

Keyword search is still better when you *do* remember the exact word. Use both.

## Measured (Jetson Orin Nano 8GB, bge-m3 Q8_0, llama.cpp)

| | |
|---|---|
| Index build (full) | 355 files / 13,265 chunks in **4m48s** (bottleneck is embedding, not storage) |
| Incremental (nothing changed) | **0.01s** — mtime short-circuit |
| Query end-to-end | **0.12s** (embed + 13k cosine over SQLite BLOBs) |
| Index size | **60 MB** (1024-dim f32) |
| Binary | **4.5 MB** static, 4 crates |
| Residency | embedding server **411 MB** RSS |
| API cost | **$0** |

## Requirements

One thing: **an OpenAI-compatible embeddings endpoint**. vsearch does not embed anything itself.

```bash
# llama.cpp (recommended — small, fast, one binary)
llama-server -m bge-m3-Q8_0.gguf \
  --embedding --pooling cls -c 512 -b 512 -ub 512 \
  --port 8098 -ngl 99 --no-warmup
```

`bge-m3-Q8_0.gguf` (606 MB): <https://huggingface.co/gpustack/bge-m3-GGUF>

`-c 512` matters: bge-m3 is trained at 512 tokens. Giving it a chat model's default long context wastes ~1.4 GB of RAM for nothing.

Alternatives that also work — anything serving `/v1/embeddings`:

```bash
ollama serve && ollama pull bge-m3      # then VSEARCH_EMBED_URL=http://127.0.0.1:11434/v1/embeddings
# or llama.cpp with nomic-embed / mxbai-embed / Qwen3-Embedding
```

## Build

```bash
git clone https://github.com/<you>/vsearch && cd vsearch
cargo build --release
install -m755 target/release/vsearch ~/.local/bin/
```

No system libraries needed — SQLite is compiled into the binary (`rusqlite` with `bundled`).

## Use

```bash
vsearch index                          # build or update (idempotent, mtime-based)
vsearch query "why did the duck fail"  # top 8
vsearch query "ducks" 20 0.30          # top 20, minimum cosine 0.30
vsearch stats                          # composition, size, age, per-root chunk counts
```

Output is `score  path:line` plus the chunk body — meant to be pasted back into an agent's context, then followed with a normal file read.

### Environment

| Var | Default |
|---|---|
| `VSEARCH_ROOTS` | `~/.hermes/notebook`, `~/.hermes/memories`, `~/.hermes/skills` (or `.` if none exist) |
| `VSEARCH_DB` | `~/.hermes/vsearch/index.db` |
| `VSEARCH_EMBED_URL` | `http://127.0.0.1:8098/v1/embeddings` |
| `VSEARCH_HEALTH_URL` | derived from `VSEARCH_EMBED_URL` |
| `VSEARCH_MODEL` | `bge-m3` |

Colon-separated roots, `~/` allowed:

```bash
export VSEARCH_ROOTS="$HOME/notes:$HOME/work/wiki"
vsearch index
```

Any markdown tree works: Obsidian vault, Zettelkasten, a docs folder, your agent's memory files.

## Using it from an AI agent

`SKILL.md` in this repo is written **for an agent, not for you**. Drop it into your harness's skill/instructions directory (Hermes, Claude Code, anything that reads markdown instructions) and the agent gets: when semantic search beats grep, how to verify the backend is up, what the output means, when *not* to trust a hit, and the storage invariants it should not "optimise" away. This README is the human version.

Whether you give an agent this tool at all is the interesting question. It helps when you keep written notes and want them reachable by meaning. It does not replace reading the actual file.

## How it works

- **Chunking** — a file is split on blank lines into blocks; blocks over 320 chars are hard-wrapped. If the file contains `§` (a common memory-file separator), it splits on that instead, so one line = one retrievable fact.
- **Embedding** — batches of 4 to `/v1/embeddings`, stored as little-endian f32 `BLOB`s.
- **Storage** — SQLite (`WAL`) with three tables: `chunks(src, line, text, vec)`, `files(path, mtime)`, `meta`. Incremental updates delete-and-reinsert per file, and purge files that disappeared.
- **Search** — brute-force cosine over a streaming cursor. The query vector is compared against the `BLOB` byte-by-byte with no intermediate allocation. At 13k chunks this is faster than maintaining an ANN index, and it is exact.
- **No async, no tokio** — `ureq` for HTTP. Four dependencies total. Small surface, few failure modes.

Design note: vectors are stored as binary f32, not as JSON. The first version wrote JSON — 166 MB index and a 1.0s query, because most of that second was `serde_json` parsing decimals. Binary BLOB: 60 MB, 0.12s. Same data.

## Limits (read these)

- **A hit is not a fact.** The index returns what was written, faithfully — including things that were later corrected. Re-read the source before acting on a hit.
- **Exact search beats semantic search** when you know the exact string. Keep `rg` for that.
- **Single-user, single-process.** No locking beyond SQLite's own; don't run two `index` builds at once.
- **Full rescan on every `index`** — it stats all files (fast) but re-embeds any file whose mtime changed, whole-file. No partial-file chunk diffing.
- **No rerank, no hybrid BM25+vector.** Deliberate: it would need another model and more RAM. If you want that, use a bigger system.
- Embedding model changes require a rebuild (`vsearch index` after deleting the db, or bump `VSEARCH_DB`).

## Prior art

There is a lot of good work in this space; these are the ones that came up and what they assume that vsearch does not:

- [obsidian-second-brain](https://github.com/eugeniughelbur/obsidian-second-brain) — persistent agent memory stored in an Obsidian vault. Needs Obsidian.
- [notebrain-cli](https://github.com/nmdra/notebrain-cli) — Go CLI, closest in spirit, targets an Obsidian vault.
- [session-indexer](https://github.com/valpere/session-indexer) — semantic search over Claude Code session JSONL.
- [qmd-rust](https://github.com/PakhomLeo/qmd-rust) — SQLite FTS5 over markdown, keyword not vector.

vsearch's only real claim is narrow: it takes **any** markdown tree, needs **no** host application, and ships as **one** binary with no runtime to break. If you live in Obsidian, use an Obsidian tool — it will be better at being Obsidian.

## 中文说明

**它解决什么**：你记得"上次那个关于 X 的想法"，但原话里一个关键词都不含。关键词搜索（grep / FTS5）找不到，向量搜索能找到。

**前提**：需要一个 OpenAI 兼容的 embedding 服务，vsearch 自己不产向量。

```bash
# llama.cpp（推荐，单二进制）
llama-server -m bge-m3-Q8_0.gguf --embedding --pooling cls \
  -c 512 -b 512 -ub 512 --port 8098 -ngl 99 --no-warmup
```

`bge-m3-Q8_0.gguf`（606 MB）从 [gpustack/bge-m3-GGUF](https://huggingface.co/gpustack/bge-m3-GGUF) 下载。**`-c 512` 别省**：bge-m3 训练长度就是 512，按 chat 模型的默认长上下文跑会白占 1.4 GB 内存。

换成 ollama 也行：

```bash
ollama serve && ollama pull bge-m3
export VSEARCH_EMBED_URL=http://127.0.0.1:11434/v1/embeddings
```

**编译 + 三条命令**：

```bash
cargo build --release && install -m755 target/release/vsearch ~/.local/bin/
vsearch index                    # 建索引（幂等，按 mtime 增量）
vsearch query "为什么鸭子扑了" 5  # 语义搜索
vsearch stats                    # 索引构成
```

**索引自己的笔记**（任何 markdown 目录都行 —— Obsidian vault、Zettelkasten、文档文件夹、agent 的记忆文件）：

```bash
export VSEARCH_ROOTS="$HOME/notes:$HOME/work/wiki"
vsearch index
```

不装 Obsidian 也能用，这是它跟现成方案最大的区别。

**实测**（Jetson Orin Nano 8GB / bge-m3 Q8_0）：全量索引 355 文件 13 265 片段 **4 分 48 秒**；无改动增量 **0.01 秒**；查询端到端 **0.12 秒**；索引 **60 MB**；二进制 **4.5 MB**；embedding 服务驻留 **411 MB**；API 花费 **0**。

**几个坑**：

1. **命中不等于事实**。索引忠实返回你写过的内容，包括后来被纠正的。用之前回原文件确认。
2. 记得确切字词时用 `rg`，向量搜索在这种场景更差。
3. 换 embedding 模型必须重建索引（维度会变）。
4. 改了源文件记得重跑 `vsearch index`，它不会自动监听。

## License

MIT.
