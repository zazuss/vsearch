use rusqlite::{params, Connection};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_EMBED_URL: &str = "http://127.0.0.1:8098/v1/embeddings";
const DEFAULT_MODEL: &str = "bge-m3";
const MAX_CHUNK: usize = 320;
const BATCH: usize = 4;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/home/orin".into()))
}

fn db_path() -> PathBuf {
    if let Ok(p) = std::env::var("VSEARCH_DB") {
        return PathBuf::from(p);
    }
    home().join(".hermes/vsearch/index.db")
}

fn legacy_json() -> PathBuf {
    home().join(".hermes/vsearch/index.json")
}

/// Index roots. VSEARCH_ROOTS is colon-separated ('~/' allowed).
/// Default: ~/.hermes/{notebook,memories,skills}; falls back to the current directory.
fn roots() -> Vec<PathBuf> {
    if let Ok(r) = std::env::var("VSEARCH_ROOTS") {
        let v: Vec<PathBuf> = r
            .split(':')
            .filter(|s| !s.is_empty())
            .map(|s| match s.strip_prefix("~/") {
                Some(rest) => home().join(rest),
                None => PathBuf::from(s),
            })
            .filter(|p| p.is_dir())
            .collect();
        if !v.is_empty() {
            return v;
        }
    }
    let def: Vec<PathBuf> = ["notebook", "memories", "skills"]
        .iter()
        .map(|d| home().join(".hermes").join(d))
        .filter(|p| p.is_dir())
        .collect();
    if def.is_empty() {
        return vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))];
    }
    def
}

fn collect_md(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect_md(&p, out);
            } else if p.extension().map(|x| x == "md").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

fn mtime(p: &Path) -> u64 {
    fs::metadata(p)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Embedding backend (OpenAI-compatible /v1/embeddings). Override: VSEARCH_EMBED_URL.
fn embed_url() -> String {
    env_or("VSEARCH_EMBED_URL", DEFAULT_EMBED_URL)
}

/// Health endpoint; derived from embed_url unless VSEARCH_HEALTH_URL is set.
fn health_url() -> String {
    if let Ok(h) = std::env::var("VSEARCH_HEALTH_URL") {
        return h;
    }
    let u = embed_url();
    match u.split("/v1/").next() {
        Some(base) if base != u => format!("{base}/health"),
        _ => u,
    }
}

/// Model name sent to the backend. Override: VSEARCH_MODEL.
fn model() -> String {
    env_or("VSEARCH_MODEL", DEFAULT_MODEL)
}

fn open_db() -> Connection {
    let p = db_path();
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).ok();
    }
    let c = Connection::open(&p).unwrap_or_else(|e| {
        eprintln!("打开数据库失败 {}: {e}", p.display());
        std::process::exit(1);
    });
    c.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA temp_store=MEMORY;
         CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime INTEGER NOT NULL);
         CREATE TABLE IF NOT EXISTS chunks(
             id INTEGER PRIMARY KEY,
             src TEXT NOT NULL,
             line INTEGER NOT NULL,
             text TEXT NOT NULL,
             vec BLOB NOT NULL
         );
         CREATE INDEX IF NOT EXISTS idx_chunks_src ON chunks(src);
         CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);",
    )
    .unwrap_or_else(|e| {
        eprintln!("建表失败: {e}");
        std::process::exit(1);
    });
    c
}

fn set_meta(c: &Connection, k: &str, v: &str) {
    let _ = c.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?1,?2)", params![k, v]);
}

fn get_meta(c: &Connection, k: &str) -> Option<String> {
    c.query_row("SELECT v FROM meta WHERE k=?1", params![k], |r| r.get(0))
        .ok()
}

fn split_long(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= MAX_CHUNK {
        return vec![s.to_string()];
    }
    let mut v = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + MAX_CHUNK).min(chars.len());
        v.push(chars[i..end].iter().collect::<String>());
        i = end;
    }
    v
}

/// 分块：memories 用 § 分条，其它按空行分段
fn chunk_file(p: &Path) -> Vec<(usize, String)> {
    let content = match fs::read_to_string(p) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<(usize, String)> = Vec::new();

    if content.contains('\u{00a7}') {
        let mut line = 1usize;
        for seg in content.split('\u{00a7}') {
            let t = seg.trim();
            if !t.is_empty() {
                for c in split_long(t) {
                    out.push((line, c));
                }
            }
            line += seg.matches('\n').count() + 1;
        }
        return out;
    }

    let mut buf = String::new();
    let mut buf_line = 1usize;
    for (i, ln) in content.lines().enumerate() {
        if ln.trim().is_empty() {
            let t = buf.trim().to_string();
            if !t.is_empty() {
                for c in split_long(&t) {
                    out.push((buf_line, c));
                }
            }
            buf.clear();
            buf_line = i + 2;
        } else {
            if buf.is_empty() {
                buf_line = i + 1;
            }
            buf.push_str(ln);
            buf.push('\n');
        }
    }
    let t = buf.trim().to_string();
    if !t.is_empty() {
        for c in split_long(&t) {
            out.push((buf_line, c));
        }
    }
    out
}

fn health_ok() -> bool {
    ureq::get(&health_url())
        .timeout(std::time::Duration::from_secs(2))
        .call()
        .is_ok()
}

/// 后端没起时尝试拉起。两种方式，按顺序：
///   VSEARCH_START_CMD  任意 shell 命令（最通用）
///   VSEARCH_UNIT       systemd 用户单元名（默认 vsearch-embed）
/// 都不生效就报错，把健康检查地址打出来，别让调用方猜。
fn ensure_service() -> Result<(), String> {
    if health_ok() {
        return Ok(());
    }
    let start_cmd = env_or("VSEARCH_START_CMD", "");
    if !start_cmd.is_empty() {
        eprintln!("embedding 服务没起，执行 VSEARCH_START_CMD ...");
        let _ = Command::new("sh").args(["-c", &start_cmd]).status();
    } else {
        let unit = env_or("VSEARCH_UNIT", "vsearch-embed");
        eprintln!("embedding 服务没起，尝试拉起 {unit} ...");
        let _ = Command::new("systemctl")
            .args(["--user", "start", &unit])
            .status();
    }
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if health_ok() {
            return Ok(());
        }
    }
    Err(format!(
        "起不来。健康检查地址 {} 一直不通。\
         手动确认后端是否在跑，或设 VSEARCH_START_CMD / VSEARCH_UNIT 指向你的启动方式。",
        health_url()
    ))
}

fn embed(texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let payload = json!({ "input": texts, "model": model() });
    let resp = ureq::post(&embed_url())
        .timeout(std::time::Duration::from_secs(300))
        .send_json(payload)
        .map_err(|e| format!("embedding 请求失败: {e}"))?;
    let v: serde_json::Value = resp
        .into_json()
        .map_err(|e| format!("响应解析失败: {e}"))?;
    let data = v["data"].as_array().ok_or("响应里没有 data 字段")?;
    let mut out = Vec::with_capacity(data.len());
    for d in data {
        let arr = d["embedding"].as_array().ok_or("缺 embedding 字段")?;
        out.push(
            arr.iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        );
    }
    Ok(out)
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// 直接对 BLOB 算余弦，零分配
fn cosine_blob(q: &[f32], blob: &[u8]) -> f32 {
    let n = q.len().min(blob.len() / 4);
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let b = f32::from_le_bytes([
            blob[i * 4],
            blob[i * 4 + 1],
            blob[i * 4 + 2],
            blob[i * 4 + 3],
        ]) as f64;
        let a = q[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn flush_batch(conn: &Connection, src: &str, pending: &mut Vec<(usize, String)>, n: &mut usize) {
    if pending.is_empty() {
        return;
    }
    let texts: Vec<String> = pending.iter().map(|(_, t)| t.clone()).collect();
    match embed(&texts) {
        Ok(vecs) => {
            for ((line, text), v) in pending.drain(..).zip(vecs.into_iter()) {
                let blob = vec_to_blob(&v);
                if let Err(e) = conn.execute(
                    "INSERT INTO chunks(src, line, text, vec) VALUES(?1,?2,?3,?4)",
                    params![src, line as i64, text, blob],
                ) {
                    eprintln!("插入片段失败: {e}");
                } else {
                    *n += 1;
                }
            }
        }
        Err(e) => {
            eprintln!("批量 embed 失败: {e}");
            pending.clear();
        }
    }
}

fn cmd_index() {
    if let Err(e) = ensure_service() {
        eprintln!("{e}");
        std::process::exit(1);
    }

    let mut conn = open_db();

    let mut old_mtime: HashMap<String, u64> = HashMap::new();
    if let Ok(mut st) = conn.prepare("SELECT path, mtime FROM files") {
        if let Ok(mut rows) = st.query([]) {
            while let Ok(Some(r)) = rows.next() {
                if let (Ok(p), Ok(m)) = (r.get::<_, String>(0), r.get::<_, i64>(1)) {
                    old_mtime.insert(p, m as u64);
                }
            }
        }
    }

    let mut files: Vec<PathBuf> = Vec::new();
    for r in roots() {
        collect_md(&r, &mut files);
    }
    files.sort();

    let tx = match conn.transaction() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("开事务失败: {e}");
            std::process::exit(1);
        }
    };

    let mut n_new = 0usize;
    let mut n_skip = 0usize;
    let mut n_chunks = 0usize;

    for p in &files {
        let ps = p.display().to_string();
        let mt = mtime(p);
        if old_mtime.get(&ps) == Some(&mt) {
            n_skip += 1;
            continue;
        }
        let _ = tx.execute("DELETE FROM chunks WHERE src = ?1", params![ps]);
        let mut pending: Vec<(usize, String)> = Vec::new();
        for (line, text) in chunk_file(p) {
            pending.push((line, text));
            if pending.len() >= BATCH {
                flush_batch(&tx, &ps, &mut pending, &mut n_chunks);
            }
        }
        flush_batch(&tx, &ps, &mut pending, &mut n_chunks);
        let _ = tx.execute(
            "INSERT OR REPLACE INTO files(path, mtime) VALUES(?1,?2)",
            params![ps, mt as i64],
        );
        n_new += 1;
    }

    let existing: HashSet<String> = files.iter().map(|p| p.display().to_string()).collect();
    let mut n_purged = 0usize;
    for path in old_mtime.keys() {
        if !existing.contains(path) {
            let _ = tx.execute("DELETE FROM chunks WHERE src = ?1", params![path]);
            let _ = tx.execute("DELETE FROM files WHERE path = ?1", params![path]);
            n_purged += 1;
        }
    }

    if n_chunks > 0 {
        set_meta(&tx, "model", &model());
        set_meta(&tx, "built_at", &now_secs().to_string());
    }
    if let Err(e) = tx.commit() {
        eprintln!("提交失败: {e}");
        std::process::exit(1);
    }

    let total: i64 = conn
        .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .unwrap_or(0);
    let dim = get_meta(&conn, "dim").unwrap_or_else(|| "?".into());
    let size = fs::metadata(db_path()).map(|m| m.len()).unwrap_or(0);
    println!(
        "索引完成: 文件 {} 个（新 {} / 未变 {} / 清理 {}）/ 片段共 {} / {} 维 / 库 {:.1} MB",
        files.len(),
        n_new,
        n_skip,
        n_purged,
        total,
        dim,
        size as f64 / 1_048_576.0
    );
    if legacy_json().exists() {
        println!(
            "提示: 旧的 index.json 还在（{}），可删掉省空间",
            legacy_json().display()
        );
    }
}

fn cmd_query(q: &str, k: usize, min_score: f32) {
    if !db_path().exists() {
        eprintln!("索引不存在，先跑: vsearch index");
        std::process::exit(1);
    }
    if let Err(e) = ensure_service() {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let conn = open_db();
    let qv = match embed(&[q.to_string()]) {
        Ok(mut v) => {
            if v.is_empty() {
                eprintln!("embed 返回空");
                std::process::exit(1);
            }
            let first = v.remove(0);
            set_meta(&conn, "dim", &first.len().to_string());
            first
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let mut st = match conn.prepare("SELECT src, line, text, vec FROM chunks") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("查询失败: {e}");
            std::process::exit(1);
        }
    };
    let mut rows = match st.query([]) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("遍历失败: {e}");
            std::process::exit(1);
        }
    };

    let mut scored: Vec<(f32, String, i64, String)> = Vec::new();
    while let Ok(Some(r)) = rows.next() {
        let src: String = r.get(0).unwrap_or_default();
        let line: i64 = r.get(1).unwrap_or(0);
        let text: String = r.get(2).unwrap_or_default();
        let blob: Vec<u8> = r.get(3).unwrap_or_default();
        let s = cosine_blob(&qv, &blob);
        if s >= min_score {
            scored.push((s, src, line, text));
        }
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    if scored.is_empty() {
        println!("没有命中（阈值 {min_score}）。换个说法，或降低阈值。");
        return;
    }
    let prefix = format!("{}/", home().display());
    for (i, (s, src, line, text)) in scored.iter().enumerate() {
        let short = src.replace(&prefix, "~/");
        println!("{}. {:.3}  {} :{}", i + 1, s, short, line);
        for ln in text.lines().take(6) {
            println!("   {}", ln);
        }
        println!();
    }
}

fn cmd_stats() {
    if !db_path().exists() {
        println!("索引不存在。跑 vsearch index 建一个。");
        return;
    }
    let conn = open_db();
    let files: i64 = conn
        .query_row("SELECT count(*) FROM files", [], |r| r.get(0))
        .unwrap_or(0);
    let chunks: i64 = conn
        .query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))
        .unwrap_or(0);
    let size = fs::metadata(db_path()).map(|m| m.len()).unwrap_or(0);
    let built: u64 = get_meta(&conn, "built_at")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let age_h = now_secs().saturating_sub(built) / 3600;

    println!("存储:   SQLite ({})", db_path().display());
    println!(
        "模型:   {}",
        get_meta(&conn, "model").unwrap_or_else(|| "?".into())
    );
    println!(
        "维度:   {}",
        get_meta(&conn, "dim").unwrap_or_else(|| "?".into())
    );
    println!("文件数: {}", files);
    println!("片段数: {}", chunks);
    println!("库大小: {:.1} MB", size as f64 / 1_048_576.0);
    println!("建于:   {} 小时前", age_h);

    if let Ok(mut st) = conn.prepare(
        "SELECT CASE
                  WHEN src LIKE '%/.hermes/notebook/%' THEN 'notebook'
                  WHEN src LIKE '%/.hermes/memories/%' THEN 'memories'
                  WHEN src LIKE '%/.hermes/skills/%'   THEN 'skills'
                  ELSE '其它' END AS k,
                count(*) FROM chunks GROUP BY k ORDER BY count(*) DESC",
    ) {
        if let Ok(mut rows) = st.query([]) {
            while let Ok(Some(r)) = rows.next() {
                let k: String = r.get(0).unwrap_or_default();
                let n: i64 = r.get(1).unwrap_or(0);
                println!("  {:<10} {} 片段", k, n);
            }
        }
    }

    let lj = legacy_json();
    if lj.exists() {
        let s = fs::metadata(&lj).map(|m| m.len()).unwrap_or(0);
        println!(
            "旧 JSON 索引还在: {} ({:.1} MB) —— 可删",
            lj.display(),
            s as f64 / 1_048_576.0
        );
    }
}

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

fn main() {
    // `vsearch query ... | head` 时别因为 broken pipe 直接 panic
    unsafe {
        signal(13, 0); // SIGPIPE -> SIG_DFL
    }
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("stats");
    match cmd {
        "index" => cmd_index(),
        "query" | "q" => {
            let q = match args.get(2) {
                Some(s) => s.clone(),
                None => {
                    eprintln!("用法: vsearch query \"问题\" [条数] [最低分]");
                    std::process::exit(1);
                }
            };
            let k = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
            let min = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.35);
            cmd_query(&q, k, min)
        }
        "stats" => cmd_stats(),
        "help" | "-h" | "--help" => {
            println!(
                "vsearch {} - local semantic search over markdown notes\n",
                env!("CARGO_PKG_VERSION")
            );
            println!("USAGE:");
            println!("  vsearch index                     build or update the index");
            println!("  vsearch query \"text\" [k] [min]   semantic search (default k=8, min=0.35)");
            println!("  vsearch stats                     index composition and age");
            println!("  vsearch help                      this help");
            println!();
            println!("ENVIRONMENT:");
            println!("  VSEARCH_ROOTS       colon-separated index roots");
            println!("                      (default: ~/.hermes/notebook,~/.hermes/memories,~/.hermes/skills)");
            println!("  VSEARCH_DB          index file (default: ~/.hermes/vsearch/index.db)");
            println!("  VSEARCH_EMBED_URL   OpenAI-compatible /v1/embeddings");
            println!("                      (default: http://127.0.0.1:8098/v1/embeddings)");
            println!("  VSEARCH_HEALTH_URL  health endpoint (default: derived from VSEARCH_EMBED_URL)");
            println!("  VSEARCH_MODEL       model name (default: bge-m3)");
            println!();
            println!("An OpenAI-compatible embedding server must be running; see README.");
        }
        other => {
            eprintln!(
                "未知子命令: {other}\n用法:\n  vsearch index\n  vsearch query \"问题\" [条数] [最低分]\n  vsearch stats"
            );
            std::process::exit(1);
        }
    }
}
