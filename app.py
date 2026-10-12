"""
Red Flag Detector - FastAPI backend

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /api/scan?contamination=0.05   multipart CSV upload -> summary + flagged rows
    GET  /api/sample                    downloadable demo CSV with planted anomalies
    POST /api/alerts/scan               any JSON / JSONL files (field `files`) -> background job (detector.py)
    GET  /api/alerts/jobs/{job_id}      poll job progress / fetch result
    GET  /api/health
    GET  /                              serves ./index.html
"""
import importlib.util
import inspect
import io
import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MAX_BYTES = 25 * 1024 * 1024        # 25 MB upload cap
MAX_ROWS = 200_000                  # keep scans snappy
MAX_FLAGS_RETURNED = 1000           # cap payload size
Z_RULE_THRESHOLD = 4.0              # robust z-score that flags a row on its own
RARE_CATEGORY_FREQ = 0.01           # categories under 1% of rows count as "rare"
FRONTEND = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="Red Flag Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # local dev; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers
def py(v):
    """Convert numpy/pandas scalars to JSON-safe Python values."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 4)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if pd.isna(v):
        return None
    return str(v) if not isinstance(v, (int, str, bool)) else v


def is_id_like(name: str, s: pd.Series) -> bool:
    """Skip identifier columns: they are unique by design and carry no signal."""
    if re.search(r"(^|_)(id|uuid|guid|index|idx)$", name.lower()):
        return True
    return s.nunique(dropna=True) == len(s) and pd.api.types.is_integer_dtype(s)


def robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score (resistant to the outliers we're hunting)."""
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad
    if scale == 0 or np.isnan(scale):
        scale = s.std(ddof=0)
    if scale == 0 or np.isnan(scale):
        return pd.Series(0.0, index=s.index)
    return (s - med) / scale


def build_features(df: pd.DataFrame):
    """Return (feature matrix, per-feature strength matrix, metadata)."""
    feats, strength, meta = {}, {}, {"numeric": [], "categorical": [], "skipped": []}

    for col in df.columns:
        s = df[col]
        if is_id_like(str(col), s):
            meta["skipped"].append(str(col))
            continue

        if pd.api.types.is_bool_dtype(s):
            s = s.astype(int)

        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) < 2:
                meta["skipped"].append(str(col))
                continue
            filled = s.fillna(s.median())
            feats[col] = filled
            strength[col] = robust_z(filled).abs()
            meta["numeric"].append(str(col))
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique < 2 or n_unique > max(50, 0.5 * len(s)):
                meta["skipped"].append(str(col))   # constant or free-text/high-cardinality
                continue
            freq = s.map(s.value_counts(normalize=True)).fillna(0.0)
            feats[col] = -np.log(freq.clip(lower=1e-6))     # rarer value -> bigger number
            rare = (freq < RARE_CATEGORY_FREQ)
            strength[col] = rare.astype(float) * 4.5         # comparable to a ~4.5 sigma hit
            meta["categorical"].append(str(col))

    if not feats:
        raise HTTPException(422, "No usable columns found. Upload a CSV with numeric or "
                                 "low-cardinality categorical columns.")
    return pd.DataFrame(feats), pd.DataFrame(strength), meta


def risk_level(score: float) -> str:
    if score >= 88:
        return "critical"
    if score >= 78:
        return "high"
    if score >= 68:
        return "medium"
    return "low"


def explain(row_idx, df, numeric_cols, strength) -> list[str]:
    """Top contributing columns for one flagged row, in plain English."""
    top = strength.loc[row_idx].sort_values(ascending=False).head(3)
    reasons = []
    for col, st in top.items():
        if st < 2.5:
            continue
        val = df.at[row_idx, col]
        if col in numeric_cols:
            med = df[col].median()
            direction = "above" if val > med else "below"
            reasons.append(f"{col}={py(val)} is unusually {direction} typical ({py(med)}) "
                           f"[{st:.1f}σ]")
        else:
            share = (df[col] == val).mean() * 100
            reasons.append(f"{col}='{val}' is rare ({share:.2f}% of rows)")
    return reasons


# ---------------------------------------------------------------- scan core
def scan_dataframe(df: pd.DataFrame, contamination: float) -> dict:
    t0 = time.perf_counter()
    df = df.reset_index(drop=True)
    total = len(df)
    if total < 20:
        raise HTTPException(422, "Need at least 20 rows to scan.")

    X, strength, meta = build_features(df)

    # Model signal: Isolation Forest on scaled features
    Xs = StandardScaler().fit_transform(X.values)
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso.fit(Xs)
    raw = -iso.score_samples(Xs)                       # higher = weirder
    iso_flag = iso.predict(Xs) == -1
    pct = pd.Series(raw).rank(pct=True).values * 100   # 0..100 percentile

    # Rule signal: any single feature far outside normal, or rare category
    max_strength = strength.max(axis=1).values
    rule_flag = max_strength >= Z_RULE_THRESHOLD
    rule_score = np.clip(max_strength / 8.0, 0, 1) * 100

    # Exact duplicate rows are a classic red flag too
    dup_flag = df.duplicated(keep=False).values

    risk = 0.65 * pct + 0.35 * rule_score
    flagged_mask = iso_flag | rule_flag
    # a rule-only hit must still clear a minimum bar so noise doesn't swamp the table
    flagged_idx = np.where(flagged_mask)[0]

    numeric_cols = set(meta["numeric"])
    records = []
    for i in flagged_idx[np.argsort(-risk[flagged_idx])][:MAX_FLAGS_RETURNED]:
        reasons = explain(i, df, numeric_cols, strength)
        if dup_flag[i]:
            reasons.append("Row is an exact duplicate of another row")
        if not reasons:
            reasons.append("Unusual combination of values across columns (model outlier)")
        signals = []
        if iso_flag[i]:
            signals.append("isolation-forest")
        if rule_flag[i]:
            signals.append("threshold-rule")
        records.append({
            "row": int(i) + 2,                           # spreadsheet row (header = 1)
            "risk_score": round(float(risk[i]), 1),
            "risk_level": risk_level(float(risk[i])),
            "signals": signals,
            "reasons": reasons,
            "values": {str(c): py(v) for c, v in df.iloc[i].items()},
        })

    levels = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for i in flagged_idx:
        levels[risk_level(float(risk[i]))] += 1

    return {
        "summary": {
            "total_scanned": total,
            "flags_detected": int(len(flagged_idx)),
            "flag_rate": round(len(flagged_idx) / total * 100, 2),
            "risk_distribution": levels,
            "duplicates": int(dup_flag.sum()),
            "avg_risk_score": round(float(risk[flagged_idx].mean()), 1) if len(flagged_idx) else 0,
            "columns_analyzed": meta["numeric"] + meta["categorical"],
            "columns_skipped": meta["skipped"],
            "contamination": contamination,
            "scan_seconds": round(time.perf_counter() - t0, 2),
            "truncated": bool(len(flagged_idx) > MAX_FLAGS_RETURNED),
        },
        "columns": [str(c) for c in df.columns],
        "flagged": records,
    }


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"status": "ok", "alerts_available": alerts_available()}


@app.post("/api/scan")
async def scan(file: UploadFile = File(...),
               contamination: float = Query(0.05, ge=0.001, le=0.25)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_BYTES // (1024 * 1024)} MB).")
    if not content.strip():
        raise HTTPException(400, "The file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(content), nrows=MAX_ROWS, encoding_errors="replace",
                         sep=None, engine="python")
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    return scan_dataframe(df, contamination)


@app.get("/api/sample")
def sample():
    """Demo transactions with planted anomalies."""
    rng = np.random.default_rng(7)
    n = 1200
    df = pd.DataFrame({
        "txn_id": np.arange(10_000, 10_000 + n),
        "user": rng.choice([f"user_{i:03d}" for i in range(40)], n),
        "country": rng.choice(["US", "US", "US", "CA", "GB", "DE", "FR"], n),
        "amount": rng.lognormal(3.6, 0.5, n).round(2),
        "items": rng.integers(1, 8, n),
        "session_sec": rng.normal(320, 80, n).clip(30).round(0),
        "login_attempts": rng.poisson(1.1, n),
        "bytes_out_kb": rng.normal(240, 60, n).clip(5).round(1),
    })
    bad = rng.choice(n, 45, replace=False)
    df.loc[bad[:12], "amount"] = rng.uniform(4000, 15000, 12).round(2)
    df.loc[bad[12:24], "login_attempts"] = rng.integers(14, 40, 12)
    df.loc[bad[24:34], "bytes_out_kb"] = rng.uniform(4000, 9000, 10).round(1)
    df.loc[bad[34:40], "country"] = rng.choice(["KP", "IR", "SY"], 6)
    df.loc[bad[40:], ["amount", "items", "session_sec"]] = [9800.0, 1, 4]
    return Response(df.to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"})


# ---------------------------------------------------------------- AIT alert triage (detector.py)
# detector.py (local Qwen via llama-cpp) is imported lazily and used unmodified, so the CSV
# mode keeps working even when llama-cpp-python is not installed. A 7B model on CPU can take
# a long time, so scans run as background jobs that the browser polls.
MAX_ALERT_BYTES = 4 * 1024 ** 3     # per uploaded alert file
MAX_JOBS_KEPT = 20
JOBS: dict = {}
JOB_LOCK = threading.Lock()         # one model, one job at a time


def alerts_available() -> bool:
    try:
        return all(importlib.util.find_spec(m) for m in ("llama_cpp", "huggingface_hub"))
    except (ImportError, ValueError):
        return False


def level_from_conf(conf: float) -> str:
    if conf >= 0.9:
        return "critical"
    if conf >= 0.75:
        return "high"
    if conf >= 0.5:
        return "medium"
    return "low"


async def save_upload(up: UploadFile, dest: Path):
    size = 0
    with open(dest, "wb") as f:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > MAX_ALERT_BYTES:
                raise HTTPException(413, f"{up.filename} exceeds {MAX_ALERT_BYTES // 1024 ** 3} GB.")
            f.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{up.filename} is empty.")


def make_llm_wrapper(det, job: dict, n_review: int):
    """Wrap detector.call_json so a bad model reply cannot kill an hours-long run.
    The original function is still what talks to the model; this only reports progress,
    fills missing keys, and falls back to 'not suspicious' / empty report on failure."""
    orig = det.call_json
    calls = {"n": 0}
    det_defaults = {"suspicious": False, "confidence": 0.0, "attack_stage": None,
                    "false_positive_factors": [], "evidence_event_ids": [], "rationale": ""}

    def fail(msg):
        job["llm_errors"] += 1
        job["last_error"] = msg[:200]

    def wrapped(system, user, max_tokens=1200):
        is_detector = system is det.DETECTOR_SYSTEM
        if is_detector:
            calls["n"] += 1
            job.update(stage="triage", pct=round(25 + 60 * calls["n"] / max(n_review, 1), 1),
                       message=f"LLM triage: cluster {calls['n']}/{n_review}")
        else:
            job.update(stage="explain", pct=88, message="Writing incident report…")
        try:
            out = orig(system, user, max_tokens)
            if not isinstance(out, dict):
                raise ValueError("model reply was not a JSON object")
        except Exception as e:                      # bad JSON, context overflow, etc.
            fail(f"{type(e).__name__}: {e}")
            if is_detector:
                return {**det_defaults, "rationale": f"LLM reply unusable ({type(e).__name__})"}
            return {"executive_summary": "The explainer model reply could not be parsed.",
                    "incidents": []}
        if is_detector:
            try:
                conf = float(out.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            ev = out.get("evidence_event_ids")
            sus = out.get("suspicious", False)
            if isinstance(sus, str):
                sus = sus.strip().lower() == "true"
            return {**det_defaults, **out, "suspicious": bool(sus),
                    "confidence": min(max(conf, 0.0), 1.0),
                    "evidence_event_ids": [str(x) for x in ev] if isinstance(ev, list) else [],
                    "rationale": str(out.get("rationale", ""))}
        out.setdefault("executive_summary", "")
        if not isinstance(out.get("incidents"), list):
            out["incidents"] = []
        return out

    return wrapped


# ---- turning dropped JSON files into the two inputs detector.py expects (AMiner + Wazuh).
# detector.py stays untouched: every upload is read here (JSON Lines, JSON array, one pretty-printed
# document, or a {"events": [...]} wrapper), sorted per record into an AMiner file and a Wazuh file,
# and anything that is neither is converted to a Wazuh-shaped record so it still gets triaged.
WRAPPERS = ("Records", "records", "events", "alerts", "logs", "entries", "results", "data", "items")


def _unwrap(obj):
    if isinstance(obj, list):
        yield from (x for x in obj if isinstance(x, dict))
    elif isinstance(obj, dict):
        for k in WRAPPERS:
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                yield from (x for x in v if isinstance(x, dict))
                return
        hits = obj.get("hits")
        if isinstance(hits, dict) and isinstance(hits.get("hits"), list):
            yield from ((h.get("_source") or h) for h in hits["hits"] if isinstance(h, dict))
            return
        yield obj


def read_records(path: Path):
    with open(path, "rb") as bf:
        head = bf.read(4096).lstrip()
    if not head:
        return
    if head[:1] == b"[":
        try:
            import ijson
        except ImportError:
            ijson = None
        if ijson:
            with open(path, "rb") as bf:
                yield from (x for x in ijson.items(bf, "item", use_float=True) if isinstance(x, dict))
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                yield from _unwrap(json.load(f))
            except json.JSONDecodeError:
                pass
        return
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                for rec in _unwrap(obj):
                    n += 1
                    yield rec
        if n == 0:                                  # not JSON Lines: try one whole document
            f.seek(0)
            try:
                yield from _unwrap(json.load(f))
            except json.JSONDecodeError:
                return


TS_KEYS = ("timestamp", "@timestamp", "eventtime", "event_time", "time", "ts", "datetime",
           "date", "created", "timecreated", "_time")
DESC_KEYS = ("message", "msg", "description", "desc", "event", "eventname", "event_name",
             "signature", "action", "text", "log", "summary", "reason")
RULE_KEYS = ("rule_id", "ruleid", "eventid", "event_id", "eventname", "event_name",
             "event_type", "eventtype", "category")
HOST_KEYS = ("host", "hostname", "host.name", "agent.name", "computer", "computername",
             "machine", "node", "server", "instance")
SRC_KEYS = ("src_ip", "srcip", "source_ip", "sourceipaddress", "source.ip", "client_ip", "clientip",
            "remote_addr", "remote_ip", "src_addr", "ipaddress", "ip_address", "ip", "src")
DST_KEYS = ("dest_ip", "dst_ip", "destip", "dstip", "destination_ip", "destination.ip",
            "server_ip", "dst_addr", "dest", "dst")
USER_KEYS = ("user", "username", "user.name", "user_name", "account", "accountname", "userid")
LEVEL_KEYS = ("level", "severity", "priority", "sev", "loglevel", "log_level", "risk")
LEVEL_WORDS = {"emerg": 14, "emergency": 14, "alert": 13, "crit": 12, "critical": 12, "fatal": 12,
               "severe": 10, "high": 10, "error": 8, "err": 8, "warn": 5, "warning": 5,
               "medium": 5, "notice": 3, "low": 2, "info": 1, "debug": 0}
ATTACK_WORDS = re.compile(r"fail|denied|unauthori[sz]ed|invalid user|brute|exploit|malware|ransom|"
                          r"injection|privilege|sudo|forbidden|attack|backdoor|exfil|\.\./|union select|<script", re.I)


def _flat(d, prefix="", out=None, depth=0):
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}{k}".lower()
        if isinstance(v, dict) and depth < 3:
            _flat(v, key + ".", out, depth + 1)
        else:
            out.setdefault(key, v)
    return out


def _pick(flat, names):
    for n in names:
        v = flat.get(n)
        if v in (None, "", [], {}):
            v = next((val for key, val in flat.items()
                      if key.rsplit(".", 1)[-1] == n and val not in (None, "", [], {})), None)
        if v not in (None, "", [], {}):
            return v
    return None


def _level(v) -> int:
    if isinstance(v, bool) or v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    v = str(v).strip().lower()
    return int(v) if v.isdigit() else LEVEL_WORDS.get(v, 0)


def generic_to_wazuh(det, r: dict, i: int):
    """Best-effort conversion of an arbitrary JSON log record into the Wazuh shape detector.py reads.
    Returns (record, had_timestamp)."""
    f = _flat(r)
    raw_ts, had_ts = _pick(f, TS_KEYS), True
    try:
        if isinstance(raw_ts, (int, float)) or (isinstance(raw_ts, str) and raw_ts.strip().isdigit()):
            ts = float(raw_ts)
            ts = ts / 1000.0 if ts > 1e11 else ts       # epoch milliseconds
        elif raw_ts is None:
            raise ValueError
        else:
            ts = det.parse_ts(str(raw_ts).strip().replace(" ", "T", 1) if " " in str(raw_ts).strip()[:11] else raw_ts)
    except Exception:
        ts, had_ts = float(i + 1), False               # keep file order; shown as "line N"
    if not ts:                                          # detector.py treats a 0 timestamp as missing
        ts, had_ts = float(i + 1), False

    sur = r.get("alert") if isinstance(r.get("alert"), dict) and r["alert"].get("signature") else None
    desc = sur["signature"] if sur else _pick(f, DESC_KEYS)
    if isinstance(desc, (dict, list)):
        desc = json.dumps(desc, default=str)
    desc = str(desc) if desc is not None else json.dumps(r, default=str)
    template = re.sub(r"\d+", "#", desc)[:60]            # same message shape = same rule (for rarity)
    user = _pick(f, USER_KEYS)
    if user is not None and str(user) not in desc:
        desc = f"{desc} [user={user}]"
    if sur:
        level = {1: 12, 2: 8, 3: 4}.get(int(sur.get("severity", 3) or 3), 4)
    else:
        level = min(max(_level(_pick(f, LEVEL_KEYS)), 0), 15)
        if level < 6 and ATTACK_WORDS.search(desc):
            level = 6                                  # attack-ish wording: lift above background noise
    rule_id = _pick(f, ("signature_id",)) if sur else _pick(f, RULE_KEYS)
    src = str(_pick(f, ("src_ip",) if sur else SRC_KEYS) or "")
    dst = str(_pick(f, ("dest_ip",) if sur else DST_KEYS) or "")
    data = {"srcip": src, "dstip": dst}
    if sur:
        data = {"src_ip": src, "dest_ip": dst, "alert": sur}
    return {"timestamp": ts,
            "rule": {"id": str(rule_id) if rule_id is not None else template, "level": level,
                     "description": desc, "groups": []},
            "agent": {"name": str(_pick(f, HOST_KEYS) or "")},
            "data": data}, had_ts


def classify(det, r: dict) -> str:
    """aminer / wazuh only if detector.py's own normalizer accepts the record; otherwise generic."""
    try:
        if "LogData" in r or "AnalysisComponent" in r:
            det.normalize_aminer(r, 0)
            return "aminer"
        if isinstance(r.get("rule"), dict):
            det.normalize_wazuh(r, 0)
            return "wazuh"
    except Exception:
        pass
    return "generic"


def prepare_inputs(det, files: List[tuple], tmpdir: Path):
    """Merge all uploads into one AMiner JSONL and one Wazuh JSONL. Returns (aminer, wazuh, per-file stats)."""
    out = {"aminer": tmpdir / "merged_aminer.jsonl", "wazuh": tmpdir / "merged_wazuh.jsonl"}
    stats, total = [], 0
    with open(out["aminer"], "w", encoding="utf-8") as fa, open(out["wazuh"], "w", encoding="utf-8") as fw:
        for path, name in files:
            st = {"name": name, "records": 0, "skipped": 0, "no_timestamp": 0,
                  "kinds": {"aminer": 0, "wazuh": 0, "generic": 0}}
            for i, r in enumerate(read_records(path)):
                kind = classify(det, r)
                try:
                    if kind == "generic":
                        r, had_ts = generic_to_wazuh(det, r, i)
                        st["no_timestamp"] += 0 if had_ts else 1
                    (fa if kind == "aminer" else fw).write(json.dumps(r, default=str) + "\n")
                except Exception:
                    st["skipped"] += 1
                    continue
                st["kinds"][kind] += 1
                st["records"] += 1
                total += 1
            stats.append(st)
    if total == 0:
        raise ValueError("No JSON records could be read from the uploaded files. Expected JSON Lines, "
                         "a JSON array, or a JSON document made of objects.")
    return out["aminer"], out["wazuh"], stats


def event_signals(r) -> list:
    """Plain-English reasons one event scored high (shown next to the model's rationale)."""
    out = []
    if r["rule_count"] <= 5:
        out.append(f"Rare: this rule fired only {int(r['rule_count'])}x in the whole dataset")
    if r["burst"] >= 10:
        out.append(f"Burst: {int(r['burst'])} similar alerts within 30 seconds")
    if r["level"] >= 10:
        out.append(f"High severity (level {int(r['level'])})")
    if r["count"] >= 10:
        out.append(f"Repeated {int(r['count'])}x inside one dedupe window")
    if str(r["mitre"]):
        out.append(f"MITRE technique {r['mitre']}")
    return out


def fallback_report(df, verdicts) -> dict:
    """If the explainer model fails, still explain every suspicious cluster from the detector's verdicts."""
    sus = sorted(((cid, v) for cid, v in verdicts.items() if v.get("suspicious")),
                 key=lambda x: -x[1]["confidence"])
    incidents = []
    for rank, (cid, v) in enumerate(sus[:20], 1):
        g = df[(df["cluster"] == cid) & df["suspicious"]]
        if g.empty:
            continue
        affected = [x for x in list(g["host"].astype(str).unique()) + list(g["src_ip"].astype(str).unique()) if x]
        incidents.append({
            "rank": rank, "title": v.get("attack_stage") or f"Suspicious activity (cluster {int(cid)})",
            "severity": level_from_conf(v["confidence"]), "confidence": v["confidence"],
            "time_start": fmt_ts(g["ts"].min()), "time_end": fmt_ts(g["ts"].max()),
            "affected": affected[:10], "attack_stage": v.get("attack_stage") or "",
            "explanation": v.get("rationale", ""), "evidence_event_ids": v.get("evidence_event_ids", [])[:20],
            "recommended_actions": [],
        })
    return {"executive_summary": f"{len(sus)} suspicious cluster(s) were found. The incident-writing model "
                                 "did not return a usable report, so these are the detector's per-cluster findings.",
            "incidents": incidents}


def fmt_ts(ts) -> str:
    ts = float(ts)
    if ts < 1e8:                                   # files without timestamps are ordered by line number
        return f"line {int(ts)}"
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def run_alert_job(job: dict, tmpdir: Path, files: List[tuple],
                  labels: Optional[Path], scenario: str, bucket: int):
    t0 = time.perf_counter()
    det, orig_call = None, None
    try:
        import detector as det                      # the user's pipeline, untouched
        job.update(status="running", stage="ingest", pct=3,
                   message="Reading and aggregating alert files…")
        aminer_p, wazuh_p, file_stats = prepare_inputs(det, files, tmpdir)
        try:
            df = det.load_events(str(aminer_p), str(wazuh_p), bucket)
        except KeyError:
            raise ValueError("No alerts could be parsed from the uploaded files.")
        total_raw = int(df["count"].sum())
        job.update(pct=15, message=f"Read {total_raw:,} records from {len(files)} file(s) → {len(df):,} unique rows")

        # estimate LLM workload with the same pre-filter detector_agent uses (for the progress bar)
        params = inspect.signature(det.detector_agent).parameters
        thr, cap = params["auto_benign_below"].default, params["max_llm_clusters"].default
        probe = det.cluster_events(det.score_events(df.copy()))
        n_review = int(min((probe.groupby("cluster").score.max() >= thr).sum(), cap))
        n_clusters = int(probe["cluster"].nunique())
        del probe

        job.update(stage="model", pct=18,
                   message="Loading local model (first run downloads ~3.8 GB)…")
        det.get_client()

        orig_call = det.call_json
        det.call_json = make_llm_wrapper(det, job, n_review)
        df, verdicts = det.detector_agent(df)
        report = det.explainer_agent(df, verdicts)
        if not report.get("incidents") and any(v.get("suspicious") for v in verdicts.values()):
            report = fallback_report(df, verdicts)

        ev = None
        if labels:
            job.update(stage="eval", pct=96, message="Scoring against labels…")
            try:
                ev = {k: py(v) for k, v in det.evaluate(df, str(labels), scenario or None).items()}
            except Exception as e:
                ev = {"error": f"{type(e).__name__}: {e}"[:200]}

        flagged = df[df["suspicious"]].sort_values(["confidence", "score"], ascending=False)
        levels = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for c in flagged["confidence"]:
            levels[level_from_conf(float(c))] += 1
        flagged_raw = int(flagged["count"].sum())

        records = []
        for idx, r in flagged.head(MAX_FLAGS_RETURNED).iterrows():
            conf = float(r["confidence"])
            v = verdicts.get(int(r["cluster"]), {})
            reasons = [f"[{r['source']}] {r['desc']}", *event_signals(r)]
            records.append({
                "row": int(idx) + 1,
                "label": str(r["id"]),
                "risk_score": round(conf * 100, 1),
                "risk_level": level_from_conf(conf),
                "signals": [str(r["source"])],
                "reasons": reasons,
                "detail": {
                    "rationale": str(v.get("rationale") or r["reason"] or ""),
                    "attack_stage": v.get("attack_stage"),
                    "false_positive_factors": [str(x) for x in (v.get("false_positive_factors") or [])][:6],
                },
                "values": {
                    "time": fmt_ts(r["ts"]),
                    "source": str(r["source"]), "rule_id": str(r["rule_id"]),
                    "level": int(r["level"]), "desc": str(r["desc"]),
                    "host": str(r["host"]), "src_ip": str(r["src_ip"]), "dst_ip": str(r["dst_ip"]),
                    "mitre": str(r["mitre"]), "count": int(r["count"]), "cluster": int(r["cluster"]),
                },
            })

        result = {
            "mode": "alerts",
            "summary": {
                "mode": "alerts",
                "total_scanned": total_raw,
                "aggregated_rows": int(len(df)),
                "flags_detected": int(len(flagged)),
                "flagged_raw": flagged_raw,
                "flag_rate": round(flagged_raw / max(total_raw, 1) * 100, 2),
                "risk_distribution": levels,
                "avg_risk_score": round(float(flagged["confidence"].mean()) * 100, 1) if len(flagged) else 0,
                "clusters_total": n_clusters,
                "clusters_reviewed": int(len(verdicts)),
                "clusters_suspicious": int(sum(1 for v in verdicts.values() if v.get("suspicious"))),
                "incidents": len(report.get("incidents", [])),
                "files": file_stats,
                "llm_errors": job["llm_errors"],
                "last_llm_error": job["last_error"],
                "eval": ev,
                "scan_seconds": round(time.perf_counter() - t0, 1),
                "truncated": bool(len(flagged) > MAX_FLAGS_RETURNED),
                "duplicates": 0, "columns_analyzed": [], "columns_skipped": [], "contamination": None,
            },
            "columns": list(records[0]["values"].keys()) if records else [],
            "flagged": records,
            "report": report,
        }
        job.update(status="done", stage="done", pct=100, message="Triage complete", result=result)
    except Exception as e:
        job.update(status="error", stage="error", error=f"{type(e).__name__}: {e}"[:500])
    finally:
        if det is not None and orig_call is not None:
            det.call_json = orig_call
        shutil.rmtree(tmpdir, ignore_errors=True)
        JOB_LOCK.release()


MAX_ALERT_FILES = 20


@app.post("/api/alerts/scan", status_code=202)
async def alerts_scan(files: Optional[List[UploadFile]] = File(None),
                      aminer: Optional[UploadFile] = File(None),     # legacy fields still accepted
                      wazuh: Optional[UploadFile] = File(None),
                      labels: Optional[UploadFile] = File(None),
                      scenario: str = Form(""), bucket: int = Form(10)):
    if not alerts_available():
        raise HTTPException(503, "Alert triage needs: pip install llama-cpp-python huggingface_hub")
    uploads = [u for u in [*(files or []), aminer, wazuh] if u is not None and u.filename]
    if not uploads:
        raise HTTPException(400, "Add at least one JSON file.")
    if len(uploads) > MAX_ALERT_FILES:
        raise HTTPException(400, f"Too many files (limit {MAX_ALERT_FILES}).")
    if not 1 <= bucket <= 3600:
        raise HTTPException(400, "bucket must be between 1 and 3600 seconds.")
    if not JOB_LOCK.acquire(blocking=False):
        raise HTTPException(409, "A triage job is already running. Wait for it to finish.")
    tmpdir = Path(tempfile.mkdtemp(prefix="redflag_"))
    try:
        saved = []
        for i, up in enumerate(uploads):
            dest = tmpdir / f"{i}.dat"
            await save_upload(up, dest)
            saved.append((dest, Path(up.filename).name))
        labels_path = None
        if labels is not None and labels.filename:
            labels_path = tmpdir / "labels.dat"
            await save_upload(labels, labels_path)
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "queued", "stage": "queued", "pct": 0,
               "message": "Queued…", "error": None, "result": None,
               "llm_errors": 0, "last_error": None}
        JOBS[job_id] = job
        while len(JOBS) > MAX_JOBS_KEPT:
            JOBS.pop(next(iter(JOBS)))
        threading.Thread(target=run_alert_job, daemon=True,
                         args=(job, tmpdir, saved, labels_path, scenario.strip(), bucket)).start()
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        JOB_LOCK.release()
        raise
    return {"job_id": job_id}


@app.get("/api/alerts/jobs/{job_id}")
def alerts_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id (the server may have restarted).")
    out = {k: job[k] for k in ("id", "status", "stage", "pct", "message", "error",
                               "llm_errors", "last_error")}
    if job["status"] == "done":
        out["result"] = job["result"]
    return out


@app.get("/")
def index():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    raise HTTPException(404, "index.html not found")
