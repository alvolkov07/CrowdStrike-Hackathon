"""
AIT alert dataset -> Detector agent -> short summary of what looks suspicious

Usage:
  pip install -r requirements.txt
  python detector.py --aminer Data/ait_ads/fox_aminer.json \
         --wazuh Data/ait_ads/fox_wazuh.json --out out/

The command line now only prints/writes a SUMMARY of what could be wrong (per log file / source),
instead of per-event flags, CSVs and evaluation. The lower-level functions below (load_events,
score_events, cluster_events, detector_agent, explainer_agent, evaluate, call_json, ...) keep
their original names and signatures because app.py imports and uses them.

Requires `pip install llama-cpp-python huggingface_hub`. The ~3.8 GB model is downloaded
and cached on the first LLM request.
"""
import argparse, json, math, os, re
from datetime import datetime, timezone
from typing import Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import pandas as pd  # (for JSON-array files also: pip install ijson)

MODEL = "Qwen/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-7b-instruct-q3_k_m.gguf"
client: Optional[Llama] = None

def get_client() -> Llama:
    global client
    if client is None:
        client = Llama(
            model_path=hf_hub_download(repo_id=MODEL, filename=MODEL_FILE),
            n_ctx=8192,
            verbose=False,
        )
    return client

# 1. INGEST + NORMALIZE  (adjust field names to your files)

def iter_json_records(path):
    """Stream one record at a time; never holds the whole file in memory."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        while first.strip() == "":
            first = f.read(1)
            if first == "":
                return
        f.seek(0)
        if first == "[":                      # one big JSON array -> pip install ijson
            import ijson
            yield from ijson.items(f, "item", use_float=True)
        else:                                 # JSON Lines (the usual AIT format)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

def parse_ts(v):
    if isinstance(v, (int, float)):
        return float(v)
    v = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", v)  # +0000 -> +00:00
    return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()

def normalize_wazuh(r, i):
    rule = r.get("rule", {})
    data = r.get("data", {}) or {}
    sur = data.get("alert", {}) if isinstance(data.get("alert"), dict) else {}
    desc = sur.get("signature") or rule.get("description", "")
    return dict(
        id=f"wazuh-{i}", ts=parse_ts(r.get("timestamp") or r["@timestamp"]),
        source="suricata" if sur else "wazuh",
        rule_id=str(sur.get("signature_id") or rule.get("id", "")),
        level=int(rule.get("level", 0)), desc=desc,
        host=(r.get("agent") or {}).get("name", ""),
        src_ip=data.get("src_ip") or data.get("srcip", ""),
        dst_ip=data.get("dest_ip") or data.get("dstip", ""),
        mitre=",".join((rule.get("mitre") or {}).get("id", [])),
        groups=",".join(rule.get("groups", [])),
    )

def normalize_aminer(r, i):
    ld, ac = r.get("LogData", {}), r.get("AnalysisComponent", {})
    ts = (ld.get("DetectionTimestamp") or ld.get("Timestamps") or [0])[0]
    return dict(
        id=f"aminer-{i}", ts=float(ts), source="aminer",
        rule_id=ac.get("AnalysisComponentName", ac.get("AnalysisComponentType", "")),
        level=0, desc=ac.get("Message", ""), host="",
        src_ip="", dst_ip="", mitre="",
        groups="",
    )

def load_events(aminer_path, wazuh_path, bucket=10):
    """Stream both files and collapse duplicates on the fly.
    Identical alerts (same source/rule/desc/host/IPs) inside one `bucket`-second
    window become a single row with a `count`. Raw JSON is discarded immediately,
    so memory scales with distinct alerts, not file size."""
    agg = {}

    def add(ev):
        ev["desc"] = ev["desc"][:200]
        key = (ev["source"], ev["rule_id"], ev["desc"], ev["host"],
               ev["src_ip"], ev["dst_ip"], int(ev["ts"] // bucket))
        a = agg.get(key)
        if a is None:
            ev["count"] = 1
            agg[key] = ev
        else:
            a["count"] += 1

    for i, r in enumerate(iter_json_records(aminer_path)):
        add(normalize_aminer(r, i))
    for i, r in enumerate(iter_json_records(wazuh_path)):
        add(normalize_wazuh(r, i))

    df = pd.DataFrame(agg.values())
    del agg
    for c in ["source", "rule_id", "host", "src_ip", "dst_ip", "mitre", "groups"]:
        df[c] = df[c].astype("category")
    df["level"] = df["level"].astype("int8")
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"Loaded {int(df['count'].sum()):,} raw alerts -> {len(df):,} aggregated rows")
    return df

# DETECTOR AGENT
#    Stage A: cheap deterministic scoring (rarity, severity, bursts)
#    Stage B: cluster events into candidate incidents
#    Stage C: LLM triage per cluster with explicit false-positive reasoning

def score_events(df):
    # Rarity: rules that fire constantly are probably noise
    df["rule_count"] = df.groupby(["source", "rule_id"], observed=True)["count"].transform("sum")
    df["rarity"] = df["rule_count"].apply(lambda c: 1 / math.log2(c + 2))
    # Burst: events from same rule within +/-30s
    df["bucket"] = (df.ts // 30).astype(int)
    df["burst"] = df.groupby(["rule_id", "bucket"], observed=True)["count"].transform("sum")
    df["burst_s"] = df.burst.clip(upper=50) / 50
    df["sev"] = df.level.clip(upper=15) / 15
    df["score"] = 0.4 * df.rarity + 0.3 * df.sev + 0.3 * df.burst_s
    return df

def cluster_events(df, gap=120):
    """Time-gap clustering. Swap for (src_ip,host)-keyed clustering if you prefer."""
    df["cluster"] = (df.ts.diff() > gap).cumsum()
    return df

def cluster_brief(g, n_samples=8):
    top_rules = (g.groupby(["source", "rule_id", "desc"], observed=True)["count"]
                   .sum().nlargest(10)).items()
    samples = g.sort_values("score", ascending=False).head(n_samples)
    top_rules = list(top_rules)
    return {
        "time_start": datetime.fromtimestamp(g.ts.min(), timezone.utc).isoformat(),
        "time_end": datetime.fromtimestamp(g.ts.max(), timezone.utc).isoformat(),
        "n_events": int(g["count"].sum()),
        "sources": g.groupby("source", observed=True)["count"].sum().to_dict(),
        "hosts": [x for x in g.host.astype(str).unique() if x][:10],
        "src_ips": [x for x in g.src_ip.astype(str).unique() if x][:10],
        "dst_ips": [x for x in g.dst_ip.astype(str).unique() if x][:10],
        "mitre": [m for m in g.mitre.astype(str).unique() if m][:10],
        "top_rules": [{"source": s, "rule_id": r, "desc": d[:120], "count": int(c)}
                      for ((s, r, d), c) in top_rules],
        "sample_events": samples[["id", "source", "rule_id", "level", "desc", "host",
                                  "src_ip", "dst_ip", "score"]].round(3).to_dict("records"),
    }

DETECTOR_SYSTEM = """You are a SOC triage agent reviewing a cluster of IDS alerts
(AMiner anomaly detector, Wazuh HIDS, Suricata NIDS). Decide if the cluster is
suspicious. Be skeptical: common false positives include routine admin logins,
scheduled jobs, package updates, health checks, repeated low-level rules, and
single-source alerts with no corroboration. Raise confidence when multiple
independent sources agree, when activity follows an attack progression
(scan -> exploit -> escalation -> persistence/exfil), or when MITRE tags align.
Respond with JSON only, no prose or code fences:
{"suspicious": bool, "confidence": 0-1, "attack_stage": str|null,
 "false_positive_factors": [str], "evidence_event_ids": [str], "rationale": str}"""

def _extract_json(txt):
    """Parse model output; tolerate code fences or stray prose around the JSON object."""
    txt = txt.strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def call_json(system, user, max_tokens=1200):
    resp = get_client().create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
    )
    txt = resp["choices"][0]["message"]["content"]
    if not txt:
        raise ValueError(f"{MODEL} returned an empty response")
    return _extract_json(txt)

def detector_agent(df, auto_benign_below=0.12, max_llm_clusters=2):
    # Kept for app.py (it expects event-level flags + per-cluster verdicts).
    df = score_events(df)
    df = cluster_events(df)
    verdicts = {}
    summ = df.groupby("cluster").score.agg(["max", "mean", "size"])
    # Cheap pre-filter: clusters with only low-scoring, low-volume noise skip the LLM
    to_review = summ[summ["max"] >= auto_benign_below].sort_values("max", ascending=False)
    for cid in to_review.index[:max_llm_clusters]:
        g = df[df.cluster == cid]
        verdicts[cid] = call_json(DETECTOR_SYSTEM, json.dumps(cluster_brief(g)))
    # Event-level flags: inherit cluster verdict, require the event itself to matter
    sus = {c: v for c, v in verdicts.items() if v["suspicious"]}
    evidence = {e for v in sus.values() for e in v["evidence_event_ids"]}
    in_sus = df.cluster.isin(sus.keys())
    df["suspicious"] = in_sus & (df.id.isin(evidence) | (df.score >= 0.25))
    df["confidence"] = df.cluster.map({c: v["confidence"] for c, v in sus.items()}).fillna(0.0) * df.suspicious
    df["reason"] = df.cluster.map({c: v["rationale"][:200] for c, v in sus.items()}).fillna("").where(df.suspicious, "")
    return df, verdicts

# EXPLAINER AGENT (kept for app.py; the command line no longer uses it)

EXPLAINER_SYSTEM = """You are an incident-response analyst. You receive suspicious
alert clusters already triaged by a detector agent. Group related clusters into
incidents if they plausibly belong to one attack chain. For each incident write a
short plain-English explanation (what happened, which hosts/IPs, which stage of
the attack, why it is believed malicious) and rank incidents by priority
(severity x confidence x breadth). Cite evidence only by the event IDs provided.
Respond with JSON only:
{"executive_summary": str,
 "incidents": [{"rank": int, "title": str, "severity": "low|medium|high|critical",
   "confidence": 0-1, "time_start": str, "time_end": str, "affected": [str],
   "attack_stage": str, "explanation": str, "evidence_event_ids": [str],
   "recommended_actions": [str]}]}"""

def explainer_agent(df, verdicts):
    payload = []
    for cid, v in verdicts.items():
        if not v["suspicious"]:
            continue
        g = df[(df.cluster == cid) & df.suspicious]
        if g.empty:
            continue
        payload.append({"cluster": int(cid), "verdict": v, "brief": cluster_brief(g, 5)})
    payload.sort(key=lambda p: p["verdict"]["confidence"], reverse=True)
    if not payload:
        return {"executive_summary": "No suspicious activity found.", "incidents": []}
    return call_json(EXPLAINER_SYSTEM, json.dumps(payload[:60]), max_tokens=4000)


# eval against labels.csv (kept for app.py; not run from the command line any more)

def evaluate(df, labels_path, scenario):
    lab = pd.read_csv(labels_path)
    if "scenario" in lab.columns:
        lab = lab[lab.scenario == scenario]
    # Expected columns: start, end (epoch seconds). Adjust if yours differ.
    y = pd.Series(False, index=df.index)
    for _, r in lab.iterrows():
        y |= (df.ts >= r["start"]) & (df.ts <= r["end"])
    pred = df.suspicious
    w = df["count"]
    tp, fp, fn = w[pred & y].sum(), w[pred & ~y].sum(), w[~pred & y].sum()
    p = tp / max(tp + fp, 1); r_ = tp / max(tp + fn, 1)
    return dict(precision=p, recall=r_, f1=2 * p * r_ / max(p + r_, 1e-9),
                tp=int(tp), fp=int(fp), fn=int(fn))


# SUMMARY MODE: only report what could be suspicious, and what the problem is

def _iso(ts):
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()

def summarize_suspicious(df, auto_benign_below=0.12, max_llm_clusters=2, source_files=None):
    """Score + cluster + LLM-triage, then return a compact summary of suspicious activity.
    A bad model reply on one cluster is recorded as a note instead of crashing the run.
    No event-level flags, CSVs or label evaluation are produced."""
    source_files = source_files or {}
    df = score_events(df)
    df = cluster_events(df)
    summ = df.groupby("cluster").score.agg(["max", "mean", "size"])
    to_review = summ[summ["max"] >= auto_benign_below].sort_values("max", ascending=False)

    findings, notes = [], []
    reviewed = 0
    for cid in to_review.index[:max_llm_clusters]:
        g = df[df.cluster == cid]
        brief = cluster_brief(g)
        try:
            v = call_json(DETECTOR_SYSTEM, json.dumps(brief))
            if not isinstance(v, dict):
                raise ValueError("reply was not a JSON object")
        except Exception as e:
            notes.append(f"Cluster {int(cid)} ({brief['time_start']}): model reply unusable "
                         f"({type(e).__name__}: {str(e)[:100]})")
            continue
        reviewed += 1
        sus = v.get("suspicious", False)
        if isinstance(sus, str):
            sus = sus.strip().lower() == "true"
        if not sus:
            continue
        try:
            conf = min(max(float(v.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.0
        sources = sorted(brief["sources"])
        findings.append({
            "confidence": round(conf, 2),
            "problem": str(v.get("rationale", "")).strip() or "Model marked this cluster suspicious without a reason.",
            "attack_stage": v.get("attack_stage"),
            "time_start": brief["time_start"],
            "time_end": brief["time_end"],
            "n_alerts": brief["n_events"],
            "log_sources": sources,
            "log_files": sorted({source_files[s] for s in sources if s in source_files}),
            "hosts": brief["hosts"],
            "src_ips": brief["src_ips"],
            "dst_ips": brief["dst_ips"],
            "mitre": brief["mitre"],
            "top_alerts": [f"{r['desc']} (x{r['count']})" for r in brief["top_rules"][:3]],
            "benign_explanations_considered": [str(x) for x in (v.get("false_positive_factors") or [])][:4],
            "example_event_ids": [str(x) for x in (v.get("evidence_event_ids") or [])][:5],
        })
    findings.sort(key=lambda f: f["confidence"], reverse=True)

    total = int(df["count"].sum())
    if findings:
        headline = (f"{len(findings)} suspicious cluster(s) found out of {df['cluster'].nunique()} "
                    f"({reviewed} reviewed, {total:,} raw alerts).")
    else:
        headline = (f"No suspicious activity found ({reviewed} cluster(s) reviewed, "
                    f"{total:,} raw alerts).")
    if notes:
        headline += f" {len(notes)} cluster(s) could not be assessed."
    return {"headline": headline, "findings": findings, "notes": notes}

def format_summary(s):
    out = [s["headline"], ""]
    for n, f in enumerate(s["findings"], 1):
        out.append(f"{n}. [{f['confidence']:.0%}] {f['time_start']} -> {f['time_end']}  "
                   f"({f['n_alerts']} alerts)")
        out.append(f"   Problem: {f['problem']}")
        if f["attack_stage"]:
            out.append(f"   Stage:   {f['attack_stage']}")
        where = ", ".join(f["log_files"]) or ", ".join(f["log_sources"])
        out.append(f"   Logs:    {where}")
        who = f["hosts"] + f["src_ips"] + f["dst_ips"]
        if who:
            out.append(f"   Involves: {', '.join(dict.fromkeys(who))}")
        if f["mitre"]:
            out.append(f"   MITRE:   {', '.join(f['mitre'])}")
        if f["top_alerts"]:
            out.append(f"   Top alerts: {'; '.join(f['top_alerts'])}")
        out.append("")
    for note in s["notes"]:
        out.append(f"Note: {note}")
    return "\n".join(out).rstrip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aminer", required=True)
    ap.add_argument("--wazuh", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels", help="ignored in summary mode (kept so old commands still work)")
    ap.add_argument("--scenario", help="ignored in summary mode")
    ap.add_argument("--bucket", type=int, default=10,
                    help="seconds; identical alerts inside one bucket are merged")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_events(a.aminer, a.wazuh, a.bucket)
    summary = summarize_suspicious(df, source_files={
        "aminer": os.path.basename(a.aminer),
        "wazuh": os.path.basename(a.wazuh),
        "suricata": os.path.basename(a.wazuh),   # Suricata alerts arrive inside the Wazuh file
    })
    text = format_summary(summary)
    with open(f"{a.out}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(f"{a.out}/summary.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)