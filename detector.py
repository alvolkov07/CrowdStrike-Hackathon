"""
AIT alert dataset -> Detector agent -> Explainer agent

Usage:
  pip install anthropic pandas
  export ANTHROPIC_API_KEY=...
  python ait_agent_pipeline.py --aminer fox_aminer.json --wazuh fox_wazuh.json \
         --out out/ [--labels labels.csv --scenario fox]

labels.csv is used ONLY for evaluation, never shown to the agents.
"""
import argparse, json, math, os, re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd  # (for JSON-array files also: pip install ijson)
import anthropic

MODEL = "claude-sonnet-5-5"
client = anthropic.Anthropic()

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
        id=f"wazuh-{i}", ts=parse_ts(r["timestamp"]),
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

def call_json(system, user, max_tokens=1200):
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    txt = "".join(b.text for b in resp.content if b.type == "text")
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)

def detector_agent(df, auto_benign_below=0.12, max_llm_clusters=300):
    df = score_events(df)
    df = cluster_events(df)
    verdicts = {}
    summ = df.groupby("cluster").score.agg(["max", "mean", "size"])
    # Cheap pre-filter: clusters with only low-scoring, low-volume noise skip the LLM
    to_review = summ[summ["max"] >= auto_benign_below].sort_values("max", ascending=False)
    for cid in to_review.index[:max_llm_clusters]:
        g = df[df.cluster == cid]
        try:
            verdicts[cid] = call_json(DETECTOR_SYSTEM, json.dumps(cluster_brief(g)))
        except Exception as e:  # fail open to the heuristic, but record it
            verdicts[cid] = {"suspicious": bool(g.score.max() > 0.5), "confidence": 0.3,
                             "attack_stage": None, "false_positive_factors": [],
                             "evidence_event_ids": [], "rationale": f"LLM error: {e}"}
    # Event-level flags: inherit cluster verdict, require the event itself to matter
    sus = {c: v for c, v in verdicts.items() if v["suspicious"]}
    evidence = {e for v in sus.values() for e in v["evidence_event_ids"]}
    in_sus = df.cluster.isin(sus.keys())
    df["suspicious"] = in_sus & (df.id.isin(evidence) | (df.score >= 0.25))
    df["confidence"] = df.cluster.map({c: v["confidence"] for c, v in sus.items()}).fillna(0.0) * df.suspicious
    df["reason"] = df.cluster.map({c: v["rationale"][:200] for c, v in sus.items()}).fillna("").where(df.suspicious, "")
    return df, verdicts

#EXPLAINER AGENT: summarize + rank

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


# eval against labels.csv (not seen by agents)

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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aminer", required=True)
    ap.add_argument("--wazuh", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--labels"); ap.add_argument("--scenario")
    ap.add_argument("--bucket", type=int, default=10,
                    help="seconds; identical alerts inside one bucket are merged")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load_events(a.aminer, a.wazuh, a.bucket)
    df, verdicts = detector_agent(df)
    df.to_csv(f"{a.out}/flagged_events.csv", index=False)

    report = explainer_agent(df, verdicts)
    json.dump(report, open(f"{a.out}/incident_report.json", "w"), indent=2)

    if a.labels:
        print("Eval:", evaluate(df, a.labels, a.scenario))
    print(report["executive_summary"])