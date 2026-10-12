# CrowdStrike Hackathon — Red Flag Detector

Red Flag Detector is a local-first security-log triage and anomaly-scanning
project. It has two ways to use the same repository:

1. A browser website with a FastAPI backend. It scans transaction-style CSVs
   with statistical anomaly detection and can triage uploaded security logs
   with a local language model.
2. A command-line program that reads AMiner and Wazuh JSON logs and writes a
   concise suspicious-activity summary.

The project is a prototype for local analysis, not a hardened, multi-user
production service.

## Repository guide

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI application: serves the website, implements CSV scanning and sample-data routes, accepts JSON log uploads, and runs/polls alert-triage jobs. |
| `detector.py` | Shared alert pipeline and command-line entry point: parses and normalizes AMiner/Wazuh events, scores and time-clusters them, asks the local LLM to triage up to six clusters, and formats summaries/reports. |
| `index.html` | Complete browser UI (HTML, CSS, and JavaScript in one file). It switches between CSV and JSON modes, calls the API, displays results, and exports CSV/JSON. |
| `requirements-web.txt` | Dependencies for running the website and CSV scanner. |
| `requirements.txt` | Dependencies used by the detector and local-model alert-triage path. Install this as well as `requirements-web.txt` to use JSON triage through the website. |
| `small_aminer.json`, `small_wazuh.json` | Small tracked JSON Lines examples, one AMiner-shaped and one Wazuh-shaped. Useful for inspecting supported record formats; they are not a test suite. |
| `LICENSE` | MIT license for this project. |
| `.gitignore` | Excludes local environments, datasets, outputs, caches, and other generated files. |

### Local-only folders

- `Data/` contains the larger AIT scenario datasets and labels. It is ignored
  by Git and is not included in a clone. Keep security logs and labels local
  unless they have been approved for sharing.
- `out/` and similarly named output folders are local/generated output
  locations and are ignored by Git.
- `.venv/` is the local Python environment. Create it on each machine; do not
  copy or commit it.
- `__pycache__/` contains generated Python bytecode.

There are currently **no checked-in automated test files or test runner
configuration**. The two `small_*.json` files are example inputs, not tests.
See [Validation](#validation) for checks that can be run without starting a
long model inference.

There is no packaged standalone `.exe`. Run the website and CLI as Python
modules/scripts using the project's virtual environment, as shown below.

## Requirements and installation

- Python 3.10 or 3.11 (the documented/tested range for this project).
- Windows instructions below use PowerShell.
- The website's CSV mode does not need the local language model.
- JSON alert triage uses `llama-cpp-python` and `huggingface_hub`. The first
  model-backed run downloads the Qwen 2.5 7B Instruct GGUF model from Hugging
  Face (roughly 4 GB) and caches it locally. Inference uses your machine's
  memory and CPU; it can be slow.

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The second requirements file installs the LLM binding for command-line use and
JSON triage. On Windows, if `llama-cpp-python` fails to build or reports that
`llama.dll` is not a valid Win32 application, install the matching prebuilt
64-bit CPU wheel into this same environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps --only-binary=:all: `
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python
```

The default supported setup is CPU inference. GPU acceleration requires a
`llama-cpp-python` build configured for the target hardware; see the
[llama-cpp-python installation guide](https://llama-cpp-python.readthedocs.io/en/latest/).

## Run the website

Start the backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. Keep the terminal running while using the site.
Press **Ctrl+C** in that terminal to stop it. `--reload` restarts the server
when watched source files change; omit it for a non-reloading local run.

The page is served by FastAPI from `index.html`, and the JavaScript calls the
API on the same origin. Opening the HTML as a local file is not the recommended
workflow; for direct-file development the UI accepts an `?api=http://host:port`
query parameter.

The visual design is a single-page dark, cyber-themed interface with two
analysis tabs, drag-and-drop and file pickers, progress views, result cards,
filters/search/sorting, and exports. Tailwind CSS and Space Grotesk/JetBrains
Mono fonts are loaded from public CDNs, so those presentation assets require
internet access; scanning requests and uploaded data go to the local backend.

## Website API

| Method and route | Purpose |
| --- | --- |
| `GET /` | Serves the browser UI from `index.html`. |
| `GET /api/health` | Returns backend status and whether the optional alert-triage imports are available. |
| `GET /api/sample` | Returns a generated 1,200-row CSV with synthetic planted anomalies. |
| `POST /api/scan?contamination=0.05` | Accepts one multipart field named `file` containing a CSV; returns scan summary and flagged rows. `contamination` must be from 0.001 through 0.25. |
| `POST /api/alerts/scan` | Accepts multipart JSON/JSONL log file(s) in repeated `files` fields; optional fields are `labels`, `scenario`, and `bucket`. Returns HTTP 202 and a `job_id`. Legacy `aminer` and `wazuh` fields are also accepted. |
| `GET /api/alerts/jobs/{job_id}` | Returns queued/running job progress or the completed result. Job IDs/results are in memory and are lost when the server restarts. |

The alert endpoint accepts up to 20 log files, each up to 4 GiB, and permits
only one alert-triage job at a time per server process. The CSV endpoint limits
uploads to 25 MiB and scans at most 200,000 rows. Result payloads return at
most the top 1,000 flagged rows.

### Alert input formats

The web backend accepts JSON Lines/NDJSON, JSON arrays, a JSON document, and
common wrappers such as `events`, `alerts`, `records`, `logs`, `items`, and
Elasticsearch-style `hits.hits`. It identifies AMiner and Wazuh records,
recognizes Suricata alerts inside Wazuh records, and best-effort maps generic
JSON objects into Wazuh-shaped events. Unreadable JSON Lines are skipped and
reported in per-file statistics.

The CLI takes AMiner and Wazuh inputs separately. AMiner timestamps are read
from `LogData`; Wazuh timestamps are taken from `timestamp` or `@timestamp`.
The CLI's JSON-array support uses `ijson`.

## Analysis design

### CSV anomaly scan

`app.py` builds features from numeric and low-cardinality categorical columns.
Identifier-like, constant, and very high-cardinality columns are skipped;
numeric missing values are filled with the median. It combines:

- Isolation Forest, using scaled feature values and the selected contamination
  rate.
- A robust median/MAD z-score rule for extreme numeric values and a rarity rule
  for uncommon categories.

The combined score ranks flagged rows and produces risk levels, signals, and
plain-language reasons. The demo CSV is generated deterministically with
planted outliers. Exact duplicate rows are noted on a flagged row, but
duplication alone is not a separate flagging rule.

### Security-alert triage

The alert pipeline in `detector.py` and `app.py` follows this flow:

1. Parse records and normalize AMiner, Wazuh, Suricata, or generic JSON into a
   common event shape. The web API writes temporary merged inputs and deletes
   its temporary directory when the job finishes.
2. Collapse identical events from the same source/rule/message/host/IPs inside
   the configurable time bucket (default 10 seconds), retaining a count.
3. Score events using rule rarity, severity, and short-window bursts, then
   split them into time-based candidate clusters when adjacent events are more
   than 120 seconds apart.
4. Skip low-scoring clusters and send at most **six** highest-scoring clusters
   to the local model. Six is a cap on clusters reviewed per run, not a
   request to create exactly six clusters.
5. Ask the model to judge suspiciousness, confidence, evidence, and a primary
   attack category. Categories are normalized to: **Scans**, **Webshell
   upload**, **Password breach**, **Privilege breach**, **Remote command
   execution**, and **Data exfiltration**. Unsupported/uncertain labels become
   `null`.
6. Build an incident report and map findings back to alert rows for the
   browser. If model JSON cannot be parsed, the web job records the error and
   uses a conservative fallback instead of terminating the server.

The 10-second deduplication bucket and 120-second clustering gap are separate
settings: the web UI exposes the deduplication bucket; the CLI accepts it with
`--bucket`. The 120-second gap is currently a code default in
`detector.cluster_events`.

Labels are optional in the website upload and are used only for evaluation;
they are not sent to the model. By contrast, the current CLI is summary-only:
it still accepts `--labels` and `--scenario` for compatibility, but both are
ignored and no evaluation is run by that entry point.

## Command-line program

Run it from the repository root with a matching AMiner/Wazuh pair:

```powershell
.\.venv\Scripts\python.exe detector.py `
  --aminer Data\ait_ads\fox_aminer.json `
  --wazuh Data\ait_ads\fox_wazuh.json `
  --out out\fox `
  --bucket 10
```

Arguments:

| Argument | Required | Meaning |
| --- | --- | --- |
| `--aminer PATH` | Yes | AMiner JSON Lines or JSON-array input. |
| `--wazuh PATH` | Yes | Wazuh JSON Lines or JSON-array input. |
| `--out DIR` | No | Output directory; defaults to `out`. Created if needed. |
| `--bucket SECONDS` | No | Deduplicate matching events inside this many seconds; defaults to 10. |
| `--labels PATH` | No | Accepted for compatibility but ignored in summary mode. |
| `--scenario NAME` | No | Accepted for compatibility but ignored in summary mode. |

The CLI prints a text summary and writes:

- `summary.json` — structured headline, findings, evidence context, and notes.
- `summary.txt` — human-readable version of the summary.

The CLI does **not** currently write `flagged_events.csv` or
`incident_report.json`; those names in older instructions are outdated. The
website instead offers downloads for flagged alerts as CSV and the completed
alert report as JSON.

## Validation

There is no checked-in unit/integration test suite. These quick checks validate
the local Python environment and syntax without downloading or running the
language model:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q app.py detector.py
```

After starting Uvicorn, verify the service and UI:

```powershell
Invoke-RestMethod http://localhost:8000/api/health
Invoke-WebRequest http://localhost:8000/
```

For a full CSV smoke test, open the site, select **CSV ANOMALY SCAN**, and
choose **Try demo data**. A full JSON alert test initializes the local model,
may download about 4 GB on first use, and can take substantial time and memory.

## Security and operational notes

- This is intended for local, trusted use. The development CORS middleware
  permits all origins; there is no authentication, authorization, or
  multi-user isolation. Do not expose it to an untrusted network as-is.
- Uploaded logs are processed on the local machine by the backend and local
  model. Hugging Face is contacted to download the model if it is not cached.
- Temporary upload files are removed when an alert job completes. In-memory
  jobs and results disappear on server restart.
- Preserve the ignored `Data/` files and local outputs; do not add sensitive
  logs, credentials, model weights, or the `.venv/` to Git.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
