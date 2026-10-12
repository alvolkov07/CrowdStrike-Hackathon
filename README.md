# CrowdStrike Hackathon

Analyze AIT alerts from AMiner and Wazuh, triage suspicious clusters with
the Qwen 2.5 7B Instruct GGUF model locally, and produce an incident report.

## Requirements

- Python 3.10 or 3.11
- The first LLM request downloads the Qwen GGUF model from Hugging Face
  (roughly 3.8 GB) and caches it locally.
- Inference runs on your computer through llama.cpp; CPU inference is supported
  but can be slow. The model uses several gigabytes of memory while running.

## Windows setup

Run these commands in PowerShell from the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default `llama-cpp-python` installation uses CPU inference. For GPU
acceleration, install a build configured for your hardware as described in the
[llama-cpp-python installation guide](https://llama-cpp-python.readthedocs.io/en/latest/).

## Run

For example, analyze the `fox` dataset:

```powershell
python detector.py `
  --aminer Data\ait_ads\fox_aminer.json `
  --wazuh Data\ait_ads\fox_wazuh.json `
  --out out `
  --labels Data\labels.csv `
  --scenario fox
```

The script writes `flagged_events.csv` and `incident_report.json` to the output
folder. The labels file is used only to calculate evaluation metrics; it is not
sent to the model.

Replace `fox` in the input filenames and `--scenario` with another available
scenario name to analyze a different dataset.

## Data and outputs

`Data/` and `out/` are ignored by Git. Keep dataset files local unless the team
has agreed that they are appropriate to share. Do not commit credentials or
sensitive logs.
