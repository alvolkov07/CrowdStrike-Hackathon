# CrowdStrike Hackathon

Analyze AIT alerts from AMiner and Wazuh, triage suspicious clusters with
Qwen 2.5 7B Instruct, and produce an incident report.

## Requirements

- Python 3.10 or 3.11
- About 16 GB or more of NVIDIA GPU memory is recommended for the 7B model.
  CPU inference is supported but can be very slow and memory-intensive.
- The first run downloads the Qwen model weights from Hugging Face (roughly
  15 GB).

## Windows setup

Run these commands in PowerShell from the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch does not detect your NVIDIA GPU, install the PyTorch build that
matches your CUDA setup using the selector at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

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
