#!/usr/bin/env python3
"""
Download all logs from LatBOND v3.3.7 pilot jobs into one JSON file.

Usage:
    python download_logs_v337.py
    python download_logs_v337.py --output my_logs.json
    python download_logs_v337.py --job scnn
"""

import argparse
import subprocess
import json
import sys
from datetime import datetime


PROJECT_ID = "YOUR_GCP_PROJECT"
REGION = "northamerica-northeast2"


def run_gcloud(cmd_args):
    """Run a gcloud command with shell=True for Windows compatibility."""
    cmd = " ".join(cmd_args)
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result


def get_v337_jobs(filter_name=None):
    result = run_gcloud([
        "gcloud", "ai", "custom-jobs", "list",
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        "--filter=displayName:latbond-v337",
        "--format=json",
        "--sort-by=createTime",
    ])
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return []
    jobs = json.loads(result.stdout) if result.stdout.strip() else []
    if filter_name:
        jobs = [j for j in jobs if filter_name.lower() in j.get("displayName", "").lower()]
    return jobs


def get_all_logs(job_id):
    """Fetch all logs using resource.labels.job_id (catches both platform and container logs)."""
    log_filter = f'resource.labels.job_id="{job_id}"'
    result = run_gcloud([
        "gcloud", "logging", "read",
        f'"{log_filter}"',
        f"--project={PROJECT_ID}",
        "--format=json",
        "--limit=10000",
        "--order=asc",
    ])
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Download all v3.3.7 logs to JSON")
    parser.add_argument("--output", type=str, default="v337_pilot_logs.json",
                        help="Output filename (default: v337_pilot_logs.json)")
    parser.add_argument("--job", type=str, default=None,
                        help="Filter by name (e.g., 'scnn', 'cxfmr', 'mat')")
    args = parser.parse_args()

    print("=" * 60)
    print("  LatBOND v3.3.7 — Download All Logs")
    print("=" * 60)

    jobs = get_v337_jobs(args.job)
    if not jobs:
        print("[ERROR] No v337 jobs found.")
        sys.exit(1)

    print(f"  Found {len(jobs)} job(s)\n")

    all_logs = {}
    total_entries = 0

    for i, job in enumerate(jobs):
        display_name = job.get("displayName", "unknown")
        job_name = job.get("name", "")
        job_id = job_name.split("/")[-1]
        state = job.get("state", "UNKNOWN")

        print(f"  [{i+1}/{len(jobs)}] {display_name} [{state}] ... ", end="", flush=True)

        entries = get_all_logs(job_id)
        total_entries += len(entries)

        all_logs[display_name] = {
            "state": state,
            "job_id": job_id,
            "create_time": job.get("createTime", ""),
            "update_time": job.get("updateTime", ""),
            "log_count": len(entries),
            "logs": [
                {
                    "timestamp": e.get("timestamp", ""),
                    "source": e.get("logName", "").split("/")[-1],
                    "message": e.get("textPayload", "") or
                               e.get("jsonPayload", {}).get("message", str(e.get("jsonPayload", "")))
                }
                for e in entries
            ],
        }

        print(f"{len(entries)} entries")

    output = {
        "downloaded_at": datetime.now().isoformat(),
        "project": PROJECT_ID,
        "region": REGION,
        "total_jobs": len(jobs),
        "total_log_entries": total_entries,
        "jobs": all_logs,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Saved {total_entries} log entries from {len(jobs)} jobs")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
