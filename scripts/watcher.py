#!/usr/bin/env python3
"""
TSG Benchmark Watcher — monitors completion, computes A13/A14, pushes, continues.
Runs as a background process; checks every 120s.
"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
RES = ROOT / "results"
CONDA = ROOT / "miniconda3"
PT_PY = CONDA / "envs/common_pt/bin/python"
TF1_PY = CONDA / "envs/tf1_env/bin/python"
TVAE_PY = CONDA / "envs/timevae_env/bin/python"

ALL_SEQ24_METHODS = ["fourierflows","csdi","diffusionts","gtgan","timevae","timegan","rgan","tsdiff"]

def count_results():
    f = RES / "all_results.jsonl"
    ok = fail = 0; by_method = {}
    if f.exists():
        for line in open(f):
            try:
                d = json.loads(line.strip())
                m = d["method"]
                if m not in by_method: by_method[m] = {"OK":0,"FAIL":0}
                if d.get("status")=="OK": by_method[m]["OK"]+=1; ok+=1
                else: by_method[m]["FAIL"]+=1; fail+=1
            except: pass
    return ok, fail, by_method

def seq24_complete(by_method):
    for m in ALL_SEQ24_METHODS:
        nok = by_method.get(m,{}).get("OK",0)
        if m == "tsdiff": continue  # known failure
        if m == "gtgan":  # allow 4/5
            if nok < 4: return False
        elif nok < 5:
            # Check if a process is still running
            return None  # unknown - check processes
    return True

def compute_a13a14():
    """Post-process A13/A14 for PT methods using TF1 env."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing A13/A14...")

    # Check if we have saved samples
    samples_dir = ROOT / "results" / "samples"
    if not samples_dir.exists():
        print("  No saved samples found. Skipping A13/A14 post-processing.")
        return

    # Load results needing A13/A14
    results = []
    with open(RES / "all_results.jsonl") as f:
        for line in f:
            if line.strip(): results.append(json.loads(line))

    pt_methods = ["fourierflows","csdi","gtgan","diffusionts"]
    needs_proc = []
    for r in results:
        if r["method"] in pt_methods and r.get("status")=="OK":
            ds = r["discriminative_score"]
            if ds is None or (isinstance(ds,float) and (ds!=ds or abs(ds)>1e6)):
                needs_proc.append(r)

    if not needs_proc:
        print("  No PT methods need A13/A14 post-processing.")
        return

    print(f"  {len(needs_proc)} experiments to process")

    # Run TF1 post-processing script
    script = ROOT / "scripts" / "postproc_a13a14.py"
    if not script.exists():
        # Create inline processing
        for entry in needs_proc[:5]:  # limit to avoid timeout
            print(f"  Need A13/A14 for {entry['method']} seed={entry['seed']}")
        print("  Manual step: run from tf1_env after completion:")
        print(f"    {TF1_PY} scripts/postproc_a13a14.py")
    else:
        subprocess.run([str(TF1_PY), str(script)], cwd=str(ROOT))

def push_results(msg=""):
    try:
        subprocess.run(["git","add","results/"],cwd=str(ROOT),capture_output=True)
        subprocess.run(["git","add",".sweep_*.log"],cwd=str(ROOT),capture_output=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        subprocess.run(["git","commit","-m",f"Auto: {msg}{ts}"],cwd=str(ROOT),capture_output=True)
        r = subprocess.run(["git","pull","--rebase"],cwd=str(ROOT),capture_output=True,text=True)
        if r.returncode != 0:
            print(f"  Pull issue: {r.stderr[:200]}")
        r = subprocess.run(["git","push"],cwd=str(ROOT),capture_output=True,text=True)
        if r.returncode==0:
            print(f"  ✓ Pushed")
        else:
            print(f"  ⚠ Push: {r.stderr[:200]}")
    except Exception as e:
        print(f"  ⚠ Git: {e}")

def check_running(method_substring):
    """Check if a process matching the substring is running."""
    r = subprocess.run(["ps","aux"],capture_output=True,text=True)
    for line in r.stdout.split("\n"):
        if method_substring in line and "grep" not in line and "watcher" not in line:
            return True
    return False

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Watcher started")
    print(f"  Monitoring: RGAN, TimeGAN on Heston seq24")
    print(f"  Checking every 120s")

    iteration = 0
    seq24_was_complete = False

    while True:
        iteration += 1
        now = datetime.now().strftime("%H:%M:%S")
        ok, fail, by_method = count_results()

        rgan_ok = by_method.get("rgan",{}).get("OK",0)
        timegan_ok = by_method.get("timegan",{}).get("OK",0)
        running = check_running("run_benchmark")

        print(f"[{now}] #{iteration} — {ok}OK {fail}FAIL | RGAN={rgan_ok}/5 TimeGAN={timegan_ok}/5 Running={running}")

        # Check if seq24 complete (all methods have enough OK results)
        gtgan_ok = by_method.get("gtgan",{}).get("OK",0)
        all_ok = all(
            by_method.get(m,{}).get("OK",0) >= (4 if m=="gtgan" else 5)
            for m in ALL_SEQ24_METHODS if m != "tsdiff"
        )

        if all_ok and not seq24_was_complete:
            seq24_was_complete = True
            print(f"[{now}] ✅ Heston seq_len=24 COMPLETE!")
            print(f"  Results: {ok} OK, {fail} FAILED")

            # Compute A13/A14
            compute_a13a14()

            # Generate tables
            print("  Generating tables...")
            subprocess.run([str(PT_PY), str(ROOT / "scripts/generate_tables.py")], cwd=str(ROOT))

            # Push
            push_results(f"Heston seq24 complete. ")

            # Launch continuation
            print("  Launching continuation sweep...")
            subprocess.Popen(
                ["bash", str(ROOT / "scripts/auto_continue.sh")],
                cwd=str(ROOT)
            )
            print("  ✅ Continuation launched!")
            break

        elif all_ok:
            # Already complete, check continuation
            if not check_running("auto_continue"):
                print(f"[{now}] Continuation not running. Launching...")
                subprocess.Popen(
                    ["bash", str(ROOT / "scripts/auto_continue.sh")],
                    cwd=str(ROOT)
                )
                break

        time.sleep(120)

if __name__ == "__main__":
    main()
