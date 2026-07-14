#!/usr/bin/env python3
"""
TSG Benchmark Auto-Pilot — unattended overnight orchestrator.
Monitors running experiments, auto-launches next batch as GPUs free up.
Runs: Heston(24,64,128) → Stocks → Energy → Sinusoidal.
Saves results, pushes to GitHub, generates tables on completion.
"""
import json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent; os.chdir(ROOT)
CONDA = ROOT / "miniconda3"; RES = ROOT / "results"; RES.mkdir(exist_ok=True)
TF1_PY = CONDA / "envs/tf1_env/bin/python"
PT_PY = CONDA / "envs/common_pt/bin/python"
TVAE_PY = CONDA / "envs/timevae_env/bin/python"

METHODS = ["timegan","rgan","gtgan","timevae","fourierflows","csdi","tsdiff","diffusionts"]
SEEDS = [0,1,2,3,4]
# Seq-len per dataset
DS = {"heston":[24,64,128],"stocks":[24],"energy":[24],"sinusoidal":[24]}

def py(m): return TF1_PY if m in ("timegan","rgan") else TVAE_PY if m=="timevae" else PT_PY
def key(m,d,sl,s): return (m,d,sl,s)

def get_done():
    done=set()
    f=RES/"all_results.jsonl"
    if f.exists():
        for line in open(f):
            try:
                d=json.loads(line.strip())
                if d.get("status")=="OK": done.add(key(d["method"],d["dataset"],d["seq_len"],d["seed"]))
            except: pass
    return done

def count():
    ok=fail=0; f=RES/"all_results.jsonl"
    if f.exists():
        for line in open(f):
            try:
                d=json.loads(line.strip())
                if d.get("status")=="OK": ok+=1
                elif d.get("status")=="FAILED": fail+=1
            except: pass
    return ok,fail

def free_gpus():
    try:
        r=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
                        capture_output=True,text=True,timeout=5)
        gs=[]
        for l in r.stdout.strip().split("\n"):
            if l.strip():
                i,m=l.strip().split(", "); i,m=int(i),int(m)
                gs.append((i,m,m<200))
        return gs
    except: return [(i,0,True) for i in range(4)]

def launch(method,dataset,seq_len,gpu):
    p=py(method); log=f".sweep_{method}_{dataset}_s{seq_len}.log"
    cmd=[str(p),"-u",str(ROOT/"run_benchmark.py"),"--methods",method,"--datasets",dataset,
         "--seq-len",str(seq_len),"--seed","all","--gpus","0"]
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(gpu); env["PYTHONUNBUFFERED"]="1"
    with open(log,"a") as f: f.write(f"\n### LAUNCH {method} {dataset} s{seq_len} GPU{gpu} {datetime.now()}\n")
    return subprocess.Popen(cmd,stdout=open(log,"a"),stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))

def push(msg):
    subprocess.run(["git","add","results/"],cwd=str(ROOT),capture_output=True)
    subprocess.run(["git","commit","-m",msg],cwd=str(ROOT),capture_output=True)
    r=subprocess.run(["git","push"],cwd=str(ROOT),capture_output=True,text=True)
    if r.returncode==0: print(f"  ✓ Pushed: {msg}")
    else: print(f"  ⚠ Push: {r.stderr[:200]}")

print(f"\n{'#'*60}")
print(f"# AUTO-PILOT started {datetime.now()}")
print(f"{'#'*60}")

running={}; last_push=time.time()
iteration=0

while True:
    iteration+=1
    ok,fail=count(); done=get_done()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] #{iteration} — {ok}OK {fail}FAILED {len(done)}unique")

    # Check running procs
    for pid,info in list(running.items()):
        if info["proc"].poll() is not None:
            rc=info["proc"].poll()
            print(f"  PID{pid} ({info['method']} {info['dataset']} s{info['seq_len']}) done rc={rc}")
            del running[pid]
        else:
            # Check log for progress
            logfile=info.get("log")
            if logfile:
                try:
                    last=subprocess.run(["tail","-1",logfile],capture_output=True,text=True).stdout.strip()[:80]
                    if last: print(f"  RUNNING {info['method']}: {last}")
                except: pass

    gpus=free_gpus()
    free=[g[0] for g in gpus if g[2]]
    print(f"  GPUs: {[(g[0],g[1]) for g in gpus]} Free: {free}")

    # Launch next experiments
    launched=0
    for dataset,sq in DS.items():
        for seq_len in sq:
            for method in METHODS:
                if method=="tsdiff": continue  # known failure
                if launched>=len(free): break

                all_done=all(key(method,dataset,seq_len,s) in done for s in SEEDS)
                if all_done: continue

                already=any(i["method"]==method and i["dataset"]==dataset and i["seq_len"]==seq_len
                           for i in running.values())
                if already: continue

                gpu=free[launched]
                proc=launch(method,dataset,seq_len,gpu)
                running[proc.pid]={"method":method,"dataset":dataset,"seq_len":seq_len,"proc":proc,
                                  "gpu":gpu,"log":f".sweep_{method}_{dataset}_s{seq_len}.log"}
                print(f"  🚀 {method} {dataset} s{seq_len} GPU{gpu} PID{proc.pid}")
                launched+=1

    if launched==0 and not running and iteration >= 10:
        print(f"\n{'#'*60}")
        print(f"✅ ALL DONE! {ok}OK {fail}FAILED")
        subprocess.run([str(PT_PY),str(ROOT/"scripts/generate_tables.py")],cwd=str(ROOT))
        push(f"Full sweep complete: {ok}OK {fail}FAILED {datetime.now().strftime('%Y-%m-%d')}")
        break

    if time.time()-last_push>3600:
        push(f"Auto progress: {ok}OK {fail}FAILED {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        last_push=time.time()

    time.sleep(120)
