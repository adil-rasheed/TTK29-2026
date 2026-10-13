#!/usr/bin/env python3
"""Execute and pair Part VI notebooks in dependency order; record each run.

Run from anywhere with the course environment. One kernel at a time keeps CPU
wall timings interpretable. Use --only <stem> to repeat a corrected notebook.
"""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
import jupytext,nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[2]
ORDER=['mpc_model_box','vessel_loop_models','mpc_field_models','mpc_surge_horizon',
       'mpc_lqr','mpc_constrained','mpc_horizon_realtime','mpc_vessel',
       'koop_dmdc_control','koop_lifting','koop_mpc','koop_vessel','koop_hybrid_models',
       'rl_policy_gradient','rl_shaping','rl_vs_mpc','rl_learned_twins',
       'safe_domain','safe_tube_mpc','safe_vessel','safe_shielding_linear','safe_hybrid_bounds','safe_shielding']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',nargs='+');args=ap.parse_args()
    paths={p.stem:p for i in range(27,31) for p in (ROOT/'python').glob(f'{i}-*/notebooks/*.py')}
    manifest=ROOT/'python/control_execution_results.json'
    previous={r['notebook']:r for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    runs=[]
    for name in ORDER:
        if args.only and name not in args.only:continue
        p=paths[name];nb=jupytext.read(p);start=time.perf_counter()
        try:
            NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(p.parent)}}).execute()
            status='passed'
        except Exception as e:
            status='failed';print(str(e),flush=True)
        nbformat.write(nb,p.with_suffix('.ipynb'))
        row={'notebook':name,'status':status,'wall_seconds':time.perf_counter()-start}
        runs.append(row);print(json.dumps(row),flush=True)
        previous[name]=row
        manifest.write_text(json.dumps([previous[n] for n in ORDER if n in previous],indent=2)+'\n')
    if any(r['status']!='passed' for r in runs):raise SystemExit(1)
if __name__=='__main__':main()
