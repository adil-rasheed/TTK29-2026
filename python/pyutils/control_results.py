"""Strict JSON results and provenance for Part VI's executed notebooks."""
from __future__ import annotations
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np


def numeric(value,depth=0):
    if depth>5: return None
    if isinstance(value,(str,bool,int)):return value
    if isinstance(value,(float,np.number)):
        x=float(value)
        return x if np.isfinite(x) else None
    if isinstance(value,np.ndarray) and value.dtype.kind in 'bifu':
        if value.size<=12000:return value.tolist()
        return {'shape':list(value.shape),'min':float(np.nanmin(value)),
                'max':float(np.nanmax(value)),'mean':float(np.nanmean(value)),
                'rms':float(np.sqrt(np.nanmean(value**2)))}
    if isinstance(value,(list,tuple)):
        out=[numeric(v,depth+1) for v in value]
        return out if all(v is not None for v in out) else None
    if isinstance(value,dict):
        return {str(k):n for k,v in value.items() if (n:=numeric(v,depth+1)) is not None}
    if hasattr(value,'z') and hasattr(value,'solve_ms'):
        return {'z':numeric(value.z),'u':numeric(value.u),'solve_ms':numeric(value.solve_ms),
                'median_ms':float(np.median(value.solve_ms)),'max_ms':float(np.max(value.solve_ms))}
    return None


def write_results(path,metrics,protocol=None,source=None):
    import scipy
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'provenance':{'python':sys.version.split()[0],
              'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
              'timing':'wall clock, CPU, pure Python; includes optimiser; machine dependent'},
             'protocol':protocol or {},'metrics':metrics}
    if source:
        payload['provenance']['source']=Path(source).name
        payload['provenance']['source_sha256']=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(payload,indent=2,allow_nan=False, default=lambda v: v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n');temp.replace(path)
    return payload


def notebook_results(namespace,name):
    """Keep named numeric outputs (including arrays used to draw the figures).

    Large training arrays are summarised; scalar outputs and metric tables are
    retained exactly. Call at the end, so an assertion failure cannot publish
    a successful-looking result file.
    """
    metrics={}
    for k,v in namespace.copy().items():
        if k.startswith('_') or callable(v):continue
        x=numeric(v)
        if x is not None and x!={}:metrics[k]=x
    source=namespace.get('__file__')
    return write_results(Path('../data')/(name+'_results.json'),metrics,
                         {'experiment':'legacy notebook; its documented seeds and protocol'},source)
