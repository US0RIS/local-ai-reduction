from __future__ import annotations
import numpy as np

def reconstruction_metrics(reference,approx):
    a=np.asarray(reference,dtype=np.float64).reshape(-1); b=np.asarray(approx,dtype=np.float64).reshape(-1); e=a-b
    mse=float(np.mean(e*e)); power=float(np.mean(a*a)); nmse=mse/max(power,1e-30); cos=float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    return {"mse":mse,"nmse":nmse,"cosine":cos,"snr_db":float(-10*np.log10(max(nmse,1e-30)))}

def linear_output_nmse(weight,approx,n_inputs=64,seed=0):
    rng=np.random.default_rng(seed); x=rng.standard_normal((weight.shape[1],n_inputs),dtype=np.float32); y=weight.astype(np.float32)@x; yh=approx.astype(np.float32)@x
    return float(np.mean((y-yh)**2)/max(float(np.mean(y**2)),1e-30))
