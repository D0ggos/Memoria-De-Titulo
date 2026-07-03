"""Extrae descriptores por sistema de DB_ssf_RS_500_c.mat -> descriptors.csv + arrays.npz"""
import numpy as np, scipy.io as sio, pandas as pd, time, sys

MAT = "../../DB_ssf_RS_500_c.mat"
OUT = "."

d = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
BASE = d['BASE']
cases = int(d['cases'])
print("BASE", BASE.shape, "cases", cases)

def verts(x, r, c):
    xs = list(x) if (isinstance(x, np.ndarray) and x.dtype == object) else [x]
    return [np.asarray(v, float).reshape(r, c) for v in xs]

def ctrb_rank(A, B):
    n = A.shape[0]
    M = [B]
    for _ in range(1, n):
        M.append(A @ M[-1])
    return np.linalg.matrix_rank(np.hstack(M))

rows = []
ol_re, ol_im, cl_re, cl_im = [], [], [], []   # autovalores para histogramas
t0 = time.time()
n_empty = 0
for oi in range(BASE.shape[0]):
    for ii in range(BASE.shape[1]):
        for vi in range(BASE.shape[2]):
            for ci in range(BASE.shape[3]):
                e = BASE[oi, ii, vi, ci]
                if not hasattr(e, '_fieldnames'):
                    n_empty += 1
                    continue
                nx, nu, N = oi + 1, ii + 1, vi + 1
                A = verts(e.A, nx, nx); B = verts(e.B, nx, nu)
                K = np.asarray(e.K, float).reshape(nu, nx)

                ol_max = []; normA2 = []; normB2 = []; condA = []; ctrb_ok = True
                cl_max = []
                for Ai, Bi in zip(A, B):
                    wo = np.linalg.eigvals(Ai)
                    ol_max.append(wo.real.max())
                    normA2.append(np.linalg.norm(Ai, 2)); condA.append(np.linalg.cond(Ai))
                    normB2.append(np.linalg.norm(Bi, 2))
                    if ctrb_rank(Ai, Bi) < nx: ctrb_ok = False
                    wc = np.linalg.eigvals(Ai + Bi @ K)
                    cl_max.append(wc.real.max())
                    ol_re.extend(wo.real); ol_im.extend(wo.imag)
                    cl_re.extend(wc.real); cl_im.extend(wc.imag)

                Abar = np.mean(A, axis=0); Bbar = np.mean(B, axis=0)
                dispA = max(np.linalg.norm(Ai - Abar, 2) for Ai in A) if N > 1 else 0.0
                dispB = max(np.linalg.norm(Bi - Bbar, 2) for Bi in B) if N > 1 else 0.0

                rows.append(dict(
                    nx=nx, nu=nu, N=N, case=ci + 1,
                    absc_ol_max=max(ol_max), absc_ol_mean=float(np.mean(ol_max)),
                    frac_unstable_vert=float(np.mean([x > 0 for x in ol_max])),
                    normA2_max=max(normA2), normA2_mean=float(np.mean(normA2)),
                    normB2_max=max(normB2), normB2_mean=float(np.mean(normB2)),
                    condA_max=max(condA),
                    ctrb_full=ctrb_ok,
                    dispA=dispA, dispB=dispB,
                    absc_cl_max=max(cl_max), alpha_ach=-max(cl_max),
                    stabilized=bool(max(cl_max) < 0),
                    normK2=np.linalg.norm(K, 2), normK_fro=np.linalg.norm(K, 'fro'),
                ))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/descriptors.csv", index=False)
np.savez_compressed(f"{OUT}/arrays.npz",
                    ol_re=np.array(ol_re), ol_im=np.array(ol_im),
                    cl_re=np.array(cl_re), cl_im=np.array(cl_im))
print(f"sistemas={len(df)}  celdas_vacias={n_empty}  t={time.time()-t0:.1f}s")
print("memoria CSV cols:", list(df.columns))
print(df.describe(include='all').T[['count','mean','min','max']])
PY = None
