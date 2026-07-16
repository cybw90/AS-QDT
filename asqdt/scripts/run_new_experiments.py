#!/usr/bin/env python3
"""Two missing experiments — fast version."""
import sys, os, time, warnings, json
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asqdt import config as cfg
from asqdt.quantum.kernel import QuantumCircuit
from asqdt.detection.mmd import mmd_squared
from asqdt.cpms.data import generate_dataset, run_episode, extract_one_window
from asqdt.cpms.plant import CPMSPlant, ObserverDT, ATTACK_GENERATORS, legacy_detector_response, compute_impact
from asqdt.cpms.cstr import generate_cstr_dataset
from asqdt.cpms.tank import generate_tank_dataset
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS, exist_ok=True)

def pca_fit(X, dim):
    X = np.nan_to_num(X); mu = X.mean(0); C = X - mu
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    Vt = Vt[:min(dim, Vt.shape[0])]
    # --- Deterministic sign convention (svd_flip-style) -------------------
    # Make the largest-|.| entry of each principal component positive.
    # LAPACK/BLAS builds return singular vectors with arbitrary signs;
    # distance-based classical kernels (RBF/Laplacian) are invariant to
    # component sign flips, but the RY/RZ quantum encoding is not, so the
    # quantum-kernel design was backend-dependent without this convention.
    signs = np.sign(Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)])
    signs[signs == 0] = 1.0
    return mu, (Vt * signs[:, None]).T

def pw_scores(K_ref, kmaker, X_ref, X_test):
    mr = (np.sum(K_ref) - np.trace(K_ref)) / (len(K_ref) * (len(K_ref) - 1))
    return np.array([mr + kmaker(X_test[i:i+1], X_test[i:i+1])[0,0] - 2*np.mean(kmaker(X_ref, X_test[i:i+1])) for i in range(len(X_test))])

def pw_thresh(K_ref, kmaker, X_ref, alpha=0.05):
    return np.percentile(pw_scores(K_ref, kmaker, X_ref, X_ref), 100*(1-alpha))

def f1(y, s, t): return f1_score(y, (s > t).astype(int), zero_division=0)

def design_k(Xn, Xa, nq, depth, seed, ni=3):
    qc = QuantumCircuit(nq, depth); rng = np.random.RandomState(seed)
    th = qc.initial_params(rng); bm, bt = -np.inf, th.copy()
    for _ in range(ni):
        m = mmd_squared(qc.gram_matrix(Xn, th), qc.gram_matrix(Xa, th), qc.cross_gram(Xn, Xa, th))
        if m > bm: bm, bt = m, th.copy()
        g = qc.spsa_gradient(Xn, Xa, th, lambda x,a,t: mmd_squared(qc.gram_matrix(x,t), qc.gram_matrix(a,t), qc.cross_gram(x,a,t)), rng=rng)
        th += cfg.KERNEL_LR * (g - 0.02 * th)
    return qc, bt

def design_g(Xn, Xa):
    med = np.median(np.linalg.norm(Xn[:,None]-Xn[None,:], axis=2))
    gs = [g/(med**2+1e-8) for g in [0.1,0.5,1,2,5]]; bg, bv = gs[2], -np.inf
    for g in gs:
        v = mmd_squared(sk_rbf(Xn,Xn,g), sk_rbf(Xa,Xa,g), sk_rbf(Xn,Xa,g))
        if v > bv: bv, bg = v, g
    return bg

def main():
    t0 = time.time(); nq, depth, td = 4, 3, 12
    R = {}

    # ═══ EXP 1 ═══
    print("="*70+"\n  EXP 1: HELD-OUT EVALUATION\n"+"="*70)
    for bname, gfn, gkw in [
        ('CNC', generate_dataset, dict(mismatch_frac=0.05, W_L=10, E_max=2.0, legacy_method='chi_sq')),
        ('CSTR', generate_cstr_dataset, dict(mismatch_frac=0.05, n_nom=20, n_atk=10, H=120)),
        ('Tank', generate_tank_dataset, dict(mismatch_frac=0.05, n_nom=20, n_atk=10, H=120))]:
        print(f"\n  [{bname}] Design...")
        ths, gs = [], []
        for ds in [42, 123, 256]:
            d = gfn(seed=ds, **gkw); Xn, atk = d[0], d[1]
            mu, V = pca_fit(Xn, td); Xnp = (Xn-mu)@V
            Xap = (np.nan_to_num(np.vstack([a['windows'] for a in atk.values()]))-mu)@V
            _, th = design_k(Xnp, Xap, nq, depth, ds); ths.append(th)
            gs.append(design_g(Xnp, Xap[:len(Xap)//2]))
        at, ag = np.median(np.array(ths), axis=0), np.median(gs)

        print(f"  [{bname}] Test...")
        br = {}
        for ts in [789, 1024]:
            d = gfn(seed=ts, **gkw); Xn, atk = d[0], d[1]
            mu, V = pca_fit(Xn, td); Xnp = (Xn-mu)@V
            qc = QuantumCircuit(nq, depth)
            Kq, Kr, Kl = qc.gram_matrix(Xnp, at), sk_rbf(Xnp,Xnp,ag), sk_lap(Xnp,Xnp,ag)
            kq = lambda X,Y,th=at,qc_=qc: qc_.cross_gram(X,Y,th) if not(X is Y) else qc_.gram_matrix(X,th)
            kr = lambda X,Y,g=ag: sk_rbf(X,Y,g)
            kl = lambda X,Y,g=ag: sk_lap(X,Y,g)
            tq,tr,tl = pw_thresh(Kq,kq,Xnp), pw_thresh(Kr,kr,Xnp), pw_thresh(Kl,kl,Xnp)
            for an, ai in atk.items():
                Xat = (np.nan_to_num(ai['windows'])-mu)@V
                yt = np.concatenate([np.zeros(len(Xnp)), np.ones(len(Xat))])
                Xall = np.vstack([Xnp, Xat])
                if an not in br: br[an] = {'fq':[],'fr':[],'fl':[],'mq':[],'mr':[]}
                br[an]['fq'].append(f1(yt, pw_scores(Kq,kq,Xnp,Xall), tq))
                br[an]['fr'].append(f1(yt, pw_scores(Kr,kr,Xnp,Xall), tr))
                br[an]['fl'].append(f1(yt, pw_scores(Kl,kl,Xnp,Xall), tl))
                br[an]['mq'].append(mmd_squared(Kq, qc.gram_matrix(Xat,at), qc.cross_gram(Xnp,Xat,at)))
                br[an]['mr'].append(mmd_squared(Kr, sk_rbf(Xat,Xat,ag), sk_rbf(Xnp,Xat,ag)))

        R[f'exp1_{bname}'] = {}
        print(f"\n  {bname} HELD-OUT (test seeds 789,1024):")
        print(f"    {'Attack':<22} {'Q-Dir F1':>10} {'RBF F1':>10} {'Lap F1':>10} {'Q MMD':>10} {'R MMD':>10}")
        for an, v in br.items():
            s = f"    {an:<22} {np.mean(v['fq']):.3f}±{np.std(v['fq']):.3f} {np.mean(v['fr']):.3f}±{np.std(v['fr']):.3f} {np.mean(v['fl']):.3f}±{np.std(v['fl']):.3f} {np.mean(v['mq']):.4f} {np.mean(v['mr']):.4f}"
            print(s)
            R[f'exp1_{bname}'][an] = {k: f"{np.mean(v[k]):.4f}±{np.std(v[k]):.4f}" for k in v}

    # ═══ EXP 2 ═══
    print("\n"+"="*70+"\n  EXP 2: FIXED-ATTACK THEOREM-INSTANTIATING\n"+"="*70)
    seed=42; WL=10; mep=30; nep=30; delta=0.05; Emax=2.0
    fa = ATTACK_GENERATORS['fdi'](cfg.HORIZON, np.random.RandomState(seed+999), E_max=Emax)

    nw, anr = [], []
    for ep in range(mep):
        p = CPMSPlant(seed=seed+5000+ep); t = ObserverDT(mismatch_frac=0.05, seed=seed+1)
        d = run_episode(p, t, p.reference(cfg.HORIZON), sensor_mask=np.ones(cfg.N_OUTPUT))
        nw.append(extract_one_window(d['residuals'], WL)); anr.append(d['residuals'])
    Sri = np.linalg.inv(np.cov(np.vstack(anr).T) + 1e-6*np.eye(cfg.N_OUTPUT))

    aw, imps, legs = [], [], []
    for ep in range(nep):
        pa = CPMSPlant(seed=seed+6000+ep); ta = ObserverDT(mismatch_frac=0.05, seed=seed+1)
        r = pa.reference(cfg.HORIZON)
        pn = CPMSPlant(seed=seed+6000+ep); tn = ObserverDT(mismatch_frac=0.05, seed=seed+1)
        dn = run_episode(pn, tn, r, sensor_mask=np.ones(cfg.N_OUTPUT))
        da = run_episode(pa, ta, r, fa, sensor_mask=np.ones(cfg.N_OUTPUT))
        aw.append(extract_one_window(da['residuals'], WL))
        imps.append(compute_impact(da['z_traj'], dn['z_traj']))
        legs.append(legacy_detector_response(da['residuals'], method='chi_sq', Sigma_r_inv=Sri))

    nw=np.array(nw); aw=np.array(aw)
    nm,ns = nw.mean(0), nw.std(0)+1e-8
    nw=(nw-nm)/ns; aw=(np.nan_to_num(aw)-nm)/ns
    mu,V = pca_fit(nw, td); Xn=(nw-mu)@V; Xa=(aw-mu)@V
    inset = sum(1 for i in range(nep) if imps[i]>=0.5 and legs[i]<=100.0)

    Xnd,Xad = Xn[:15],Xa[:15]; Xnt,Xat = Xn[15:],Xa[15:]
    mt,nt = len(Xnt), len(Xat)
    qc, th = design_k(Xnd, Xad, nq, depth, seed, ni=4)
    gam = design_g(Xnd, Xad)

    Knn=qc.gram_matrix(Xnt,th); Kaa=qc.gram_matrix(Xat,th); Kna=qc.cross_gram(Xnt,Xat,th)
    mq = mmd_squared(Knn, Kaa, Kna)
    mr = mmd_squared(sk_rbf(Xnt,Xnt,gam), sk_rbf(Xat,Xat,gam), sk_rbf(Xnt,Xat,gam))
    beta = 4*1.0*np.sqrt(0.5*np.log(2/delta)*(1/mt+1/nt))

    kq = lambda X,Y,th_=th,qc_=qc: qc_.cross_gram(X,Y,th_) if not(X is Y) else qc_.gram_matrix(X,th_)
    kr_ = lambda X,Y,g=gam: sk_rbf(X,Y,g)
    tq=pw_thresh(Knn,kq,Xnt); tr_=pw_thresh(sk_rbf(Xnt,Xnt,gam),kr_,Xnt)
    yt=np.concatenate([np.zeros(mt),np.ones(nt)]); Xa_=np.vstack([Xnt,Xat])
    f1q=f1(yt, pw_scores(Knn,kq,Xnt,Xa_), tq)
    f1r=f1(yt, pw_scores(sk_rbf(Xnt,Xnt,gam),kr_,Xnt,Xa_), tr_)

    print(f"\n  Fixed attack: energy={np.sum(fa**2):.4f}, in-set: {inset}/{nep}")
    print(f"  Impact: [{min(imps):.2f}, {max(imps):.2f}], Legacy: [{min(legs):.2f}, {max(legs):.2f}]")
    print(f"\n  HELD-OUT (m={mt}, n={nt}):")
    print(f"  Q-Direct MMD²: {mq:.6f}  |  RBF MMD²: {mr:.6f}")
    print(f"  β_{{m,n,δ}}:     {beta:.4f}  |  τ_δ vacuous: {'YES' if beta>max(abs(mq),abs(mr)) else 'NO'}")
    print(f"  Q-Direct F1:   {f1q:.3f}   |  RBF F1:    {f1r:.3f}")

    R['exp2'] = {'in_set': f"{inset}/{nep}", 'mmd_q': float(mq), 'mmd_rbf': float(mr),
                 'beta': float(beta), 'vacuous': bool(beta>max(abs(mq),abs(mr))),
                 'f1_q': float(f1q), 'f1_rbf': float(f1r), 'm': mt, 'n': nt}

    with open(os.path.join(RESULTS, 'new_experiments.json'), 'w') as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\n  Total: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
