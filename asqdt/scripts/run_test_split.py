#!/usr/bin/env python3
"""
Three-way disjoint attack-split evaluation (advisor item 2.1).

The shipped pipeline fits kernels/bandwidths on 10 design-pool attack episodes
(ep 0..9, attack RNG seed+300+ep) and — under the corrected honest protocol —
calibrates thresholds and draws negatives from held-out NOMINAL episodes. But
the attack POSITIVES in run_v2_honest are still those same design-pool episodes.

This script adds genuinely disjoint attack splits by extending the episode
index range with fresh plant-noise seeds and fresh attack-parameter RNGs:
    design      ep 0..9    (unchanged: kernel/bandwidth/surrogate fit here)
    validation  ep 10..19  (fresh)   -- reserved (model selection)
    test        ep 20..39  (fresh, 20 episodes) -- positives for all main tables

Everything else (kernel theta, gamma, PCA, standardization, nominal
calibration/test windows) is frozen from the design pass. Only the attack
positives change: this isolates the effect of scoring on unseen attacks.

Outputs asqdt/results/test_split.json with, per (bench, family):
  batch MMD (Q/RBF/Lap) on the test split,
  held-out AUC/F1 (Q/RBF) on the test split (30 held-out nominal negatives),
  in-set counts on design vs test.
Run: python -m asqdt.scripts.run_test_split
"""
import sys, os, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.metrics.pairwise import rbf_kernel as sk_rbf, laplacian_kernel as sk_lap
from sklearn.metrics import f1_score, roc_auc_score

from asqdt import config as cfg
from asqdt.scripts.run_v2_revision import load_or_build_mp, RESULTS, save_stage
from asqdt.scripts.run_v2_honest import fresh_noms, ensure_tank_stats, kscore
from asqdt.evaluation.fastq import gram_from_sv, mmd_u
from asqdt.cpms.plant import (CPMSPlant, ObserverDT, ATTACK_GENERATORS,
                              legacy_detector_response, compute_impact)
from asqdt.cpms.data import run_episode, extract_one_window
from asqdt.cpms import cstr as C, tank as T

ALPHA = cfg.DELTA
LAB = {'fdi':'FDI','command':'Sinusoidal','ramp':'Ramp','composite':'Composite',
       'cstr_fdi':'FDI','cstr_thermal':'Thermal','cstr_concentration':'Conc.',
       'tank_bias':'Bias','tank_oscillation':'Oscill.','tank_drift':'Drift'}


def gen_attack_windows(bname, seed, mp, ep_lo, ep_hi):
    """Generate fresh attack windows for episodes [ep_lo, ep_hi) with membership.
    Uses the SAME standardization stats as the design pass (mp nom_mean/std)."""
    d = mp['data']
    fams = {}
    if bname == 'CNC':
        nm, ns, W_L = d['nom_mean'], d['nom_std'], 10
        Sri = d['Sigma_r_inv']; P = cfg.ATTACK_SET['CNC']
        for aname, agen in ATTACK_GENERATORS.items():
            W, imp, leg = [], [], []
            for ep in range(ep_lo, ep_hi):
                p = CPMSPlant(seed=seed+200+ep); t = ObserverDT(0.05, seed=seed+1)
                ref = p.reference(cfg.HORIZON)
                a = agen(cfg.HORIZON, np.random.RandomState(seed+300+ep), E_max=P['E_max'])
                pn = CPMSPlant(seed=seed+200+ep); tn = ObserverDT(0.05, seed=seed+1)
                dn = run_episode(pn, tn, ref, sensor_mask=np.ones(6))
                da = run_episode(p, t, ref, a, np.ones(6))
                W.append((extract_one_window(da['residuals'], W_L)-nm)/ns)
                imp.append(compute_impact(da['z_traj'], dn['z_traj']))
                leg.append(legacy_detector_response(da['residuals'], method='chi_sq', Sigma_r_inv=Sri))
            fams[aname] = dict(W=np.array(W), imp=imp, leg=leg,
                               inset=[imp[i]>=P['rho'] and leg[i]<=P['epsilon'] for i in range(len(imp))])
    elif bname == 'CSTR':
        nm, ns, W_L = d['nom_mean'], d['nom_std'], 8; P = cfg.ATTACK_SET['CSTR']
        norms=[]
        for ep in range(20):
            p=C.CSTRPlant(seed=seed+100+ep); t=C.CSTRObserverDT(0.05, seed=seed+1)
            norms.append(C.run_cstr_episode(p,t,p.reference(120),sensor_mask=np.ones(3))['residuals'])
        nn=np.linalg.norm(np.vstack(norms),axis=1); mu0=nn.mean()+2*nn.std()
        for aname, agen in C.CSTR_ATTACK_GENERATORS.items():
            W, imp, leg = [], [], []
            for ep in range(ep_lo, ep_hi):
                p=C.CSTRPlant(seed=seed+200+ep); t=C.CSTRObserverDT(0.05, seed=seed+1)
                ref=p.reference(120); a=agen(120, np.random.RandomState(seed+300+ep), E_max=P['E_max'])
                pn=C.CSTRPlant(seed=seed+200+ep); tn=C.CSTRObserverDT(0.05, seed=seed+1)
                dn=C.run_cstr_episode(pn,tn,ref,sensor_mask=np.ones(3))
                da=C.run_cstr_episode(p,t,ref,a,np.ones(3))
                k0=120//2-W_L//2
                W.append((da['residuals'][k0:k0+W_L].flatten()-nm)/ns)
                imp.append(compute_impact(da['z_traj'],dn['z_traj'],W_z=C.W_Z_CSTR))
                leg.append(C.cusum_legacy(da['residuals'],mu_0=mu0))
            fams[aname]=dict(W=np.nan_to_num(np.array(W)),imp=imp,leg=leg,
                             inset=[imp[i]>=P['rho'] and leg[i]<=P['epsilon'] for i in range(len(imp))])
    else:
        W_L=8; P=cfg.ATTACK_SET['Tank']
        nw=[]; allr=[]
        for ep in range(20):
            p=T.CoupledTankPlant(seed=seed+100+ep); t=T.TankObserverDT(0.05,seed=seed+1)
            dd=T.run_tank_episode(p,t,p.reference(120),sensor_mask=np.ones(2))
            allr.append(dd['residuals']); k0=120//2-W_L//2
            nw.append(dd['residuals'][k0:k0+W_L].flatten())
        nw=np.nan_to_num(np.array(nw)); nm=nw.mean(0); ns=nw.std(0)+1e-8
        cov=np.cov(np.vstack(allr).T)+1e-6*np.eye(2); Sig=np.linalg.inv(cov)
        for aname, agen in T.TANK_ATTACK_GENERATORS.items():
            W, imp, leg = [], [], []
            for ep in range(ep_lo, ep_hi):
                p=T.CoupledTankPlant(seed=seed+200+ep); t=T.TankObserverDT(0.05,seed=seed+1)
                ref=p.reference(120); a=agen(120,np.random.RandomState(seed+300+ep),E_max=P['E_max'])
                pn=T.CoupledTankPlant(seed=seed+200+ep); tn=T.TankObserverDT(0.05,seed=seed+1)
                dn=T.run_tank_episode(pn,tn,ref,sensor_mask=np.ones(2))
                da=T.run_tank_episode(p,t,ref,a,np.ones(2))
                k0=120//2-W_L//2
                W.append((np.nan_to_num(da['residuals'][k0:k0+W_L].flatten())-nm)/ns)
                imp.append(compute_impact(da['z_traj'],dn['z_traj']))
                leg.append(T.chi_squared_legacy(da['residuals'],Sig))
            fams[aname]=dict(W=np.nan_to_num(np.array(W)),imp=imp,leg=leg,
                             inset=[imp[i]>=P['rho'] and leg[i]<=P['epsilon'] for i in range(len(imp))])
    return fams


def main():
    MP = load_or_build_mp(set())
    ensure_tank_stats(MP)
    out = {}
    for bname in ('CNC','CSTR','Tank'):
        fams = list(MP[bname][cfg.SEEDS[0]]['data']['atk'].keys())
        acc = {a: dict(qmmd=[],rmmd=[],lmmd=[],qauc=[],rauc=[],qf1=[],rf1=[],
                       inset_test=[],inset_design=[]) for a in fams}
        for seed in cfg.SEEDS:
            mp = MP[bname][seed]; d = mp['data']
            mu,V,gamma = mp['mu'],mp['V'],mp['gamma']
            qcache,S_nom,K_q = mp['qcache'],mp['S_nom'],mp['K_q']
            Xn_p = mp['Xn_p']
            Kr = sk_rbf(Xn_p,Xn_p,gamma); Kl = sk_lap(Xn_p,Xn_p,gamma)
            # held-out nominal calibration(30)+test(30) — same as honest protocol
            F = (fresh_noms(bname,seed,d,60)-mu)@V
            Xcal, Xtest_neg = F[:30], F[30:]
            # test-split attack windows: ep 20..39 (disjoint from design ep0..9)
            test_fams = gen_attack_windows(bname,seed,mp,20,40)
            # design in-set counts for reference
            P=cfg.ATTACK_SET[bname]
            for a in fams:
                Xa = (np.nan_to_num(test_fams[a]['W'])-mu)@V
                # batch MMD on test split
                Sa=qcache.sv(Xa); Kaa=gram_from_sv(Sa); np.fill_diagonal(Kaa,1.0)
                acc[a]['qmmd'].append(mmd_u(K_q,Kaa,gram_from_sv(S_nom,Sa)))
                acc[a]['rmmd'].append(mmd_u(Kr,sk_rbf(Xa,Xa,gamma),sk_rbf(Xn_p,Xa,gamma)))
                acc[a]['lmmd'].append(mmd_u(Kl,sk_lap(Xa,Xa,gamma),sk_lap(Xn_p,Xa,gamma)))
                # held-out AUC/F1: negatives = 30 held-out nominal, positives = test-split attacks
                y=np.concatenate([np.zeros(30),np.ones(len(Xa))])
                sq_neg=kscore(K_q,gram_from_sv(S_nom,qcache.sv(Xtest_neg)))
                sq_pos=kscore(K_q,gram_from_sv(S_nom,qcache.sv(Xa)))
                sr_neg=kscore(Kr,sk_rbf(Xn_p,Xtest_neg,gamma))
                sr_pos=kscore(Kr,sk_rbf(Xn_p,Xa,gamma))
                tau_q=np.percentile(kscore(K_q,gram_from_sv(S_nom,qcache.sv(Xcal))),95)
                tau_r=np.percentile(kscore(Kr,sk_rbf(Xn_p,Xcal,gamma)),95)
                sq=np.concatenate([sq_neg,sq_pos]); sr=np.concatenate([sr_neg,sr_pos])
                acc[a]['qauc'].append(roc_auc_score(y,sq)); acc[a]['rauc'].append(roc_auc_score(y,sr))
                acc[a]['qf1'].append(f1_score(y,(sq>tau_q).astype(int),zero_division=0))
                acc[a]['rf1'].append(f1_score(y,(sr>tau_r).astype(int),zero_division=0))
                acc[a]['inset_test'].append(int(np.sum(test_fams[a]['inset'])))
                di=d['atk'][a]
                acc[a]['inset_design'].append(int(np.sum(
                    [di['impacts'][e]>=P['rho'] and di['legacy_responses'][e]<=P['epsilon']
                     for e in range(len(di['impacts']))])))
            print(f"  {bname} seed={seed} done", flush=True)
        out[bname]={a:{k:(float(np.mean(v)),float(np.std(v))) for k,v in acc[a].items()} for a in fams}
    save_stage('test_split', out)
    # print LaTeX-ready
    print("\n=== TEST-SPLIT held-out AUC/F1 (positives = disjoint ep20..39) ===")
    for b in ('CNC','CSTR','Tank'):
        for a in out[b]:
            r=out[b][a]
            print(f"  {b:5s} {LAB[a]:11s} Qmmd={r['qmmd'][0]:+.3f} Rmmd={r['rmmd'][0]:+.3f} | "
                  f"Q-AUC={r['qauc'][0]:.2f} R-AUC={r['rauc'][0]:.2f} | "
                  f"Q-F1={r['qf1'][0]:.2f} R-F1={r['rf1'][0]:.2f} | "
                  f"inset test={r['inset_test'][0]:.0f}/20 design={r['inset_design'][0]:.0f}/10")

if __name__=='__main__':
    main()
