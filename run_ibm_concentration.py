#!/usr/bin/env python3
"""
run_ibm_concentration.py — hardware concentration check for Supplementary R.

Estimates the fidelity kernel k_theta(x, x') = |<0| U(x)^dag U(x') |0>|^2 on an
IBM Quantum backend via compute-uncompute circuits (apply U(x'), then the
inverse of U(x), measure; k = P[all-zeros]) for the 75 window pairs and two
encoding scales exported by export_hw_inputs.py (150 circuits total).

Purpose: verify on hardware that the co-designed kernel's concentration at
c=1 (one overlapping similarity band) and the lift at c=0.1 are physical
properties of the encoding, not simulation artifacts. No detection claim.

Requirements:
    pip install "qiskit>=1.4" "qiskit-ibm-runtime>=0.34"
    Save your token once:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                          token="...", set_as_default=True)

Usage:
    python run_ibm_concentration.py --selftest            # no QPU: verify circuits
    python run_ibm_concentration.py --submit              # least-busy backend
    python run_ibm_concentration.py --submit --backend ibm_kingston
    python run_ibm_concentration.py --analyze JOB_ID      # retrieve + report

Budget: 150 circuits x 4096 shots on a Heron backend is a small fraction of
the Open Plan's 10 min/28 days QPU allowance (one job, typically <1 min QPU).

Caveat carried into Supp R: raw P(all-zeros) is biased DOWNWARD by readout
and gate errors, so hardware k values sit uniformly below the exact ones;
the object of interest is the BAND STRUCTURE (overlap at c=1, separation and
lift at c=0.1), which is robust to a uniform multiplicative error floor.
"""
import argparse, json, os, sys
import numpy as np

HW_INPUTS = os.environ.get('HW_INPUTS', 'asqdt/results/hw_inputs.json')
RESULTS_OUT = 'ibm_concentration_results.json'


# ---------------------------------------------------------------- circuits --
def build_fidelity_circuit(x, xp, theta, nq, depth):
    """Compute-uncompute circuit: U(x)^dag U(xp) |0>, measure all qubits.

    Encoding must mirror asqdt.quantum.kernel.QuantumCircuit.statevector:
      per layer l (0-indexed), per qubit i:
          RY(theta[l,i,0] * x[(l*nq+i) % d]) then RZ(theta[l,i,1] * x[...])
      then entangler CNOT(0,1), (1,2), ..., (nq-2,nq-1), CNOT(nq-1,0).
    Qubit index i maps directly to Qiskit qubit i; P(all-zeros) is unaffected
    by endianness.
    """
    from qiskit import QuantumCircuit as QkCircuit
    d = len(x)
    qc = QkCircuit(nq, nq)

    def apply_U(vec):
        for l in range(depth):
            for i in range(nq):
                j = (l * nq + i) % d
                qc.ry(float(theta[l][i][0] * vec[j]), i)
                qc.rz(float(theta[l][i][1] * vec[j]), i)
            for i in range(nq - 1):
                qc.cx(i, i + 1)
            if nq > 2:
                qc.cx(nq - 1, 0)

    apply_U(xp)                       # U(x')
    n_forward = len(qc.data)
    apply_U(x)                        # then invert U(x) gate-by-gate
    inverse_part = qc.data[n_forward:]
    del qc.data[n_forward:]
    for instr in reversed(inverse_part):
        qc.append(instr.operation.inverse(), instr.qubits)
    qc.measure(range(nq), range(nq))
    return qc


def all_circuits(cfg_json, scales=None):
    theta = cfg_json['theta']; nq, depth = cfg_json['n_qubits'], cfg_json['depth']
    vec = {k: np.array(v) for k, v in cfg_json['vectors'].items()}
    labels, circs = [], []
    for c in (scales or cfg_json['scales']):
        for bname, plist in cfg_json['bands'].items():
            for idx, (sa, i, sb, j) in enumerate(plist):
                circs.append(build_fidelity_circuit(
                    c * vec[sa][i], c * vec[sb][j], theta, nq, depth))
                labels.append((str(c), bname, idx))
    return labels, circs


# ---------------------------------------------------------------- selftest --
def selftest(cfg_json):
    """Exact Statevector check of every circuit against the exported reference."""
    from qiskit.quantum_info import Statevector
    labels, circs = all_circuits(cfg_json)
    worst = 0.0
    for (c, bname, idx), qc in zip(labels, circs):
        qc_nomeas = qc.remove_final_measurements(inplace=False)
        p0 = float(abs(Statevector.from_instruction(qc_nomeas)[0]) ** 2)
        ref = cfg_json['ref_fidelity'][c][bname][idx]
        worst = max(worst, abs(p0 - ref))
    ok = worst < 1e-9
    print(f"[selftest] {len(circs)} circuits vs exact reference: "
          f"max |P(0..0) - k_ref| = {worst:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------ submit --
def submit(cfg_json, backend_name, shots):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    service = QiskitRuntimeService()
    backend = (service.backend(backend_name) if backend_name else
               service.least_busy(operational=True, simulator=False,
                                  min_num_qubits=cfg_json['n_qubits']))
    print(f"[submit] backend: {backend.name}")
    labels, circs = all_circuits(cfg_json)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(circs)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa, shots=shots)
    print(f"[submit] job id: {job.job_id()}  ({len(circs)} circuits x {shots} shots)")
    json.dump({'job_id': job.job_id(), 'backend': backend.name, 'shots': shots,
               'labels': labels}, open('ibm_job_meta.json', 'w'))
    print("[submit] saved ibm_job_meta.json — run --analyze after completion.")
    return job.job_id()


# ----------------------------------------------------------------- analyze --
def analyze(cfg_json, job_id):
    from qiskit_ibm_runtime import QiskitRuntimeService
    meta = json.load(open('ibm_job_meta.json'))
    if job_id is None:
        job_id = meta['job_id']
    service = QiskitRuntimeService()
    job = service.job(job_id)
    result = job.result()
    nq = cfg_json['n_qubits']
    zeros = '0' * nq

    hw = {}
    for (c, bname, idx), pub in zip([tuple(l) for l in meta['labels']], result):
        counts = pub.data.c.get_counts() if hasattr(pub.data, 'c') else \
                 next(iter(pub.data.__dict__.values())).get_counts()
        total = sum(counts.values())
        hw.setdefault(c, {}).setdefault(bname, []).append(counts.get(zeros, 0) / total)

    print(f"\nSupplementary R table  (backend {meta['backend']}, "
          f"{meta['shots']} shots/circuit, 25 pairs/band)")
    print(f"{'scale':>6} {'band':>8} {'exact sim':>16} {'hardware':>16}")
    out = {'job_id': job_id, 'backend': meta['backend'], 'shots': meta['shots'],
           'bands': {}}
    for c in hw:
        for bname in ('ref-ref', 'ref-nom', 'ref-atk'):
            ref = np.array(cfg_json['ref_fidelity'][c][bname])
            h = np.array(hw[c][bname])
            print(f"{c:>6} {bname:>8} {ref.mean():>8.3f}±{ref.std():.3f} "
                  f"{h.mean():>8.3f}±{h.std():.3f}")
            out['bands'][f"{c}/{bname}"] = {
                'exact_mean': float(ref.mean()), 'exact_std': float(ref.std()),
                'hw_mean': float(h.mean()), 'hw_std': float(h.std()),
                'hw_values': h.tolist()}
    json.dump(out, open(RESULTS_OUT, 'w'), indent=1)
    print(f"\nsaved {RESULTS_OUT}")
    print("Reading: at c=1 the three hardware bands should overlap (concentration);"
          "\nat c=0.1 they should lift toward the exact ~0.82-0.87 values, minus a"
          "\nuniform hardware error floor. Band STRUCTURE, not absolute level, is"
          "\nthe reported finding.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--analyze', nargs='?', const='', default=None,
                    metavar='JOB_ID')
    ap.add_argument('--backend', default=None)
    ap.add_argument('--shots', type=int, default=4096)
    a = ap.parse_args()
    cfg_json = json.load(open(HW_INPUTS))
    if a.selftest:
        sys.exit(0 if selftest(cfg_json) else 1)
    if a.submit:
        if not selftest(cfg_json):
            sys.exit("selftest failed — not submitting")
        submit(cfg_json, a.backend, a.shots); return
    if a.analyze is not None:
        analyze(cfg_json, a.analyze or None); return
    ap.print_help()


if __name__ == '__main__':
    main()
