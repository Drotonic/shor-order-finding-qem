"""N,a-driven experiment preparation, execution and saved-count analysis.

Planning imports neither CUDA nor IBM account clients. Cloud submission is an
explicit ``--execute`` action; analysis never submits jobs or regenerates shots.
The publication's fixed portable_c1 entry points remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

SIM_SCALES = (1., 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.)
IBM_SCALES = (1., 1.4, 1.8, 2.2)


def positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def experiment_config(N, a, *, mode="simulator", scales=None, rates=None,
                      shots_per_scale=None, baseline_shots=None, seed=42,
                      batch_size=48, folding_patterns=10, backend_name=None):
    """Create an execution specification, without calculating or using order r."""
    N = positive_integer(N, "N", 3)
    if N % 2 == 0:
        raise ValueError("N must be odd; an even N already has factor 2")
    a = positive_integer(a, "a", 2)
    if a >= N or math.gcd(N, a) != 1:
        raise ValueError("Choose 1 < a < N with gcd(a,N)=1; a shared gcd is already a factor")
    if mode not in ("simulator", "ibm"):
        raise ValueError("mode must be simulator or ibm")
    bits = 2 * N.bit_length()
    grid = [float(x) for x in (scales if scales is not None else (SIM_SCALES if mode == 'simulator' else IBM_SCALES))]
    if (len(grid) < 4 or any(not math.isfinite(x) or x < 1 or x > 3 for x in grid)
            or grid[0] != 1 or any(x >= y for x, y in zip(grid, grid[1:]))):
        raise ValueError("Use at least four distinct increasing scales from 1 to 3, starting at 1")
    shots = positive_integer(shots_per_scale if shots_per_scale is not None else
                             (20_000 if mode == "simulator" else 10_000), "shots_per_scale")
    baseline = positive_integer(baseline_shots if baseline_shots is not None else len(grid)*shots,
                                "baseline_shots")
    if baseline < shots:
        raise ValueError("baseline_shots must include the shared scale-one shots")
    if (len(grid) + 3) * (1 << bits) * 8 > 512 * 1024**2:
        raise ValueError("Dense count arrays exceed the 512 MiB planning limit; choose smaller N")
    config = dict(N=N, a=a, bits=bits, scales=grid, shots_per_scale=shots,
                  baseline_shots=baseline, seed=positive_integer(seed, "seed", 0),
                  batch_size=positive_integer(batch_size, "batch_size"),
                  folding_patterns=positive_integer(folding_patterns, "folding_patterns"),
                  mode=mode, optimization_level=1)
    if mode == "simulator":
        values = [float(x) for x in (rates if rates is not None else [.0001])]
        if not values or any(not math.isfinite(x) or x < 0 or x >= 1 for x in values):
            raise ValueError("rates must be finite probabilities in [0,1)")
        if len(set(values)) != len(values):
            raise ValueError("rates must not repeat")
        if any(x >= y for x,y in zip(values, values[1:])):
            raise ValueError("rates must be strictly increasing")
        config["rates"] = values
    else:
        if rates is not None:
            raise ValueError("Hardware runs use backend noise, not a simulator p2 rate")
        if shots > 10_000_000 or baseline > 10_000_000:
            raise ValueError("Hardware shot allocations must not exceed 10,000,000 executions")
        if not isinstance(backend_name, str) or not backend_name.strip():
            raise ValueError("IBM mode requires an explicitly chosen backend_name")
        config["backend_name"] = backend_name
    return config


def plan(config):
    scales = config["scales"]
    per_scale = config["shots_per_scale"]
    baseline = config["baseline_shots"]
    per_rate = len(scales)*per_scale + (baseline-per_scale) + baseline
    return dict(configuration=config, qubits=2*config["N"].bit_length()+3,
                measured_bits=config["bits"], outcomes=1 << config["bits"],
                shots_per_condition=per_rate,
                total_circuit_shots=per_rate*len(config.get("rates", [None])),
                reset_baseline="scale-one shots + independent extension",
                raw_baseline="independent; lambda=1 only",
                known_order_used_for_synthesis=False,
                qpu_submitted=False, gpu_executed=False)


def reference_order(N, a):
    """Classical small-instance truth label for evaluation, not circuit synthesis."""
    value = 1
    for order in range(1, N+1):
        value = value*a % N
        if value == 1:
            return order
    raise ValueError("No multiplicative order found")


def _check_ideal_budget(bits, order):
    # Existing finite-QPE metric helper constructs multiple (order,D) arrays.
    if 4 * order * (1 << bits) * 8 > 512 * 1024**2:
        raise ValueError("Ideal-reference evaluation exceeds the 512 MiB working-array guard; saved counts are unchanged")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze_counts(counts_path, *, N, a, output_dir):
    """C1, beta=2, ridge=0 analysis for arbitrary (>=4) scales and shot totals.

    Input schema is deliberately separate from the archived fixed-20k workflow.
    No merged baseline is arbitrarily split to manufacture bootstrap parents.
    """
    from .portable_c1 import build_suite, decoder_selected_orders, METHODS
    from .sparse_zne_core import analyze_compressed
    from .metrics import distribution_metrics, ideal_distribution

    N = positive_integer(N, "N", 3)
    a = positive_integer(a, "a", 2)
    if a >= N or math.gcd(a, N) != 1:
        raise ValueError("Invalid N,a for reversible order finding")
    counts_path, output_dir = Path(counts_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Analysis destination already exists: {output_dir}")
    with np.load(counts_path, allow_pickle=False) as z:
        required = ("scale_counts", "raw_counts", "reset_extension_counts", "effective_lambdas")
        if set(required) - set(z.files):
            raise ValueError("Missing generic workflow count arrays")
        arrays = {k: z[k].copy() for k in required}
        for key, value in (("N", N), ("a", a)):
            if key in z and z[key].item() != value:
                raise ValueError(f"Count metadata {key} does not match analysis request")
    scales, raw, extension = (arrays[k] for k in required[:3])
    dimension = 1 << (2*N.bit_length())
    if scales.ndim != 2 or scales.shape[1] != dimension or scales.shape[0] < 4:
        raise ValueError("Expected at least four scales with D=2**(2*N.bit_length())")
    for name, values, shape in (("scales", scales, scales.shape), ("raw",raw,(dimension,)),
                                ("extension",extension,(dimension,))):
        if values.shape != shape or values.dtype.kind not in 'iu' or np.any(values < 0):
            raise ValueError(f"{name} must be nonnegative integer counts of shape {shape}")
    totals = scales.sum(axis=1, dtype=np.int64)
    if np.any(totals <= 0) or np.any(totals != totals[0]) or raw.sum() <= 0:
        raise ValueError("Scale shot totals must be positive and equal; Raw must be complete")
    baseline = int(totals[0]+extension.sum())
    if raw.sum() != baseline:
        raise ValueError("Raw and Reset baselines must have equal declared shot totals")
    lambdas = np.asarray(arrays["effective_lambdas"], dtype=float)
    if (lambdas.shape != totals.shape or not np.isfinite(lambdas).all() or lambdas[0] != 1
            or np.any(np.diff(lambdas) <= 0)):
        raise ValueError("Effective scales must increase strictly from 1")
    support = np.flatnonzero(np.any(scales > 0, axis=0))
    suite = build_suite(lambdas)
    result = analyze_compressed(scales[:,support]/totals[:,None], support, dimension,
                                *suite, variants=("C1",))["C1"]
    vectors = {"raw": raw/raw.sum(), "reset": (scales[0]+extension)/baseline}
    vectors.update({m: result["after"][m].materialize()[0] for m in METHODS})
    # Truth label is computed only after reconstruction; it is never an input to fitting.
    order = reference_order(N,a)
    bits = dimension.bit_length()-1
    _check_ideal_budget(bits,order)
    decoded = decoder_selected_orders(N,a,bits)
    success = decoded == order
    ideal = ideal_distribution(bits,order)
    rows = []
    for method,p in vectors.items():
        if (p.shape != (dimension,) or not np.isfinite(p).all() or np.any(p < 0)
                or not np.isclose(p.sum(),1,rtol=0,atol=1e-11)):
            raise FloatingPointError("Invalid reconstructed distribution")
        row = asdict(distribution_metrics(p,bits,order))
        rows.append(dict(method=method, **row, exact_order_probability=float(p[success].sum())))
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir/"distributions.npz",
        **{"after__"+m:p for m,p in vectors.items()}, ideal_distribution=ideal,
        observed_support=support, selected_decoder_orders=decoded, effective_lambdas=lambdas)
    with (output_dir/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(N=N,a=a,order_evaluation_only=order,bits=bits,
        reconstruction="C1",weight_exponent=2,weights_normalization="mean_one",
        ridge_all_models=0,offset=1/dimension,scales=lambdas,shots_per_scale=totals,
        baseline_shots=baseline,points=rows,
        ideal_exact_order_probability=float(ideal[success].sum()),
        uniform_exact_order_probability=float(success.mean()),
        probability_interpretation="ZNE success is reconstructed probability mass, not physical output shots",
        new_quantum_shots=0,bootstrap_replicates=0,source_sha256=_sha(counts_path),
        selection=result["selection"],output_sha256={
            name:_sha(output_dir/name) for name in ('distributions.npz','metrics.csv')})
    _json(output_dir/"summary.json",summary)
    return summary


def _add_config(parser, mode):
    parser.add_argument("--N",type=int,required=True)
    parser.add_argument("--a",type=int,required=True)
    parser.add_argument("--scales",type=float,nargs="+")
    parser.add_argument("--shots",type=int,help="shots per scale")
    parser.add_argument("--baseline-shots",type=int)
    parser.add_argument("--seed",type=int,default=42)
    if mode == "simulator":
        parser.add_argument("--rates-percent",type=float,nargs="+",default=[.01],
                            help="percent values; 0.01 means probability 0.0001")
        parser.add_argument("--batch-size",type=int,default=48)
    else:
        parser.add_argument("--backend",required=True)
        parser.add_argument("--folding-patterns",type=int,default=10)


def _from_args(args, mode):
    return experiment_config(args.N,args.a,mode=mode,scales=args.scales,
        rates=[p/100 for p in args.rates_percent] if mode=="simulator" else None,
        shots_per_scale=args.shots,baseline_shots=args.baseline_shots,seed=args.seed,
        batch_size=getattr(args,"batch_size",48),folding_patterns=getattr(args,"folding_patterns",10),
        backend_name=getattr(args,"backend",None))


def _print(value):
    print(json.dumps(_clean(value),indent=2,allow_nan=False))


def _analyze_or_verify(counts,dest,N,a):
    if dest.exists():
        saved=json.loads((dest/"summary.json").read_text())
        if saved['N']!=N or saved['a']!=a or saved["source_sha256"] != _sha(counts):
            raise ValueError("Existing analysis does not match collected counts")
        if set(saved.get('output_sha256',{})) != {'distributions.npz','metrics.csv'}:
            raise ValueError("Existing analysis lacks complete output checksums")
        for name,digest in saved['output_sha256'].items():
            if _sha(dest/name)!=digest:raise ValueError("Existing analysis checksum differs: "+name)
        return saved
    return analyze_counts(counts,N=N,a=a,output_dir=dest)


def _analyze_collected(prepared):
    config=json.loads((prepared/"manifest.json").read_text())["config"]
    return _analyze_or_verify(prepared/"counts.npz",prepared/"analysis",config['N'],config['a'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("plan", "simulate"):
        s=sub.add_parser(name);_add_config(s,"simulator")
        if name=="simulate":
            s.add_argument("--output-dir",type=Path,required=True)
            s.add_argument("--execute",action="store_true",help="authorize local GPU shot generation")
            s.add_argument("--resume",action="store_true")
    for name in ("ibm-plan", "ibm-prepare", "ibm"):
        s=sub.add_parser(name);_add_config(s,"ibm")
        if name!="ibm-plan":
            s.add_argument("--output-dir",type=Path,required=True)
            s.add_argument("--prompt-token",action="store_true")
            s.add_argument("--offline",action="store_true",help="compile-only GenericBackendV2, not QPU calibration")
        if name=="ibm":
            s.add_argument("--execute",action="store_true",help="authorize real billable QPU submissions")
            s.add_argument("--wait",action="store_true",help="wait for jobs and analyze after completion")
            s.add_argument("--max-jobs",type=int,default=100)
            s.add_argument("--max-execution-time",type=int,default=300)
    for name in ("ibm-run","ibm-collect"):
        s=sub.add_parser(name);s.add_argument("--prepared",type=Path,required=True)
        s.add_argument("--prompt-token",action="store_true")
        s.add_argument("--wait",action="store_true")
        if name=="ibm-run":
            s.add_argument("--execute",action="store_true")
            s.add_argument("--max-jobs",type=int,default=100)
            s.add_argument("--max-execution-time",type=int,default=300)
    s=sub.add_parser("ibm-recover",help="attach an independently verified existing job ID; never submit")
    s.add_argument("--prepared",type=Path,required=True)
    s.add_argument("--job-index",type=int,required=True)
    s.add_argument("--job-id",required=True)
    s.add_argument("--verified-job",action="store_true",help="confirm backend, PUB ordering and shots match the ledger")
    s=sub.add_parser("analyze")
    for k in ("N","a"):s.add_argument("--"+k,type=int,required=True)
    s.add_argument("--input",type=Path,required=True);s.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(argv)
    if args.command in ("plan","ibm-plan"):
        _print(plan(_from_args(args,"simulator" if args.command=="plan" else "ibm")));return 0
    if args.command=="analyze":
        _print(analyze_counts(args.input,N=args.N,a=args.a,output_dir=args.output_dir));return 0
    if args.command=="simulate":
        config=_from_args(args,"simulator")
        if not args.execute:
            _print(plan(config));return 0
        from . import simulation_workflow as sim
        if args.resume:
            previous=json.loads((args.output_dir/"manifest.json").read_text())["config"]
            for k,v in config.items():
                if previous.get(k)!=v:raise ValueError("Resume configuration mismatch: "+k)
        else:sim.prepare(config,args.output_dir)
        outputs=sim.run(args.output_dir,execute=True,resume=args.resume)
        for rate,path in outputs.items():
            destination=args.output_dir/"analysis"/Path(path).stem
            _analyze_or_verify(path,destination,config['N'],config['a'])
        _print({"counts":{str(k):str(v) for k,v in outputs.items()},"analysis":str(args.output_dir/"analysis")});return 0
    from . import ibm_runtime as ibm
    if args.command=="ibm-recover":
        if not args.verified_job:
            raise ValueError("Verify the existing IBM job's backend/PUBs/shots, then pass --verified-job")
        _print(ibm.recover_submission(args.prepared,args.job_index,args.job_id));return 0
    service=None
    if args.command in ("ibm","ibm-prepare"):
        config=_from_args(args,"ibm")
        if args.command=="ibm" and not args.execute:
            _print(plan(config));return 0
        if args.offline:
            if args.command=="ibm":raise ValueError("--offline is compile-only; use ibm-prepare")
            from qiskit.providers.fake_provider import GenericBackendV2
            backend=GenericBackendV2(2*args.N.bit_length()+3,basis_gates=['id','rz','sx','x','cx'],
                                     control_flow=True,seed=args.seed,noise_info=False)
            config["backend_name"]=backend.name
            config["offline_compile_only"]=True
        else:
            service=ibm.service_from_environment(prompt=args.prompt_token)
            backend=ibm.select_backend(service,args.backend)
        prepared=args.output_dir
        info=ibm.prepare(config,prepared,backend)
        if args.command=="ibm-prepare":_print(info);return 0
    else:prepared=args.prepared
    if args.command=="ibm-run" and not args.execute:
        _print(ibm.submit(prepared,service=None,execute=False,
                         max_execution_time=args.max_execution_time,max_jobs=args.max_jobs));return 0
    if service is None:service=ibm.service_from_environment(prompt=args.prompt_token)
    if args.command in ("ibm","ibm-run"):
        _print(ibm.submit(prepared,service=service,execute=True,
                         max_execution_time=args.max_execution_time,max_jobs=args.max_jobs))
    arrays=ibm.collect(prepared,service=service,wait=args.wait)
    if arrays is None:_print({"status":"pending","next":"Run ibm-collect with the same --prepared directory"})
    else:_print(_analyze_collected(prepared))
    return 0


def cli():
    """Console boundary: never print SDK exceptions that might echo credentials."""
    try:
        return main()
    except Exception as exc:
        if len(sys.argv)>1 and sys.argv[1].startswith('ibm'):
            print(f"{type(exc).__name__}: IBM workflow stopped. Check the command/configuration and jobs.json. "
                  "A submission marked ambiguous must be reconciled with an existing IBM job ID; "
                  "do not delete its ledger or blindly resubmit. Exception details are not printed to protect credentials.",
                  file=sys.stderr)
        else:
            print(f"{type(exc).__name__}: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(cli())
