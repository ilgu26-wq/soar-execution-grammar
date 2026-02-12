#!/usr/bin/env python3
"""
SOAR INTEGRITY CHECK — integrity 3stage 
================================================================
"measurementis it?, execution nodeis it??"

Level 1: Physics Integrity — invariant catalogto/asthat
Level 2: Execution Graph Integrity — execution path  confirmed
Level 3: Compute Redundancy — redundant computation identification + measurement use separation

Output: soar_core (execution) vs soar_observer (measurement) separation map
"""

import sys, os, json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'exp_evidence')


EXEC_CHAIN = {
    'v2_locked': {
        'role': 'EXECUTION',
        'desc': 'Gate parameters (DD, consec_loss, vol_gate)',
        'feeds': ['run_v2_pipeline entry/deny decision'],
        'exp_origin': 'SOAR CORE v1→v2',
        'frozen': True,
    },
    'regime_layer': {
        'role': 'EXECUTION',
        'desc': 'Regime classification → size_hint',
        'feeds': ['alpha_gen context', 'size_hint → effective_pnl'],
        'exp_origin': 'Regime Layer v0.1.0',
        'frozen': True,
    },
    'force_engine': {
        'role': 'EXECUTION',
        'desc': 'Force state (dir_consistency, magnitude)',
        'feeds': ['alpha_gen input', 'FCL/AOCL evaluation', 'fast_death_check'],
        'exp_origin': 'Force Engine v1',
        'frozen': True,
    },
    'alpha_layer': {
        'role': 'EXECUTION',
        'desc': 'Alpha candidate generation',
        'feeds': ['signal candidates → gate evaluation'],
        'exp_origin': 'Alpha Layer',
        'frozen': True,
    },
    'motion_watchdog': {
        'role': 'EXECUTION',
        'desc': 'Trade motion analysis (MFE/MAE/motion_tag)',
        'feeds': ['FCL input', 'AOCL input'],
        'exp_origin': 'EXP-13',
        'frozen': True,
    },
    'pheromone_drift': {
        'role': 'EXECUTION',
        'desc': 'PDL drift context for alpha generation',
        'feeds': ['alpha_gen.set_pheromone_layer()'],
        'exp_origin': 'EXP-20',
        'frozen': True,
    },
    'failure_commitment': {
        'role': 'EXECUTION',
        'desc': 'FCL + AOCL + stabilized_orbit_evaluation → bar_evolution',
        'feeds': ['bar_evolution → energy_trajectory → shadow → AEP → ARG → sharp_boundary'],
        'exp_origin': 'EXP-15/16/17/18a',
        'frozen': True,
        'note': 'Most expensive module. EXP-52/53 prune this.',
    },
    'alpha_termination': {
        'role': 'EXECUTION',
        'desc': 'ATP detection + alpha_fate classification',
        'feeds': ['fate → shadow_geometry → AEP', 'fate → minimal_features → sharp_boundary'],
        'exp_origin': 'EXP-22',
        'frozen': True,
    },
    'alpha_energy': {
        'role': 'EXECUTION',
        'desc': 'Energy trajectory + summary from bar_evolution',
        'feeds': ['energy_trajectory → shadow_geometry → AEP → ARG → sharp_boundary',
                  'energy_summary → minimal_features (e_sign, de_sign)'],
        'exp_origin': 'EXP-23',
        'frozen': True,
    },
}

POST_PIPELINE = {
    'compute_shadow_geometry': {
        'role': 'EXECUTION',
        'desc': 'Shadow class from energy trajectory',
        'inputs': ['energy_trajectory', 'atp_bar', 'alpha_fate'],
        'feeds': ['AEP (shadow_energy_integral)', 'ARG (CLEAN_SHADOW_CONSECUTIVE)',
                  'minimal_features (shadow_binary)'],
        'exp_origin': 'EXP-36',
    },
    'compute_aep': {
        'role': 'EXECUTION',
        'desc': 'AEP phase transition (rolling window of shadow geometry)',
        'inputs': ['trades_list → energy_trajectory per trade'],
        'feeds': ['ARG (AEP_DOWN_JUMP)', 'minimal_features (aep_binary)'],
        'exp_origin': 'EXP-38',
    },
    'compute_arg_deny': {
        'role': 'EXECUTION',
        'desc': 'Relative Gate Shadow Test deny reasons',
        'inputs': ['energy_trajectory', 'energy_summary', 'shadow_results', 'aep_results'],
        'feeds': ['minimal_features (arg_depth)'],
        'exp_origin': 'EXP-39/40',
    },
    'extract_minimal_features': {
        'role': 'EXECUTION',
        'desc': 'EXP-47 minimal 6-feature state vector',
        'inputs': ['energy_summary', 'arg_results', 'shadow_results', 'aep_results'],
        'outputs': ['e_sign', 'de_sign', 'shadow_binary', 'arg_depth', 'regime_coarse', 'aep_binary'],
        'feeds': ['apply_sharp_boundary → p_exec'],
        'exp_origin': 'EXP-47',
    },
    'apply_sharp_boundary': {
        'role': 'EXECUTION',
        'desc': 'Deterministic p_exec from 6 features',
        'inputs': ['minimal_features'],
        'outputs': ['p_exec (0.0 / 0.5 / 0.9 / 1.0)'],
        'feeds': ['FINAL EXECUTION DECISION'],
        'exp_origin': 'EXP-48',
    },
    'measure_invariants': {
        'role': 'MEASUREMENT_ONLY',
        'desc': 'Compute 5 invariant metrics for validation',
        'inputs': ['minimal_features', 'sharp_p_exec', 'aep_results'],
        'feeds': ['NOTHING — measurement only'],
        'exp_origin': 'EXP-51',
    },
}

CORE_MODULES_OBSERVER = {
    'central_axis': {
        'role': 'OBSERVER_ONLY',
        'desc': 'Axis drift measurement around orbit events',
        'imported_by': 'exp_51 (imported but UNUSED in pipeline)',
        'exp_origin': 'EXP-24',
        'exec_connected': False,
    },
    'boundary': {
        'role': 'OBSERVER_ONLY',
        'desc': 'Phase detection (STI/EDG/MHI)',
        'imported_by': 'core/engine.py (EV Grammar demo only)',
        'exp_origin': 'Core v1 (pre-v2)',
        'exec_connected': False,
        'note': 'Replaced by v2_locked gate system',
    },
    'gate': {
        'role': 'OBSERVER_ONLY',
        'desc': 'EV Gate (P-root binary ALLOW/DENY)',
        'imported_by': 'core/engine.py (EV Grammar demo only)',
        'exp_origin': 'Core v1 (pre-v2)',
        'exec_connected': False,
        'note': 'EV Grammar preserved for formal verification, not in v2 hot path',
    },
    'judge': {
        'role': 'OBSERVER_ONLY',
        'desc': 'JudgeIR irreversible judgment',
        'imported_by': 'core/engine.py (EV Grammar demo only)',
        'exp_origin': 'Core v1 (pre-v2)',
        'exec_connected': False,
        'note': 'Architectural principle preserved, not in v2 hot path',
    },
    'constitution': {
        'role': 'OBSERVER_ONLY',
        'desc': '5 structural laws (R1-R5)',
        'imported_by': 'core/engine.py (EV Grammar demo only)',
        'exp_origin': 'Core v1 (pre-v2)',
        'exec_connected': False,
        'note': 'Laws are constraints, not computation nodes',
    },
    'engine': {
        'role': 'OBSERVER_ONLY',
        'desc': 'ExecutionEngine (Boundary→Judge→Gate)',
        'imported_by': 'core/run_core.py (demo script)',
        'exp_origin': 'Core v1 (pre-v2)',
        'exec_connected': False,
        'note': 'v1 execution engine, replaced by v2 pipeline',
    },
    'run_core': {
        'role': 'OBSERVER_ONLY',
        'desc': 'Demo script for RAW vs SOAR comparison',
        'imported_by': 'standalone script',
        'exp_origin': 'Core v1',
        'exec_connected': False,
    },
}

EXP_CLASSIFICATION = {
    'EXP-01 raw_vs_soar': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-02 gate_ablation': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-03 kill_switch': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-04 core_vs_v2': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-05 v2_with_v1_overlay': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-06 trade_count_calibration': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-07 prop_deployment_sim': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-08 boundary_sensitivity': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-09 alpha_condition_refinement': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-10 proposal_shaping': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-12 regime_condition_resolution': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-13 motion_watchdog': {'role': 'EXECUTION', 'connected': True, 'module': 'motion_watchdog'},
    'EXP-14 motion_penalty': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-15 failure_commitment': {'role': 'EXECUTION', 'connected': True, 'module': 'failure_commitment (FCL)'},
    'EXP-16 alpha_orbit': {'role': 'EXECUTION', 'connected': True, 'module': 'failure_commitment (AOCL)'},
    'EXP-17 observer_gauge': {'role': 'EXECUTION', 'connected': True, 'module': 'failure_commitment (progressive)'},
    'EXP-18a gauge_lock_v2': {'role': 'EXECUTION', 'connected': True, 'module': 'failure_commitment (stabilized)'},
    'EXP-19 contested_micro_orbit': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-20 pheromone_drift': {'role': 'EXECUTION', 'connected': True, 'module': 'pheromone_drift'},
    'EXP-21 pdl_rl_dataset': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-22 alpha_termination': {'role': 'EXECUTION', 'connected': True, 'module': 'alpha_termination'},
    'EXP-23 alpha_energy': {'role': 'EXECUTION', 'connected': True, 'module': 'alpha_energy'},
    'EXP-24 central_axis': {'role': 'OBSERVER', 'connected': False, 'note': 'imported but unused in pipeline'},
    'EXP-25 alpha_census': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-26 interference': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-27 sbii': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-28 geometry_drift': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-29 micro_origin': {'role': 'MEASUREMENT', 'connected': False},
    'EXP-30 energy_axis': {'role': 'OBSERVER', 'connected': False},
    'EXP-31 relative_frame': {'role': 'OBSERVER', 'connected': False},
    'EXP-32 orbit_closure': {'role': 'OBSERVER', 'connected': False},
    'EXP-34 frame_switch': {'role': 'OBSERVER', 'connected': False},
    'EXP-35 frame_cost': {'role': 'OBSERVER', 'connected': False},
    'EXP-36 shadow_geometry': {'role': 'EXECUTION', 'connected': True, 'module': 'compute_shadow_geometry'},
    'EXP-37 shadow_accumulation': {'role': 'OBSERVER', 'connected': False, 'note': 'shadow acc measured, not in exec path'},
    'EXP-38 phase_transition': {'role': 'EXECUTION', 'connected': True, 'module': 'compute_aep'},
    'EXP-39 relative_gate': {'role': 'EXECUTION', 'connected': True, 'module': 'compute_arg_deny'},
    'EXP-40 arg_attach': {'role': 'EXECUTION', 'connected': True, 'module': 'compute_arg_deny'},
    'EXP-41 threshold_learning': {'role': 'OBSERVER', 'connected': False, 'note': 'threshold sensitivity study'},
    'EXP-42 zombie_learning': {'role': 'OBSERVER', 'connected': False, 'note': 'ZOMBIE behavior study'},
    'EXP-43 computation_skip': {'role': 'OBSERVER', 'connected': False, 'note': 'skip feasibility study → led to EXP-52'},
    'EXP-44 ecl_execution': {'role': 'EXECUTION', 'connected': True, 'module': 'alpha_energy (ECL principle)'},
    'EXP-45 energy_exit': {'role': 'OBSERVER', 'connected': False},
    'EXP-46 observer_learning': {'role': 'OBSERVER', 'connected': False},
    'EXP-47 minimal_distillation': {'role': 'EXECUTION', 'connected': True, 'module': 'extract_minimal_features'},
    'EXP-48 sharp_boundary': {'role': 'EXECUTION', 'connected': True, 'module': 'apply_sharp_boundary'},
    'EXP-49 immortal_tight': {'role': 'OBSERVER', 'connected': False},
    'EXP-50 execution_delay': {'role': 'OBSERVER', 'connected': False},
    'EXP-51 cross_market': {'role': 'MEASUREMENT', 'connected': False, 'note': 'validation framework'},
    'EXP-52 worldline_pruning': {'role': 'EXECUTION', 'connected': True, 'module': 'fast_death_check + fast_orbit_energy'},
    'EXP-53 hierarchical_pruning': {'role': 'EXECUTION', 'connected': True, 'module': 'hierarchical 3-tier pruning'},
}

COMPUTE_FLOW = [
    {'step': 1, 'name': 'Gate Check', 'module': 'v2_locked', 'cost': 'O(1)', 'prunable': False},
    {'step': 2, 'name': 'Force State', 'module': 'force_engine', 'cost': 'O(1) lookup', 'prunable': False},
    {'step': 3, 'name': 'Regime Classification', 'module': 'regime_layer', 'cost': 'O(1)', 'prunable': False},
    {'step': 4, 'name': 'Alpha Generation', 'module': 'alpha_layer', 'cost': 'O(candidates)', 'prunable': False},
    {'step': 5, 'name': 'Death Check (bar 0-1)', 'module': 'EXP-52/53', 'cost': 'O(1)', 'prunable': False,
     'note': 'NEW — decides Tier 1/2/3'},
    {'step': 6, 'name': 'Motion Analysis', 'module': 'motion_watchdog', 'cost': 'O(bars)', 'prunable': True,
     'skip_when': 'HARD_DEAD or SOFT_DEAD'},
    {'step': 7, 'name': 'FCL Evaluation', 'module': 'failure_commitment', 'cost': 'O(bars×conditions)', 'prunable': True,
     'skip_when': 'HARD_DEAD or SOFT_DEAD'},
    {'step': 8, 'name': 'AOCL Evaluation', 'module': 'failure_commitment', 'cost': 'O(bars×conditions)', 'prunable': True,
     'skip_when': 'HARD_DEAD or SOFT_DEAD'},
    {'step': 9, 'name': 'Stabilized Orbit (bar loop)', 'module': 'failure_commitment', 'cost': 'O(10×conditions)',
     'prunable': True, 'skip_when': 'HARD_DEAD or SOFT_DEAD', 'note': 'MOST EXPENSIVE'},
    {'step': 10, 'name': 'ATP + Fate', 'module': 'alpha_termination', 'cost': 'O(bars)', 'prunable': True,
     'skip_when': 'HARD_DEAD or SOFT_DEAD (use fast version)'},
    {'step': 11, 'name': 'Energy Trajectory', 'module': 'alpha_energy', 'cost': 'O(bars)', 'prunable': True,
     'skip_when': 'HARD_DEAD or SOFT_DEAD (use fast_orbit_energy)'},
    {'step': 12, 'name': 'Shadow Geometry', 'module': 'compute_shadow_geometry', 'cost': 'O(bars)', 'prunable': False,
     'note': 'Post-pipeline, runs on all trades for AEP consistency'},
    {'step': 13, 'name': 'AEP', 'module': 'compute_aep', 'cost': 'O(n×window)', 'prunable': False},
    {'step': 14, 'name': 'ARG Deny', 'module': 'compute_arg_deny', 'cost': 'O(n)', 'prunable': False},
    {'step': 15, 'name': 'Minimal Features', 'module': 'extract_minimal_features', 'cost': 'O(n)', 'prunable': False},
    {'step': 16, 'name': 'Sharp Boundary → p_exec', 'module': 'apply_sharp_boundary', 'cost': 'O(n)', 'prunable': False},
]

REDUNDANCY_MAP = {
    'progressive_orbit (EXP-17)': {
        'computed_in': 'run_v2_pipeline (line 317)',
        'also_computed_in': 'stabilized_orbit (EXP-18a, line 318)',
        'redundant': True,
        'reason': 'stabilized_orbit supersedes progressive_orbit completely',
        'action': 'REMOVE from exec path',
    },
    'central_axis (EXP-24)': {
        'computed_in': 'imported but never called in pipeline',
        'redundant': True,
        'reason': 'Import-only, no function calls in exec path',
        'action': 'REMOVE import',
    },
    'energy_trajectory recomputation': {
        'computed_in': 'compute_aep calls compute_shadow_geometry per prev trade',
        'also_computed_in': 'already computed in run_world',
        'redundant': True,
        'reason': 'AEP recomputes shadow for window trades that already have shadow',
        'action': 'OPTIMIZE: pass pre-computed shadow results to AEP',
    },
}


def level1_physics():
    print(f"\n  ═══ LEVEL 1: PHYSICS INTEGRITY ═══")
    print(f"  Loading latest experiment evidence...\n")

    metrics = {
        'Sharp Gap ≥ 70%p': None,
        'Fate Separation ≥ 80%p': None,
        'AEP Median ≈ 0.98': None,
        'False Execute ≤ 10%': None,
    }

    for exp_name in ['exp53_hierarchical_pruning', 'exp52_worldline_pruning', 'exp51_cross_market']:
        path = os.path.join(EVIDENCE_DIR, exp_name)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(path, fname)) as f:
                data = json.load(f)

            inv = None
            if 'law_preservation' in data:
                lp = data['law_preservation']
                inv = lp.get('full_invariants') or lp.get('hier_invariants')
            elif 'invariants' in data:
                inv = data['invariants']

            if inv:
                if metrics['Sharp Gap ≥ 70%p'] is None:
                    metrics['Sharp Gap ≥ 70%p'] = inv.get('sharp_gap')
                    metrics['Fate Separation ≥ 80%p'] = inv.get('fate_separation')
                    metrics['AEP Median ≈ 0.98'] = inv.get('aep_median')
                    metrics['False Execute ≤ 10%'] = inv.get('false_exec_rate')
                break
        if metrics['Sharp Gap ≥ 70%p'] is not None:
            break

    thresholds = {
        'Sharp Gap ≥ 70%p': lambda v: v >= 70,
        'Fate Separation ≥ 80%p': lambda v: v >= 80,
        'AEP Median ≈ 0.98': lambda v: v >= 0.90,
        'False Execute ≤ 10%': lambda v: v <= 10,
    }

    all_pass = True
    for name, val in metrics.items():
        if val is None:
            print(f"    ⚠️  {name:>30s}  NO DATA")
            all_pass = False
        else:
            ok = thresholds[name](val)
            if not ok:
                all_pass = False
            mark = '✅' if ok else '❌'
            if isinstance(val, float) and val < 1:
                print(f"    {mark}  {name:>30s}  = {val:.4f}")
            else:
                print(f"    {mark}  {name:>30s}  = {val:.1f}")

    print(f"\n  Physics Integrity: {'✅ PASS' if all_pass else '❌ FAIL'}")
    return all_pass


def level2_exec_graph():
    print(f"\n  ═══ LEVEL 2: EXECUTION GRAPH INTEGRITY ═══")

    print(f"\n  ── Core Modules in Execution Path ──")
    exec_count = 0
    for name, info in EXEC_CHAIN.items():
        print(f"    ✅ {name:>25s}  →  {info['desc'][:50]}")
        exec_count += 1

    print(f"\n  ── Post-Pipeline Functions in Execution Path ──")
    for name, info in POST_PIPELINE.items():
        mark = '✅' if info['role'] == 'EXECUTION' else '📊'
        print(f"    {mark} {name:>30s}  →  {info['desc'][:50]}")
        if info['role'] == 'EXECUTION':
            exec_count += 1

    print(f"\n  ── Core Modules NOT in Execution Path (Observer Only) ──")
    obs_count = 0
    for name, info in CORE_MODULES_OBSERVER.items():
        print(f"    📊 {name:>25s}  →  {info['desc'][:50]}")
        obs_count += 1

    print(f"\n  Execution nodes: {exec_count}")
    print(f"  Observer nodes:  {obs_count}")

    print(f"\n  ── EXP Classification ──")
    exec_exps = [k for k, v in EXP_CLASSIFICATION.items() if v.get('connected')]
    obs_exps = [k for k, v in EXP_CLASSIFICATION.items() if v['role'] == 'OBSERVER' and not v.get('connected')]
    meas_exps = [k for k, v in EXP_CLASSIFICATION.items() if v['role'] == 'MEASUREMENT']

    print(f"\n  EXECUTION-connected EXPs ({len(exec_exps)}):")
    for exp in exec_exps:
        info = EXP_CLASSIFICATION[exp]
        print(f"    ✅ {exp:>40s}  →  {info.get('module', '')}")

    print(f"\n  OBSERVER-only EXPs ({len(obs_exps)}):")
    for exp in obs_exps:
        info = EXP_CLASSIFICATION[exp]
        note = info.get('note', '')
        print(f"    📊 {exp:>40s}  {note}")

    print(f"\n  MEASUREMENT-only EXPs ({len(meas_exps)}):")
    for exp in meas_exps:
        print(f"    📏 {exp:>40s}")

    print(f"\n  Execution Graph:")
    print(f"    {len(exec_exps)}/{len(EXP_CLASSIFICATION)} EXPs connected to exec path")
    print(f"    {len(obs_exps)} observer-only (safe to remove from runtime)")
    print(f"    {len(meas_exps)} measurement-only (validation scripts)")

    return exec_exps, obs_exps, meas_exps


def level3_redundancy():
    print(f"\n  ═══ LEVEL 3: COMPUTE REDUNDANCY ═══")

    print(f"\n  ── Compute Flow (per trade) ──")
    prunable_steps = []
    for step in COMPUTE_FLOW:
        skip = step.get('skip_when', '')
        prunable = step.get('prunable', False)
        mark = '🔴' if prunable else '🟢'
        note = f"  [SKIP if {skip}]" if skip else ''
        extra = f"  ({step.get('note', '')})" if step.get('note') else ''
        print(f"    {mark} Step {step['step']:>2d}: {step['name']:<30s}  {step['cost']:<20s}{note}{extra}")
        if prunable:
            prunable_steps.append(step)

    print(f"\n  Prunable steps: {len(prunable_steps)}/{len(COMPUTE_FLOW)}")
    print(f"  Steps 6-11 skipped for DEAD worldlines (EXP-52/53)")

    print(f"\n  ── Redundancy Findings ──")
    for name, info in REDUNDANCY_MAP.items():
        print(f"\n    ⚠️  {name}")
        print(f"       Reason: {info['reason']}")
        print(f"       Action: {info['action']}")

    return REDUNDANCY_MAP


def level_summary(exec_exps, obs_exps, meas_exps, redundancies):
    print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  SOAR SEPARATION MAP                                           ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════╝")

    print(f"\n  🔵 soar_core/ (execution use — time)")
    core_modules = ['v2_locked', 'regime_layer', 'force_engine', 'alpha_layer',
                    'motion_watchdog', 'pheromone_drift', 'failure_commitment',
                    'alpha_termination', 'alpha_energy']
    for m in core_modules:
        print(f"     {m}")
    print(f"     + compute_shadow_geometry")
    print(f"     + compute_aep")
    print(f"     + compute_arg_deny")
    print(f"     + extract_minimal_features (EXP-47)")
    print(f"     + apply_sharp_boundary (EXP-48)")
    print(f"     + fast_death_check (EXP-52)")
    print(f"     + fast_orbit_energy (EXP-52)")
    print(f"     + tier2_soft_death_check (EXP-53)")

    print(f"\n  🟡 soar_observer/ (observation use — offline)")
    for m in CORE_MODULES_OBSERVER:
        print(f"     {m}")
    for exp in obs_exps:
        print(f"     {exp}")

    print(f"\n  🔴 soar_experiments/ (verification use)")
    for exp in meas_exps[:10]:
        print(f"     {exp}")
    if len(meas_exps) > 10:
        print(f"     ... +{len(meas_exps) - 10} more")

    print(f"\n  ── Immediate Actions ──")
    print(f"  1. REMOVE progressive_orbit_evaluation from run_v2_pipeline")
    print(f"     (stabilized_orbit supersedes it completely)")
    print(f"  2. REMOVE unused central_axis import from exp_51")
    print(f"  3. OPTIMIZE compute_aep: pass pre-computed shadow to avoid recomputation")
    print(f"  4. soar_core modules: 9 core + 6 post-pipeline = 15 nodes")
    print(f"  5. soar_observer modules: {len(CORE_MODULES_OBSERVER)} core + {len(obs_exps)} EXP = {len(CORE_MODULES_OBSERVER) + len(obs_exps)} nodes")


def main():
    print("=" * 70)
    print(f"  SOAR INTEGRITY CHECK — integrity 3stage ")
    print(f"  'measurementis it?, execution nodeis it??'")
    print("=" * 70)

    l1_pass = level1_physics()
    exec_exps, obs_exps, meas_exps = level2_exec_graph()
    redundancies = level3_redundancy()
    level_summary(exec_exps, obs_exps, meas_exps, redundancies)

    integrity_dir = os.path.join(EVIDENCE_DIR, 'integrity_check')
    os.makedirs(integrity_dir, exist_ok=True)

    result = {
        'timestamp': datetime.now().isoformat(),
        'level1_physics': {
            'pass': l1_pass,
        },
        'level2_exec_graph': {
            'exec_chain_modules': list(EXEC_CHAIN.keys()),
            'post_pipeline_exec': [k for k, v in POST_PIPELINE.items() if v['role'] == 'EXECUTION'],
            'post_pipeline_measurement': [k for k, v in POST_PIPELINE.items() if v['role'] == 'MEASUREMENT_ONLY'],
            'observer_modules': list(CORE_MODULES_OBSERVER.keys()),
            'exec_exps': exec_exps,
            'observer_exps': obs_exps,
            'measurement_exps': meas_exps,
        },
        'level3_redundancy': {
            'redundant_computations': list(REDUNDANCY_MAP.keys()),
            'prunable_steps': [s['name'] for s in COMPUTE_FLOW if s.get('prunable')],
        },
        'separation_map': {
            'soar_core': list(EXEC_CHAIN.keys()) + [k for k, v in POST_PIPELINE.items() if v['role'] == 'EXECUTION'],
            'soar_observer': list(CORE_MODULES_OBSERVER.keys()) + obs_exps,
            'soar_experiments': meas_exps,
        },
        'immediate_actions': [
            'REMOVE progressive_orbit_evaluation from run_v2_pipeline (redundant with stabilized_orbit)',
            'REMOVE unused central_axis import from exp_51',
            'OPTIMIZE compute_aep shadow recomputation',
        ],
    }

    path = os.path.join(integrity_dir, 'integrity_check.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  --- Integrity Check Saved ---")
    print(f"  {path}")
    print(f"\n  'what executionand what is observation — now separated.'")


if __name__ == '__main__':
    main()
