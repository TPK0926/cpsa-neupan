"""
Baseline comparison experiment for conformal risk calibration paper.

Runs vanilla, CP methods, CPSA v4, and honest baselines (input perturbation,
CBF safety filter, ACI) across multiple environments and noise levels.

Each (env, noise, method) config is run as a subprocess via run_config.py,
which uses the same proven episode loop as the paper replicate experiments.

Usage:
    python experiments/exp_baselines.py
    python experiments/exp_baselines.py --envs corridor,dyna_obs --noises 0,2 --episodes 30
"""

import sys, os, json, time, subprocess, argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, '..')
PYTHON = os.environ.get('PYTHON', sys.executable)
RUN_CONFIG = os.path.join(BASE_DIR, 'run_config.py')

ENVIRONMENTS = ['corridor', 'convex_obs', 'pf_obs', 'dyna_obs', 'non_obs']
NOISE_LEVELS_CM = [0, 2, 3, 5]
METHODS = ['vanilla', 'cpsa_v4_full', 'cpsa_v4_online', 'cbf', 'aci']


def run_one_config(env, noise_cm, method, n_episodes=30):
    """Run one (env, noise, method) config via run_config.py subprocess."""
    cmd = [PYTHON, RUN_CONFIG, env, str(noise_cm), method, str(n_episodes)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        for line in result.stdout.strip().split('\n'):
            if line.startswith('RESULT_JSON:'):
                return json.loads(line[len('RESULT_JSON:'):])
        # Fallback: check stderr for the line too
        for line in result.stderr.strip().split('\n'):
            if line.startswith('RESULT_JSON:'):
                return json.loads(line[len('RESULT_JSON:'):])
        print(f"    WARNING: no RESULT_JSON in output. stderr tail: {result.stderr[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after 2h")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--envs', type=str, default=','.join(ENVIRONMENTS))
    parser.add_argument('--noises', type=str, default=','.join(str(n) for n in NOISE_LEVELS_CM))
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--methods', type=str, default=','.join(METHODS))
    args = parser.parse_args()

    env_names = [e.strip() for e in args.envs.split(',')]
    noise_levels = [int(n.strip()) for n in args.noises.split(',')]
    method_names = [m.strip() for m in args.methods.split(',')]

    print("=" * 70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Environments: {env_names}")
    print(f"Noise levels: {noise_levels} cm")
    print(f"Methods: {method_names}")
    print(f"Episodes per config: {args.episodes}")
    total = len(env_names) * len(noise_levels) * len(method_names)
    print(f"Total configs: {total}")
    print(f"Est. time: {total * 20:.0f}-{total * 40:.0f} min")

    all_results = {}
    count = 0
    t_start = time.time()

    for env in env_names:
        for noise_cm in noise_levels:
            for method in method_names:
                count += 1
                key = f"{env}_{noise_cm}cm_{method}"
                print(f"[{count}/{total}] {key} ... ", end='', flush=True)
                t0 = time.time()

                result = run_one_config(env, noise_cm, method, args.episodes)
                elapsed = time.time() - t0

                if result is not None:
                    result['key'] = key
                    all_results[key] = result
                    sr = result.get('success_rate', 0) * 100
                    cr = result.get('collision_rate', 0) * 100
                    md = result.get('avg_min_distance', 0) * 100
                    print(f"SR={sr:.1f}% CR={cr:.1f}% MinD={md:.1f}cm ({elapsed:.0f}s)")
                else:
                    all_results[key] = {'error': True, 'key': key}
                    print(f"FAILED ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE. {count} configs in {total_elapsed/60:.0f} min")

    # Save
    out_dir = os.path.join(BASE_DIR, 'exp_baselines_output')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'exp_baselines.json')
    output = {
        'config': {
            'environments': env_names,
            'noise_levels_cm': noise_levels,
            'n_episodes': args.episodes,
            'methods': method_names,
        },
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for env in env_names:
        for noise_cm in noise_levels:
            prefix = f"{env}_{noise_cm}cm_"
            entries = [(k, v) for k, v in all_results.items() if k.startswith(prefix) and 'error' not in v]
            if not entries:
                continue
            print(f"\n{env}, {noise_cm}cm:")
            print(f"  {'Method':<25} {'SR%':>6} {'CR%':>6} {'MinDcm':>7}")
            for k, v in sorted(entries):
                print(f"  {v.get('method','?'):<25} {v['success_rate']*100:>5.1f}  "
                      f"{v['collision_rate']*100:>5.1f}  {v['avg_min_distance']*100:>6.1f}")


if __name__ == '__main__':
    main()
