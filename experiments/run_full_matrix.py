"""
Serial full-matrix experiment. No concurrency, no shell issues.
Run: nohup python experiments/run_full_matrix.py > /tmp/exp.log 2>&1 &
Resume-safe: skips configs already in results.jsonl
"""

import sys, os, json, time, subprocess, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE = os.path.join(BASE_DIR, 'exp_full_matrix_output', 'results.jsonl')
RUN_CONFIG = os.path.join(BASE_DIR, 'run_config.py')
PYTHON = os.environ.get('PYTHON', sys.executable)

ENVS = ['corridor', 'convex_obs', 'pf_obs', 'dyna_obs', 'non_obs']
NOISES = [0, 2]
METHODS = ['cpsa_v4_full', 'cpsa_v4_noTemporal', 'cpsa_v4_noPassage',
           'cpsa_v4_staticTau', 'cpsa_v4_staticQstar']

total = len(ENVS) * len(NOISES) * len(METHODS)


def is_done(key):
    if not os.path.exists(OUTFILE):
        return False
    with open(OUTFILE) as f:
        for line in f:
            if key in line:
                return True
    return False


def run_one(env, noise, method):
    """Run one config. Returns (success, result_dict_or_error)."""
    key = f"{env}_{noise}cm_{method}"
    cmd = [PYTHON, RUN_CONFIG, env, str(noise), method]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=os.path.join(BASE_DIR, '..'), timeout=3600)
        # Extract JSON result
        for line in r.stdout.strip().split('\n'):
            if line.startswith('RESULT_JSON:'):
                d = json.loads(line.split('RESULT_JSON:', 1)[1])
                d['key'] = key
                return True, d
        return False, f"No RESULT_JSON in output (stderr: {r.stderr[:200]})"
    except subprocess.TimeoutExpired:
        return False, "Timeout (1h)"
    except Exception as e:
        return False, str(e)


def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    t_start = time.time()
    count = 0
    done_count = 0

    print(f"Starting: {total} configs ({len(ENVS)}envs x {len(NOISES)}noise x {len(METHODS)}methods)")
    print(f"Output: {OUTFILE}", flush=True)

    for env in ENVS:
        for noise in NOISES:
            for method in METHODS:
                count += 1
                key = f"{env}_{noise}cm_{method}"
                elapsed = (time.time() - t_start) / 3600

                if is_done(key):
                    done_count += 1
                    print(f"[{count}/{total}] SKIP {key} (already done)", flush=True)
                    continue

                print(f"[{count}/{total}] {key} ... ", end='', flush=True)
                t0 = time.time()
                ok, result = run_one(env, noise, method)

                if ok:
                    with open(OUTFILE, 'a') as f:
                        f.write(json.dumps(result) + '\n')
                    sr = result.get('success_rate', 0) * 100
                    dt = time.time() - t0
                    done_count += 1
                    print(f"SR={sr:.0f}% ({dt:.0f}s) [{done_count}/{total} done, {elapsed:.1f}h elapsed]", flush=True)
                else:
                    dt = time.time() - t0
                    print(f"FAILED: {result} ({dt:.0f}s)", flush=True)

    total_time = (time.time() - t_start) / 3600
    print(f"\nALL DONE. {done_count}/{total} successful. Total: {total_time:.1f}h", flush=True)


if __name__ == '__main__':
    main()
