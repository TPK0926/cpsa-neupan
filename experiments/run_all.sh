#!/bin/bash
# Full matrix experiment with incremental saving.
# Usage: bash experiments/run_all.sh

set -e

PYTHON=${PYTHON:-python3}
OUTDIR=experiments/exp_full_matrix_output
mkdir -p $OUTDIR
OUTFILE=$OUTDIR/results.jsonl

ENVS="corridor convex_obs pf_obs dyna_obs non_obs"
NOISES="0 2"
METHODS="vanilla cp_global cp_dw cpsa_v4_full cpsa_v4_noTemporal cpsa_v4_noPassage cpsa_v4_staticTau cpsa_v4_staticQstar"

# Count total
total_envs=$(echo $ENVS | wc -w)
total_noises=$(echo $NOISES | wc -w)
total_methods=$(echo $METHODS | wc -w)
total=$((total_envs * total_noises * total_methods))
echo "Total configs: $total"
echo ""

count=0
for env in $ENVS; do
    for noise in $NOISES; do
        for method in $METHODS; do
            count=$((count + 1))
            key="${env}_${noise}cm_${method}"

            # Skip if already done
            if grep -q "\"$key\"" $OUTFILE 2>/dev/null; then
                echo "[$count/$total] SKIP $key (done)"
                continue
            fi

            echo -n "[$count/$total] $key ... "
            START=$(date +%s)

            FULL_OUT=$($PYTHON experiments/run_config.py $env $noise $method 2>/dev/null)
            # Extract JSON line with RESULT_JSON prefix
            RESULT=$(echo "$FULL_OUT" | grep '^RESULT_JSON:' | sed 's/^RESULT_JSON://')
            if [ $? -eq 0 ] && [ -n "$RESULT" ]; then
                # Add key and timestamp
                echo "$RESULT" | $PYTHON -c "
import json, sys
d = json.loads(sys.stdin.read())
d['key'] = '$key'
d['timestamp'] = '$(date -Iseconds)'
print(json.dumps(d))
" >> $OUTFILE

                DUR=$(( $(date +%s) - START ))
                SR=$(echo "$RESULT" | $PYTHON -c "import json,sys;d=json.loads(sys.stdin.read());print(f\"{d['success_rate']*100:.0f}%\")" 2>/dev/null || echo "?")
                echo "SR=$SR (${DUR}s)"
            else
                echo "FAILED"
                echo "{\"key\":\"$key\",\"error\":true}" >> $OUTFILE
            fi
        done
    done
done

echo ""
echo "DONE. Results in $OUTFILE"

# Merge into single JSON
$PYTHON -c "
import json
results = {}
with open('$OUTFILE') as f:
    for line in f:
        line = line.strip()
        if line:
            d = json.loads(line)
            key = d.pop('key')
            results[key] = d
with open('$OUTDIR/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Merged {len(results)} configs into results.json')
"
