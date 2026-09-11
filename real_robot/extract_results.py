#!/usr/bin/env python
"""从 roslaunch 输出日志提取 [EP_RESULT] 行, 汇总成 JSON。"""
import sys, json, re, os

if len(sys.argv) < 2:
    print("Usage: python extract_results.py <log_file_or_dir>")
    print("  log_file: 单个日志文件")
    print("  dir: 目录, 批量处理所有 .log 文件")
    sys.exit(1)

def extract(fpath):
    results = []
    with open(fpath) as f:
        for line in f:
            m = re.search(r'\[EP_RESULT\] outcome=(\w+) \| minD=([\d.]+)m \| path=([\d.]+)m \| time=([\d.]+)s', line)
            if m:
                results.append({
                    'outcome': m.group(1),
                    'min_dist_m': float(m.group(2)),
                    'path_m': float(m.group(3)),
                    'time_s': float(m.group(4)),
                })
    return results

def summary(results, name):
    total = len(results)
    arrived = sum(1 for r in results if r['outcome'] == 'arrived')
    collision = sum(1 for r in results if r['outcome'] == 'collision')
    sr = arrived / total * 100 if total else 0
    md = sum(r['min_dist_m'] for r in results) / total * 100 if total else 0
    print(f"{name}: SR={sr:.0f}% ({arrived}/{total})  CR={collision/total*100:.0f}%  avgMinD={md:.1f}cm")

path = sys.argv[1]
if os.path.isdir(path):
    for f in sorted(os.listdir(path)):
        if f.endswith('.log'):
            results = extract(os.path.join(path, f))
            summary(results, f)
elif os.path.isfile(path):
    results = extract(path)
    summary(results, os.path.basename(path))
    out = path.replace('.log', '.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  -> saved {out} ({len(results)} episodes)")
