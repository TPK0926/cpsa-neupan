#!/usr/bin/env python3
"""Fast BARN world generator (Python 3).

Uses numpy for C-space and distance map computation.
Batch-generates worlds then selects by difficulty.

Usage:
  python3 generate_barn_worlds.py [--num 30]
"""

import argparse
import heapq
import json
import os
import random
import math
import numpy as np
from collections import deque


GRID_SIZE = 30
CYL_RADIUS = 0.075
CONTAIN_WALL_LENGTH = 5
ROBOT_RADIUS_CELLS = 1  # 3x3 cells = 0.45m footprint for 0.5x0.4m robot

WALL_RGB = [0.152, 0.379, 0.720]
OBS_RGB = [0.648, 0.192, 0.192]


def cellular_automaton(rows, cols, fill_pct, seed, smooth_iter=4):
    rng = random.Random(seed)
    g = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1:
                g[r, c] = 1
            else:
                g[r, c] = 1 if rng.random() < fill_pct else 0
    for _ in range(smooth_iter):
        n = np.zeros_like(g)
        for r in range(rows):
            for c in range(cols):
                cnt = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if dr == 0 and dc == 0:
                            continue
                        if nr < 0 or nr >= rows:
                            cnt += 1
                        elif nc < 0 or nc >= cols:
                            pass  # side boundaries open for path
                        elif g[nr, nc] == 1:
                            cnt += 1
                n[r, c] = 1 if cnt >= 5 else (0 if cnt <= 1 else g[r, c])
        g = n
    return g


def compute_cspace_np(grid, robot_r):
    """C-space via binary dilation."""
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * robot_r + 1, 2 * robot_r + 1), dtype=bool)
    return binary_dilation(grid.astype(bool), structure=struct).astype(np.int8)


def compute_cspace_pure(grid, robot_r):
    """C-space without scipy."""
    rows, cols = grid.shape
    cs = np.zeros_like(grid)
    obs_r, obs_c = np.where(grid == 1)
    for i in range(len(obs_r)):
        r0, c0 = obs_r[i], obs_c[i]
        r_lo = max(0, r0 - robot_r)
        r_hi = min(rows, r0 + robot_r + 1)
        c_lo = max(0, c0 - robot_r)
        c_hi = min(cols, c0 + robot_r + 1)
        cs[r_lo:r_hi, c_lo:c_hi] = 1
    return cs


def bfs_reachable(cspace, start_r, start_c):
    rows, cols = cspace.shape
    if cspace[start_r, start_c] == 1:
        return set()
    visited = set()
    visited.add((start_r, start_c))
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and cspace[nr, nc] == 0:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


def astar(cspace, start, goal):
    rows, cols = cspace.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and cspace[nr, nc] == 0:
                    if dr != 0 and dc != 0 and cspace[cur[0]+dr, cur[1]] == 1 and cspace[cur[0], cur[1]+dc] == 1:
                        continue
                    cost = math.sqrt(dr*dr + dc*dc)
                    tg = g_score[cur] + cost
                    nb = (nr, nc)
                    if tg < g_score.get(nb, 1e9):
                        came_from[nb] = cur
                        g_score[nb] = tg
                        h = math.sqrt((nr-goal[0])**2 + (nc-goal[1])**2)
                        heapq.heappush(open_set, (tg + h, nb))
    return None


def min_clearance_along_path(grid, path):
    """Min clearance from path points to nearest obstacle in original grid."""
    rows, cols = grid.shape
    min_cl = float('inf')
    for pr, pc in path:
        for d in range(1, 15):
            found = False
            for dr in range(-d, d + 1):
                for dc in range(-d, d + 1):
                    if max(abs(dr), abs(dc)) != d:
                        continue
                    nr, nc = pr + dr, pc + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        dist = math.sqrt(dr*dr + dc*dc)
                        if dist < min_cl:
                            min_cl = dist
                        found = True
                        break
                    if grid[nr, nc] == 1:
                        dist = math.sqrt(dr*dr + dc*dc)
                        if dist < min_cl:
                            min_cl = dist
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return min_cl


def write_world(filepath, obstacle_map, cyl_radius=CYL_RADIUS):
    rows, cols = obstacle_map.shape
    r_shift = -(rows - 1) * cyl_radius * 2
    c_shift_wall = 1.95

    lines = []
    lines.append("""<?xml version="1.0" ?>
<sdf version='1.6'>
  <world name='barn_world'>
    <light name='sun' type='directional'>
      <cast_shadows>0</cast_shadows>
      <pose frame=''>0 0 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.5 0.5 -1</direction>
    </light>
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
        <visual name='visual'><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry><material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Grey</name></script></material></visual>
      </link>
    </model>
    <gravity>0 0 -9.8</gravity>
    <physics name='default_physics' default='0' type='ode'>
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>100</real_time_update_rate>
    </physics>
""")

    cyl_id = 0
    cyl_data = []

    def add_cyl(x, y, rgb):
        nonlocal cyl_id
        lines.append(f'    <model name="unit_cylinder_{cyl_id}">\n'
                     f'      <static>1</static>\n'
                     f'      <pose frame="">{x:.4f} {y:.4f} 0.5 0 0 0</pose>\n'
                     f'      <link name="link">\n'
                     f'        <collision name="collision"><geometry><cylinder><radius>{cyl_radius}</radius><length>1</length></cylinder></geometry></collision>\n'
                     f'        <visual name="visual"><geometry><cylinder><radius>{cyl_radius}</radius><length>1</length></cylinder></geometry><material><ambient>{rgb[0]} {rgb[1]} {rgb[2]} 1</ambient><diffuse>{rgb[0]} {rgb[1]} {rgb[2]} 1</diffuse></material></visual>\n'
                     f'      </link>\n'
                     f'    </model>\n')
        cyl_id += 1

    # Containment walls
    c_lower = cyl_radius
    c_upper = cyl_radius + CONTAIN_WALL_LENGTH
    r_lower = -cyl_radius
    r_upper = r_shift - cyl_radius

    # Back wall
    rc = r_lower
    while rc >= r_upper:
        add_cyl(rc, c_lower, WALL_RGB)
        rc -= cyl_radius * 2

    # Top and bottom walls (containment)
    cc = c_lower + cyl_radius * 2
    while cc <= c_upper:
        add_cyl(r_lower, cc, WALL_RGB)
        add_cyl(r_upper, cc, WALL_RGB)
        cc += cyl_radius * 2

    # Obstacle field
    c_obs = cc  # column shift for obstacle area
    # Left wall of obstacle field
    rc = r_lower
    while rc >= r_upper:
        add_cyl(rc, c_obs, WALL_RGB)
        rc -= cyl_radius * 2

    for r in range(rows):
        for c in range(cols):
            if obstacle_map[r, c] == 1:
                x = r_shift + r * cyl_radius * 2
                y = c_obs + c * cyl_radius * 2
                rgb = WALL_RGB if (r == 0 or r == rows - 1) else OBS_RGB
                add_cyl(x, y, rgb)

    # Right wall
    c_right = c_obs + (cols - 1) * cyl_radius * 2
    rc = r_lower
    while rc >= r_upper:
        add_cyl(rc, c_right, WALL_RGB)
        rc -= cyl_radius * 2

    # Top and bottom right
    cc = c_obs
    while cc <= c_right:
        add_cyl(r_lower, cc, WALL_RGB)
        add_cyl(r_upper, cc, WALL_RGB)
        cc += cyl_radius * 2

    lines.append("""    <gui>
      <camera name='user_camera'>
        <pose frame=''>0 5 10 0 0.4 1.57</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>
  </world>
</sdf>
""")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    return r_shift, c_obs


def grid_to_world(row, col, r_shift, c_shift, cyl_r):
    x = r_shift + row * cyl_r * 2
    y = c_shift + col * cyl_r * 2
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=30)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--seed-start', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Try scipy first, fall back to pure
    try:
        from scipy.ndimage import binary_dilation
        compute_cspace = compute_cspace_np
        print("Using scipy for C-space computation")
    except ImportError:
        compute_cspace = compute_cspace_pure
        print("Using pure numpy for C-space computation")

    print(f"Generating {args.num} BARN worlds (grid {GRID_SIZE}x{GRID_SIZE})...")

    # Phase 1: batch generate valid worlds with varied fill rates
    valid_worlds = []
    seed = args.seed_start
    while len(valid_worlds) < args.num and seed < args.seed_start + 10000:
        # Vary fill rate across the range for diversity
        t = len(valid_worlds) / max(args.num, 1)
        base_fill = 0.10 + t * 0.20  # 0.10 to 0.30
        fill = base_fill + (random.random() - 0.5) * 0.05
        fill = max(0.08, min(0.32, fill))

        grid = cellular_automaton(GRID_SIZE, GRID_SIZE, fill, seed)
        cspace = compute_cspace(grid, ROBOT_RADIUS_CELLS)

        left_open = np.where(cspace[:, 0] == 0)[0]
        if len(left_open) == 0:
            seed += 1
            continue

        found = False
        for start_row in left_open:
            region = bfs_reachable(cspace, int(start_row), 0)
            right_open = [(r, c) for r, c in region if c == GRID_SIZE - 1]
            if not right_open:
                continue

            sr = int(start_row)
            gr = int(right_open[len(right_open)//2][0])
            path = astar(cspace, (sr, 0), (gr, GRID_SIZE - 1))

            if path and len(path) > 5:
                obs_pct = float(grid.sum()) / (GRID_SIZE * GRID_SIZE)
                valid_worlds.append({
                    'seed': seed, 'fill': fill, 'grid': grid,
                    'cspace': cspace, 'path': path,
                    'start': (sr, 0), 'goal': (gr, GRID_SIZE - 1),
                    'obs_pct': obs_pct,
                    'path_len': len(path),
                })
                found = True
                break

        if found and len(valid_worlds) % 10 == 0:
            print(f"  Found {len(valid_worlds)} valid worlds...")

        seed += 1

        seed += 1

    print(f"Total valid: {len(valid_worlds)}")

    # Classify by obstacle density for diversity
    for w in valid_worlds:
        pct = w['obs_pct']
        if pct < 0.15:
            w['difficulty'] = 'easy'
        elif pct < 0.25:
            w['difficulty'] = 'medium'
        else:
            w['difficulty'] = 'hard'

    selected = valid_worlds[:args.num]

    # Phase 3: write files
    index_entries = []
    for i, w in enumerate(selected):
        filename = f'barn_{i:04d}.world'
        filepath = os.path.join(args.output_dir, filename)
        r_shift, c_shift = write_world(filepath, w['grid'])

        sx, sy = grid_to_world(w['start'][0], w['start'][1], r_shift, c_shift, CYL_RADIUS)
        gx, gy = grid_to_world(w['goal'][0], w['goal'][1], r_shift, c_shift, CYL_RADIUS)

        index_entries.append({
            'world_id': i,
            'filename': filename,
            'seed': int(w['seed']),
            'fill_pct': round(float(w['fill']), 3),
            'obs_pct': round(float(w['obs_pct']), 3),
            'difficulty': w['difficulty'],
            'start': {'row': int(w['start'][0]), 'col': int(w['start'][1]),
                      'x': round(float(sx), 4), 'y': round(float(sy), 4)},
            'goal': {'row': int(w['goal'][0]), 'col': int(w['goal'][1]),
                     'x': round(float(gx), 4), 'y': round(float(gy), 4)},
            'path_len': len(w['path']),
        })

    counts = {}
    for e in index_entries:
        counts[e['difficulty']] = counts.get(e['difficulty'], 0) + 1

    # Save grids as numpy files for evaluation
    grids_dir = os.path.join(args.output_dir, 'grids')
    os.makedirs(grids_dir, exist_ok=True)
    for i, w in enumerate(selected):
        np.save(os.path.join(grids_dir, f'grid_{i:04d}.npy'), w['grid'])

    index_path = os.path.join(args.output_dir, 'barn_index.json')
    with open(index_path, 'w') as f:
        json.dump({
            'description': 'BARN benchmark worlds for CPSA-v4 evaluation',
            'grid_size': GRID_SIZE,
            'cyl_radius': CYL_RADIUS,
            'robot_size': '0.5x0.4m',
            'robot_radius_cells': ROBOT_RADIUS_CELLS,
            'num_worlds': len(index_entries),
            'difficulty_counts': counts,
            'worlds': index_entries,
        }, f, indent=2)

    print(f"\nGenerated {len(index_entries)} worlds in {args.output_dir}")
    for d in ['easy', 'medium', 'hard']:
        print(f"  {d}: {counts.get(d, 0)}")
    print(f"  Index: {index_path}")


if __name__ == '__main__':
    main()
