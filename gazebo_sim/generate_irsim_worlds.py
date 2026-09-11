#!/usr/bin/env python3
"""Generate Gazebo worlds from IR-SIM env.yaml files, scaled for TB3.

Scale factor: TB3 (0.14m) / IR-SIM robot (1.6m) ≈ 0.0875
LiDAR range kept at 3.5m (TB3 default), which covers scaled worlds.
"""

import yaml, os, numpy as np

NEUPAN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds')
SCALE = 0.12  # ~1.4x geometric ratio, tighter but navigable

os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENES = {
    'tb3_irs_corridor': {
        'env': 'example/corridor/diff/env.yaml',
        'start': (0.0, 2.6),  # corridor entrance, centered
        'goal': (3.5, 2.6),
        'goal_threshold': 0.5, 'max_steps': 600,
        'desc': 'IR-SIM corridor: 6.1x2.0m, 4 inner blocks',
    },
    'tb3_irs_convex': {
        'env': 'example/convex_obs/diff/env.yaml',
        'start': (0.2, 2.6),
        'goal': (3.0, 2.6),
        'goal_threshold': 0.5, 'max_steps': 800,
        'desc': 'IR-SIM convex_obs: circles + polygon, 3.7x3.7m',
    },
    'tb3_irs_nonobs': {
        'env': 'example/non_obs/diff/env.yaml',
        'start': (0.2, 2.6),
        'goal': (3.0, 2.6),
        'goal_threshold': 0.5, 'max_steps': 800,
        'desc': 'IR-SIM non_obs: irregular polygon obstacles, 3.7x3.7m',
    },
    'tb3_irs_narrow': {
        'env': 'example/narrow_corridor/diff/env.yaml',
        'start': (0.2, 0.3),
        'goal': (3.5, 0.3),
        'goal_threshold': 0.3, 'max_steps': 800,
        'desc': 'IR-SIM narrow_corridor: 4.4x0.6m tight passage',
    },
}

def scale_pt(x, y, offset=[0,0]):
    """Scale IR-SIM coordinate to TB3 world."""
    # Apply IR-SIM offset then scale
    sx = (x + offset[0]) * SCALE
    sy = (y + offset[1]) * SCALE
    return sx, sy

def make_world(name, cfg):
    env = yaml.safe_load(open(os.path.join(NEUPAN_ROOT, cfg['env'])))
    world_cfg = env['world']
    offset = world_cfg.get('offset', [0, 0])
    obstacles = env['obstacle']
    robot_cfg = env['robot'][0]

    # Scaled world dimensions
    ww = world_cfg['width'] * SCALE
    wh = world_cfg['height'] * SCALE
    sx, sy = cfg['start']  # manual start position

    xml = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="{name}">
    <physics type="ode"><max_step_size>0.001</max_step_size><real_time_factor>1</real_time_factor></physics>
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
'''

    # Process obstacles
    all_states = []
    all_shapes = []
    for obs_group in obstacles:
        states = obs_group['state']
        shapes = obs_group['shape']
        n = obs_group['number']
        # Handle both [[x,y,th], ...] and [x, y, th] formats
        if isinstance(states[0], (list, tuple)):
            state_list = states
        else:
            state_list = [states]
        for i in range(min(n, len(state_list))):
            all_states.append(state_list[i])
            all_shapes.append(shapes[min(i, len(shapes)-1)])

    for i, (state, shape) in enumerate(zip(all_states, all_shapes)):
        ox, oy = scale_pt(state[0], state[1], offset)
        oz = 0.5
        theta = state[2] if len(state) > 2 else 0

        if shape['name'] == 'rectangle':
            rl = max(shape['length'] * SCALE, 0.15)
            rw = max(shape['width'] * SCALE, 0.10)
            xml += f'''    <model name="obs_{i}"><static>true</static><pose>{ox:.3f} {oy:.3f} {oz} 0 0 {theta:.3f}</pose>
      <link name="link"><collision name="c"><geometry><box><size>{rl:.3f} {rw:.3f} 1.0</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{rl:.3f} {rw:.3f} 1.0</size></box></geometry>
          <material><ambient>0.9 0.4 0.3 1</ambient></material></visual></link></model>
'''
        elif shape['name'] == 'circle':
            rr = shape['radius'] * SCALE
            xml += f'''    <model name="obs_{i}"><static>true</static><pose>{ox:.3f} {oy:.3f} {oz} 0 0 0</pose>
      <link name="link"><collision name="c"><geometry><cylinder><radius>{rr:.3f}</radius><length>1.0</length></cylinder></geometry></collision>
        <visual name="v"><geometry><cylinder><radius>{rr:.3f}</radius><length>1.0</length></cylinder></geometry>
          <material><ambient>0.85 0.35 0.25 1</ambient></material></visual></link></model>
'''
        elif shape['name'] == 'polygon':
            verts = shape.get('vertices', [])
            if verts:
                # Approximate polygon with box at centroid
                vx = [v[0] for v in verts]; vy = [v[1] for v in verts]
                cx, cy = np.mean(vx)*SCALE, np.mean(vy)*SCALE
                bw = (max(vx)-min(vx))*SCALE; bh = (max(vy)-min(vy))*SCALE
                xml += f'''    <model name="obs_{i}"><static>true</static><pose>{ox+cy*0:.3f} {oy:.3f} {oz} 0 0 0</pose>
      <link name="link"><collision name="c"><geometry><box><size>{bw:.3f} {bh:.3f} 1.0</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{bw:.3f} {bh:.3f} 1.0</size></box></geometry>
          <material><ambient>0.6 0.5 0.7 1</ambient></material></visual></link></model>
'''

    # TB3 spawn
    xml += f'''
    <include><uri>model://turtlebot3_burger</uri><name>turtlebot3</name>
      <pose>{sx:.3f} {sy:.3f} 0.01 0 0 0</pose></include>
  </world>
</sdf>'''

    path = os.path.join(OUTPUT_DIR, f'{name}.world')
    with open(path, 'w') as f:
        f.write(xml)
    print(f"  Created {path}")
    print(f"    World: {ww:.1f}x{wh:.1f}m, Robot start: ({sx:.2f},{sy:.2f})")
    return {'start': (sx, sy), 'world_size': (ww, wh)}

# Generate all worlds
print("Generating IR-SIM replica worlds...")
world_configs = {}
for name, cfg in SCENES.items():
    print(f"\n{name}: {cfg['desc']}")
    info = make_world(name, cfg)
    world_configs[name] = {
        'world_file': f'worlds/{name}.world',
        'goal': cfg['goal'],
        'goal_threshold': cfg['goal_threshold'],
        'max_steps': cfg['max_steps'],
        'desc': cfg['desc'],
        'start': cfg['start'],
    }

# Print WORLD_CONFIGS snippet
print("\n=== WORLD_CONFIGS for tb3_reactive_nav.py ===")
import pprint
pprint.pprint(world_configs)
