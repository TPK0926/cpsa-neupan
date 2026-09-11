#!/usr/bin/env python3
"""Train DUNE model for BARN robot (0.5x0.4m)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neupan import neupan

if __name__ == '__main__':
    yaml_file = os.path.join(os.path.dirname(__file__), 'dune_train_barn.yaml')
    planner = neupan.init_from_yaml(yaml_file)
    planner.train_dune()
