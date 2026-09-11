'''
neupan is the main class for the NeuPan algorithm. It wraps the PAN class and provides a more user-friendly interface.

Developer: Han Ruihua (hanrh@connect.hku.hk)
'''

import yaml
import torch
from neupan.robot import robot
from neupan.blocks import InitialPath, PAN
from neupan import configuration
from neupan.util import time_it, file_check, get_transform
import numpy as np
from neupan.configuration import np_to_tensor, tensor_to_np
from math import cos, sin
from neupan.risk_calibration import RiskCalibrator
from neupan.risk_calibration.risk_budget import RiskBudgetAllocator
from neupan.blocks.cpsa_risk_net import CPSARiskNet, OnlineCPSAAdapter

class neupan(torch.nn.Module):

    """
    Args:
        receding: int, the number of steps in the receding horizon.
        step_time: float, the time step in the MPC framework.
        ref_speed: float, the reference speed of the robot.
        device: str, the device to run the algorithm on. 'cpu' or 'cuda'.
        robot_kwargs: dict, the keyword arguments for the robot class.
        ipath_kwargs: dict, the keyword arguments for the initial path class.
        pan_kwargs: dict, the keyword arguments for the PAN class.
        adjust_kwargs: dict, the keyword arguments for the adjust class
        train_kwargs: dict, the keyword arguments for the train class
        time_print: bool, whether to print the forward time of the algorithm.
        collision_threshold: float, the threshold for the collision detection. If collision, the algorithm will stop.
    """

    def __init__(
        self,
        receding: int = 10,
        step_time: float = 0.1,
        ref_speed: float = 4.0,
        device: str = "cpu",
        robot_kwargs: dict = None,
        ipath_kwargs: dict = None,
        pan_kwargs: dict = None,
        adjust_kwargs: dict = None,
        train_kwargs: dict = None,
        **kwargs,
    ) -> None:
        super(neupan, self).__init__()

        # mpc parameters
        self.T = receding
        self.dt = step_time
        self.ref_speed = ref_speed

        configuration.device = torch.device(device)
        configuration.time_print = kwargs.get("time_print", False)
        self.collision_threshold = kwargs.get("collision_threshold", 0.1)

        # initialization
        self.cur_vel_array = np.zeros((2, self.T))
        self.robot = robot(receding, step_time, **robot_kwargs)

        self.ipath = InitialPath(
            receding, step_time, ref_speed, self.robot, **ipath_kwargs
        )

        pan_kwargs["adjust_kwargs"] = adjust_kwargs
        pan_kwargs["train_kwargs"] = train_kwargs
        self.dune_train_kwargs = train_kwargs

        self.pan = PAN(receding, step_time, self.robot, **pan_kwargs)

        self.info = {"stop": False, "arrive": False, "collision": False}

        # Risk calibration state
        self._risk_calibration = None
        self._collision_threshold_adjustment = 0.0
        self._budget_allocator = None
        self._cpsa_net = None
        self._cpsa_q_star = 0.0
        self._cpsa_prev_d_pred = None
        self._cpsa_total_budget = 0.0
        self._cpsa_feat_mean = None
        self._cpsa_feat_std = None
        self._cpsa_tau_max = 0.08
        self._cpsa_prev_radius = None
        self._robot_width = 2.0
        self._cpsa_online_adapter = None  # OnlineCPSAAdapter for deployment-time finetuning
        self._cpsa_episode_buffer = []    # accumulate (features, d_pred) per episode

    @classmethod
    def init_from_yaml(cls, yaml_file, **kwargs):
        abs_path = file_check(yaml_file)

        with open(abs_path, "r") as f:
            config = yaml.safe_load(f)
            config.update(kwargs)

        config["robot_kwargs"] = config.pop("robot", dict())
        config["ipath_kwargs"] = config.pop("ipath", dict())
        config["pan_kwargs"] = config.pop("pan", dict())
        config["adjust_kwargs"] = config.pop("adjust", dict())
        config["train_kwargs"] = config.pop("train", dict())

        return cls(**config)

    @time_it("neupan forward")
    def forward(self, state, points, velocities=None):
        """
        state: current state of the robot, matrix (3, 1), x, y, theta
        points: current input obstacle point positions, matrix (2, N), N is the number of obstacle points.
        velocities: current velocity of each obstacle point, matrix (2, N), N is the number of obstacle points. vx, vy
        """

        assert state.shape[0] >= 3

        if self.ipath.check_arrive(state):
            self.info["arrive"] = True
            return np.zeros((2, 1)), self.info

        nom_input_np = self.ipath.generate_nom_ref_state(
            state, self.cur_vel_array, self.ref_speed
        )

        # convert to tensor
        nom_input_tensor = [np_to_tensor(n) for n in nom_input_np]
        obstacle_points_tensor = np_to_tensor(points) if points is not None else None
        point_velocities_tensor = (
            np_to_tensor(velocities) if velocities is not None else None
        )

        opt_state_tensor, opt_vel_tensor, opt_distance_tensor = self.pan(
            *nom_input_tensor, obstacle_points_tensor, point_velocities_tensor
        )

        # Apply per-point risk budget allocation using DUNE's predicted distances
        if (self._risk_calibration is not None
                and self._risk_calibration.get("budget_strategy") == "cpsa"
                and self._cpsa_net is not None
                and not self.pan.no_obs):
            if hasattr(self.pan.dune_layer, 'obstacle_points') and self.pan.dune_layer.obstacle_points is not None:
                obs_pts = self.pan.dune_layer.obstacle_points
                if obs_pts.shape[1] > 0:
                    margins = self._compute_cpsa_margins(obs_pts)
                    n_points = min(len(margins), self.pan.nrmp_max_num)
                    if hasattr(self.pan.dune_layer, 'distances_0'):
                        d_pred = tensor_to_np(self.pan.dune_layer.distances_0)
                        sort_idx = np.argsort(d_pred)[:n_points]
                        margins_sorted = margins[sort_idx]
                    else:
                        margins_sorted = margins[:n_points]
                    self.pan.nrmp_layer.set_per_point_margins(margins_sorted)
                    self.info["cpsa_margins"] = margins_sorted
                    self.info["cpsa_margin_points"] = obs_pts[:, sort_idx] if obs_pts.shape[1] >= n_points else obs_pts[:, :n_points]
                    self.info["cpsa_all_margins"] = margins  # full tau before sorting
                    self.info["cpsa_all_points"] = obs_pts    # full obstacle points

        elif (self._budget_allocator is not None
                and self._budget_allocator.strategy != "global"
                and self._risk_calibration is not None):
            q_hat = self._risk_calibration["q_hat"]
            if not self.pan.no_obs and hasattr(self.pan.dune_layer, 'distances_0'):
                d_pred = tensor_to_np(self.pan.dune_layer.distances_0)
                n_points = self.pan.nrmp_max_num
                margins = self._budget_allocator.allocate(
                    q_hat, d_pred=d_pred, horizon=self.T, n_points=n_points
                )
                if margins.ndim == 2:
                    self.pan.nrmp_layer.set_per_point_margins(margins[0])
                else:
                    self.pan.nrmp_layer.set_per_point_margins(margins.flatten())

        opt_state_np, opt_vel_np = tensor_to_np(opt_state_tensor), tensor_to_np(
            opt_vel_tensor
        )

        self.cur_vel_array = opt_vel_np

        self.info["state_tensor"] = opt_state_tensor
        self.info["vel_tensor"] = opt_vel_tensor
        self.info["distance_tensor"] = opt_distance_tensor
        self.info['ref_state_tensor'] = nom_input_tensor[2]
        self.info['ref_speed_tensor'] = nom_input_tensor[3]

        self.info["ref_state_list"] = [
            state[:, np.newaxis] for state in nom_input_np[2].T
        ]
        self.info["opt_state_list"] = [state[:, np.newaxis] for state in opt_state_np.T]

        if self.check_stop():
            self.info["stop"] = True
            return np.zeros((2, 1)), self.info
        else:
            self.info["stop"] = False

        action = opt_vel_np[:, 0:1]

        return action, self.info

    def check_stop(self):
        effective_threshold = self.collision_threshold - self._collision_threshold_adjustment
        return self.min_distance < max(effective_threshold, 0.0)


    def scan_to_point(
        self,
        state: np.ndarray,
        scan: dict,
        scan_offset: list[float] = [0, 0, 0],
        angle_range: list[float] = [-np.pi, np.pi],
        down_sample: int = 1,
    ) -> np.ndarray | None:

        """
        input:
            state: [x, y, theta]
            scan: {}
                ranges: list[float], the range of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)
        """
        point_cloud = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range > scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)

        if len(point_cloud) == 0:
            return None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R @ point_array + s_trans

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        return points

    def scan_to_point_velocity(
        self,
        state,
        scan,
        scan_offset=[0, 0, 0],
        angle_range=[-np.pi, np.pi],
        down_sample=1,
    ):
        """
        input:
            state: [x, y, theta]
            scan: {}
                ranges: list[float], the ranges of the scan
                angle_min: float, the minimum angle of the scan
                angle_max: float, the maximum angle of the scan
                range_max: float, the maximum range of the scan
                range_min: float, the minimum range of the scan
                velocity: list[float], the velocity of the scan

            scan_offset: [x, y, theta], the relative position of the sensor to the robot state coordinate

        return point cloud: (2, n)
        """
        point_cloud = []
        velocity_points = []

        ranges = np.array(scan["ranges"])
        angles = np.linspace(scan["angle_min"], scan["angle_max"], len(ranges))
        scan_velocity = scan.get("velocity", np.zeros((2, len(ranges))))

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            if scan_range < (scan["range_max"] - 0.02) and scan_range >= scan["range_min"]:
                if angle > angle_range[0] and angle < angle_range[1]:
                    point = np.array(
                        [[scan_range * cos(angle)], [scan_range * sin(angle)]]
                    )
                    point_cloud.append(point)
                    velocity_points.append(scan_velocity[:, i : i + 1])

        if len(point_cloud) == 0:
            return None, None

        point_array = np.hstack(point_cloud)
        s_trans, s_R = get_transform(np.c_[scan_offset])
        temp_points = s_R.T @ (
            point_array - s_trans
        )

        trans, R = get_transform(state)
        points = (R @ temp_points + trans)[:, ::down_sample]

        velocity = np.hstack(velocity_points)[:, ::down_sample]

        return points, velocity


    def train_dune(self):
        self.pan.dune_layer.train_dune(self.dune_train_kwargs)


    def reset(self):
        self.ipath.point_index = 0
        self.ipath.curve_index = 0
        self.info["stop"] = False
        self.info["arrive"] = False
        self.cur_vel_array = np.zeros_like(self.cur_vel_array)
        self._cpsa_prev_d_pred = None
        self._cpsa_prev_radius = None
        self._cpsa_episode_buffer = []


    def set_initial_path_from_state(self, state):
        """
        Args:
            states: [x, y, theta] or 3x1 vector

        """
        self.ipath.init_check(state)

    def set_reference_speed(self, speed: float):

        """
        Args:
            speed: float, the reference speed of the robot
        """

        self.ipath.ref_speed = speed
        self.ref_speed = speed

    def update_initial_path_from_goal(self, start, goal):

        """
        Args:
            start: [x, y, theta] or 3x1 vector
            goal: [x, y, theta] or 3x1 vector
        """

        self.ipath.update_initial_path_from_goal(start, goal)

    def enable_risk_calibration(
        self,
        calibration_file: str = None,
        epsilon: float = 0.05,
        q_hat: float = None,
        budget_strategy: str = "global",
        collision_q_hat: float = None,
        cp_checkpoint: str = None,
        noise_std: float = 0.0,
        env_type: int = 0,
        cpsa_config: dict = None,
    ):
        """Enable Risk-Calibrated Navigation with conformal risk margin.

        Args:
            calibration_file: path to calibration JSON (from calibrate_dune.py)
            epsilon: risk budget (default 0.05 = 5% collision probability)
            q_hat: directly set the conformal risk quantile (overrides file)
            budget_strategy: "global" (scalar margin), "uniform",
                "distance_weighted" (per-point margin), or "cpsa"
                (per-point margin from CPSA risk network)
            collision_q_hat: q_hat for the underestimation direction, used to
                adjust collision_threshold and avoid false positive stops.
            cp_checkpoint: path to trained CPSA network checkpoint (for "cpsa")
            noise_std: lidar noise std for CPSA adaptive behavior
            env_type: environment type for CPSA features
            cpsa_config: dict with ablation switches
                - disable_global_ctx: bool
                - disable_temporal: bool
                - disable_passage: bool
                - static_tau_max: bool
                - static_q_star: bool
        """
        # Resolve q_hat
        if budget_strategy == "cpsa":
            # CPSA uses static CP margin as safety floor + per-point delta
            if q_hat is None and calibration_file is not None:
                calibrator = RiskCalibrator()
                calibrator.load(calibration_file)
                q_hat = calibrator.compute_q_hat(epsilon)
            if q_hat is None:
                q_hat = 0.0  # floor disabled if not provided
        else:
            if q_hat is None and calibration_file is not None:
                calibrator = RiskCalibrator()
                calibrator.load(calibration_file)
                q_hat = calibrator.compute_q_hat(epsilon)
            if q_hat is None:
                raise ValueError("Provide either calibration_file or q_hat")

        # Set budget allocator
        self._budget_allocator = RiskBudgetAllocator(strategy=budget_strategy)

        if budget_strategy == "global":
            self.pan.nrmp_layer.set_risk_margin(q_hat)
        elif budget_strategy == "cpsa":
            if cp_checkpoint is None:
                raise ValueError("cp_checkpoint required for cpsa strategy")
            self._load_cpsa(cp_checkpoint)
            self.pan.nrmp_layer.set_risk_margin(0.0)
        else:
            self.pan.nrmp_layer.set_risk_margin(0.0)

        # Collision threshold adjustment
        if collision_q_hat is not None and collision_q_hat > 0:
            self._collision_threshold_adjustment = collision_q_hat
        else:
            self._collision_threshold_adjustment = 0.0

        self._risk_calibration = {
            "epsilon": epsilon,
            "q_hat": q_hat,
            "budget_strategy": budget_strategy,
            "collision_q_hat": collision_q_hat or 0.0,
            "noise_std": noise_std,
            "env_type": env_type,
            "cpsa_config": cpsa_config or {},
        }

    def disable_risk_calibration(self):
        """Reset risk margin to zero (disable calibration)."""
        self.pan.nrmp_layer.set_risk_margin(0.0)
        self.pan.nrmp_layer._per_point_margins = None
        self._risk_calibration = None
        self._collision_threshold_adjustment = 0.0
        self._budget_allocator = None
        self._cpsa_net = None
        self._cpsa_q_star = 0.0
        self._cpsa_prev_d_pred = None
        self._cpsa_total_budget = 0.0
        self._cpsa_feat_mean = None
        self._cpsa_feat_std = None

    def _load_cpsa(self, cp_checkpoint):
        """Load CPSA risk network for per-point adaptive margins."""
        ckpt = torch.load(cp_checkpoint, map_location='cpu', weights_only=False)
        config = ckpt.get('config', {})

        self._cpsa_net = CPSARiskNet(
            feature_dim=config.get('feature_dim', 17),
            hidden_dims=config.get('hidden_dims', [128, 64, 32]),
        )
        self._cpsa_net.load_state_dict(ckpt['model_state'], strict=False)
        self._cpsa_net.eval()

        self._cpsa_q_star = ckpt.get('q_star', 0.0)
        self._cpsa_q_star_conditional = ckpt.get('q_star_conditional', {})
        self._cpsa_prev_d_pred = None
        self._cpsa_prev_radius = None
        self._cpsa_total_budget = ckpt.get('q_star', 0.0)
        self._cpsa_feat_mean = ckpt.get('feature_mean', None)
        self._cpsa_feat_std = ckpt.get('feature_std', None)
        self._cpsa_feat_dim = config.get('feature_dim', 17)

        # Online adaptation: enabled via cpsa_config
        self._cpsa_online_adapter = None

    def enable_cpsa_online(self, lr: float = 1e-4, buffer_size: int = 2000):
        """Enable online CPSA fine-tuning during deployment."""
        if self._cpsa_net is None:
            return
        self._cpsa_online_adapter = OnlineCPSAAdapter(
            self._cpsa_net,
            feat_mean=self._cpsa_feat_mean,
            feat_std=self._cpsa_feat_std,
            buffer_size=buffer_size,
            lr=lr,
        )

    def cpsa_episode_feedback(self, d_gt, outcome='unknown', noise_std=0.0,
                             lateral_margin=5.0, is_approaching=None):
        """Feed episode result back to CPSA online adapter for fine-tuning."""
        if self._cpsa_online_adapter is None:
            return
        if not self._cpsa_episode_buffer:
            return
        features = torch.cat([f for f, _ in self._cpsa_episode_buffer], dim=0)
        d_pred = torch.cat([d for _, d in self._cpsa_episode_buffer], dim=0)
        d_gt_t = torch.tensor(d_gt, dtype=torch.float32) if not isinstance(d_gt, torch.Tensor) else d_gt
        self._cpsa_online_adapter.feedback(
            features=features,
            d_pred=d_pred,
            d_gt=d_gt_t,
            outcome=outcome,
            noise_std=noise_std,
            lateral_margin=lateral_margin,
            is_approaching=is_approaching,
        )
        self._cpsa_episode_buffer = []

    def _compute_lateral_margin(self, pts):
        """Compute lateral (left/right) margin from point cloud."""
        N = pts.shape[0]
        if N < 2:
            return 10.0, 10.0, 10.0

        y = pts[:, 1]

        left_mask = y > 0
        right_mask = y < 0

        left_min = torch.abs(y[left_mask]).min().item() if left_mask.any() else 10.0
        right_min = torch.abs(y[right_mask]).min().item() if right_mask.any() else 10.0

        lateral_margin = min(left_min, right_min)
        return lateral_margin, left_min, right_min

    def _compute_cpsa_margins(self, points_tensor):
        """Compute per-point margins from CPSA risk network.

        Args:
            points_tensor: (2, N) obstacle points in robot-local coords
        Returns:
            margins: (N,) numpy array of per-point safety margins (tau)
        """
        if self._cpsa_net is None:
            return None

        with torch.no_grad():
            pts = points_tensor.T
            N = pts.shape[0]

            if hasattr(self.pan.dune_layer, 'distances_0'):
                d_pred = self.pan.dune_layer.distances_0
            else:
                d_pred = torch.ones(N, device=pts.device)

            eps = 1e-6
            radius = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + eps)
            angle = torch.arctan2(pts[:, 1], pts[:, 0])
            proximity_ratio = torch.abs(d_pred) / (radius + eps)

            if N > 5:
                diff = pts.unsqueeze(1) - pts.unsqueeze(0)
                dists = torch.sqrt((diff**2).sum(dim=2) + eps)
                k = min(5, N - 1)
                knn_dists, _ = torch.topk(dists, k + 1, dim=1, largest=False)
                local_density = knn_dists[:, 1:].mean(dim=1)
                in_radius_count = (dists < 1.0).float().sum(dim=1) - 1
                in_radius = torch.clamp(in_radius_count, 0, 50) / 50.0
            elif N > 1:
                diff = pts.unsqueeze(1) - pts.unsqueeze(0)
                dists = torch.sqrt((diff**2).sum(dim=2) + eps)
                local_density = dists[dists > 0].mean().expand(N)
                in_radius = torch.zeros(N, device=pts.device)
            else:
                local_density = torch.full((N,), 10.0, device=pts.device)
                in_radius = torch.zeros(N, device=pts.device)

            speed_val = 0.0
            if self.cur_vel_array.shape[1] > 0:
                speed_val = float(np.linalg.norm(self.cur_vel_array[:, 0]))
            speed = torch.full((N,), speed_val, device=pts.device)

            d_pred_rank = torch.argsort(torch.argsort(torch.abs(d_pred))).float() / max(N - 1, 1)

            noise_std_val = 0.0
            if self._risk_calibration is not None:
                noise_std_val = self._risk_calibration.get('noise_std', 0.0)
            noise_std = torch.full((N,), noise_std_val, device=pts.device)

            d_pred_delta = torch.zeros(N, device=pts.device)
            if self._cpsa_prev_d_pred is not None and self._cpsa_prev_d_pred.shape[0] == N:
                d_pred_delta = d_pred - self._cpsa_prev_d_pred

            radius_delta = torch.zeros(N, device=pts.device)
            if self._cpsa_prev_radius is not None and self._cpsa_prev_radius.shape[0] == N:
                radius_delta = radius - self._cpsa_prev_radius

            lateral_margin_val, left_min, right_min = self._compute_lateral_margin(pts)
            lateral_margin = torch.full((N,), lateral_margin_val, device=pts.device)
            passage_ratio_val = (left_min + right_min) / max(self._robot_width, 0.1)
            passage_ratio = torch.full((N,), passage_ratio_val, device=pts.device)

            env_type_val = 0.0
            if self._risk_calibration is not None:
                env_type_val = float(self._risk_calibration.get('env_type', 0)) / 4.0
            env_type = torch.full((N,), env_type_val, device=pts.device)

            feat_dim = getattr(self, '_cpsa_feat_dim', 17)
            base_features = [
                d_pred, radius, angle, proximity_ratio, d_pred ** 2,
                torch.cos(angle), torch.sin(angle),
                local_density, in_radius,
                speed, d_pred_rank, noise_std,
                d_pred_delta, radius_delta,
                lateral_margin, passage_ratio, env_type,
            ]
            if feat_dim < len(base_features):
                base_features = base_features[:feat_dim]

            features = torch.stack(base_features, dim=1)

            # Collect raw features for online adaptation (before normalization)
            if self._cpsa_online_adapter is not None:
                self._cpsa_episode_buffer.append((features.detach().cpu(), d_pred.detach().cpu()))

            if self._cpsa_feat_mean is not None and self._cpsa_feat_std is not None:
                feat_mean = self._cpsa_feat_mean.to(features.device)
                feat_std = self._cpsa_feat_std.to(features.device)
                if feat_mean.shape[0] < features.shape[1]:
                    pad_len = features.shape[1] - feat_mean.shape[0]
                    feat_mean = torch.cat([feat_mean, torch.zeros(pad_len, device=features.device)])
                    feat_std = torch.cat([feat_std, torch.ones(pad_len, device=features.device)])
                features = (features - feat_mean) / (feat_std + 1e-6)

            features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

            # Ablation: zero out feature groups
            cpsa_cfg = self._risk_calibration.get('cpsa_config', {}) if self._risk_calibration else {}
            if cpsa_cfg.get('disable_temporal'):
                features[:, 12:14] = 0.0  # d_pred_delta, radius_delta
            if cpsa_cfg.get('disable_passage'):
                features[:, 14:16] = 0.0  # lateral_margin, passage_ratio

            tau_base = self._cpsa_net(
                features,
                disable_global_ctx=cpsa_cfg.get('disable_global_ctx', False),
            )

            # Apply conformal correction with dynamic q_star
            q = self._cpsa_q_star
            if self._cpsa_q_star_conditional and not cpsa_cfg.get('static_q_star'):
                cond_keys = sorted(self._cpsa_q_star_conditional.keys())
                closest = min(cond_keys, key=lambda k: abs(k - noise_std_val))
                q = self._cpsa_q_star_conditional[closest]

            tau_raw_mean = tau_base.mean().item()
            if cpsa_cfg.get('static_q_star'):
                q_scale = 1.0
            else:
                q_scale = max(0.4, 1.0 - tau_raw_mean / 0.04)
            q = q * q_scale

            # Add static CP margin as safety floor: CPSA never worse than Level 1
            q_hat_floor = 0.0
            if self._risk_calibration is not None:
                q_hat_floor = self._risk_calibration.get('q_hat', 0.0)

            # Per-point safety margin with conformal correction
            tau_final = torch.relu(tau_base + q + q_hat_floor)

            # Upper bound: prevent excessively large margins
            tau_max = 0.08  # 8cm cap
            tau_final = torch.clamp(tau_final, 0.0, tau_max)

            if N > 3 and not cpsa_cfg.get('disable_smoothing'):
                tau_np = tau_final.cpu().numpy()
                sorted_idx = np.argsort(d_pred.cpu().numpy())
                tau_sorted = tau_np[sorted_idx]
                kernel = np.array([0.25, 0.5, 0.25])
                tau_smooth = np.convolve(tau_sorted, kernel, mode='same')
                unsort_idx = np.argsort(sorted_idx)
                tau_np = tau_smooth[unsort_idx]
                tau_final = torch.tensor(tau_np, device=pts.device, dtype=torch.float32)

            self._cpsa_prev_d_pred = d_pred.detach().clone()
            self._cpsa_prev_radius = radius.detach().clone()

        return tau_final.cpu().numpy()

    def get_risk_report(self):
        """Return current risk calibration state."""
        if self._risk_calibration is None:
            return {
                "enabled": False, "q_hat": 0.0,
                "collision_threshold_adjustment": 0.0,
            }
        return {
            "enabled": True,
            **self._risk_calibration,
            "collision_threshold_adjustment": self._collision_threshold_adjustment,
            "effective_collision_threshold": (
                self.collision_threshold - self._collision_threshold_adjustment
            ),
        }

    def update_adjust_parameters(self, **kwargs):

        """
        update the adjust parameters value: q_s, p_u, eta, d_max, d_min

        Args:
            q_s: float, the weight of the state cost
            p_u: float, the weight of the speed cost
            eta: float, the weight of the collision avoidance cost
            d_max: float, the maximum distance to the obstacle
            d_min: float, the minimum distance to the obstacle
        """

        self.pan.nrmp_layer.update_adjust_parameters_value(**kwargs)

    @property
    def min_distance(self):
        return self.pan.min_distance

    @property
    def dune_points(self):
        return self.pan.dune_points

    @property
    def nrmp_points(self):
        return self.pan.nrmp_points

    @property
    def initial_path(self):
        return self.ipath.initial_path

    @property
    def adjust_parameters(self):
        return self.pan.nrmp_layer.adjust_parameters

    @property
    def waypoints(self):
        return self.ipath.waypoints
