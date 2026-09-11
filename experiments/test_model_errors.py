import sys; sys.path.insert(0, '.')
import torch, numpy as np, cvxpy as cp
from neupan.robot import robot
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, to_device

r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
G_np, h_np = r.G, r.h
G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)

def lp_gt(G, h, pt):
    mu = cp.Variable((G.shape[0],1), nonneg=True)
    p = cp.Parameter((2,1)); p.value = pt
    prob = cp.Problem(cp.Maximize(mu.T @ (G@p-h)), [cp.norm(G.T@mu)<=1])
    prob.solve(solver=cp.ECOS)
    return prob.value

for model_name in ['model_0.pth', 'model_500.pth', 'model_1000.pth', 'model_2000.pth', 'model_5000.pth']:
    path = f'example/model/diff_robot_default/{model_name}'
    model = to_device(ObsPointNet(2, G_np.shape[0]))
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    errors = []
    rng = np.random.RandomState(42)
    for _ in range(200):
        pt = rng.uniform(-25, 25, (2,1))
        pt_t = np_to_tensor(pt)
        with torch.no_grad():
            mu_pred = model(pt_t.T).T
        d_pred = float(torch.squeeze(mu_pred.T @ (G_t@pt_t - h_t)))
        d_gt = lp_gt(G_np, h_np, pt)
        if d_gt is not None:
            errors.append(d_pred - d_gt)

    errors = np.array(errors)
    r_asy = np.maximum(0, errors)
    r_sym = np.abs(errors)
    print(f'{model_name:15s}: mean_err={np.mean(errors)*1000:.1f}mm, '
          f'std={np.std(errors)*1000:.1f}mm, '
          f'max_asy={np.max(r_asy)*100:.2f}cm, '
          f'max_sym={np.max(r_sym)*100:.2f}cm, '
          f'P95_asy={np.percentile(r_asy,95)*100:.2f}cm')
