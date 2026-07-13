import numpy as np
from PIL import Image
from autograd import Tensor, clear_graph
from gaussian2d import build_covariance_2d
from render import render_gaussians_alpha_composite_tensor
from loss import l1_loss
from optim import Adam

H, W = 64, 64
target_image = np.array(Image.open("target.png")).astype(np.float64) / 255.0

np.random.seed(42)
N = 200

means = [
    Tensor(np.random.rand(2) * np.array([W, H]), requires_grad=True)
    for _ in range(N)
]
covs = [Tensor(build_covariance_2d(5, 5, 0)) for _ in range(N)]
colors = [Tensor(np.random.rand(3), requires_grad=True) for _ in range(N)]
opacities = [Tensor(np.array(0.5), requires_grad=True) for _ in range(N)]
depths = [float(i) for i in range(N)]

params = means + colors + opacities
optimizer = Adam(params, lr=0.05)
target = Tensor(target_image)

for step in range(501):
    optimizer.zero_grad()
    pred = render_gaussians_alpha_composite_tensor(
        means, covs, colors, opacities, depths, H, W
    )
    loss = l1_loss(pred, target)
    loss.backward()
    clear_graph(loss)
    optimizer.step()

    if step % 100 == 0:
        img = np.clip(pred.data.reshape(H, W, 3), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(f"adam_step{step:03d}.png")
        print(f"Step {step:4d}: loss = {loss.data:.6f} -> adam_step{step:03d}.png")
