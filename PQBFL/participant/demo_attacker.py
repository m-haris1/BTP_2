import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Try to infer a reasonable input shape for common datasets
def _candidate_input_shape(model, dataset_type='MNIST'):
    # MNIST: 1x28x28, CIFAR: 3x32x32, UCI_HAR (if used) -> fallback to 1x1x128
    candidates = {
        'MNIST': (1, 1, 28, 28),
        'CIFAR': (1, 3, 32, 32),
        'UCI_HAR': (1, 1, 128)
    }
    return candidates.get(dataset_type, (1, 1, 28, 28))

def _extract_target_grads(model, raw_update):
    """
    raw_update: dict keyed by state_dict keys with tensors representing (new-old)
    Return a list of target tensors aligned with model.parameters()
    """
    state_keys = list(model.state_dict().keys())
    target_grads = []
    for k, p in zip(state_keys, model.parameters()):
        t = raw_update.get(k, None)
        if t is None:
            target_grads.append(torch.zeros_like(p).to(p.device))
        else:
            target_grads.append(t.detach().to(p.device))
    return target_grads

def demo_gradient_attack(model, raw_update, dataset_type='MNIST', steps=300, lr=0.1, verbose=False):
    """
    Attempt a simple DLG-style reconstruction by matching parameter gradients/updates.
    Returns reconstructed numpy image (HWC) or None on failure.
    Note: this is a demo and may not reconstruct well for large models/datasets.
    """
    device = next(model.parameters()).device
    model.eval()

    # infer input shape and create candidate
    shape = _candidate_input_shape(model, dataset_type)
    cand = torch.randn(shape, requires_grad=True, device=device, dtype=torch.float32)
    # soft label logits (optimize)
    num_classes = getattr(model, 'num_classes', None)
    if num_classes is None:
        # try infer from last layer weight
        try:
            last_w = list(model.parameters())[-1]
            num_classes = last_w.shape[0]
        except Exception:
            num_classes = 10
    label_logits = torch.randn((1, num_classes), requires_grad=True, device=device)

    target_grads = _extract_target_grads(model, raw_update)

    opt = torch.optim.Adam([cand, label_logits], lr=lr)
    mse = torch.nn.MSELoss()

    for i in range(steps):
        opt.zero_grad()
        # forward
        try:
            preds = model(cand)
        except Exception:
            # maybe need channel-last or flatten: try flatten input
            try:
                preds = model(cand.view(cand.size(0), -1))
            except Exception:
                return None

        # compute differentiable loss using soft labels
        soft_labels = F.softmax(label_logits, dim=1)
        pred_logsoft = F.log_softmax(preds, dim=1)
        loss_ce = - (soft_labels * pred_logsoft).sum(dim=1).mean()

        # compute grads of model params w.r.t. this loss
        grads = torch.autograd.grad(loss_ce, list(model.parameters()), create_graph=True)
        # build match loss vs target_grads (treat raw_update as target gradients)
        match_loss = 0.0
        for g, tg in zip(grads, target_grads):
            # normalize scales to reduce numeric mismatch
            match_loss = match_loss + mse(g, tg)
        total_loss = match_loss
        total_loss.backward()
        opt.step()

        if verbose and (i % max(1, steps//5) == 0):
            print(f"DLG iter {i}/{steps} loss={total_loss.item():.6f}")

    recon = cand.detach().cpu().numpy()
    # convert CHW->HWC for images
    if recon.ndim == 4:
        recon_img = recon[0]
        if recon_img.shape[0] <= 3:
            recon_img = np.transpose(recon_img, (1,2,0))
    else:
        recon_img = recon

    return recon_img

def show_image(img_arr, out_path=None, vmin=None, vmax=None):
    """
    Display or save an image array (HWC). If single-channel, use gray cmap.
    """
    if img_arr is None:
        print("No reconstruction to show.")
        return
    os.makedirs(os.path.dirname(out_path) if out_path else '.', exist_ok=True)
    plt.figure(figsize=(3,3))
    if img_arr.ndim == 2 or (img_arr.ndim==3 and img_arr.shape[2]==1):
        plt.imshow(img_arr.squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.clip((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8), 0, 1))
    plt.axis('off')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved reconstruction to {out_path}")
    else:
        plt.show()