# main_salience_v2.py
# A stateful, multi-signal salience with per-tensor budgets + hysteresis

from __future__ import print_function
import argparse
import math
import os
import gzip
import struct
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# ----------------------------
# Global knobs
# ----------------------------
sparsityParam = 0.7
gradDecay = 0.999
gradGrow = 1.001

display = False

if display:
    plotSize = (512, 28 * 28)
    myData = np.random.random(plotSize)
    fig = plt.figure()
    im = plt.imshow(myData, interpolation='nearest')

# ---------------------------------------------------------
# Minimal IDX-based dataset loaders (no torchvision)
# ---------------------------------------------------------

MNIST_URLS = {
    "base": "https://storage.googleapis.com/cvdf-datasets/mnist/"
}

FASHION_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

KMNIST_URLS = {
    "train_images": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz",
    "test_images":  "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz",
}


def download_url(url, root, filename):
    os.makedirs(root, exist_ok=True)
    fpath = os.path.join(root, filename)
    if os.path.exists(fpath):
        return fpath
    print("Downloading", url, "to", fpath)
    urllib.request.urlretrieve(url, fpath)
    return fpath


def read_idx_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data


def read_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


class MNISTLikeDataset(torch.utils.data.Dataset):
    """
    Minimal Dataset for MNIST-like IDX datasets (MNIST / FashionMNIST / KMNIST).
    Normalization matches torchvision: mean=0.1307, std=0.3081
    """
    def __init__(self, root, urls, train=True, download=True):
        super().__init__()
        self.root = root
        self.urls = urls
        self.train = train
        self.mean = 0.1307
        self.std = 0.3081

        if download:
            self._download()

        if train:
            img_file = os.path.join(root, "train-images-idx3-ubyte.gz")
            lbl_file = os.path.join(root, "train-labels-idx1-ubyte.gz")
        else:
            img_file = os.path.join(root, "t10k-images-idx3-ubyte.gz")
            lbl_file = os.path.join(root, "t10k-labels-idx1-ubyte.gz")

        imgs = read_idx_images(img_file)     # (N, 28, 28)
        labels = read_idx_labels(lbl_file)   # (N,)
        self.images = imgs
        self.labels = labels

    def _download(self):
        base = self.urls.get("base", None)
        if base is not None:
            files = [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ]
            for fname in files:
                url = base + fname
                download_url(url, self.root, fname)
        else:
            mapping = [
                ("train-images-idx3-ubyte.gz", "train_images"),
                ("train-labels-idx1-ubyte.gz", "train_labels"),
                ("t10k-images-idx3-ubyte.gz", "test_images"),
                ("t10k-labels-idx1-ubyte.gz", "test_labels"),
            ]
            for fname, key in mapping:
                url = self.urls[key]
                download_url(url, self.root, fname)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]              # (28,28), uint8
        label = int(self.labels[idx])

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        img = (img - self.mean) / self.std
        return img, label


# ---------------------------------------------------------
# Masked linear and network
# ---------------------------------------------------------

def masked_linear(input, weight, weightMask=None, biasMask=None, bias=None):
    r"""
    Applies a linear transformation to the incoming data: y = xA^T + b.
    """
    if input.dim() == 2 and bias is not None:
        if weightMask is not None and biasMask is not None:
            ret = torch.addmm(bias * biasMask, input, (weight * weightMask).t())
        else:
            ret = torch.addmm(bias, input, weight.t())
    else:
        if weightMask is not None:
            output = input.matmul((weight * weightMask).t())
        else:
            output = input.matmul(weight.t())
        if bias is not None:
            if biasMask is not None:
                output += bias * biasMask
            else:
                output += bias
        ret = output
    return ret


class MaskedLinear(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, weightMask=None, biasMask=None):
        return masked_linear(
            input, self.weight,
            weightMask=weightMask,
            biasMask=biasMask,
            bias=self.bias
        )

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = MaskedLinear(28 * 28, 256)
        self.fc2 = MaskedLinear(256, 256)
        self.fc3 = MaskedLinear(256, 256)
        self.fc4 = MaskedLinear(256, 10)

    def forward(self, x, masks=[None, None, None, None, None, None, None, None]):
        x = x.view(-1, 28 * 28)

        x = self.fc1(x, masks[0], masks[1])
        x = F.relu(x)
        act1 = x

        x = self.fc2(x, masks[2], masks[3])
        x = F.relu(x)
        act2 = x

        x = self.fc3(x, masks[4], masks[5])
        x = F.relu(x)
        act3 = x

        x = self.fc4(x, masks[6], masks[7])
        act4 = x
        return F.log_softmax(x, dim=1), [act1, act1, act2, act2, act3, act3, act4, act4]


# ------------------------------------------------------------------
# Stateful multi-signal salience with per-tensor budget + hysteresis
# ------------------------------------------------------------------

class SalienceV2:
    """
    Tracks per-parameter salience state from a "usage" tensor derived from activations.
    Salience is updated via running statistics and used to pick top-k
    elements per tensor (layer-local budget), with hysteresis to reduce churn.

    For weights: usage is broadcast from per-output activation statistics.
    For biases: usage is the per-output activation statistics.
    """

    def __init__(
        self,
        ema_decay=0.99,
        alpha_abs=1.0,
        beta_var=0.25,
        gamma_persist=0.25,
        persist_threshold=0.0,
        normalize_per_tensor=True,
        norm_ema_decay=0.99,
        use_hysteresis=True,
        keep_ratio=0.5,
        eps=1e-8,
    ):
        self.ema_decay = float(ema_decay)
        self.alpha_abs = float(alpha_abs)
        self.beta_var = float(beta_var)
        self.gamma_persist = float(gamma_persist)
        self.persist_threshold = float(persist_threshold)
        self.normalize_per_tensor = bool(normalize_per_tensor)
        self.norm_ema_decay = float(norm_ema_decay)
        self.use_hysteresis = bool(use_hysteresis)
        self.keep_ratio = float(keep_ratio)
        self.eps = float(eps)

        # state per param index (int key)
        self.ema_abs = {}
        self.ema_sq = {}
        self.ema_actrate = {}

        # normalization stats (scalar per tensor)
        self.sal_mean = {}
        self.sal_var = {}

        # previous mask for hysteresis
        self.prev_mask = {}

    @torch.no_grad()
    def _ensure(self, key, shape, device, dtype=torch.float32):
        if key in self.ema_abs:
            return
        z = torch.zeros(shape, device=device, dtype=dtype)
        self.ema_abs[key] = z.clone()
        self.ema_sq[key] = z.clone()
        self.ema_actrate[key] = z.clone()
        self.sal_mean[key] = torch.zeros((), device=device, dtype=dtype)
        self.sal_var[key] = torch.ones((), device=device, dtype=dtype)
        self.prev_mask[key] = torch.zeros(shape, device=device, dtype=torch.bool)

    @torch.no_grad()
    def update_and_score(self, key: int, usage: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        usage: tensor same shape as weight (already broadcast if needed)
        weight: parameter tensor
        returns: salience score tensor same shape as weight
        """
        # stable dtype for stats
        u = usage.detach()
        if u.dtype != torch.float32:
            u = u.float()

        self._ensure(key, u.shape, u.device, u.dtype)

        d = self.ema_decay
        abs_u = u.abs()
        sq_u = u * u
        act = (abs_u > self.persist_threshold).float()

        self.ema_abs[key].mul_(d).add_(abs_u, alpha=(1.0 - d))
        self.ema_sq[key].mul_(d).add_(sq_u, alpha=(1.0 - d))
        self.ema_actrate[key].mul_(d).add_(act, alpha=(1.0 - d))

        var_proxy = (self.ema_sq[key] - (self.ema_abs[key] ** 2)).clamp_min(0.0)

        raw = (
            self.alpha_abs * self.ema_abs[key]
            + self.beta_var * var_proxy
            + self.gamma_persist * self.ema_actrate[key]
        )

        if self.normalize_per_tensor:
            nd = self.norm_ema_decay
            m = raw.mean()
            v = raw.var(unbiased=False).clamp_min(self.eps)
            self.sal_mean[key].mul_(nd).add_(m, alpha=(1.0 - nd))
            self.sal_var[key].mul_(nd).add_(v, alpha=(1.0 - nd))
            z = (raw - self.sal_mean[key]) / self.sal_var[key].sqrt()
            # z can be negative; shift + clamp to keep score nonnegative and stable
            raw = (z + 2.0).clamp_min(0.0)

        score = weight.detach().abs()
        if score.dtype != raw.dtype:
            score = score.float()
        score = score * raw
        return score

    @torch.no_grad()
    def topk_mask(self, key: int, score: torch.Tensor, keep_frac: float) -> torch.Tensor:
        """
        Per-tensor budget: keep top (keep_frac) fraction by score, with optional hysteresis.
        """
        keep_frac = float(keep_frac)
        flat = score.flatten()
        n = flat.numel()
        k = max(1, int(round(keep_frac * n)))
        if k >= n:
            base = torch.ones_like(flat, dtype=torch.bool).view_as(score)
            self.prev_mask[key] = base
            return base

        # base top-k
        _, idx = torch.topk(flat, k, largest=True, sorted=False)
        base = torch.zeros_like(flat, dtype=torch.bool)
        base[idx] = True
        base = base.view_as(score)

        if not self.use_hysteresis:
            self.prev_mask[key] = base
            return base

        prev = self.prev_mask.get(key, torch.zeros_like(base))
        if prev.shape != base.shape:
            prev = torch.zeros_like(base)

        keep_k = int(round(self.keep_ratio * k))
        keep_k = max(0, min(keep_k, k))

        if keep_k == 0 or prev.sum().item() == 0:
            self.prev_mask[key] = base
            return base

        prev_idx = prev.flatten().nonzero(as_tuple=False).flatten()
        prev_scores = flat[prev_idx]

        # keep best keep_k among previous active positions
        if keep_k >= prev_idx.numel():
            keep_prev = prev.clone()
        else:
            _, top_prev = torch.topk(prev_scores, keep_k, largest=True, sorted=False)
            keep_idx = prev_idx[top_prev]
            keep_prev = torch.zeros_like(flat, dtype=torch.bool)
            keep_prev[keep_idx] = True
            keep_prev = keep_prev.view_as(score)

        remaining_k = max(0, k - int(keep_prev.sum().item()))
        if remaining_k == 0:
            out = keep_prev
            self.prev_mask[key] = out
            return out

        eligible = (~keep_prev).flatten()
        eligible_scores = flat.clone()
        eligible_scores[~eligible] = -float("inf")

        if remaining_k >= int(eligible.sum().item()):
            add = eligible.view_as(score)
        else:
            _, add_idx = torch.topk(eligible_scores, remaining_k, largest=True, sorted=False)
            add = torch.zeros_like(flat, dtype=torch.bool)
            add[add_idx] = True
            add = add.view_as(score)

        out = keep_prev | add
        self.prev_mask[key] = out
        return out


# ------------------
# Pruning & training
# ------------------

@torch.no_grad()
def pruningMasks_v2(x, model, sparsity, salience_state: SalienceV2,
                    prevMasks=[None, None, None, None, None, None, None, None]):
    weights = list(model.parameters())
    masks = [torch.ones_like(p, dtype=torch.float32) for p in weights]

    # Forward once to get activations for usage statistics
    _, acts = model.forward(x, prevMasks)

    keep_frac = 1.0 - float(sparsity)

    for j, (p, act) in enumerate(zip(weights, acts)):
        # act: (batch, out_features)
        act_mean = act.mean(dim=0)  # (out_features,)

        if p.dim() == 1:
            # bias: usage aligns with output units
            usage = act_mean
        else:
            # weight: broadcast usage of output units across input dimension
            usage = act_mean.unsqueeze(1).expand_as(p)

        score = salience_state.update_and_score(key=j, usage=usage, weight=p)

        # per-tensor top-k selection
        m = salience_state.topk_mask(key=j, score=score, keep_frac=keep_frac)

        # match your mask dtype expectations (float mask used in masked_linear)
        masks[j] = m.to(dtype=p.dtype)

    return masks


def train(args, model, device, train_loader, optimizer, epoch, gradMasks, salience_state):
    global myData, im, plt
    model.train()

    last_masks = None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # compute pruning masks (data-dependent) and update gradMasks
        masks = pruningMasks_v2(data, model, sparsityParam, salience_state)
        last_masks = masks

        for j, gradMask in enumerate(gradMasks):
            gradMask *= torch.min(
                (1 - masks[j]) * gradGrow + masks[j] * gradDecay,
                torch.ones_like(gradMask)
            )
            gradMask = torch.max(gradMask, torch.ones_like(gradMask) * 0.01)
            gradMasks[j] = gradMask  # persistent update

        if display and batch_idx % args.log_interval == 0:
            myData = myData * 0 + gradMasks[0].cpu().view(-1)[:plotSize[0] * plotSize[1]] \
                .view(plotSize[0], plotSize[1]).numpy()
            im.set_array(myData)
            plt.draw()
            plt.pause(1e-17)

        optimizer.zero_grad()
        output, _ = model(data, masks)
        loss = F.nll_loss(output, target)
        loss.backward()

        for j, p in enumerate(model.parameters()):
            p.grad *= gradMasks[j]

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return last_masks, gradMasks


def test(testName, args, model, device, test_loader,
         masks=[None, None, None, None, None, None, None, None]):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output, _ = model(data, masks)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        testName, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

def main():
    parser = argparse.ArgumentParser(description='Hebbian masked MNIST/Fashion/KMNIST (salience v2, no torchvision)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train per phase (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_root = '../data/MNIST'
    fashion_root = '../data/FashionMNIST'
    kmnist_root = '../data/KMNIST'

    # Datasets
    train_dataset = MNISTLikeDataset(mnist_root, MNIST_URLS, train=True, download=True)
    test_dataset = MNISTLikeDataset(mnist_root, MNIST_URLS, train=False, download=True)

    train_dataset2 = MNISTLikeDataset(fashion_root, FASHION_URLS, train=True, download=True)
    test_dataset2 = MNISTLikeDataset(fashion_root, FASHION_URLS, train=False, download=True)

    train_dataset3 = MNISTLikeDataset(kmnist_root, KMNIST_URLS, train=True, download=True)
    test_dataset3 = MNISTLikeDataset(kmnist_root, KMNIST_URLS, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_loader3 = torch.utils.data.DataLoader(train_dataset3, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader3 = torch.utils.data.DataLoader(test_dataset3, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    masks1 = None
    masks2 = None
    masks3 = None

    print("Parameter shapes:")
    for layer in model.parameters():
        print(tuple(layer.size()))

    gradMasks = [torch.ones_like(layer) for layer in model.parameters()]

    # salience state
    salience_state = SalienceV2(
        ema_decay=0.99,
        alpha_abs=1.0,
        beta_var=0.25,
        gamma_persist=0.25,
        persist_threshold=0.0,
        normalize_per_tensor=True,
        norm_ema_decay=0.99,
        use_hysteresis=True,
        keep_ratio=0.5,
    )

    # ---- Phase 1: MNIST ----
    for epoch in range(1, args.epochs + 1):
        masks1, gradMasks = train(args, model, device, train_loader, optimizer, epoch, gradMasks, salience_state)
        test(' Db1 w/o mask', args, model, device, test_loader)
        test(' Db1 with mask', args, model, device, test_loader, masks1)

    # ---- Phase 2: FashionMNIST ----
    for epoch in range(1, args.epochs + 1):
        masks2, gradMasks = train(args, model, device, train_loader2, optimizer, epoch, gradMasks, salience_state)
        test(' Db2 w/o mask', args, model, device, test_loader2)
        test(' Db2 with mask', args, model, device, test_loader2, masks2)

        test(' Db1 w/o mask', args, model, device, test_loader)
        test(' Db1 with mask', args, model, device, test_loader, masks1)

        totalOverlap = 0
        for im_idx in range(len(masks1)):
            maskSum = (masks1[im_idx] + masks2[im_idx]) / 2
            totalOverlap += maskSum[maskSum == 1].sum()
        print('Total overlap (2 datasets):', totalOverlap.item() if torch.is_tensor(totalOverlap) else totalOverlap)

    # ---- Phase 3: KMNIST ----
    for epoch in range(1, args.epochs + 1):
        masks3, gradMasks = train(args, model, device, train_loader3, optimizer, epoch, gradMasks, salience_state)
        test(' Db3 w/o mask', args, model, device, test_loader3)
        test(' Db3 with mask', args, model, device, test_loader3, masks3)

        test(' Db2 w/o mask', args, model, device, test_loader2)
        test(' Db2 with mask', args, model, device, test_loader2, masks2)

        test(' Db1 w/o mask', args, model, device, test_loader)
        test(' Db1 with mask', args, model, device, test_loader, masks1)

        totalOverlap = 0
        for im_idx in range(len(masks1)):
            maskSum = (masks1[im_idx] + masks2[im_idx] + masks3[im_idx]) / 3
            totalOverlap += maskSum[maskSum == 1].sum()
        print('Total overlap (3 datasets):', totalOverlap.item() if torch.is_tensor(totalOverlap) else totalOverlap)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
