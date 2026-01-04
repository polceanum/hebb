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

sparsityParam = 0.7
gradDecay = 0.999
gradGrow = 1.001

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
    Minimal Dataset for MNIST-like IDX datasets (MNIST, FashionMNIST, KMNIST).

    Normalization matches torchvision:
      mean = 0.1307, std = 0.3081
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
            # MNIST via base URL
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
            # FashionMNIST / KMNIST use explicit URLs
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
        img = self.images[idx]  # (28, 28), uint8
        label = int(self.labels[idx])

        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # (1, 28, 28)
        img = (img - self.mean) / self.std
        return img, label

# ---------------------------------------------------------
# Masked linear + VAE type net
# ---------------------------------------------------------

def masked_linear(input, weight, weightMask=None, biasMask=None, bias=None):
    r"""
    Applies a linear transformation to the incoming data: y = xA^T + b
    with optional masks on weight and bias.
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
        return masked_linear(input, self.weight,
                             weightMask=weightMask,
                             biasMask=biasMask,
                             bias=self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = MaskedLinear(28*28, 512)
        self.fc2 = MaskedLinear(512, 256)
        self.fc3 = MaskedLinear(256, 128)

        self.latent_size = 16
        self.hidden2mean = MaskedLinear(128, self.latent_size)
        self.hidden2logv = MaskedLinear(128, self.latent_size)
        self.latent2net = MaskedLinear(self.latent_size, 128*10 + 10)

    def forward(self, x, masks=[None]*12):
        # x: (N, 1, 28, 28)
        x = x.view(-1, 28*28)

        x = self.fc1(x, masks[0], masks[1])
        x = F.relu(x)
        x = self.fc2(x, masks[2], masks[3])
        x = F.relu(x)
        x = self.fc3(x, masks[4], masks[5])
        x = F.relu(x)

        mean = self.hidden2mean(x, masks[6], masks[7])
        logv = self.hidden2logv(x, masks[8], masks[9])

        std = torch.exp(0.5 * logv)

        # Sample z on same device as x
        z = torch.randn(x.size(0), self.latent_size, device=x.device)
        z = z * std + mean

        netWeights = self.latent2net(z, masks[10], masks[11]).mean(dim=0)

        # First 128*10 are weights, last 10 are bias
        W = netWeights[:-10].view(128, 10)
        b = netWeights[-10:]
        x = x @ W + b

        return F.log_softmax(x, dim=1), mean, logv

# ---------------------------------------------------------
# Pruning, KL, train, test
# ---------------------------------------------------------

def pruningMasks(x, y, model, sparsity, silent=False):
    """SNIP-like pruning to build masks for each parameter tensor."""
    masks = [torch.ones_like(layer) for layer in model.parameters()]
    weights = list(model.parameters())

    model.zero_grad()
    grads = [torch.zeros_like(w) for w in weights]

    logits, mean, logv = model.forward(x)
    L = F.nll_loss(logits, y)
    grads = [g.abs() + ag.abs() for g, ag in zip(grads, torch.autograd.grad(L, weights))]

    with torch.no_grad():
        saliences = [(grad * weight).view(-1).abs().cpu()
                     for weight, grad in zip(weights, grads)]
        saliences = torch.cat(saliences)

        thresh = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0])

        for j, mask in enumerate(masks):
            mask[(grads[j] * weights[j]).abs() <= thresh] = 0

    model.zero_grad()
    return masks


def KLloss(mean, logv):
    # Standard VAE KL divergence
    KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
    return 0.001 * KL_loss


def train(args, model, device, train_loader, optimizer, epoch, gradMasks):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        masks = pruningMasks(data, target, model, sparsityParam)

        for j, gradMask in enumerate(gradMasks):
            gradMask *= torch.min((1 - masks[j]) * gradGrow + masks[j] * gradDecay,
                                  torch.ones_like(gradMask))
            gradMask = torch.max(gradMask, torch.ones_like(gradMask) * 0.01)

        optimizer.zero_grad()
        output, mean, logv = model(data, masks)

        kl_loss = KLloss(mean, logv)
        loss = F.nll_loss(output, target) + kl_loss
        loss.backward()

        for j, p in enumerate(model.parameters()):
            p.grad *= gradMasks[j]

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKLLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), kl_loss.item()))

    return masks, gradMasks


def test(testName, args, model, device, test_loader, masks=[None]*12):
    model.eval()
    test_loss = 0
    test_lossAided = 0
    correct = 0
    correctAided = 0

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output, mean, logv = model(data, masks)
            test_loss += (F.nll_loss(output, target, reduction='sum').item() +
                          KLloss(mean, logv).item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        aidedMasks = pruningMasks(data, target, model, sparsityParam)
        with torch.no_grad():
            output, mean, logv = model(data, aidedMasks)
            test_lossAided += (F.nll_loss(output, target, reduction='sum').item() +
                               KLloss(mean, logv).item())
            pred = output.argmax(dim=1, keepdim=True)
            correctAided += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_lossAided /= len(test_loader.dataset)

    print('\nTest{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), '
          'Average lossAided: {:.4f}, AccuracyAided: {}/{} ({:.0f}%)'.format(
              testName, test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset),
              test_lossAided, correctAided, len(test_loader.dataset),
              100. * correctAided / len(test_loader.dataset)))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Masked VAE-ish MNIST (no torchvision)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true', default=False)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Dataset roots (mirroring your structure)
    mnist_root = './data/MNIST'
    fashion_root = './data/FashionMNIST'
    kmnist_root = './data/KMNIST'

    # Datasets
    train_dataset = MNISTLikeDataset(mnist_root, MNIST_URLS, train=True, download=True)
    test_dataset = MNISTLikeDataset(mnist_root, MNIST_URLS, train=False, download=True)

    train_dataset2 = MNISTLikeDataset(fashion_root, FASHION_URLS, train=True, download=True)
    test_dataset2 = MNISTLikeDataset(fashion_root, FASHION_URLS, train=False, download=True)

    train_dataset3 = MNISTLikeDataset(kmnist_root, KMNIST_URLS, train=True, download=True)
    test_dataset3 = MNISTLikeDataset(kmnist_root, KMNIST_URLS, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(
        test_dataset2, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_loader3 = torch.utils.data.DataLoader(
        train_dataset3, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader3 = torch.utils.data.DataLoader(
        test_dataset3, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, nesterov=True)

    for layer in model.parameters():
        print(layer.size())

    gradMasks = [torch.ones_like(layer) for layer in model.parameters()]

    # Your original script loops 10 times over the 4 DBs; here we keep the 10
    # cycles but across 3 DBs (MNIST, Fashion, KMNIST), since EMNIST is removed.
    for i in range(10):
        print(f"\n=== Cycle {i+1}/10 ===")

        masks1 = None
        masks2 = None
        masks3 = None

        # Db1: MNIST
        for epoch in range(1, args.epochs + 1):
            masks1, gradMasks = train(args, model, device, train_loader, optimizer, epoch, gradMasks)
            test('Db1 w/o mask', args, model, device, test_loader)
            test('Db1 with mask', args, model, device, test_loader, masks1)

        # Db2: FashionMNIST
        for epoch in range(1, args.epochs + 1):
            masks2, gradMasks = train(args, model, device, train_loader2, optimizer, epoch, gradMasks)
            test('Db2 w/o mask', args, model, device, test_loader2)
            test('Db2 with mask', args, model, device, test_loader2, masks2)

            test('Db1 w/o mask', args, model, device, test_loader)
            test('Db1 with mask', args, model, device, test_loader, masks1)

        # Db3: KMNIST
        for epoch in range(1, args.epochs + 1):
            masks3, gradMasks = train(args, model, device, train_loader3, optimizer, epoch, gradMasks)
            test('Db3 w/o mask', args, model, device, test_loader3)
            test('Db3 with mask', args, model, device, test_loader3, masks3)

            test('Db2 w/o mask', args, model, device, test_loader2)
            test('Db2 with mask', args, model, device, test_loader2, masks2)

            test('Db1 w/o mask', args, model, device, test_loader)
            test('Db1 with mask', args, model, device, test_loader, masks1)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
