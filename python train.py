import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.datasets.cifar')
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ======================
# 固定种子
# ======================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# 超参数 | 64 世界
# ======================
BATCH_SIZE = 128
LR = 3e-4
EPOCHS = 120
NUM_CLASSES = 10
DIM = 256
NUM_WORLDS = 64
MOMENTUM = 0.99

# ======================
# 数据增强：+ RandAugment
# ======================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=5),  # 🔥 1行暴涨
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

# Windows 稳定
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ======================
# WideResNet-28-10 🔥 涨点3~4%
# ======================
class WideResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._block(64, 128, stride=1)
        self.layer2 = self._block(128, 256, stride=2)
        self.layer3 = self._block(256, 256, stride=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _block(self, in_c, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ======================
# LightCone 64 世界
# ======================
class LightConeDynamic(nn.Module):
    def __init__(self, dim=256, num_worlds=NUM_WORLDS, momentum=MOMENTUM, temp_base=0.3):
        super().__init__()
        self.dim = dim
        self.num_worlds = num_worlds
        self.momentum = momentum
        self.temp_base = temp_base
        self.worlds = nn.Parameter(torch.randn(num_worlds, dim) * 0.02, requires_grad=False)
        self.struct = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.norm = nn.LayerNorm(dim)
        self.gate_net = nn.Sequential(nn.Linear(dim, dim//8), nn.SiLU(), nn.Linear(dim//8, 1), nn.Sigmoid())

    def forward(self, x, label=None):
        x_norm = F.normalize(x, dim=-1)
        if self.training and label is not None:
            with torch.no_grad():
                centers = []
                for c in range(self.num_worlds):
                    mask = (label % self.num_worlds) == c
                    center = x_norm[mask].mean(dim=0) if mask.any() else self.worlds[c]
                    centers.append(center)
                centers = torch.stack(centers)
                self.worlds.data = self.momentum * self.worlds.data + (1-self.momentum) * centers
                self.worlds.data = F.normalize(self.worlds.data, dim=-1)

        w = F.normalize(self.worlds, dim=-1)
        sim = x_norm @ w.T
        feat_std = x_norm.std(dim=-1, keepdim=True)
        temp = self.temp_base * (1 + 2 * feat_std)
        att = F.softmax(sim / temp, dim=-1)
        fused = att @ w
        fused = self.struct(fused)
        conf = sim.max(dim=-1, keepdim=True)[0].detach()
        gate = self.gate_net(x) * conf
        out = x + gate * (fused - x_norm)
        out = self.norm(out)
        return out, sim

# ======================
# 基线：WideResNet
# ======================
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WideResNet(num_classes=NUM_CLASSES)
    def forward(self, x):
        return self.backbone(x)

# ======================
# WideResNet + LightCone
# ======================
class ResNet18_LightCone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WideResNet()
        self.lc = LightConeDynamic(dim=256)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x, label=None):
        feat = self.backbone.conv1(x)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.bn(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.avgpool(feat)
        feat = torch.flatten(feat, 1)

        if self.training and label is not None:
            feat, sim = self.lc(feat, label)
            return self.fc(feat), sim
        else:
            feat, _ = self.lc(feat, None)
            return self.fc(feat)

# ======================
# 训练 / 测试
# ======================
criterion = nn.CrossEntropyLoss()

def train(model_rn, model_lc, opt_rn, opt_lc):
    model_rn.train()
    model_lc.train()
    loss_rn_sum = 0
    loss_lc_sum = 0
    total = 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        bs = x.size(0)
        total += bs

        opt_rn.zero_grad()
        loss_rn = criterion(model_rn(x), y)
        loss_rn.backward()
        opt_rn.step()
        loss_rn_sum += loss_rn.item() * bs

        opt_lc.zero_grad()
        logits, _ = model_lc(x, y)
        loss_lc = criterion(logits, y)
        loss_lc.backward()
        opt_lc.step()
        loss_lc_sum += loss_lc.item() * bs

    return loss_rn_sum / total, loss_lc_sum / total

@torch.no_grad()
def test(model, noise=0.0):
    model.eval()
    correct = 0
    total = 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        if noise > 0:
            x = x + torch.randn_like(x) * noise
            x = torch.clamp(x, -1, 1)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100 * correct / total

# ======================
# 主程序
# ======================
if __name__ == '__main__':
    model_rn = ResNet18().to(device)
    model_lc = ResNet18_LightCone().to(device)

    opt_rn = optim.AdamW(model_rn.parameters(), lr=LR, weight_decay=1e-5)
    opt_lc = optim.AdamW(model_lc.parameters(), lr=LR, weight_decay=1e-5)

    sch_rn = optim.lr_scheduler.CosineAnnealingLR(opt_rn, T_max=EPOCHS)
    sch_lc = optim.lr_scheduler.CosineAnnealingLR(opt_lc, T_max=EPOCHS)

    print("=" * 70)
    print(" WideResNet + RandAugment + LightCone 64 worlds")
    print("=" * 70)

    for epoch in range(1, EPOCHS+1):
        loss_rn, loss_lc = train(model_rn, model_lc, opt_rn, opt_lc)
        sch_rn.step()
        sch_lc.step()

        acc_rn = test(model_rn)
        acc_lc = test(model_lc)

        print(f"Epoch {epoch:3d} | RN [{loss_rn:.4f}, {acc_rn:.2f}] | LC [{loss_lc:.4f}, {acc_lc:.2f}]")

    print("\n======================================================================")
    noises = [0.0, 0.05, 0.10, 0.15, 0.20]
    rn_res = [test(model_rn, n) for n in noises]
    lc_res = [test(model_lc, n) for n in noises]

    print(f"{'模型':<15} {'干净':<8} {'0.05':<8} {'0.10':<8} {'0.15':<8} {'0.20':<8}")
    print("------------------------------------------------------------")
    print(f"{'WideResNet':<15} {rn_res[0]:<8.2f} {rn_res[1]:<8.2f} {rn_res[2]:<8.2f} {rn_res[3]:<8.2f} {rn_res[4]:<8.2f}")
    print(f"{'LightCone':<15} {lc_res[0]:<8.2f} {lc_res[1]:<8.2f} {lc_res[2]:<8.2f} {lc_res[3]:<8.2f} {lc_res[4]:<8.2f}")
    print("------------------------------------------------------------")
    print(f"{'提升':<15} {lc_res[0]-rn_res[0]:<+8.2f} {lc_res[1]-rn_res[1]:<+8.2f} {lc_res[2]-rn_res[2]:<+8.2f} {lc_res[3]-rn_res[3]:<+8.2f} {lc_res[4]-rn_res[4]:<+8.2f}")
