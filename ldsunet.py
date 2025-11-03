import argparse
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed: int = 42):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, width=64, height=256):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = (width, height)
        self.samples = self._pair_files()

    def _pair_files(self):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        imgs = [p for p in self.img_dir.iterdir() if p.suffix.lower() in extensions]
        masks = [p for p in self.mask_dir.iterdir() if p.suffix.lower() in extensions]
        img_map = {p.stem.lower(): p for p in imgs}
        pairs = []
        for m in masks:
            key = m.stem.lower()
            if key in img_map:
                pairs.append((img_map[key], m))
        if not pairs:
            n = min(len(imgs), len(masks))
            pairs = list(zip(sorted(imgs)[:n], sorted(masks)[:n]))
        return sorted(pairs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB").resize(self.target_size, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.target_size, Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        mask = torch.from_numpy(mask[None, ...])
        return img, mask, img_path.name

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth=16, layers=3, bn=True):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(layers):
            ops = [nn.Conv2d(ch, growth, kernel_size=3, padding=1, bias=not bn)]
            if bn:
                ops.append(nn.BatchNorm2d(growth))
            ops.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Sequential(*ops))
            ch += growth
        self.out_ch = ch

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            y = layer(torch.cat(feats, dim=1))
            feats.append(y)
        return torch.cat(feats, dim=1)

class PlainBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not bn),
            nn.BatchNorm2d(out_ch) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not bn),
            nn.BatchNorm2d(out_ch) if bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)
        self.out_ch = out_ch

    def forward(self, x):
        return self.net(x)

class TransitionDown(nn.Module):
    def __init__(self, in_ch, out_ch=None, bn=True):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        ops = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=not bn)]
        if bn:
            ops.append(nn.BatchNorm2d(out_ch))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))

class TransitionUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class SelectionGate(nn.Module):
    def __init__(self, c_skip, c_gate):
        super().__init__()
        self.theta = nn.Conv2d(c_skip, 1, kernel_size=1, bias=True)
        self.phi   = nn.Conv2d(c_gate, 1, kernel_size=1, bias=True)

    def forward(self, x_skip, x_gate):
        s = self.theta(x_skip) + self.phi(x_gate)
        a = torch.sigmoid(s)
        return x_skip * a

class LDSUNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        base=32,
        growth=16,
        layers_per_block=3,
        use_dense=True,
        use_gate=True,
        use_mskip=True,
        bn=True,
    ):
        super().__init__()
        self.use_dense = use_dense
        self.use_gate = use_gate
        self.use_mskip = use_mskip

        def enc_block(in_channels, stage_base):
            if self.use_dense:
                blk = DenseBlock(in_channels, growth, layers_per_block, bn=bn)
                out_c = blk.out_ch
            else:
                blk = PlainBlock(in_channels, stage_base, bn=bn)
                out_c = blk.out_ch
            return blk, out_c

        def dec_block(in_channels_cat):
            if self.use_dense:
                blk = DenseBlock(in_channels_cat, growth, layers_per_block, bn=bn)
                out_c = blk.out_ch
            else:
                out_tgt = max(in_channels_cat // 2, base)
                blk = PlainBlock(in_channels_cat, out_tgt, bn=bn)
                out_c = blk.out_ch
            return blk, out_c

        self.db1, c1 = enc_block(in_ch, base)
        self.td1 = TransitionDown(c1, base, bn=bn)

        self.db2, c2 = enc_block(base, base * 2)
        self.td2 = TransitionDown(c2, base * 2, bn=bn)

        self.db3, c3 = enc_block(base * 2, base * 4)
        self.td3 = TransitionDown(c3, base * 4, bn=bn)

        self.db4, c4 = enc_block(base * 4, base * 8)
        self.td4 = TransitionDown(c4, base * 8, bn=bn)

        if self.use_dense:
            self.db5 = DenseBlock(base * 8, growth, layers_per_block, bn=bn)
            c5 = self.db5.out_ch
        else:
            self.db5 = PlainBlock(base * 8, base * 8, bn=bn)
            c5 = self.db5.out_ch

        in_db6 = base * 8 + (c4 if self.use_mskip else 0)
        in_db7 = base * 4 + (c3 if self.use_mskip else 0)
        in_db8 = base * 2 + (c2 if self.use_mskip else 0)
        in_db9 = base     + (c1 if self.use_mskip else 0)

        self.tu4 = TransitionUp(c5, base * 8)
        if self.use_gate and self.use_mskip:
            self.g4 = SelectionGate(c4, base * 8)
        self.db6, c6 = dec_block(in_db6)

        self.tu3 = TransitionUp(c6, base * 4)
        if self.use_gate and self.use_mskip:
            self.g3 = SelectionGate(c3, base * 4)
        self.db7, c7 = dec_block(in_db7)

        self.tu2 = TransitionUp(c7, base * 2)
        if self.use_gate and self.use_mskip:
            self.g2 = SelectionGate(c2, base * 2)
        self.db8, c8 = dec_block(in_db8)

        self.tu1 = TransitionUp(c8, base)
        if self.use_gate and self.use_mskip:
            self.g1 = SelectionGate(c1, base)
        self.db9, c9 = dec_block(in_db9)

        self.final = nn.Conv2d(c9, out_ch, kernel_size=1)

        self._c1, self._c2, self._c3, self._c4 = c1, c2, c3, c4

    def forward(self, x):
        d1 = self.db1(x); p1 = self.td1(d1)
        d2 = self.db2(p1); p2 = self.td2(d2)
        d3 = self.db3(p2); p3 = self.td3(d3)
        d4 = self.db4(p3); p4 = self.td4(d4)

        bn = self.db5(p4)

        u4 = self.tu4(bn)
        if self.use_mskip:
            s4 = d4 if not self.use_gate else self.g4(d4, u4)
            x  = self.db6(torch.cat([u4, s4], dim=1))
        else:
            x  = self.db6(u4)

        u3 = self.tu3(x)
        if self.use_mskip:
            s3 = d3 if not self.use_gate else self.g3(d3, u3)
            x  = self.db7(torch.cat([u3, s3], dim=1))
        else:
            x  = self.db7(u3)

        u2 = self.tu2(x)
        if self.use_mskip:
            s2 = d2 if not self.use_gate else self.g2(d2, u2)
            x  = self.db8(torch.cat([u2, s2], dim=1))
        else:
            x  = self.db8(u2)

        u1 = self.tu1(x)
        if self.use_mskip:
            s1 = d1 if not self.use_gate else self.g1(d1, u1)
            x  = self.db9(torch.cat([u1, s1], dim=1))
        else:
            x  = self.db9(u1)

        return self.final(x)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, probs, target):
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()

def dice_score(pred, target, eps=1e-7):
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2 * inter + eps) / (union + eps)

def iou_score(pred, target, eps=1e-7):
    inter = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - inter
    return (inter + eps) / (union + eps)

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    dices, ious = [], []
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    total, n = 0.0, 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += bce(logits, y).item() * x.size(0)
            n += x.size(0)
            probs = torch.sigmoid(logits)
            pred = (probs >= threshold).float()
            dices.append(dice_score(pred, y))
            ious.append(iou_score(pred, y))
    return total / max(1, n), float(np.mean(dices)), float(np.mean(ious))

def save_model(model, loader, device, out_dir, threshold=0.5, max_items=16, show=False, open_folder=False):
    import subprocess, sys

    plt.rcParams["figure.dpi"] = 120
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    saved_paths = []

    with torch.no_grad():
        for x, y, names in loader:
            x, y = x.to(device), y.to(device)

            start_time = time.perf_counter()
            logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            probs = torch.sigmoid(logits)
            pred  = (probs >= threshold).float()

            bs = x.size(0)
            for i in range(bs):
                if saved >= max_items:
                    abs_dir = out_dir.resolve()
                    print(f"Saved {saved} panels to: {abs_dir}")
                    if open_folder and sys.platform == "darwin":
                        subprocess.run(["open", str(abs_dir)])
                    return saved_paths

                img_np = np.transpose(x[i].detach().cpu().numpy(), (1, 2, 0))
                img_gray = img_np.mean(axis=2) if img_np.shape[2] == 3 else img_np.squeeze()
                gt_np  = y[i, 0].detach().cpu().numpy()
                pr_np  = pred[i, 0].detach().cpu().numpy()

                inter = np.logical_and(pr_np > 0, gt_np > 0).sum()
                union = np.logical_or(pr_np > 0, gt_np > 0).sum()
                dice = (2 * inter) / (gt_np.sum() + pr_np.sum() + 1e-7)
                iou  = inter / (union + 1e-7)

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                title_main = f"Sample: {names[i]}"
                title_metrics = f"Dice={dice:.3f} | IoU={iou:.3f} | Inference={elapsed_ms:.1f} ms"
                fig.suptitle(f"{title_main}\n{title_metrics}", y=0.97, fontsize=11)

                ax[0].imshow(np.clip(img_gray, 0, 1), cmap="gray", vmin=0, vmax=1)
                ax[0].set_title("Image"); ax[0].axis("off")

                ax[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
                ax[1].set_title("Ground Truth"); ax[1].axis("off")

                ax[2].imshow(pr_np, cmap="gray", vmin=0, vmax=1)
                ax[2].set_title("Prediction"); ax[2].axis("off")

                plt.tight_layout(rect=[0, 0.03, 1, 0.92])

                base = Path(names[i]).stem
                save_path = out_dir / f"vis_{base}.png"
                plt.savefig(save_path, dpi=150, bbox_inches="tight")

                if show:
                    plt.show(block=False)
                    plt.pause(0.001)

                plt.close(fig)
                saved_paths.append(str(save_path))
                saved += 1

    abs_dir = out_dir.resolve()
    print(f"Saved {saved} panels to: {abs_dir}")
    return saved_paths

def make_run_name(args):
    enabled = []
    if not args.no_dense:
        enabled.append("dense")
    if not args.no_gate:
        enabled.append("gate")
    if not args.no_mskip:
        enabled.append("mskip")

    if not enabled:
        return "U_Net_basic"

    if len(enabled) == 3:
        return "lds_full"

    if len(enabled) == 1:
        return f"lds_only_{enabled[0]}"

    return "lds_" + "_".join(enabled)

def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = make_run_name(args)
    default_out = "runs/lds_unet_best.pt"
    default_vis = "runs/vis_lds"
    if args.out == default_out:
        args.out = str(Path("runs") / run_name / "checkpoint.pt")
    if args.save_vis == default_vis:
        args.save_vis = str(Path("runs") / run_name / "vis")

    print(f"[Model] use_dense={not args.no_dense} | use_gate={not args.no_gate} | use_mskip={not args.no_mskip}")
    print(f"[Paths] checkpoint -> {args.out}")
    print(f"[Paths] visuals    -> {args.save_vis}")

    ds = SegDataset(args.images, args.masks, width=args.width, height=args.height)
    if len(ds) == 0:
        raise RuntimeError("No paired images/masks found. Check the --images and --masks paths.")

    n_train = max(1, int(0.8 * len(ds)))
    train_ds = torch.utils.data.Subset(ds, list(range(n_train)))
    val_indices = list(range(n_train, len(ds))) if len(ds) - n_train > 0 else [0]
    val_ds   = torch.utils.data.Subset(ds, val_indices)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    model = LDSUNet(
        in_ch=3,
        out_ch=1,
        base=32,
        growth=16,
        layers_per_block=3,
        use_dense=not args.no_dense,
        use_gate=not args.no_gate,
        use_mskip=not args.no_mskip,
        bn=True,
    ).to(device)

    if args.weights and Path(args.weights).exists():
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")

    if args.eval_only:
        val_bce, val_dice, val_iou = evaluate(model, val_loader, device, args.threshold)
        print(f"[EVAL] BCE={val_bce:.4f} | Dice={val_dice:.4f} | IoU(Jaccard)={val_iou:.4f}")
        if args.save_vis:
            save_model(
                model, val_loader, device, args.save_vis, args.threshold,
                max_items=args.max_vis, show=args.show_vis, open_folder=args.open_folder
            )
        return

    bce_logits = nn.BCEWithLogitsLoss()
    dice_loss  = DiceLoss()
    def loss_fn(logits, y):
        if args.use_dice:
            p = torch.sigmoid(logits)
            return 0.5 * bce_logits(logits, y) + 0.5 * dice_loss(p, y)
        else:
            return bce_logits(logits, y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    best_dice = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        train_loss = total / len(train_loader.dataset)

        val_bce, val_dice, val_iou = evaluate(model, val_loader, device, args.threshold)
        print(f"Epoch {epoch+1}/{args.epochs} | train_BCE={train_loss:.4f} | val_BCE={val_bce:.4f} | val_Dice={val_dice:.4f} | val_IoU={val_iou:.4f}")
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), out_path)
            print(f" Saved BEST (Dice={best_dice:.4f}) -> {out_path}")

    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"Loaded best weights from: {out_path}")
    if args.save_vis:
        save_model(model, val_loader, device, args.save_vis, args.threshold, max_items=args.max_vis)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="data/images", help="Folder with input images")
    p.add_argument("--masks",  type=str, default="data/masks",  help="Folder with corresponding masks")
    p.add_argument("--width", type=int, default=64, help="Resize width")
    p.add_argument("--height", type=int, default=256, help="Resize height")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--use-dice", action="store_true", help="Use 0.5 * BCEWithLogits + 0.5 * dice loss")
    p.add_argument("--out", type=str, default="runs/lds_unet_best.pt", help="Path to save best checkpoint")
    p.add_argument("--save-vis", type=str, default="runs/vis_lds", help="Folder to save panels")
    p.add_argument("--max-vis", type=int, default=16, help="How many panels to save")
    p.add_argument("--weights", type=str, default="", help="Load weights before train/eval")
    p.add_argument("--eval-only", action="store_true", help="Only evaluate with no training")
    p.add_argument("--show-vis", action="store_true", help="Show the figures as they are generated")
    p.add_argument("--open-folder", action="store_true", help="Auto-open the output folder")
    p.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility")
    p.add_argument("--no-dense", action="store_true", help="Disable DenseBlocks")
    p.add_argument("--no-gate",  action="store_true", help="Disable selection gates on skip connections")
    p.add_argument("--no-mskip", "--no-multiscale-skip", dest="no_mskip", action="store_true", help="Disable multi-scale skip concatenations")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
