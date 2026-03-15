from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ai_image_detector.ensemble import EnsembleDetector, load_models
from ai_image_detector.model import build_model


def main():
    ap = argparse.ArgumentParser(description="Distill ensemble teacher into a compact student")
    ap.add_argument("--data", default="./data_best")
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--out", default="./artifacts_distill")
    ap.add_argument("--student-backbone", choices=["tiny", "effb0"], default="tiny")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.6, help="teacher loss weight")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_models(args.teacher, device)
    teacher = EnsembleDetector(loaded.models).to(device)
    teacher.eval()

    student = build_model(backbone=args.student_backbone, pretrained_backbone=True).to(device)

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(Path(args.data) / "train", transform=tf)
    val_ds = datasets.ImageFolder(Path(args.data) / "val", transform=tf)
    ai_idx = int(train_ds.class_to_idx["ai"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    opt = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        student.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            target = (y == ai_idx).float()
            with torch.no_grad():
                t_logit = teacher(x)

            s_logit = student(x)
            hard_loss = F.binary_cross_entropy_with_logits(s_logit, target)
            soft_loss = F.mse_loss(torch.sigmoid(s_logit), torch.sigmoid(t_logit))
            loss = args.alpha * soft_loss + (1.0 - args.alpha) * hard_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        student.eval()
        corr = 0
        tot = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                target = (y == ai_idx).long()
                pred = (torch.sigmoid(student(x)) >= 0.5).long()
                corr += (pred == target).sum().item()
                tot += target.numel()
        acc = corr / max(tot, 1)
        print(f"epoch={epoch} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "img_size": args.img_size,
                    "threshold": 0.5,
                    "temperature": 1.0,
                    "model_id": f"distilled-{args.student_backbone}",
                    "backbone": args.student_backbone,
                },
                out / "best.pt",
            )

    print(f"saved={out / 'best.pt'} best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
