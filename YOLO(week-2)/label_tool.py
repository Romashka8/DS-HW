"""
Ручная разметка датасета в формате YOLO.

Использование:
  python label_tool.py --src path/to/images --out path/to/dataset \\
                       --split 0.8/0.1/0.1 \\
                       --names buffalo,elephant,rhino,zebra

На выходе:
  out/
    images/{train,val,test}/...
    labels/{train,val,test}/...
    data.yaml

Управление:
  0..9          выбрать активный класс
  ЛКМ-drag      нарисовать bbox активного класса
  Enter         сохранить картинку + метки и перейти к следующей
  Z / Backspace отменить последний bbox
  N             пропустить картинку (не сохранять)
  P             предыдущая картинка
  Q / Esc       выйти (data.yaml пишется при выходе)

Зависимости: pip install opencv-python numpy
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
          (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 255),
          (255, 128, 0), (200, 200, 200)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv_imread(path: Path):
    # cv2.imread не понимает не-ASCII пути на Windows
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def parse_split(spec: str):
    parts = [float(x) for x in spec.split("/")]
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-3:
        raise SystemExit(f"--split: три числа через '/', сумма=1.0 (было {spec})")
    return parts


def assign_splits(images, ratios, seed):
    rng = random.Random(seed)
    shuf = list(images)
    rng.shuffle(shuf)
    n = len(shuf)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    plan = {}
    for i, p in enumerate(shuf):
        plan[p] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return plan


def fit_to_screen(img, max_w=1400, max_h=860):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return (cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s) if s < 1.0 else (img, 1.0)


class Annotator:
    def __init__(self, names):
        self.names = names
        self.active = 0
        self.boxes = []          # (cls, x1, y1, x2, y2) — координаты в display
        self.start = None
        self.cur = None
        self.dragging = False

    def class_label(self, c):
        return self.names[c] if self.names and c < len(self.names) else f"class_{c}"

    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.start, self.cur, self.dragging = (x, y), (x, y), True
        elif ev == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.cur = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start
            x2, y2 = x, y
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                self.boxes.append((self.active,
                                   min(x1, x2), min(y1, y2),
                                   max(x1, x2), max(y1, y2)))
            self.start = self.cur = None


def render(disp, ann, status):
    canvas = disp.copy()
    for cls, x1, y1, x2, y2 in ann.boxes:
        col = COLORS[cls % len(COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
        cv2.putText(canvas, ann.class_label(cls),
                    (x1, max(15, y1 - 6)), FONT, 0.5, col, 1, cv2.LINE_AA)
    if ann.dragging and ann.start and ann.cur:
        col = COLORS[ann.active % len(COLORS)]
        cv2.rectangle(canvas, ann.start, ann.cur, col, 1, cv2.LINE_AA)
    # активный цвет — квадратик в правом верхнем углу
    col = COLORS[ann.active % len(COLORS)]
    cv2.rectangle(canvas, (canvas.shape[1] - 28, 4),
                  (canvas.shape[1] - 4, 28), col, -1)
    cv2.rectangle(canvas, (canvas.shape[1] - 28, 4),
                  (canvas.shape[1] - 4, 28), (0, 0, 0), 1)
    # статус-строка под изображением
    strip = np.full((28, canvas.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(strip, status, (8, 19), FONT, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([canvas, strip])


def save_sample(img_path, boxes_disp, scale, out_root, split, names_seen):
    img = cv_imread(img_path)
    if img is None:
        print(f"!! не открылось: {img_path}")
        return
    H, W = img.shape[:2]
    lines = []
    for cls, x1, y1, x2, y2 in boxes_disp:
        x1o, y1o, x2o, y2o = (v / scale for v in (x1, y1, x2, y2))
        x1o, x2o = max(0, min(W, x1o)), max(0, min(W, x2o))
        y1o, y2o = max(0, min(H, y1o)), max(0, min(H, y2o))
        bw, bh = abs(x2o - x1o), abs(y2o - y1o)
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1o + x2o) / 2 / W, (y1o + y2o) / 2 / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw/W:.6f} {bh/H:.6f}")
        names_seen.add(cls)
    img_dst = out_root / "images" / split / img_path.name
    lbl_dst = out_root / "labels" / split / (img_path.stem + ".txt")
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, img_dst)
    lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(out_root, names, names_seen):
    if not names:
        max_id = max(names_seen) if names_seen else -1
        names = [f"class_{i}" for i in range(max_id + 1)]
    lines = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        *(f"  {i}: {n}" for i, n in enumerate(names)),
    ]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def already_labeled(out_root, stem):
    for split in ("train", "val", "test"):
        if (out_root / "labels" / split / (stem + ".txt")).exists():
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Manual YOLO labeler")
    ap.add_argument("--src", type=Path, required=True, help="папка с изображениями (рекурсивно)")
    ap.add_argument("--out", type=Path, required=True, help="корень выходного датасета")
    ap.add_argument("--split", default="0.8/0.1/0.1", help="train/val/test, напр. 0.8/0.1/0.1")
    ap.add_argument("--names", default=None, help="имена классов через запятую")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="пропускать уже размеченные")
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",")] if args.names else None
    ratios = parse_split(args.split)
    images = find_images(args.src)
    if not images:
        raise SystemExit(f"Не нашёл изображений в {args.src}")
    splits = assign_splits(images, ratios, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        images = [p for p in images if not already_labeled(args.out, p.stem)]
        print(f"Resume: осталось разметить {len(images)} из {len(splits)}")

    names_seen = set()
    win = "label_tool"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    ann = Annotator(names)
    cv2.setMouseCallback(win, ann.on_mouse)

    i = 0
    history = []  # стэк (img_path, split) для отката P
    try:
        while i < len(images):
            img_path = images[i]
            orig = cv_imread(img_path)
            if orig is None:
                i += 1
                continue
            disp, scale = fit_to_screen(orig)
            ann.boxes, ann.start, ann.cur, ann.dragging = [], None, None, False

            while True:
                msg = (f"[{i+1}/{len(images)}] {img_path.name}  split={splits[img_path]}  "
                       f"class={ann.active}:{ann.class_label(ann.active)}  boxes={len(ann.boxes)}  "
                       f"Enter=save  Z=undo  N=skip  P=prev  Q=quit")
                cv2.imshow(win, render(disp, ann, msg))
                k = cv2.waitKey(20) & 0xFFFF
                if k == 0xFFFF:
                    continue
                if ord('0') <= k <= ord('9'):
                    ann.active = k - ord('0')
                elif k in (13, 10):  # Enter
                    save_sample(img_path, ann.boxes, scale, args.out,
                                splits[img_path], names_seen)
                    history.append(img_path)
                    i += 1
                    break
                elif k in (ord('z'), ord('Z'), 8):  # undo box
                    if ann.boxes:
                        ann.boxes.pop()
                elif k in (ord('n'), ord('N')):
                    i += 1
                    break
                elif k in (ord('p'), ord('P')) and i > 0:
                    i -= 1
                    break
                elif k in (ord('q'), ord('Q'), 27):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    finally:
        write_yaml(args.out, names, names_seen)
        cv2.destroyAllWindows()
        print(f"data.yaml записан: {(args.out / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
