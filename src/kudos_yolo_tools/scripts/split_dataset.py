#!/usr/bin/env python3
"""
split_dataset.py
images/all, labels/all 에 모인 데이터를 train / val 로 나눈다.

[왜 무작위로 섞으면 안 되나]
연속 프레임은 거의 똑같다. 한 장은 train, 바로 다음 장은 val 에 들어가면
val 이 "외운 문제"가 되어 mAP 가 실제보다 크게 부풀려진다.
→ 연속된 프레임 묶음(block) 단위로 통째로 train 또는 val 에 보낸다.

파일명 규칙: <세션>_<번호>.jpg  (auto_labeler.py 가 이렇게 저장함)
  - 같은 세션 안에서 번호순으로 block_size 장씩 묶는다.
  - 규칙에 안 맞는 파일(직접 찍은 사진 등)은 파일명 순서로 'misc' 세션에 묶인다.

사용법:
  python3 split_dataset.py --root ~/yolo_data/dataset --val-ratio 0.2
  # 결과: images/train, images/val, labels/train, labels/val (원본 all/ 은 그대로 둠)

  # 실사진 폴더를 따로 나눌 때
  python3 split_dataset.py --root ~/yolo_data/real --block-size 10
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
STEM_RE = re.compile(r'^(.*)_(\d+)$')


def read_classes(label_path):
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if parts:
                out.append(int(float(parts[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='images/all, labels/all 이 있는 폴더')
    ap.add_argument('--src', default='all', help='원본 하위폴더 이름')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--block-size', type=int, default=40,
                    help='한 묶음 장수. save_every_n=3, 30fps 기준 40장 ≈ 4초 주행')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--move', action='store_true', help='복사 대신 이동')
    ap.add_argument('--names', default='', help='클래스 이름 쉼표구분 (통계 출력용)')
    args = ap.parse_args()

    img_dir = os.path.join(args.root, 'images', args.src)
    lbl_dir = os.path.join(args.root, 'labels', args.src)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    if not files:
        print('[오류] 이미지 없음: %s' % img_dir)
        return

    # 세션별로 묶고 번호순 정렬
    sessions = defaultdict(list)
    for fn in files:
        stem = os.path.splitext(fn)[0]
        m = STEM_RE.match(stem)
        if m:
            sessions[m.group(1)].append((int(m.group(2)), fn))
        else:
            sessions['misc'].append((len(sessions['misc']), fn))

    blocks = []
    for sess in sorted(sessions):
        items = [fn for _, fn in sorted(sessions[sess])]
        for i in range(0, len(items), args.block_size):
            blocks.append(items[i:i + args.block_size])

    rng = random.Random(args.seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    target_val = int(round(len(files) * args.val_ratio))
    val_set, n_val = set(), 0
    for bi in order:
        if n_val >= target_val:
            break
        val_set.add(bi)
        n_val += len(blocks[bi])

    names = [s for s in args.names.split(',') if s] if args.names else []
    stats = {'train': defaultdict(int), 'val': defaultdict(int)}
    n_img = {'train': 0, 'val': 0}
    op = shutil.move if args.move else shutil.copy2

    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.root, 'labels', split), exist_ok=True)

    for bi, block in enumerate(blocks):
        split = 'val' if bi in val_set else 'train'
        for fn in block:
            stem = os.path.splitext(fn)[0]
            src_lbl = os.path.join(lbl_dir, stem + '.txt')
            op(os.path.join(img_dir, fn), os.path.join(args.root, 'images', split, fn))
            dst_lbl = os.path.join(args.root, 'labels', split, stem + '.txt')
            if os.path.exists(src_lbl):
                for c in read_classes(src_lbl):
                    stats[split][c] += 1
                op(src_lbl, dst_lbl)
            else:
                open(dst_lbl, 'w').close()   # 라벨 없음 = 배경(negative) 이미지
            n_img[split] += 1

    print('블록 %d개 (세션 %d개) -> train %d장 / val %d장'
          % (len(blocks), len(sessions), n_img['train'], n_img['val']))
    all_cls = sorted(set(stats['train']) | set(stats['val']))
    print('\n%-16s %8s %8s' % ('class', 'train', 'val'))
    for c in all_cls:
        nm = names[c] if c < len(names) else str(c)
        warn = '   ** val 에 없음 — block-size 를 줄이거나 데이터 추가' if stats['val'][c] == 0 else ''
        print('%-16s %8d %8d%s' % (nm, stats['train'][c], stats['val'][c], warn))


if __name__ == '__main__':
    main()
