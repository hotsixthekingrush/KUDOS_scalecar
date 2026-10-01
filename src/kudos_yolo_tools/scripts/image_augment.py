#!/usr/bin/env python3
"""
image_augment.py
수집이 끝난 데이터셋에 "실제 카메라처럼 보이게 하는" 변형을 입혀
데이터를 불리고 Sim-to-Real Gap을 줄이는 스크립트.

[왜 필요한가]
Gazebo 렌더링 이미지는 지나치게 깨끗하다. 실제 Orbbec DaBai 카메라로
찍으면 센서 노이즈, 조명 얼룩, 약간의 블러, 색감 차이가 생긴다.
이런 변형을 미리 학습시켜두면 실전에서 성능 저하가 줄어든다.

[중요]
밝기/노이즈/블러처럼 "위치가 안 바뀌는" 변형만 적용한다.
회전이나 크롭처럼 위치가 바뀌는 변형은 라벨 좌표도 같이 바꿔야 하므로
여기서는 다루지 않는다 (YOLOv8 학습 시 내장 augmentation이 처리함).

사용법:
    python3 image_augment.py --src dataset --dst dataset_aug --copies 2
"""

import os
import shutil
import random
import argparse

import cv2
import numpy as np


def random_brightness_contrast(img):
    """조명 밝기·대비 변화 — 대회장 조명이 연습 환경과 다를 상황 대비"""
    alpha = random.uniform(0.6, 1.5)     # 대비
    beta = random.uniform(-45, 45)       # 밝기
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def random_color_shift(img):
    """화이트밸런스 차이 재현 — 채널별로 살짝 다른 배율"""
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] *= random.uniform(0.88, 1.12)
    return np.clip(out, 0, 255).astype(np.uint8)


def random_noise(img):
    """센서 노이즈 재현"""
    sigma = random.uniform(2, 12)
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_blur(img):
    """주행 중 흔들림 / 초점 흐림 재현"""
    if random.random() < 0.5:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    # 모션 블러 (주행 방향으로 흐림)
    size = random.choice([3, 5, 7])
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)


def random_shadow(img):
    """바닥 그림자 / 조명 얼룩 재현 — HSV 임계값을 흔드는 주범"""
    h, w = img.shape[:2]
    overlay = img.copy()
    x1, x2 = sorted(random.sample(range(w), 2))
    y1, y2 = sorted(random.sample(range(h), 2))
    factor = random.uniform(0.45, 0.8)
    overlay[y1:y2, x1:x2] = (overlay[y1:y2, x1:x2] * factor).astype(np.uint8)
    return cv2.addWeighted(overlay, 0.6, img, 0.4, 0)


AUGS = [
    (random_brightness_contrast, 0.9),
    (random_color_shift,          0.6),
    (random_noise,                0.7),
    (random_blur,                 0.5),
    (random_shadow,               0.4),
]


def augment(img):
    out = img.copy()
    for fn, p in AUGS:
        if random.random() < p:
            out = fn(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='dataset', help='원본 데이터셋 폴더')
    ap.add_argument('--dst', default='dataset_aug', help='출력 폴더')
    ap.add_argument('--copies', type=int, default=2, help='원본 1장당 만들 변형본 수')
    ap.add_argument('--split', default='train', help='train / val')
    args = ap.parse_args()

    src_img = os.path.join(args.src, 'images', args.split)
    src_lbl = os.path.join(args.src, 'labels', args.split)
    dst_img = os.path.join(args.dst, 'images', args.split)
    dst_lbl = os.path.join(args.dst, 'labels', args.split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    files = sorted(f for f in os.listdir(src_img) if f.lower().endswith(('.jpg', '.png')))
    if not files:
        print(f'[오류] {src_img} 에 이미지가 없습니다.')
        return

    total = 0
    for fn in files:
        stem = os.path.splitext(fn)[0]
        img = cv2.imread(os.path.join(src_img, fn))
        lbl_path = os.path.join(src_lbl, stem + '.txt')
        if img is None or not os.path.exists(lbl_path):
            continue

        # 원본도 함께 복사
        cv2.imwrite(os.path.join(dst_img, fn), img)
        shutil.copy(lbl_path, os.path.join(dst_lbl, stem + '.txt'))
        total += 1

        # 변형본 생성 — 위치가 안 바뀌므로 라벨은 그대로 재사용
        for i in range(args.copies):
            aug = augment(img)
            new_stem = f'{stem}_aug{i}'
            cv2.imwrite(os.path.join(dst_img, new_stem + '.jpg'), aug)
            shutil.copy(lbl_path, os.path.join(dst_lbl, new_stem + '.txt'))
            total += 1

    print(f'완료: {len(files)}장 -> {total}장 (원본 포함)')
    print(f'출력 경로: {args.dst}')


if __name__ == '__main__':
    main()
