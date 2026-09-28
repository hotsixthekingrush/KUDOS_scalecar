#!/usr/bin/env python3
"""
train_yolo.py
YOLOv8 전이학습 + 검증 + TensorRT 변환까지 한 번에 처리하는 스크립트.

[전체 흐름]
  1) COCO 사전학습 YOLOv8n 가중치에서 시작 (전이학습)
  2) 우리 커스텀 데이터셋으로 fine-tuning
  3) validation mAP 확인 -> 목표 미달이면 데이터 보강 후 재학습
  4) 목표 달성하면 TensorRT(.engine)로 변환해 Jetson에 배포

[왜 YOLOv8n(nano)인가]
Jetson Orin Nano는 CUDA 코어가 1024개로 임베디드급이다.
s/m/l 같은 큰 모델은 정확도가 조금 오르는 대신 추론이 느려져
여러 미션을 동시에 처리할 때 병목이 된다. 고정 트랙·고정 카메라라는
좁은 조건에서는 n 모델로도 충분한 성능이 나온다.

사용법:
  # 1단계: 베이스라인 학습
  python3 train_yolo.py --mode train --data dataset.yaml --epochs 100

  # 2단계: 검증 (mAP 확인)
  python3 train_yolo.py --mode val --weights runs/detect/train/weights/best.pt

  # 3단계: TensorRT 변환 (Jetson에서 실행할 것)
  python3 train_yolo.py --mode export --weights runs/detect/train/weights/best.pt
"""

import argparse
import os


# 목표 성능 — 이 수치를 넘으면 데이터 수집을 멈춰도 된다는 기준
TARGET_MAP50 = 0.90      # mAP@0.5 90% 이상
TARGET_MAP50_95 = 0.65   # 더 엄격한 기준 (참고용)


def do_train(args):
    from ultralytics import YOLO

    # 전이학습: COCO 사전학습 가중치에서 시작
    #   detect  : yolov8n.pt      — 박스만. 가장 빠름
    #   segment : yolov8n-seg.pt  — 박스 + 마스크. 차단기 기울기 각도를 마스크로 계산할 때
    # auto_labeler 는 다각형 라벨을 저장하므로 두 방식 모두 같은 데이터로 학습 가능.
    base = args.model or ('yolov8n-seg.pt' if args.task == 'segment' else 'yolov8n.pt')
    model = YOLO(base)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        # --- 조기 종료: val loss가 30에폭 동안 개선 없으면 중단 ---
        # 정해진 에폭 수를 채우는 것보다, 수렴하면 멈추는 게 효율적
        patience=30,

        # --- YOLOv8 내장 augmentation ---
        # 커스텀 데이터가 적을 때 이걸 끄면 오히려 손해다. 켜둘 것.
        hsv_h=0.015,      # 색상 변화 (차선/표지판 색이 핵심이므로 작게)
        hsv_s=0.7,        # 채도
        hsv_v=0.4,        # 명도 — 조명 변화 대응
        degrees=5.0,      # 회전 (카메라가 거의 고정이므로 작게)
        translate=0.1,
        scale=0.4,        # 거리 변화 대응 (가까이/멀리)
        fliplr=0.0,       # ★ 좌우반전 금지! 좌회전/우회전 마커가 뒤바뀜
        mosaic=1.0,       # 여러 이미지를 합성 — 데이터 적을 때 효과 큼
        close_mosaic=10,  # 마지막 10에폭은 mosaic 끄고 실제 분포로 마무리

        # 절대경로로 지정 — 최신 ultralytics 는 상대경로를 runs/<task>/ 아래에
        # 한 번 더 중첩시켜서(runs/detect/runs/detect/...) 가중치 위치가 헷갈린다.
        project=os.path.abspath(os.path.join('runs', args.task)),
        name=args.name,
        exist_ok=True,
    )
    save_dir = getattr(getattr(model, 'trainer', None), 'save_dir', None)
    print('\n학습 완료. 가중치: %s' % os.path.join(str(save_dir), 'weights', 'best.pt'))
    return results


def do_val(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    map50 = metrics.box.map50
    map5095 = metrics.box.map

    print('\n' + '=' * 55)
    print(f'  mAP@0.5      : {map50:.4f}   (목표 {TARGET_MAP50})')
    print(f'  mAP@0.5:0.95 : {map5095:.4f}   (참고 {TARGET_MAP50_95})')
    print('=' * 55)

    # 클래스별 성능 — 어떤 클래스가 부족한지 알아야 데이터를 어디에 더 쓸지 판단됨
    # ※ ap50 배열은 "val 에 등장한 클래스"만 담고 있다. 순서는 ap_class_index 를 따른다.
    #   (enumerate 로 번호를 매기면 val 에 없는 클래스가 있을 때 이름이 밀린다)
    try:
        names = model.names
        seen = set()
        print('\n[클래스별 mAP@0.5]  ← 낮은 클래스 위주로 데이터를 보강하세요')
        for ci, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            ci = int(ci)
            seen.add(ci)
            mark = '  ' if ap >= TARGET_MAP50 else '  ** 부족'
            print(f'  {names[ci]:<16} {ap:.4f}{mark}')
        missing = sorted(set(names) - seen)
        for ci in missing:
            print(f'  {names[ci]:<16}   —      ** val 에 이 클래스가 없음 (측정 불가)')
        if missing:
            print('\n※ val 에 없는 클래스는 전체 mAP 에 반영되지 않았습니다. 그 클래스는 아직 검증 안 된 상태.')
    except Exception as e:
        print('클래스별 출력 실패:', e)

    if map50 >= TARGET_MAP50:
        print('\n목표 달성. TensorRT 변환 단계로 넘어가도 됩니다.')
    else:
        print('\n목표 미달. 위에서 "부족" 표시된 클래스의 데이터를 추가 수집하세요.')
        print('  - Gazebo에서 해당 객체가 잘 보이는 각도/거리로 더 수집')
        print('  - 또는 image_augment.py 로 변형본을 늘리기')


def do_export(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)

    # TensorRT 엔진은 "그 GPU에서 직접 빌드"해야 한다.
    # 즉 이 명령은 반드시 Jetson Orin Nano 위에서 실행할 것.
    # 개발 PC(RTX 3060)에서 만든 엔진은 Jetson에서 작동하지 않는다.
    model.export(
        format='engine',
        imgsz=args.imgsz,
        half=True,        # FP16 — 정확도 손실은 미미하고 속도는 크게 향상
        device=0,
        simplify=True,
    )
    print('\nTensorRT 엔진 생성 완료 (.engine 파일)')
    print('주의: 이 파일은 생성한 GPU에서만 동작합니다.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['train', 'val', 'export'])
    ap.add_argument('--data', default='dataset.yaml')
    ap.add_argument('--weights', default='runs/detect/train/weights/best.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0', help='GPU 번호. CPU면 cpu')
    ap.add_argument('--task', default='detect', choices=['detect', 'segment'])
    ap.add_argument('--model', default='', help='시작 가중치 직접 지정 (기본: yolov8n / yolov8n-seg)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--name', default='train', help='runs/<task>/<name> 에 저장')
    args = ap.parse_args()

    if args.mode == 'train':
        do_train(args)
    elif args.mode == 'val':
        do_val(args)
    else:
        do_export(args)


if __name__ == '__main__':
    main()
