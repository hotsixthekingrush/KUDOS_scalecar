# YOLO 카메라 인지 파이프라인 — 구현 및 학습 가이드

2026 AUTO RACE 대회, 카메라 인지 파트 담당자용.

---

## 전체 흐름 한눈에

```
[1] Gazebo에서 자동 라벨링으로 데이터 수집   <- 가장 오래 걸림, 제일 먼저 시작
        ↓
[2] 이미지 augmentation으로 데이터 불리기
        ↓
[3] YOLOv8n 전이학습 (개발 PC, RTX 3060)
        ↓
[4] validation mAP 확인 -> 부족하면 [1]로 돌아가 보강
        ↓
[5] 실물 LIMO Pro 받으면 실제 촬영 데이터 추가 (Human-in-the-loop)
        ↓
[6] TensorRT 변환 (반드시 Jetson Orin Nano 위에서)
        ↓
[7] ROS2 추론 노드로 실시간 인식
```

**핵심 원칙**: [1]~[4]는 실물 없이도 지금 당장 할 수 있습니다.
실물을 기다리지 말고 지금 시작하세요.

---

## [0] 설치

```bash
pip install ultralytics opencv-python pyyaml
# ROS2 쪽
sudo apt install ros-humble-cv-bridge ros-humble-gazebo-msgs ros-humble-tf2-ros
sudo apt install ros-humble-message-filters
```

---

## [1] 자동 라벨링으로 데이터 수집

### 1-1. 월드 파일에 플러그인 추가

`autorace_2026.world` 의 `<world>` 태그 안에 아래를 넣으세요.
(물체 위치를 ROS 토픽으로 받기 위해 필요)

```xml
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
  <ros><namespace>/gazebo</namespace></ros>
  <update_rate>30.0</update_rate>
</plugin>
```

### 1-2. 대상 좌표 확인

`auto_labeler.py` 상단의 `TARGETS` 를 실제 월드에 맞게 수정하세요.

- **3D 모델이 있는 것** (표지판, 차단기, 동적차량) → `gazebo_name` 으로 지정.
  이름은 월드 파일의 `<model name="...">` 와 정확히 일치해야 합니다.
- **노면에 그려진 것** (횡단보도) → 3D 모델이 아니므로 `world_pose` 로 직접 좌표 지정.

`size` 는 3D 바운딩박스 크기(m)입니다. 실제 모델 크기와 맞아야
Bounding Box가 정확하게 나옵니다.

### 1-3. 실행

터미널 3개를 띄웁니다.

```bash
# 터미널 1 — Gazebo
gazebo autorace_2026.world

# 터미널 2 — 자동 라벨러
ros2 run <pkg> auto_labeler.py --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p output_dir:=./dataset

# 터미널 3 — 도메인 랜덤화 (물체 위치를 계속 바꿔줌)
ros2 run <pkg> domain_randomizer.py --ros-args -p interval_sec:=2.0
```

그다음 **LIMO Pro를 트랙 위에서 수동 조종**하며 돌아다니세요.
(teleop 또는 이미 만든 lane_tracing_node 로 자율주행시켜도 됩니다)
돌아다니는 동안 이미지와 라벨이 자동으로 쌓입니다.

### 1-4. ★ 반드시 확인할 것

`dataset/debug/` 폴더에 Bounding Box가 그려진 이미지가 저장됩니다.
**몇 장 열어보고 박스가 물체에 정확히 맞는지 눈으로 확인하세요.**

박스가 어긋나 있다면:
- `size` 값이 실제 모델 크기와 다름
- `z_offset` 이 잘못됨 (물체 중심 높이)
- `camera_optical_frame` / `world_frame` 파라미터가 실제 TF 이름과 다름
  → `ros2 run tf2_tools view_frames` 로 실제 프레임 이름 확인

이 확인을 건너뛰고 수천 장을 모으면, 전부 잘못된 라벨이라 학습이 망가집니다.

### 1-5. 목표 수량

- 클래스당 **200~300장**으로 먼저 시작
- 다양성이 수량보다 중요합니다:
  - 거리별 (가까이/멀리 → Bounding Box 크기가 다양하게)
  - 각도별 (정면/비스듬히)
  - 배경별 (트랙 여러 구간이 배경에 섞이도록)

---

## [2] Augmentation으로 데이터 불리기

```bash
# train 데이터
python3 image_augment.py --src dataset --dst dataset_aug --copies 2 --split train

# val 데이터는 변형하지 않는 게 원칙 (성능을 정직하게 측정하기 위해)
# 원본만 복사
mkdir -p dataset_aug/images/val dataset_aug/labels/val
cp dataset/images/val/* dataset_aug/images/val/
cp dataset/labels/val/* dataset_aug/labels/val/
```

### train/val 나누기 주의사항

같은 촬영 세션(연속된 프레임)이 train과 val에 섞이면,
거의 똑같은 이미지가 양쪽에 들어가서 **mAP가 실제보다 부풀려집니다.**

→ 시간대를 나눠서 분리하세요.
예: 앞쪽 80% 프레임 = train, 뒤쪽 20% = val

---

## [3] 학습

```bash
python3 train_yolo.py --mode train --data dataset.yaml --epochs 100 --batch 16
```

### 중요한 설정 두 가지

**`fliplr=0.0` (좌우반전 금지)**
좌회전 마커(A)와 우회전 마커(B)가 좌우반전하면 서로 뒤바뀝니다.
이걸 켜두면 모델이 좌/우를 영원히 구분 못 합니다. **절대 켜지 마세요.**

**`patience=30` (조기 종료)**
에폭 수를 정해놓고 채우는 것보다, validation loss가 개선을 멈추면
자동으로 끝내는 게 효율적입니다. 보통 50~150 에폭 사이에서 수렴합니다.

---

## [4] 검증 — mAP 확인

```bash
python3 train_yolo.py --mode val --weights runs/detect/train/weights/best.pt
```

### 목표 기준

- **mAP@0.5 ≥ 0.90**
  일반적인 기준(0.5~0.7)보다 높게 잡는 이유: 대회에서는 인식 실패가
  곧바로 미션 실패로 직결되기 때문입니다.

### 숫자만 믿지 마세요

클래스별 mAP가 출력되니 **어떤 클래스가 부족한지 확인**하고,
그 클래스 위주로 데이터를 보강하세요.

그리고 **실제 영상으로 눈으로 확인**하는 게 mAP 숫자보다 중요합니다.
특히 미션 판정 순간(횡단보도 정지선 근처, 차단기 열리는 순간)에
Bounding Box가 흔들리거나 놓치는지 직접 재생해서 보세요.

```bash
yolo predict model=runs/detect/train/weights/best.pt source=테스트영상.mp4 show=True
```

---

## [5] 실물 데이터 추가 (LIMO Pro 받은 뒤)

시뮬레이션만으로 학습한 모델은 실제 카메라에서 성능이 떨어질 수 있습니다
(Sim-to-Real Gap). 실물을 받으면:

1. 실제 트랙에서 영상 촬영
2. 지금까지 학습한 모델로 **1차 자동 예측**
   ```bash
   yolo predict model=best.pt source=실제영상/ save_txt=True save_conf=True
   ```
3. 예측 결과를 **사람이 검수·수정**만 함 (Roboflow, CVAT, LabelImg 등)
4. 수정된 데이터를 합쳐서 재학습

이게 Human-in-the-loop 방식입니다. 처음부터 손으로 그리는 것보다
훨씬 빠르고, 이미 대부분 맞게 예측되므로 수정만 하면 됩니다.

---

## [6] TensorRT 변환

```bash
# ★ 반드시 Jetson Orin Nano 위에서 실행 ★
python3 train_yolo.py --mode export --weights best.pt
```

**왜 Jetson에서 해야 하나**: TensorRT 엔진은 그 GPU의 아키텍처에
맞춰 최적화됩니다. 개발 PC(RTX 3060)에서 만든 `.engine` 파일은
Jetson에서 로드조차 안 됩니다.

변환 후 **mAP를 다시 측정**하세요. FP16으로 정밀도를 낮췄을 때
정확도가 얼마나 떨어지는지 확인해야 합니다. 보통 미미하지만,
데이터가 적으면 손실이 예상보다 클 수 있습니다.

---

## [7] 일정 제안

실물 LIMO Pro가 아직 없고 시험기간(10/8~10/22)이 끼어있는 상황 기준.

| 시기 | 할 일 |
|---|---|
| 9월 3주 | 자동 라벨링 파이프라인 구축 + 디버그 이미지로 정확도 확인 |
| 9월 4주 | 데이터 수집 (클래스당 200~300장) + augmentation |
| 10월 1주 | 베이스라인 학습 + mAP 확인 + 부족 클래스 보강 |
| **10/8~10/22** | **시험기간 — 개발 중단** |
| 10월 4주 | 재개, 모델 다듬기 |
| 11월 1~2주 | 실물 받으면 실제 데이터 추가, Human-in-the-loop |
| 11월 3주 | TensorRT 변환 + 실차 검증, **11/16부터 신규 기능 동결** |
| 11/18~20 | 시험주행 + 본선 |

---

## 자주 막히는 지점

**Q. TF 조회 실패 경고가 계속 떠요**
→ `ros2 run tf2_tools view_frames` 로 실제 프레임 이름을 확인하세요.
`camera_optical_frame` 이 `camera_link_optical` 이거나 다른 이름일 수 있습니다.
`world_frame` 도 `odom` 이 아니라 `world` 일 수 있습니다.

**Q. Bounding Box가 물체보다 크거나 작아요**
→ `TARGETS` 의 `size` 값을 실제 모델 크기에 맞게 조정하세요.
Gazebo에서 모델을 클릭하면 크기를 확인할 수 있습니다.

**Q. 라벨 파일이 전부 비어있어요**
→ 물체가 카메라 시야 안에 없거나, `MIN_BOX_PX` 기준보다 작게 보이는 겁니다.
카메라를 물체 쪽으로 향하게 하고 다시 수집하세요.

**Q. mAP가 0.99처럼 비정상적으로 높아요**
→ train/val이 제대로 분리 안 됐을 가능성이 큽니다.
같은 장면이 양쪽에 들어가 있는지 확인하세요.
