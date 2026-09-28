# YOLO 카메라 인지 파이프라인 — 구현 및 학습 가이드

2026 AUTO RACE 대회, 카메라 인지 파트 담당자용.

---

## 전체 흐름 한눈에

```
[1] Gazebo에서 자동 라벨링으로 데이터 수집   <- 가장 오래 걸림, 제일 먼저 시작
        ↓
[1.5] split_dataset.py 로 train/val 분할 (연속 프레임 묶음 단위)
        ↓
[2] 이미지 augmentation으로 데이터 불리기 (train 만)
        ↓
[3] YOLOv8n 전이학습 (개발 PC, RTX 3060)
        ↓
[4] validation mAP 확인 -> 부족하면 [1]로 돌아가 보강
        ↓
[5] 실물 LIMO Pro 받으면 실제 촬영 데이터 추가 (Human-in-the-loop)
        ↓
[6] TensorRT 변환 (반드시 Jetson Orin Nano 위에서)
        ↓
[7] ROS2 추론 노드(yolo_detector_node.py)로 실시간 인식 -> /perception/mission_objects
```

## v2 변경점 (9/28)

| 문제 | 수정 |
|---|---|
| 카메라 위치를 `odom` TF 로 계산 → 스폰 위치/odom 오차만큼 박스가 전부 어긋남 | Gazebo 로봇 정답 위치(`model_states`) × 로봇→카메라 정적 TF 로 계산 |
| `marker_right` 대상이 없어 우회전 마커 라벨 0장 | 대상 목록을 `config/targets.yaml` 로 분리, 좌/우 마커 별도 모델 |
| 벽·터널 뒤에 가려진 물체에도 라벨 생성 | 깊이 영상으로 가림 검사 (`max_occluded_ratio`) |
| 재실행하면 `frame_000000` 부터 다시 저장 → 기존 데이터 덮어씀 | 파일명에 세션 접두어 (`0928_201530_000001`) |
| train/val 분할 도구 없음 | `split_dataset.py` |
| 검증 시 클래스별 mAP 이름이 밀릴 수 있음 (val 에 없는 클래스가 있을 때) | `ap_class_index` 기준으로 출력 |
| 최신 ultralytics 에서 가중치가 `runs/detect/runs/detect/...` 로 저장됨 | 절대경로 + 실제 저장 경로 출력 |
| 추론 노드 없음 | `yolo_detector_node.py` + `config/yolo_detector.yaml` |

라벨은 이제 **다각형(segmentation 형식)** 으로 저장된다. detect 학습에도 그대로 쓰이고
(ultralytics 가 다각형으로 박스를 계산), 차단기 각도가 필요하면 같은 데이터로 `--task segment` 학습이 된다.

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

### 1-2. 대상 / 로봇 이름 확인

`config/targets.yaml` 을 실제 월드에 맞게 수정하세요. (코드 수정 불필요)

- **3D 모델이 있는 것** (표지판, 차단기, 동적차량) → `gazebo_name` 으로 지정.
  이름은 월드 파일의 `<model name="...">` 와 정확히 일치해야 합니다.
- **노면에 그려진 것** (횡단보도) → `world_pose: [x, y, z, yaw]` 로 직접 지정.
- **좌/우 마커는 별도 모델 2개**(`sign_marker_left`, `sign_marker_right`)로 월드에 넣어야
  합니다. 한 모델에 텍스처만 바꾸면 라벨러가 좌/우를 구분할 수 없습니다.
  `domain_randomizer.py` 가 매번 둘 중 하나만 제자리에 둡니다.

로봇 이름과 루트 링크도 확인하세요:
- `robot_model_name`: `ros2 topic echo /gazebo/model_states --once` 의 name 목록에 있는 로봇 이름.
  틀리면 노드가 가능한 이름 목록을 에러로 출력해 줍니다.
- `robot_root_frame`: 로봇 URDF 의 최상위 링크 (보통 `base_footprint`)
- 카메라 이미지의 `frame_id` 가 `*_optical_frame` 이 아니라 카메라 링크라면
  `camera_frame_is_optical:=false`

### 1-3. 실행

터미널 3개를 띄웁니다.

```bash
# 터미널 1 — Gazebo
gazebo autorace_2026.world

# 터미널 2 — 자동 라벨러
ros2 run kudos_yolo_tools auto_labeler.py --ros-args \
  -p robot_model_name:=limo \
  -p output_dir:=$HOME/yolo_data/dataset
# 깊이 토픽이 없으면 -p depth_topic:='' (가림 검사 끔)

# 터미널 3 — 도메인 랜덤화 (물체 위치를 계속 바꿔줌)
ros2 run kudos_yolo_tools domain_randomizer.py --ros-args -p interval_sec:=2.0
```

그다음 **LIMO Pro를 트랙 위에서 수동 조종**하며 돌아다니세요.
(teleop 또는 이미 만든 lane_tracing_node 로 자율주행시켜도 됩니다)
돌아다니는 동안 이미지와 라벨이 자동으로 쌓입니다.

### 1-4. ★ 반드시 확인할 것

`dataset/debug/` 폴더에 Bounding Box가 그려진 이미지가 저장됩니다.
**몇 장 열어보고 박스가 물체에 정확히 맞는지 눈으로 확인하세요.**

박스가 어긋나 있다면:
- 전부 같은 방향으로 밀려 있음 → `robot_root_frame` / `camera_frame_is_optical` 확인
- 박스가 얇은 막대처럼 나옴 → `size` 의 x, y 를 서로 바꾸기 (판의 얇은 축이 반대)
- 높이만 틀림 → `offset` 의 z
- 프레임 이름 → `ros2 run tf2_tools view_frames`

이 확인을 건너뛰고 수천 장을 모으면, 전부 잘못된 라벨이라 학습이 망가집니다.

### 1-5. 목표 수량

- 클래스당 **200~300장**으로 먼저 시작
- 다양성이 수량보다 중요합니다:
  - 거리별 (가까이/멀리 → Bounding Box 크기가 다양하게)
  - 각도별 (정면/비스듬히)
  - 배경별 (트랙 여러 구간이 배경에 섞이도록)

---

## [1.5] train / val 분할

```bash
python3 split_dataset.py --root ~/yolo_data/dataset --val-ratio 0.2 \
  --names crosswalk,marker_left,marker_right,gate_bar,parking_sign,dynamic_car
```
연속 프레임 40장 묶음 단위로 나눕니다. 출력의 `** val 에 없음` 경고가 뜨는 클래스는
`--block-size` 를 줄이거나 그 클래스 데이터를 더 모으세요.

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

※ 분할([1.5])을 먼저 하고 augmentation 을 해야 합니다. 순서가 반대면
같은 원본의 변형본이 train/val 양쪽에 들어가 mAP 가 부풀려집니다.

---

## [3] 학습

```bash
python3 train_yolo.py --mode train --data dataset.yaml --epochs 100 --batch 16
# 차단기 각도까지 필요하면 (같은 데이터로 가능)
python3 train_yolo.py --mode train --task segment --data dataset.yaml --epochs 100
```
`dataset.yaml` 의 `path:` 는 절대경로로 바꿔두는 게 안전합니다.

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
python3 train_yolo.py --mode val --data dataset.yaml --weights runs/detect/train/weights/best.pt
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

## [6.5] 추론 노드 — 판단팀 연동

```bash
ros2 run kudos_yolo_tools yolo_detector_node.py --ros-args \
  --params-file src/kudos_yolo_tools/config/yolo_detector.yaml \
  -p weights:=/절대/경로/best.pt
rqt_image_view /perception/yolo_debug     # 박스 + 거리 확인
ros2 topic echo /perception/mission_objects
```

- 출력 메시지 `ar_msgs/MissionObjectArray` 는 **아직 ar_msgs 에 없음** →
  `docs/ar_msgs_proposal/README_proposal.md` 를 세진님께 전달해서 추가 요청
- 거리: 횡단보도 = 바닥평면 교점, 표지판류 = RGB-D 깊이 (실패 시 크기 역산)
- 개발 PC 에서는 Gazebo 영상 + `.pt` 로 먼저 돌려보고, Jetson 에서는 `.engine` 으로 교체

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
→ `config/targets.yaml` 의 `size` 값을 실제 모델 크기에 맞게 조정하세요.
Gazebo에서 모델을 클릭하면 크기를 확인할 수 있습니다.

**Q. 라벨 파일이 전부 비어있어요**
→ 물체가 카메라 시야 안에 없거나, `MIN_BOX_PX` 기준보다 작게 보이는 겁니다.
카메라를 물체 쪽으로 향하게 하고 다시 수집하세요.

**Q. mAP가 0.99처럼 비정상적으로 높아요**
→ train/val이 제대로 분리 안 됐을 가능성이 큽니다.
같은 장면이 양쪽에 들어가 있는지 확인하세요.
