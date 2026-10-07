# 메모--
추가해야하는 패키지?같은거 막 물어봤는데 계속 없다고 해서 잘 모르겠어 일단 굳이 하자면

라인트레이싱 코드 자체가 요구하는 추가 패키지는 이 3개입니다.
```
python3-opencv
python3-numpy
ros-humble-cv-bridge
```

설치 명령은:
```
sudo apt install -y python3-opencv python3-numpy ros-humble-cv-bridge
```
다만 기존 통합본에서 이미 NumPy/OpenCV 등을 쓰고 있다면 그것들은 “새로 추가된 의존성”은 아닙니다. 기존 코드와 비교하기 전에는 세 개 중 무엇이 진짜 신규인지는 확정할 수 없습니다.



# line_tracing_pkg

KUDOS 스케일카 자율주행 - 카메라 기반 **차선 추종(Line Tracing)** 노드.
예선 계획보고서 3-1(카메라 인지) / 3-3(제어) 설계를 구현한 독립 ROS2 패키지입니다.

나중에 조원(`hotsixthekingrush/KUDOS_scalecar`)의 가제보 환경과 합칠 때는
**코드를 고칠 필요 없이 `config/line_tracing_params.yaml`만 수정**하면 됩니다.

## 파이프라인

```
카메라 이미지 (/camera/image_raw)
  -> ROI + IPM (Bird's-eye View 변환)
  -> HSV 변환 -> 노란색 실선 이진화 (cv2.inRange)
  -> Morphology Opening (Salt-and-pepper 노이즈 제거)
  -> Sliding Window Search -> 2차 곡선 피팅
  -> 횡방향 오차(e) + 헤딩 오차 계산
  -> PD 제어 -> 조향각 -> omega = v*tan(delta)/L
  -> /cmd_vel (geometry_msgs/Twist) 발행
```

가장자리 실선 1개를 기준으로 `lane_offset_m` 만큼 떨어진 지점을 추종 목표로
삼습니다 (보고서의 "가장자리 노란색 실선" 검출 방식과 동일).

## 빌드 & 실행

```bash
# 워크스페이스 src/ 아래에 이 폴더를 복사한 뒤
cd ~/ros2_ws
colcon build --packages-select line_tracing_pkg
source install/setup.bash

ros2 launch line_tracing_pkg line_tracing.launch.py
# 다른 파라미터 파일 사용 시:
ros2 launch line_tracing_pkg line_tracing.launch.py config_file:=/path/to/other_params.yaml
```

## 조원 가제보 환경과 합칠 때 체크할 것

`config/line_tracing_params.yaml` 에서 아래 값들을 실제 환경에 맞게 바꾸세요.

| 파라미터 | 설명 |
|---|---|
| `camera_topic` | 조원 URDF/world에 정의된 실제 카메라 토픽명 |
| `cmd_vel_topic` | 판단 모듈(FSM 노드)이 최종 발행하는 토픽과 겹치지 않는지 확인 (연동 시 이 노드가 아니라 판단 모듈이 `/cmd_vel`을 최종 발행하도록 리팩터링 필요할 수 있음) |
| `image_width` / `image_height` | 카메라 센서 해상도 |
| `ipm_src_points` | 카메라 장착 각도/위치에 따른 ROI 사다리꼴 4점 (재보정 필수) |
| `xm_per_pix` / `ym_per_pix` | 버드아이뷰 픽셀당 실거리 (체커보드 등으로 캘리브레이션) |
| `hsv_yellow_lower/upper` | 가제보 조명 vs 실제 트랙 조명에 따라 재조정 |
| `wheelbase_m` | LIMO Pro 축거 실측값으로 교체 |
| `lane_side` | 추종할 실선이 차량 기준 좌/우 어느 쪽인지 |

## 판단(FSM) 모듈과의 연동 지점

- 이 노드는 `/line_tracing/lane_detected` (`std_msgs/Bool`) 를 발행합니다.
  보고서의 Supervisor/SMACH 판단 모듈이 이 토픽을 구독해서 lane_following
  state의 실패 판단이나 터널 fail-safe 전이 조건에 활용할 수 있습니다.
- 현재 포함된 fail-safe는 **차선 미검출 지속시간에 따른 단기 대응**
  (직전 명령 유지 -> 저속 직진 -> 정지)까지만 구현했습니다.
  LiDAR 기반 `wall_following` 전환은 판단 모듈(FSM) 영역이므로 포함하지
  않았습니다 — 통합 시 판단 노드에서 `lane_detected=false` 지속시간을 보고
  wall_following state로 전이시키는 로직을 추가하면 됩니다.
- 통합 아키텍처(보고서 기준: Supervisor+Planner가 하나의 노드)로 합칠 때는
  `LaneDetector`, `PDController` 클래스를 그대로 가져가서 판단 노드 내부의
  `lane_following` State 안에서 호출하는 방식으로 리팩터링하면 자연스럽게
  들어맞습니다.

## 디버깅

- `/line_tracing/debug_image` 토픽으로 Sliding Window + 피팅 곡선이
  그려진 이미지를 rqt_image_view 등으로 확인할 수 있습니다.
- 곡선이 엉뚱하게 잡히면 대부분 `ipm_src_points`(ROI) 또는
  `hsv_yellow_lower/upper` 문제입니다. `publish_debug_image: true` 상태에서
  버드아이뷰/이진화 중간 결과를 먼저 확인하세요.
