# kudos_perception — 카메라 인지 노드

2026 AUTO RACE / 팀 KUDOS / 카메라 인지 담당

---

## 변경 이력
- 메시지(LaneError, ColorZone)를 공용 패키지 `ar_msgs` 로 이관. 이 패키지에는 msg 없음.
- 이미지 구독 QoS 를 `qos_profile_sensor_data` 로 변경 (best-effort 발행 대응).
- 종료 시 `rclpy.ok()` 확인 후 shutdown.
- 대회용 `camera_perception_race.yaml` 추가 (publish_debug=false).
- 부호 규약("좌회전 필요 = 양수") 판단팀 채택 확정.

## 책임 경계 (중요)

```
[camera_perception_node]  ← 이 패키지
    /perception/lane_error
    /perception/color_zone
          ↓
[mission_supervisor_node]  ← 판단팀 (SMACH FSM + 상위제어)
    /cmd_vel
          ↓
[limo_base] → STM32
```

**이 노드는 `/cmd_vel` 을 발행하지 않는다.** 주행 명령은 판단 모듈이 단독 발행한다.

**차선 유실 시 대응도 이 노드의 책임이 아니다.** `valid=false` 로 알리기만 하고,
1초 미만 유실 시 직전 명령 유지 / 벽 감지 시 wall_following / 장시간 유실 시 정차
같은 단계적 fail-safe 는 lane_following State 내부 로직이 판단한다.

---

## 인터페이스 계약

### `/perception/lane_error` (ar_msgs/LaneError)

| 필드 | 단위 | 의미 |
|---|---|---|
| `valid` | - | false 면 나머지 값 전부 무의미 |
| `lateral_offset_m` | m | 로봇에서 본 차선 중앙의 횡방향 위치 |
| `heading_error_rad` | rad | 차선이 로봇 전방축과 이루는 각도 |
| `curvature` | 1/m | 전방 주시 지점의 곡률 (부호 있음) |
| `confidence` | 0~1 | 검출 신뢰도 |
| `single_lane_estimated` | - | 한쪽 차선만 보여 lane_width 로 중앙을 추정한 경우 true |

**부호 규약 — 셋 다 "좌회전 필요 = 양수"로 통일했다.**

ROS 표준(x 전방, y 좌측)을 따르므로, 제어 쪽에서 아래처럼 그대로 쓸 수 있다.

```python
angular_z = Kp * lateral_offset_m + Kh * heading_error_rad
```

세 값의 부호가 일치하므로 서로 상충하지 않는다.
곡률도 같은 규약이라 `heading_error` 와 부호를 비교해 이상치 검출에 쓸 수 있다.

> **유도 근거**: BEV 에서 전방은 y 감소 방향이므로 `d²x/ds² = +2a`.
> `a>0` 은 우커브를 뜻하는데 heading 규약은 좌가 양수이므로
> 곡률은 `κ = -(2a)/(1+(dx/dy)²)^1.5` 로 부호를 뒤집었다.
> (이 부분은 처음 구현 때 부호가 반대로 나가던 것을 검증 중 잡아낸 것이므로,
>  혹시 제어 쪽에서 곡률 부호가 이상하면 이 지점을 먼저 확인할 것)

### `/perception/color_zone` (ar_msgs/ColorZone)

| 필드 | 의미 |
|---|---|
| `zone` | 0=NONE, 1=RED, 2=BLUE |
| `red_ratio`, `blue_ratio` | ROI 내 색상 픽셀 비율 (디버깅용) |

**Hysteresis 와 연속 프레임 확인을 인지 단계에서 이미 적용한 "안정된 판정"이다.**
판단 모듈은 이 값을 그대로 믿고 `acc_dcc` State 전이에 쓰면 된다.
(라이다팀이 `radius` 에 마진을 포함해 보내기로 한 것과 같은 원칙 — 보정은 정보가
많은 인지 쪽에서 하고, 판단은 받은 값을 신뢰한다)

---

## 판단팀에 확인 요청한 사항

**차선 기준 관련 (팀원 질문에 대한 답)**

이 구현은 Sliding Window 가 히스토그램 피크로 좌/우 선을 찾으므로,
**항상 "로봇에서 가장 가까운 좌우 선" = 현재 속한 차선**을 기준으로 한다.
차선을 옮기면 자연히 새 차선 기준으로 오차가 계산되므로, 판단팀 가정대로 동작한다.

**단, 미션 4(차선변경) 수행 중에는 주의가 필요하다.** 차선을 넘는 도중에는
로봇이 선을 밟고 있어 히스토그램이 엉뚱한 쌍을 잡을 수 있다.
`left_right` State 진행 중에는 `lane_error` 를 신뢰하지 말고,
마커까지의 거리 기반 로직으로 조향하는 것을 권장한다.
(`confidence` 가 낮아지거나 `single_lane_estimated=true` 가 뜨는 것으로 감지 가능)

---

## 빌드 및 실행

```bash
# 워크스페이스 src 에 배치
cd ~/26contest/src
# (이 패키지 폴더를 여기에 둔다)

cd ~/26contest
# ar_msgs 를 먼저 빌드해야 한다 (메시지 정의가 거기 있음)
colcon build --packages-select ar_msgs kudos_perception --symlink-install
source install/setup.bash

ros2 run kudos_perception camera_perception_node.py \
  --ros-args --params-file src/kudos_perception/config/camera_perception.yaml
```

대회 당일에는 디버그 영상 발행을 끈 설정을 쓴다 (매 프레임 대역폭 절약):
```bash
  --ros-args --params-file src/kudos_perception/config/camera_perception_race.yaml
```

### 디버그 영상 확인
```bash
ros2 run rqt_image_view rqt_image_view /perception/lane_debug
```

### 출력 확인
```bash
ros2 topic echo /perception/lane_error
ros2 topic echo /perception/color_zone
```

---

## 튜닝 순서 (실물 받은 뒤)

`config/camera_perception.yaml` 의 `[튜닝필요]` 항목을 이 순서로 잡는다.

**1단계 — IPM 사다리꼴 (`ipm_src`)**
가장 먼저, 가장 중요하다. 이게 틀리면 이후 모든 수치가 틀어진다.
노면에 **크기를 아는 직사각형**(예: 40cm × 40cm 테이프)을 붙이고,
`rqt_image_view` 로 그 네 꼭짓점의 픽셀 좌표를 읽어 넣는다.

**2단계 — BEV 실제 크기 (`bev_width_m`, `bev_length_m`)**
1단계에서 쓴 직사각형의 실제 치수를 그대로 넣는다.
이 두 값이 px→m 환산의 유일한 근거다.

**검증**: 로봇을 차선 중앙에서 정확히 10cm 옆으로 옮겨놓고
`lateral_offset_m` 이 0.10 근처로 나오는지 확인한다. 안 맞으면 1~2단계를 다시.

**3단계 — 차선 HSV (`lane_hsv_lower/upper`)**
대회장 조명에서 재측정한다. 디버그 영상에서 차선만 깨끗이 흰색으로
나오는지 보며 조정.

**4단계 — 차선 폭 (`lane_width_m`)**
실측값으로 교체. 한쪽 선만 보일 때의 중앙 추정 정확도를 좌우한다.

**5단계 — 색상 구역 문턱값**
레드존/블루존 위에 로봇을 올려놓고 `red_ratio`/`blue_ratio` 를 echo 로 보며
`zone_enter_ratio`(진입) / `zone_exit_ratio`(이탈) 를 잡는다.
반드시 **진입 > 이탈** 이어야 Hysteresis 가 작동한다.

---

## 검증 완료 항목

구현 중 아래 시나리오를 수치로 검증했다.

| 시나리오 | 결과 |
|---|---|
| 차선 중앙 정렬 | `lateral_offset ≈ 0` |
| 차선이 왼쪽 5cm | `lateral_offset = +0.050 m` |
| 한쪽 선만 검출 | lane_width 로 중앙 추정, `single_lane_estimated=true` |
| 차선 완전 소실 | `valid=false` |
| 좌커브 | `heading > 0`, `curvature > 0` |
| 우커브 | `heading < 0`, `curvature < 0` |
| 곡률 절대값 | 반지름 1m 원 → `κ = 1.000` (이론값 일치) |

---

## 아직 안 된 것

- **YOLO 노드는 별도**(`camera_yolo`)로 분리 예정. 이 노드는 HSV 계열만 담당.
- 차단기 각도 추정을 OBB 로 갈지 segmentation 으로 갈지 팀 논의 중.
  결정되면 자동 라벨러(`auto_labeler.py`)도 그에 맞춰 수정 필요.
