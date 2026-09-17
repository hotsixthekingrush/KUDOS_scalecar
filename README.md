[README.md](https://github.com/user-attachments/files/32337674/README.md)
# KUDOS Scale Car

ROS 2 Humble + Gazebo Classic 기반의 **LIMO Pro 자율주행 스케일카 시뮬레이션 프로젝트**입니다.

실제 스케일카 자율주행 경진대회 환경을 Gazebo에서 재현하고, 대회 미션을 반복적으로 테스트할 수 있도록 제작했습니다.  
경기장 생성 코드는 Python으로 구성되어 있으며, 라바콘 위치와 동적장애물 속도 등 주요 파라미터를 비교적 쉽게 수정할 수 있습니다.

---

## 개발 환경

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic
- Python 3
- AgileX LIMO Pro

---

## 주요 구성

```text
KUDOS_scalecar/
└── src/
    ├── limo_base/
    ├── limo_car/
    ├── limo_description/
    ├── limo_msgs/
    └── webot_arena/
        ├── launch/
        ├── media/
        │   └── materials/
        ├── meshes/
        │   └── traffic_cone.stl
        ├── scripts/
        │   ├── make_arena.py
        │   └── mission_motion_controller.py
        └── worlds/
            └── limo_competition.world
```

### 핵심 파일

| 파일 | 역할 |
|---|---|
| `src/webot_arena/scripts/make_arena.py` | 경기장 geometry 및 미션 오브젝트 생성 |
| `src/webot_arena/scripts/mission_motion_controller.py` | 회전교차로 동적장애물과 차단기 움직임 제어 |
| `src/webot_arena/worlds/limo_competition.world` | `make_arena.py`로 생성되는 Gazebo world |
| `src/limo_car/launch/ackermann_gazebo.launch.py` | LIMO + 경기장 Gazebo 실행 |
| `src/webot_arena/meshes/traffic_cone.stl` | 라바콘 mesh |
| `src/webot_arena/media/materials/` | 갈림길/주차 표지판 texture 및 material |

---

## 구현된 경기장 미션

- 출발 구역
- 빨간색 / 파란색 속도 변경 구역
- 횡단보도
- 라바콘이 배치된 곡선 구간
- 갈림길 및 방향 표지판
- 회전교차로
- 회전교차로 동적장애물 2대
- 터널
- 자동 개폐 차단기
- 주차 구역 및 주차 표지판

---

# 설치

## 1. Repository clone

```bash
cd ~/Workspace

git clone https://github.com/hotsixthekingrush/KUDOS_scalecar.git

cd KUDOS_scalecar
```

## 2. ROS 2 환경 불러오기

```bash
source /opt/ros/humble/setup.bash
```

## 3. 빌드

```bash
colcon build \
  --symlink-install \
  --packages-select \
  limo_msgs \
  limo_description \
  limo_car \
  webot_arena
```

빌드 후:

```bash
source install/setup.bash
```

> 환경에 따라 추가 ROS/Gazebo dependency가 필요할 수 있습니다.

---

# 실행

```bash
cd ~/Workspace/KUDOS_scalecar

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch limo_car ackermann_gazebo.launch.py
```

---

# 경기장 수정

경기장 geometry를 변경할 때는 다음 파일을 수정합니다.

```text
src/webot_arena/scripts/make_arena.py
```

수정 후 world를 다시 생성합니다.

```bash
cd ~/Workspace/KUDOS_scalecar/src/webot_arena/scripts

python3 -m py_compile make_arena.py
python3 make_arena.py
```

Gazebo가 이미 실행 중이었다면 종료 후 다시 실행해야 수정된 world가 반영됩니다.

```bash
pkill -9 -f gzserver
pkill -9 -f gzclient
```

그다음:

```bash
cd ~/Workspace/KUDOS_scalecar

source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch limo_car ackermann_gazebo.launch.py
```

---

# 라바콘 위치 수정

`make_arena.py` 상단의 `CONE_OFFSETS`를 수정합니다.

예:

```python
CONE_OFFSETS = {
    "cone_01": {"dx": 0.00, "dy": 0.00},
    "cone_02": {"dx": 0.00, "dy": 0.00},
    "cone_03": {"dx": 0.00, "dy": 0.00},
}
```

`dx`, `dy`의 단위는 **m**입니다.

```text
dx = +0.10  → +X 방향 10 cm
dx = -0.10  → -X 방향 10 cm

dy = +0.10  → +Y 방향 10 cm
dy = -0.10  → -Y 방향 10 cm
```

Gazebo 전역 좌표축 기준:

```text
+X : 빨간색 축 화살표 방향
+Y : 초록색 축 화살표 방향
+Z : 위쪽
```

예를 들어:

```python
"cone_02": {"dx": 0.20, "dy": -0.10},
```

이면 `cone_02`는 **기본 위치 기준**으로:

- +X 방향 20 cm
- -Y 방향 10 cm

이동합니다.

> `dx`, `dy`는 현재 위치에 계속 누적되는 값이 아닙니다.  
> 항상 `기본 계산 위치 + offset` 방식으로 적용됩니다.

라바콘 위치를 수정한 뒤에는 반드시:

```bash
cd ~/Workspace/KUDOS_scalecar/src/webot_arena/scripts
python3 make_arena.py
```

를 다시 실행하고 Gazebo를 재시작해야 합니다.

---

# 동적장애물 속도 수정

회전교차로 동적장애물은 다음 파일에서 제어합니다.

```text
src/webot_arena/scripts/mission_motion_controller.py
```

기본 속도는 ROS 2 parameter인:

```text
obstacle_speed
```

입니다.

현재 기본값:

```python
self.declare_parameter('obstacle_speed', 0.20)
```

단위는 **m/s**입니다.

```text
0.10 m/s : 느림
0.15 m/s : 조금 느림
0.20 m/s : 기본
0.25 m/s : 조금 빠름
0.30 m/s : 빠름
```

controller 실행 시 속도를 직접 지정할 수도 있습니다.

```bash
cd ~/Workspace/KUDOS_scalecar

source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/webot_arena/scripts/mission_motion_controller.py \
  --ros-args \
  -p obstacle_speed:=0.20
```

실행 중인 controller의 속도를 바꾸려면:

```bash
ros2 param set \
/mission_motion_controller \
obstacle_speed 0.15
```

동적장애물은 `v = rω` 관계를 이용해 원형 경로를 주행하며, 기본 회전 반경은 `0.68 m`입니다.

---

# 차단기 동작

차단기 역시 `mission_motion_controller.py`에서 제어합니다.

주요 ROS 2 parameter:

```text
barrier_speed_deg
barrier_wait_closed
barrier_wait_open
```

현재 기본값:

```text
barrier_speed_deg   = 30.0 deg/s
barrier_wait_closed = 3.0 s
barrier_wait_open   = 3.0 s
```

---

# 갈림길 표지판 방향 설정

`make_arena.py`는 환경변수 `M4_SIGN_MODE`를 이용해 갈림길 표지판 방향을 설정할 수 있습니다.

랜덤:

```bash
python3 make_arena.py
```

또는:

```bash
M4_SIGN_MODE=random python3 make_arena.py
```

시작 표지판을 오른쪽 방향으로 고정:

```bash
M4_SIGN_MODE=right python3 make_arena.py
```

시작 표지판을 왼쪽 방향으로 고정:

```bash
M4_SIGN_MODE=left python3 make_arena.py
```

시작 표지판과 종료 표지판은 서로 반대 방향으로 생성됩니다.

---

# 개발 시 참고

`make_arena.py`를 수정하면 Python 파일만 저장하는 것으로는 Gazebo 화면이 바로 바뀌지 않습니다.

```text
make_arena.py 수정
        ↓
python3 make_arena.py
        ↓
limo_competition.world 재생성
        ↓
Gazebo 재시작
```

라바콘 위치처럼 world geometry만 수정하는 경우, `--symlink-install` 환경에서는 매번 전체 workspace를 다시 빌드할 필요는 없습니다.

Python 문법 오류 확인:

```bash
python3 -m py_compile \
src/webot_arena/scripts/make_arena.py

python3 -m py_compile \
src/webot_arena/scripts/mission_motion_controller.py
```

---

# Git 사용

수정한 내용을 GitHub에 반영할 때:

```bash
git status
git add .
git commit -m "Describe your changes"
git push
```

예:

```bash
git add src/webot_arena/scripts/make_arena.py
git commit -m "Adjust traffic cone positions"
git push
```

---

# Project Goal

본 프로젝트의 목표는 실제 스케일카 자율주행 경진대회의 주요 미션을 Gazebo에서 재현하여, 실제 차량에 적용하기 전에 센서 인식 및 주행 알고리즘을 반복적으로 검증할 수 있는 환경을 구축하는 것입니다.

특히 실제 대회 환경에서 발생할 수 있는 센서 오차, 인식 실패 및 다양한 미션 상황을 시뮬레이션에서 먼저 확인하고 수정할 수 있도록 하는 것을 목표로 합니다.

---

## Team

**KUDOS**

Autonomous Scale Car Project

---

## License

일부 LIMO 관련 패키지 및 asset은 원본 프로젝트의 라이선스를 따를 수 있습니다.  
사용 및 재배포 전 각 패키지의 `package.xml` 및 원본 라이선스를 확인해 주세요.
