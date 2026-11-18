# 정적 LIMO 조건부 점 누적

적용 위치는 상위 워크스페이스 `src/ar_perception`이다. `distribution` 통합본에는 자동 반영하지 않는다.

## 정책

기존 `estimate_limo_candidate`와 센서 위치 기준 LINE 법선은 변경하지 않았다.
기존 후보가 없으면 새 장애물을 만들지 않는다. LINE 결과도 그대로 유지한다.
현재 스캔이 CIRCLE일 때만, 동일 물체로 연결된 최근 스캔의 표면점을 odom 기준으로 모아 0.322 x 0.22 m 직사각형을 피팅한다.
각도는 전체 [0, pi) 범위를 탐색한다. 센서에서 보이는 유한 길이의 모서리까지의 거리로 피팅하고, 서로 인접한 두 면의 점 개수·길이, 대안 중심의 모호성, 현재 스캔과의 일치, 중심 이동량, 시간상 안정성을 확인한다.
안정성 확인 전과 모든 거부 상황에는 기존 중심을 발행한다. 결과는 x/y만 대체하며 radius=0.16, is_dynamic=false, 메시지 타입, point_count는 유지한다.

## 시간 및 추적

- 스캔 시각 기준 odom 보간. 가까운 한쪽 odom만 있을 때 최대 0.06초 차이 허용. 오래된 odom은 사용하지 않는다.
- odom 프레임 변경·시간 역행·큰 roll/pitch·중복/역행 scan 시각에는 누적 상태를 버린다.
- 중심 근접뿐 아니라 상호 표면점 일치와 일대일 대응을 확인한다. 복수 대응 또는 너무 가까운 후보는 누적하지 않는다.
- 한 스캔이라도 후보가 끊기면 해당 track을 버린다. 관측 간격 0.35초 초과도 새로운 track이다.
- 최근 최대 1초, 누적 표면점 최대 90개, 서로 다른 스캔 최소 3개. 피팅 중심이 다음 피팅에서도 안정적인지 확인하므로 실제 채택은 통상 빠르면 4번째 스캔이다.
- 기본 스캔당 누적 시간 예산 20ms, 피팅 후보 최대 3개, 추적 최대 24개. 초과 시 기존 결과를 사용한다. 이는 하드 실시간 보장이 아니다.

## 실행

기존 LiDAR 프로세스를 종료하고, Gazebo 또는 bag이 /scan, /odom, /clock을 같은 ROS 도메인으로 발행하는 상태에서 실행한다.

```bash
cd /home/td/Workspace/limo_competition_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=37
ros2 run ar_perception lidar_perception --ros-args \
  --params-file /home/td/Workspace/limo_competition_ws/src/ar_perception/config/lidar.yaml \
  -p use_sim_time:=true --log-level warn
```

비교용으로 같은 노드에서 기능을 끄거나 켤 수 있다. 끄면 기존 track은 폐기한다.

```bash
ros2 param set /lidar_perception use_point_accumulation false
ros2 param set /lidar_perception use_point_accumulation true
```

## 디버그

```bash
ros2 topic echo /perception/debug/limo_accumulation std_msgs/msg/String
```

JSON 필드:
- `stamp`, `frame_id`: 입력 스캔 시각과 좌표계.
- `obstacle_index`: 그 스캔의 ObstacleArray 내 인덱스 (RViz cluster ID와 다를 수 있음).
- `baseline`: 기존 중심 x/y.
- `proposed`: 피팅이 생성된 경우 제안 중심 x/y. 거부된 제안은 발행 장애물에 반영되지 않는다.
- `accepted`, `reason`: 채택 여부와 거부 이유.
- `track`, `scans`, `points`: 추적 번호, 축적 스캔 수, 점 수.
- `yaw_base`: 직사각형 긴 변 방향, base_link 기준 rad. pi 차이는 같은 형상이다.
- `rms`, `accumulation_ms`: 피팅 표면 오차와 누적 처리시간.

기존 RViz 마커는 요청대로 변경하지 않았으므로 **원래 LINE/CIRCLE 중심을 계속 표시한다**.
최종 중심은 `/perception/obstacles`, 비교 값은 위 JSON에서 확인한다.
`/perception/event/lidar`는 기존처럼 cone/none만 발행한다.

검증 결과 및 재현 방법은 [validation/REPORT.md](validation/REPORT.md)를 참고한다.
