# ar_msgs 추가 요청 — YOLO 출력 메시지 (세진님 검토용)

ar_msgs 는 판단팀 관리 패키지이고 곧 git 으로 옮길 예정이므로, 인지 쪽에서 직접 고치지 않고
아래 내용으로 추가를 요청합니다.

## 추가할 것
1. `MissionObject.msg`, `MissionObjectArray.msg` 를 `ar_msgs/msg/` 에 복사
2. `CMakeLists.txt` 의 `rosidl_generate_interfaces` 에 두 줄 추가 (`ar_msgs_CMakeLists.patch` 참고)
3. 확인: `colcon build --packages-select ar_msgs && ros2 interface show ar_msgs/msg/MissionObjectArray`

## 인터페이스 명세 (토픽 `/perception/mission_objects`, 약 15Hz)
- 좌표는 `base_link` 기준, 미터 단위 (x 전방, y 좌측 양수)
- 검출이 없으면 **빈 배열로 계속 발행**. 메시지가 아예 안 오면 인지 노드 이상으로 볼 것
- `class_id` 는 메시지 상수로 비교 (`MissionObject.MARKER_LEFT` 등, 숫자 직접 쓰지 말 것)
- 클래스별 신뢰도 문턱은 인지 쪽에서 이미 적용됨. `confidence` 는 참고용
- `dynamic_car` 외 클래스는 한 프레임에 최대 1개 (가장 확실한 것)
- 위치 흔들림은 인지에서 EMA 로 줄여서 보냄. **"연속 N프레임 확인 후 상태 전이"는 판단 쪽에서** (보고서 설계대로)
- `position_valid=false` 면 x_m/y_m 무시. `position_source` 로 방식 확인 가능
  - 횡단보도: `x_m` = 횡단보도 앞쪽 끝까지 거리 (바닥평면 교점). 너무 가까워 화면 밖이면 invalid
  - 마커/주차표지판/차단기: RGB-D 깊이, 실패 시 크기 역산(`POS_SIZE`, 오차 큼)
- 차단기 `angle_valid` 는 seg 모델을 쓸 때만 true. 그 전까지는 LiDAR 로 판단

## 판단팀과 합의 필요
1. 위 명세로 충분한지 (특히 횡단보도 거리 기준: 앞쪽 끝 vs 정지선)
2. 차단기 각도를 YOLO seg 로 할지, LiDAR 만으로 할지
