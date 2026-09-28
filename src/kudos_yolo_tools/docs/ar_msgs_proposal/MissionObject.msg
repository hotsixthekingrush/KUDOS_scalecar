# [제안] YOLO 미션 객체 인식 결과 1개 — ar_msgs 추가 검토용
# 인지(camera_yolo) -> 판단(mission_supervisor_node)
#
# 원칙은 LaneError 와 동일:
#   - 픽셀 좌표는 노드 밖으로 내보내지 않는다. 위치는 미터로.
#   - 좌표계는 base_link (x 전방, y 좌측 양수) — LaneError 부호 규약과 일치.
#
# class 상수 순서는 학습 데이터셋(dataset.yaml)의 클래스 ID 와 반드시 같아야 한다.

uint8 CROSSWALK    = 0
uint8 MARKER_LEFT  = 1
uint8 MARKER_RIGHT = 2
uint8 GATE_BAR     = 3
uint8 PARKING_SIGN = 4
uint8 DYNAMIC_CAR  = 5

uint8   class_id
float32 confidence

# 거리/위치 추정 성공 여부. 미션마다 방법이 다르다.
#   횡단보도: IPM 역산 / 마커: RGB-D 깊이 / 나머지: 필요시
bool    position_valid
float32 x_m          # 전방 거리 [m]
float32 y_m          # 좌측 양수 [m]

# 차단기 전용. 방식(OBB/segmentation) 확정 전이라 필드만 예약.
bool    angle_valid
float32 angle_rad    # 차단바 기울기. 0=수평(닫힘), 커질수록 열림
