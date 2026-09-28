# YOLO 미션 객체 1개 — yolo_detector_node -> mission_supervisor_node
#
# 원칙은 LaneError 와 동일:
#   - 픽셀 좌표는 노드 밖으로 내보내지 않는다. 위치는 미터.
#   - 좌표계는 base_link (x 전방, y 좌측 양수) — LaneError 부호 규약과 일치.
#
# class 상수는 학습 클래스 "이름"으로 매핑되므로 학습 순서가 바뀌어도 안전하다.

uint8 CROSSWALK    = 0
uint8 MARKER_LEFT  = 1
uint8 MARKER_RIGHT = 2
uint8 GATE_BAR     = 3
uint8 PARKING_SIGN = 4
uint8 DYNAMIC_CAR  = 5

uint8   class_id
float32 confidence          # 0.0~1.0 (클래스별 문턱은 인지 노드에서 이미 적용됨)

# 위치 추정 성공 여부. false 면 x_m, y_m 은 무의미.
bool    position_valid
float32 x_m                 # 전방 거리 [m]. 횡단보도는 "앞쪽 끝"까지 거리
float32 y_m                 # 좌측 양수 [m]

# 위치를 어떻게 구했는지 (디버깅·신뢰도 판단용)
uint8 POS_NONE   = 0
uint8 POS_GROUND = 1        # 바닥평면 교점 (횡단보도)
uint8 POS_DEPTH  = 2        # RGB-D 깊이
uint8 POS_SIZE   = 3        # 알려진 실제 크기로 역산 (깊이 실패 시 대안, 오차 큼)
uint8 position_source

# 차단기 전용 (seg 모델일 때만 채워짐). 화면상 기울기.
bool    angle_valid
float32 angle_rad           # 0 = 수평(닫힘), pi/2 = 수직(완전 열림)
