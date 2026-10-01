# 미션 1 구역 색상 판정 — camera_perception_node -> mission_supervisor_node
#
# Hysteresis 와 연속 프레임 확인을 인지 단계에서 이미 적용한 "안정된 판정"이다.
# 판단 모듈은 이 값을 그대로 믿고 acc_dcc State 전이에 사용하면 된다.

std_msgs/Header header

uint8 ZONE_NONE = 0
uint8 ZONE_RED  = 1
uint8 ZONE_BLUE = 2

uint8 zone

# 디버깅/튜닝용 원시 비율 (ROI 내 해당 색 픽셀 비율 0.0~1.0)
float32 red_ratio
float32 blue_ratio
