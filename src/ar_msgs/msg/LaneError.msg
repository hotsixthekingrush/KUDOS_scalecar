# 차선 인식 결과 — camera_perception_node -> mission_supervisor_node
#
# [부호 규약] ROS 표준(x 전방, y 좌측, z 상방)을 따름.
#   lateral_offset_m : 로봇 중심에서 본 "차선 중앙"의 횡방향 위치
#                      양수 = 차선 중앙이 로봇보다 왼쪽  -> 로봇이 왼쪽으로 가야 함
#   heading_error_rad: 차선이 로봇 전방축과 이루는 각도
#                      양수 = 차선이 전방에서 왼쪽으로 꺾임 -> 좌회전 필요
#   둘 다 양수일 때 angular.z 를 양수(반시계=좌회전)로 주면 되도록 맞춤

std_msgs/Header header

# false 이면 아래 수치는 모두 무의미. 판단 모듈의 fail-safe 로직이 처리할 것.
bool valid

float32 lateral_offset_m      # [m]
float32 heading_error_rad     # [rad]
float32 curvature             # [1/m] 부호 있음. 양수=좌커브. 제어 속도규제용
float32 confidence            # 0.0~1.0 검출 신뢰도

# 한쪽 차선만 보일 때 lane_width 로 중앙을 추정한 경우 true
bool single_lane_estimated
