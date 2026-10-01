# ar_msgs/msg/ParkingGap.msg
std_msgs/Header header   # frame_id = base_link, stamp = 사용한 scan 시각
bool valid               # 추적 중인 gap이 있으면 true
bool confirmed           # 같은 gap이 연속 관측 중이면 true
uint32 gap_id            # gap이 바뀌면 값이 바뀜
float32 center_x         # base_link 기준, 앞 +
float32 center_y         # base_link 기준, 왼쪽 +
float32 length           # 진행방향 입구 길이
float32 depth            # 안쪽 깊이, 모르면 0