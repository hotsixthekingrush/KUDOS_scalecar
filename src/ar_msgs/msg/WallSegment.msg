# 직선 벽 한 면 (base_link 기준, 단위 m)
float32 x1
float32 y1
float32 x2
float32 y2
float32 theta           # 선분 방향각 (rad)
float32 length
uint8   role            # 아래 상수 참조

uint8 ROLE_UNKNOWN=0
uint8 ROLE_BACK=1       # 뒷벽 (진행방향과 나란한 면)
uint8 ROLE_SIDE=2       # 옆벽