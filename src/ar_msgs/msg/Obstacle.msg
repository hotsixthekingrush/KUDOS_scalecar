# 라이다 클러스터 하나 (base_link 기준, 단위 m)
float32 x
float32 y
float32 radius          # 안전 마진이 포함된 유효 반지름
uint16  point_count     # 클러스터를 이루는 포인트 수
float32 vx              # 추정 속도 (정적이면 0)
float32 vy
bool    is_dynamic      # 동적 객체로 판정되었는지