# 한 프레임에서 검출된 미션 객체 목록
# header.stamp = 원본 카메라 프레임 시각, header.frame_id = base_link
# 검출이 없어도 빈 배열로 계속 발행된다 (발행이 끊기면 = 인지 노드 이상)
std_msgs/Header header
MissionObject[] objects
