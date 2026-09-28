# ar_msgs 추가 제안 — YOLO 출력 메시지

LaneError/ColorZone 을 ar_msgs 로 옮긴 것과 같은 이유(인지·판단 공용 메시지는
한 패키지에서만 관리)로, YOLO 추론 노드의 출력도 처음부터 ar_msgs 에 둔다.

판단팀 검토 요청 사항:
1. 클래스 상수를 메시지에 박아두는 방식 괜찮은지
   (판단 쪽이 매직넘버 대신 MissionObject.MARKER_LEFT 로 비교 가능)
2. 위치를 x_m/y_m(base_link) 로 주는 것으로 충분한지
3. 차단기 angle 필드는 OBB/seg 결정 후 의미 확정
