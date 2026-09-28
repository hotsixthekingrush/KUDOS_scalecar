KUDOS 카메라 인지 파트 (고균명) — src 압축본 v2 (2026-09-28)

[포함]
  src/kudos_yolo_tools   : YOLO 자동 라벨링 / 분할 / 학습 / 실시간 추론 노드   <- 고균명 담당
                           docs/ar_msgs_proposal : YOLO 출력 메시지 추가 요청안 (세진님 전달용)
  src/kudos_perception   : 차선(lane_error) + 색상구역(color_zone) 인지 노드
                           -> 라인트레이싱 담당 팀원에게 인계. v1 과 동일(수정 없음)

[미포함]
  ar_msgs  : 판단팀 관리. 워크스페이스에 이미 있어야 함.
             yolo_detector_node 를 돌리려면 MissionObject/MissionObjectArray 추가가 필요
             (docs/ar_msgs_proposal/README_proposal.md)
  build/ install/ log/

[설치]
  cd ~/<워크스페이스>
  tar -xzf kudos_camera_src_v2.tar.gz
  pip install ultralytics pyyaml
  colcon build --packages-up-to kudos_perception kudos_yolo_tools --symlink-install
  source install/setup.bash

[순서]  자세한 건 src/kudos_yolo_tools/README.md
  1) config/targets.yaml 을 월드 파일에 맞게 수정 (좌/우 마커 모델 2개!)
  2) Gazebo + auto_labeler.py + domain_randomizer.py 로 수집 → debug/ 이미지 눈으로 확인
  3) split_dataset.py → image_augment.py → train_yolo.py (train / val)
  4) yolo_detector_node.py 로 판단팀 연동
