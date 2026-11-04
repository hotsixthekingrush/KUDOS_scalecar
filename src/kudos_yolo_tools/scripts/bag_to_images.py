#!/usr/bin/env python3
"""
bag_to_images.py — rosbag(ROS2) 의 카메라 영상을 사진으로 뽑기

  python3 bag_to_images.py <bag 폴더> <저장 폴더> [--every 0.5] [--topic /camera/color/image_raw/compressed]

- every 초마다 1장 (연속 프레임은 거의 똑같아서 라벨만 늘어남)
- 흐린 사진(움직이는 중) 은 버림: --blur 60 보다 선명도가 낮으면 제외 (0 = 안 버림)
- 파일 이름 = <bag 이름>_<번호>.jpg  → 폴더/bag 이름으로 정답 색을 구분하기 좋게
"""
import argparse, os, sys, yaml, cv2, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

ap = argparse.ArgumentParser()
ap.add_argument('bag'); ap.add_argument('out')
ap.add_argument('--topic', default='/camera/color/image_raw/compressed')
ap.add_argument('--every', type=float, default=0.5)
ap.add_argument('--blur', type=float, default=60.0)
a = ap.parse_args()

meta = yaml.safe_load(open(os.path.join(a.bag, 'metadata.yaml')))['rosbag2_bagfile_information']
storage = meta.get('storage_identifier', 'sqlite3')
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id=storage),
       rosbag2_py.ConverterOptions('cdr', 'cdr'))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
if a.topic not in types:
    sys.exit('토픽 %s 없음. bag 안 토픽: %s' % (a.topic, list(types)))
compressed = types[a.topic].endswith('CompressedImage')
r.set_filter(rosbag2_py.StorageFilter(topics=[a.topic]))

os.makedirs(a.out, exist_ok=True)
name = os.path.basename(os.path.normpath(a.bag))
last, n_all, n_save, n_blur = None, 0, 0, 0
while r.has_next():
    _, data, t = r.read_next()
    n_all += 1
    if last is not None and (t - last) * 1e-9 < a.every:
        continue
    if compressed:
        m = deserialize_message(data, CompressedImage)
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
    else:
        m = deserialize_message(data, Image)
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        continue
    if a.blur > 0 and cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < a.blur:
        n_blur += 1
        continue
    last = t
    cv2.imwrite(os.path.join(a.out, '%s_%05d.jpg' % (name, n_save)), img)
    n_save += 1
print('%s: 프레임 %d → 저장 %d장 (흐려서 버림 %d) → %s' % (name, n_all, n_save, n_blur, a.out))
