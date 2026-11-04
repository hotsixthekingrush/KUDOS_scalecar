#!/usr/bin/env python3
"""
tl_check.py — 신호등 사진에 (1) 시뮬 모델로 박스 미리 치기 (2) 색 판정 정확도 확인

  python3 tl_check.py <모델.pt> <사진 폴더들...> [--conf 0.25]

폴더 이름에 red / yellow / green / off 가 들어 있으면 그게 정답 색 (예: tl_red, tl_green)
결과:
  <폴더>/labels/*.txt   YOLO 라벨 초안 (클래스 0 = traffic_light). Roboflow/CVAT 로 불러와 고치기
  <폴더>/_check/*.jpg   박스 + 판정 색을 그린 사진 (틀린 것만 'WRONG_' 붙임)
  화면 출력: 폴더별 검출률, 색 정답률, 많이 나온 오답
"""
import argparse, glob, os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import light_color as lc
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('model'); ap.add_argument('dirs', nargs='+')
ap.add_argument('--conf', type=float, default=0.25)
ap.add_argument('--s_min', type=int, default=90); ap.add_argument('--v_min', type=int, default=150)
a = ap.parse_args()
m = YOLO(a.model)
tl_ids = [i for i, n in m.names.items() if n == 'traffic_light']
if not tl_ids:
    sys.exit('모델에 traffic_light 클래스가 없음: %s' % m.names)
TRUTH = {'red': 1, 'yellow': 2, 'green': 3, 'off': 0}

for d in a.dirs:
    truth = next((v for k, v in TRUTH.items() if k in os.path.basename(os.path.normpath(d)).lower()), None)
    imgs = sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png')))
    os.makedirs(os.path.join(d, 'labels'), exist_ok=True); os.makedirs(os.path.join(d, '_check'), exist_ok=True)
    found = right = 0; wrong = {}
    for p in imgs:
        im = cv2.imread(p); h, w = im.shape[:2]
        r = m.predict(im, conf=a.conf, classes=tl_ids, verbose=False)[0]
        lines, best = [], None
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = b
            lines.append('0 %.6f %.6f %.6f %.6f' % ((x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h))
            if best is None or c > best[1]: best = (b, c)
        open(os.path.join(d, 'labels', os.path.splitext(os.path.basename(p))[0] + '.txt'), 'w').write('\n'.join(lines))
        tag = 'NOBOX_'
        if best is not None:
            found += 1
            col, why = lc.classify(im, best[0], a.s_min, a.v_min)
            ok = truth is None or col == truth
            right += ok
            if not ok: wrong[lc.NAMES[col]] = wrong.get(lc.NAMES[col], 0) + 1
            tag = '' if ok else 'WRONG_'
            x1, y1, x2, y2 = [int(v) for v in best[0]]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.putText(im, '%s %.2f' % (lc.NAMES[col], best[1]), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if tag:
            cv2.imwrite(os.path.join(d, '_check', tag + os.path.basename(p)), im)
    n = max(len(imgs), 1)
    print('%-20s 정답 %-7s | 사진 %4d | 박스 찾음 %4d (%.0f%%) | 색 맞음 %4d/%d (%.0f%%) | 오답 %s'
          % (os.path.basename(os.path.normpath(d)), lc.NAMES.get(truth, '?'), len(imgs), found, 100*found/n,
             right, found, 100*right/max(found, 1), wrong or '-'))
