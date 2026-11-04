#!/usr/bin/env python3
"""
crosswalk_from_paint.py — 트랙 페인트 메시(white.dae)에서 횡단보도를 찾아 라벨 설정에 넣고,
이미 모은 사진(meta/*.json 카메라 자세)에 횡단보도 라벨을 덧붙인다. (11/4 밤, 무인 실행용)

  python3 crosswalk_from_paint.py --world W --mesh-root R --cfg CFG --data D --lib S

찾는 방법
  1) world 의 track_paint 모델에서 white 메시 경로를 찾고 삼각형을 읽는다
  2) 꼭짓점을 공유하는 삼각형끼리 묶어 조각(연결 성분)을 만든다
  3) 가늘고 긴 직사각형 조각(폭 1.5~15cm, 길이 12~100cm, 직사각형 비율 80% 이상) = 줄 후보
  4) 같은 방향·비슷한 길이의 줄이 "나란히"(긴 축과 수직으로) 35cm 안에 3개 이상 → 횡단보도
     (점선 차선은 줄이 긴 축 방향으로 이어져서 제외, 체커기는 정사각형이라 제외)
결과
  CFG/targets_full_A.yaml, targets_full_B.yaml 에 crosswalk 대상 추가 (world_pose)
  CFG/viewpoints_full.yaml 에 cw<i>_a~d 구역, CFG/mission_points.yaml 에 점검 지점 추가
  D/labels/all/*.txt 에 클래스 9 줄 추가 (이미 있으면 건너뜀), D/debug_cw/ 확인 사진
  CFG/crosswalks.txt 에 이름 목록 (훅이 수집 only:= 에 사용). 못 찾으면 종료코드 2
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import yaml

CW = 9


def collada_triangles(path):
    """collada 의 모든 삼각형 (N,3,3) — 단위·up_axis·노드 변환 적용"""
    root = ET.parse(path).getroot()
    m = re.match(r'\{.*\}', root.tag)
    ns = m.group(0) if m else ''
    q = lambda t: ns + t
    unit = 1.0
    u = root.find('.//%s/%s' % (q('asset'), q('unit')))
    if u is not None and u.get('meter'):
        unit = float(u.get('meter'))
    up = root.find('.//%s/%s' % (q('asset'), q('up_axis')))
    up = (up.text or 'Z_UP').strip() if up is not None else 'Z_UP'

    geoms = {}
    for g in root.iter(q('geometry')):
        mesh = g.find(q('mesh'))
        if mesh is None:
            continue
        srcs = {}
        for s in mesh.findall(q('source')):
            fa = s.find(q('float_array'))
            if fa is not None and fa.text:
                srcs[s.get('id')] = np.array([float(x) for x in fa.text.split()])
        vsrc = {}
        for v in mesh.findall(q('vertices')):
            for inp in v.findall(q('input')):
                if inp.get('semantic') == 'POSITION':
                    vsrc[v.get('id')] = inp.get('source', '').lstrip('#')
        tris = []
        for prim in list(mesh):
            tag = prim.tag.replace(ns, '')
            if tag not in ('triangles', 'polylist', 'polygons'):
                continue
            inputs = prim.findall(q('input'))
            stride = max(int(i.get('offset', 0)) for i in inputs) + 1 if inputs else 1
            off, pos = None, None
            for i in inputs:
                if i.get('semantic') == 'VERTEX':
                    off = int(i.get('offset', 0))
                    pos = srcs.get(vsrc.get(i.get('source', '').lstrip('#'), ''))
            if off is None or pos is None:
                continue
            P = pos.reshape(-1, 3)
            ps = prim.findall(q('p'))
            if tag == 'polylist':
                vc = [int(x) for x in (prim.findtext(q('vcount')) or '').split()]
                idx = np.array([int(x) for x in (ps[0].text or '').split()]).reshape(-1, stride)[:, off]
                k = 0
                for n in vc:
                    poly = idx[k:k + n]; k += n
                    for j in range(1, n - 1):
                        tris.append(P[[poly[0], poly[j], poly[j + 1]]])
            else:
                for p in ps:
                    idx = np.array([int(x) for x in (p.text or '').split()]).reshape(-1, stride)[:, off]
                    if tag == 'triangles':
                        for t in idx.reshape(-1, 3):
                            tris.append(P[t])
                    else:
                        for j in range(1, len(idx) - 1):
                            tris.append(P[[idx[0], idx[j], idx[j + 1]]])
        if tris:
            geoms[g.get('id')] = np.array(tris)

    out = []

    def node_T(n):
        T = np.eye(4)
        for ch in list(n):
            tag = ch.tag.replace(ns, '')
            vals = [float(x) for x in (ch.text or '').split()]
            if tag == 'matrix' and len(vals) == 16:
                T = T @ np.array(vals).reshape(4, 4)
            elif tag == 'translate' and len(vals) == 3:
                M = np.eye(4); M[:3, 3] = vals; T = T @ M
            elif tag == 'scale' and len(vals) == 3:
                T = T @ np.diag(vals + [1.0])
            elif tag == 'rotate' and len(vals) == 4:
                ax = np.array(vals[:3]); ax = ax / (np.linalg.norm(ax) or 1); a = math.radians(vals[3])
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                M = np.eye(4); M[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
                T = T @ M
        return T

    def walk(n, T):
        T = T @ node_T(n)
        for ig in n.findall(q('instance_geometry')):
            gid = (ig.get('url') or '').lstrip('#')
            if gid in geoms:
                t = geoms[gid].reshape(-1, 3)
                out.append((t @ T[:3, :3].T + T[:3, 3]).reshape(-1, 3, 3))
        for ch in n.findall(q('node')):
            walk(ch, T)

    for sc in root.iter(q('visual_scene')):
        for n in sc.findall(q('node')):
            walk(n, np.eye(4))
    if not out:
        out = list(geoms.values())
    if not out:
        return np.zeros((0, 3, 3))
    T = np.vstack(out) * unit
    if up == 'Y_UP':
        T = np.stack([T[..., 0], -T[..., 2], T[..., 1]], axis=-1)
    elif up == 'X_UP':
        T = np.stack([-T[..., 1], T[..., 0], T[..., 2]], axis=-1)
    return T


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[a] = b


def components(tris):
    keys = {}
    vid = np.zeros(tris.shape[:2], dtype=int)
    for i in range(tris.shape[0]):
        for j in range(3):
            k = tuple(np.round(tris[i, j, :2], 4))
            vid[i, j] = keys.setdefault(k, len(keys))
    uf = UF(len(keys))
    for i in range(tris.shape[0]):
        uf.u(vid[i, 0], vid[i, 1]); uf.u(vid[i, 0], vid[i, 2])
    groups = {}
    for i in range(tris.shape[0]):
        groups.setdefault(uf.f(vid[i, 0]), []).append(i)
    return [tris[idx] for idx in groups.values()]


def tri_area(t):
    a = t[:, 1, :2] - t[:, 0, :2]; b = t[:, 2, :2] - t[:, 0, :2]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def find_crosswalks(tris, log):
    stripes = []
    for comp in components(tris):
        pts = comp.reshape(-1, 3)[:, :2].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        lo, sh = max(w, h), min(w, h)
        if sh <= 0:
            continue
        rect = float(tri_area(comp).sum()) / (w * h)
        if not (0.015 <= sh <= 0.15 and 0.12 <= lo <= 1.0 and lo / sh >= 2.5 and rect >= 0.8):
            continue
        th = math.radians(ang) if w >= h else math.radians(ang + 90)
        stripes.append({'c': np.array([cx, cy]), 'len': lo, 'wid': sh, 'u': np.array([math.cos(th), math.sin(th)]),
                        'pts': pts, 'z': float(comp[..., 2].max())})
    log.append('줄 후보 %d개' % len(stripes))
    uf = UF(len(stripes))
    for i in range(len(stripes)):
        for j in range(i + 1, len(stripes)):
            a, b = stripes[i], stripes[j]
            if abs(float(np.dot(a['u'], b['u']))) < math.cos(math.radians(10)):
                continue
            r = a['len'] / b['len']
            if not (0.7 <= r <= 1.43):
                continue
            d = b['c'] - a['c']
            dist = float(np.linalg.norm(d))
            if dist > 0.35 or abs(float(np.dot(d, a['u']))) > 0.35 * dist:
                continue
            uf.u(i, j)
    groups = {}
    for i in range(len(stripes)):
        groups.setdefault(uf.f(i), []).append(stripes[i])
    out = []
    for g in groups.values():
        if len(g) < 3:
            continue
        u = g[0]['u']
        yaw = math.atan2(u[1], u[0])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        P = np.vstack([s['pts'] for s in g]).astype(float)
        c0 = P.mean(0)
        L = (P - c0) @ R          # 열 0 = 줄 긴 축(u), 열 1 = 줄이 늘어선 방향
        lo, hi = L.min(0), L.max(0)
        c = c0 + R @ ((lo + hi) / 2)
        size = hi - lo
        out.append({'center': [float(c[0]), float(c[1]), max(s['z'] for s in g) / 2 + 0.001],
                    'yaw': yaw, 'size': [float(size[0]), float(size[1]), 0.004], 'n': len(g)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True)
    ap.add_argument('--mesh-root', default='~/26v2/KUDOS_Autorace2026_Gazebo')
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--lib', required=True, help='yolo_geometry.py, world_targets.py 가 있는 폴더')
    ap.add_argument('--dae', default='', help='white.dae 를 직접 지정')
    a = ap.parse_args()
    sys.path.insert(0, a.lib)
    import yolo_geometry as geo
    import world_targets as wt
    log = []
    world = os.path.expanduser(a.world)
    cfg, data = os.path.expanduser(a.cfg), os.path.expanduser(a.data)

    dae = os.path.expanduser(a.dae)
    T_paint = np.eye(4)
    if not dae:
        res = wt.Resolver([a.mesh_root, os.path.dirname(world)])
        root = ET.parse(world).getroot().find('world')
        for mdl in root.findall('model'):
            for link in mdl.findall('link'):
                for vis in link.findall('visual'):
                    uri = vis.findtext('geometry/mesh/uri') or ''
                    if 'white' in uri.lower() and uri.lower().endswith('.dae'):
                        dae = res.resolve(uri) or ''
                        T_paint = wt.pose_T(mdl.find('pose')) @ wt.pose_T(link.find('pose')) @ wt.pose_T(vis.find('pose'))
                        log.append('white 메시: %s (%s)' % (dae, uri))
    if not dae or not os.path.exists(dae):
        print('white.dae 못 찾음'); sys.exit(2)
    tris = collada_triangles(dae)
    tris = tris @ T_paint[:3, :3].T + T_paint[:3, 3]
    log.append('삼각형 %d개' % len(tris))
    cws = find_crosswalks(tris, log)
    for i, c in enumerate(cws):
        log.append('횡단보도 cw%d: 중심 (%.2f, %.2f) 줄방향 %.0f° 크기 줄길이 %.2f x 폭 %.2f, 줄 %d개'
                   % (i, c['center'][0], c['center'][1], math.degrees(c['yaw']), c['size'][0], c['size'][1], c['n']))
    print('\n'.join(log))
    if not cws:
        print('횡단보도 못 찾음'); sys.exit(2)

    # --- 설정 파일에 추가 ---
    names = []
    for s in ('A', 'B'):
        f = os.path.join(cfg, 'targets_full_%s.yaml' % s)
        txt = open(f).read()
        if 'crosswalk, world_pose' not in txt:
            with open(f, 'a') as fo:
                for i, c in enumerate(cws):
                    fo.write('  - {class: crosswalk, world_pose: %s, size: %s}   # cw%d white.dae 줄 %d개\n'
                             % ([round(v, 4) for v in c['center']] + [round(c['yaw'], 4)],
                                [round(v, 4) for v in c['size']], i, c['n']))
    vf = os.path.join(cfg, 'viewpoints_full.yaml')
    vp = yaml.safe_load(open(vf))
    pf = os.path.join(cfg, 'mission_points.yaml')
    mp = yaml.safe_load(open(pf))
    have = set(r['name'] for r in vp['regions'])
    for i, c in enumerate(cws):
        cx, cy = c['center'][:2]
        for k, (ax, half, across) in enumerate([(c['yaw'], c['size'][0] / 2, c['size'][1]),
                                                (c['yaw'] + math.pi / 2, c['size'][1] / 2, c['size'][0])]):
            for s, tag in ((0.0, 'ab'[0] if k == 0 else 'c'), (math.pi, 'b' if k == 0 else 'd')):
                nm = 'cw%d_%s' % (i, tag)
                names.append(nm)
                n = ax + s
                lat = max(0.05, min(0.25, across / 2 - 0.1))
                if nm not in have:
                    vp['regions'].append({'name': nm, 'type': 'facing', 'target': [round(cx, 3), round(cy, 3)],
                                          'normal_yaw': round(n, 4), 'dist': [round(half + 0.25, 3), round(half + 2.0, 3)],
                                          'lateral': [-round(lat, 3), round(lat, 3)], 'yaw_jitter': 0.25, 'weight': 1.0})
                for d in (1.5, 1.0, 0.5, 0.25):
                    x, y = cx + (half + d) * math.cos(n), cy + (half + d) * math.sin(n)
                    mp['points'].append({'name': 'cw%d %s %.2fm' % (i, tag, d), 'class': 'crosswalk',
                                         'x': round(x, 3), 'y': round(y, 3),
                                         'yaw': round(math.atan2(cy - y, cx - x), 4), 'expect_x': d})
    with open(vf, 'w') as f:
        yaml.safe_dump(vp, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    seen = set(); pts = []
    for p in mp['points']:
        if p['name'] not in seen:
            seen.add(p['name']); pts.append(p)
    mp['points'] = pts
    with open(pf, 'w') as f:
        yaml.safe_dump(mp, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    with open(os.path.join(cfg, 'crosswalks.txt'), 'w') as f:
        f.write(','.join(names) + '\n')

    # --- 이미 모은 사진에 라벨 덧붙이기 ---
    os.makedirs(os.path.join(data, 'debug_cw'), exist_ok=True)
    added = shown = 0
    for m in sorted(glob.glob(os.path.join(data, 'meta', '*.json'))):
        stem = os.path.splitext(os.path.basename(m))[0]
        lf = os.path.join(data, 'labels', 'all', stem + '.txt')
        if not os.path.exists(lf):
            continue
        old = open(lf).read()
        if any(l.startswith('%d ' % CW) for l in old.splitlines()):
            continue
        meta = json.load(open(m))
        T = np.array(meta['T_cam_world']); K = np.array(meta['K']); w, h = int(meta['w']), int(meta['h'])
        cam = geo.inv_T(T)[:3, 3]
        lines, polys = [], []
        for c in cws:
            # 카메라 쪽 앞쪽 끝이 너무 멀면(4m+) 건너뜀
            if np.hypot(cam[0] - c['center'][0], cam[1] - c['center'][1]) > 4.5:
                continue
            corners = geo.cuboid_corners(c['center'], c['size'], geo.yaw_to_rot(c['yaw']))
            uv = geo.project_points(K, geo.transform_points(T, corners))
            if uv is None:
                continue
            poly, keep = geo.clipped_hull(uv, w, h)
            if poly is None or keep < 0.4:
                continue
            x1, y1, x2, y2 = geo.poly_bbox(poly)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            lines.append(geo.yolo_line(CW, poly, w, h, 'polygon')); polys.append(poly)
        if lines:
            with open(lf, 'w') as f:
                f.write((old.rstrip('\n') + '\n' if old.strip() else '') + '\n'.join(lines))
            added += 1
            if shown < 12:
                img = cv2.imread(os.path.join(data, 'images', 'all', stem + '.jpg'))
                if img is not None:
                    for p in polys:
                        cv2.polylines(img, [np.round(p).astype(np.int32)], True, (255, 0, 255), 2)
                    cv2.imwrite(os.path.join(data, 'debug_cw', stem + '.jpg'), img)
                    shown += 1
    print('횡단보도 라벨 추가: %d장 (확인 사진 %s)' % (added, os.path.join(data, 'debug_cw')))


if __name__ == '__main__':
    main()
