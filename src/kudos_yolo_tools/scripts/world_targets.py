#!/usr/bin/env python3
"""
world_targets.py — 2026 시뮬 맵(track_2026.world)을 읽어 미션 1~5 물체의 자동 라벨 설정을 만든다. (11/4)

왜: targets_v2 에는 신호등·A/B·P 만 있었다. 보행자 표지판·방지턱 표지판·횡단보도·정차 LIMO 는
    월드 안 이름·크기·방향을 몰라서 비워 뒀다. 손으로 재는 대신 world 파일에서 직접 읽는다.

만드는 파일 (--out 폴더, 기본 ~/yolo_data/v2/config_full)
  targets_full_A.yaml / targets_full_B.yaml  auto_labeler 대상 (갈림길 표지판 A/B 실행분)
  viewpoints_full.yaml                       viewpoint_sampler 구역 (미션 물체마다 하나씩 자동)
  mission_points.yaml                        mission_check.py 가 차를 세울 위치
  world_report.txt                           찾은 것 / 못 찾은 것 (아래 화면에도 출력)

실행 (Gazebo 를 한 번 띄워서 /tmp/kudos_arena_*/track_2026.world 가 생긴 뒤)
  python3 world_targets.py
  python3 world_targets.py --world /tmp/kudos_arena_xxx/track_2026.world --mesh-root ~/26v2/KUDOS_Autorace2026_Gazebo

규칙
  - 신호등·갈림길·주차 표지판은 10/29 에 디버그 사진으로 확인한 값을 그대로 쓴다 (KNOWN)
  - 나머지는 모델/visual 이름 키워드로 분류 (CLASS_RULES). 표지판은 그림 면(face/print)만 박스로
  - 표지판 앞면 방향 = 그림 면의 얇은 축, 부호는 그림 면이 받침판/원점보다 튀어나온 쪽
  - 횡단보도 줄이 여러 모델/visual 로 나뉘어 있으면 0.8 m 안에 있는 것끼리 묶어 한 박스
  - 메시(.dae) 크기는 파일의 꼭짓점에서 계산. 파일을 못 찾으면 경고 + 기본 크기
ROS 없이 실행된다 (numpy, pyyaml 만 필요).
"""
import argparse
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

CLASSES = ['traffic_light', 'marker_left', 'marker_right', 'parking_sign', 'pedestrian_sign',
           'slow_sign', 'bump_sign', 'limo', 'slow_guide_line', 'crosswalk']

# 10/29 디버그 사진으로 확인한 대상 (make_arena.py 기준). 이름이 월드에 있으면 이 값을 그대로 쓴다
KNOWN = {
    'branch_sign_panel': {'class': 'BRANCH', 'size': [0.01, 0.224, 0.224], 'offset': [0.0305, 0, 0], 'front': '+x'},
    'parking_sign_panel': {'class': 'parking_sign', 'size': [0.01, 0.224, 0.224], 'offset': [0.0305, 0, 0], 'front': '+x'},
    'traffic_light': {'class': 'traffic_light', 'size': [0.045, 0.12, 0.205], 'offset': [0, 0, 0], 'front': '+x'},
}

# (클래스, 반드시 포함할 키워드 묶음 중 하나, 제외 키워드). 소문자 이름에 대해 검사, 위에서부터 첫 일치
# make_arena.py 는 표지판마다 <이름>_base, <이름>_pole, <이름>_panel 세 모델을 만든다 → 그림은 _panel 에만
SUPPORT = ('_base', '_pole', 'base', 'pole', 'post', 'stand')
CLASS_RULES = [
    ('pedestrian_sign', [('pedestrian', 'sign'), ('ped', 'sign'), ('walker', 'sign'), ('child', 'sign'),
                         ('school', 'sign')], SUPPORT),
    ('bump_sign', [('bump', 'sign'), ('hump', 'sign')], SUPPORT),
    ('slow_sign', [('slow', 'sign')], ('guide',) + SUPPORT),
    ('crosswalk', [('crosswalk',), ('cross_walk',), ('zebra',), ('crossing',)], ('sign',)),
    ('limo', [('traffic_car',), ('parking_car',), ('static_limo',), ('obstacle_limo',), ('parked',)], ()),
    ('slow_guide_line', [('guide',), ('zigzag',)], ('sign',)),
]
# 분류는 안 하지만 리포트·순간이동 금지 구역에 쓰는 것
OTHER_RULES = [
    ('sign_support', [('sign', '_base'), ('sign', '_pole'), ('light', '_base'), ('light', '_pole')]),
    ('pedestrian_figure', [('pedestrian',), ('person',), ('walker',), ('human',)]),
    ('speed_bump', [('bump',), ('hump',)]),
    ('stop_line', [('stop',)]),
    ('cone', [('cone',)]),
]
SIGN_CLASSES = {'pedestrian_sign', 'bump_sign', 'slow_sign', 'parking_sign'}
FACE_KEYS = ('print', 'face', 'texture', 'image', 'picture', 'symbol', 'icon')
NOT_FACE_KEYS = ('pole', 'post', 'stand', 'base', 'leg', 'pillar', 'foot', 'mount', 'stem', 'rod', 'back')
LIMO_DEFAULT = {'size': [0.322, 0.22, 0.25], 'offset': [0.0, 0.0, 0.125]}
ARENA = [0.0, 10.0, 0.0, 6.0]


# ---------------------------------------------------------------- 기하
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def pose_T(el):
    T = np.eye(4)
    if el is None or el.text is None or not el.text.strip():
        return T
    v = [float(x) for x in el.text.split()]
    v += [0.0] * (6 - len(v))
    T[:3, :3] = rpy_to_R(v[3], v[4], v[5])
    T[:3, 3] = v[:3]
    return T


def yaw_of_R(R):
    return math.atan2(R[1, 0], R[0, 0])


def box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def tf_pts(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- 파일 찾기
class Resolver:
    def __init__(self, roots):
        self.roots = [os.path.expanduser(r) for r in roots if r]
        self.cache = {}

    def find_dir(self, name):
        if name in self.cache:
            return self.cache[name]
        found = None
        for root in self.roots:
            if not os.path.isdir(root):
                continue
            if os.path.basename(root.rstrip('/')) == name:
                found = root
                break
            for dp, dns, _ in os.walk(root):
                dns[:] = [d for d in dns if not d.startswith('.') and d not in ('build', 'log')]
                if name in dns:
                    found = os.path.join(dp, name)
                    break
            if found:
                break
        self.cache[name] = found
        return found

    def resolve(self, uri):
        if not uri:
            return None
        uri = uri.strip()
        if uri.startswith('file://'):
            p = uri[7:]
            return p if os.path.exists(p) else None
        m = re.match(r'(model|package)://([^/]+)/?(.*)', uri)
        if m:
            d = self.find_dir(m.group(2))
            if d:
                p = os.path.join(d, m.group(3))
                if os.path.exists(p):
                    return p
            # 이름이 같은 파일을 직접 찾기 (마지막 수단)
            base = os.path.basename(uri)
            for root in self.roots:
                hits = glob.glob(os.path.join(root, '**', base), recursive=True)
                if hits:
                    return hits[0]
            return None
        p = os.path.expanduser(uri)
        return p if os.path.exists(p) else None


def stl_bbox(path):
    """STL(바이너리/ASCII) 꼭짓점 경계 상자. 실패 시 None"""
    try:
        data = open(path, 'rb').read()
    except Exception:
        return None
    if data[:5].lower() == b'solid' and b'facet' in data[:2000]:
        vals = re.findall(rb'vertex\s+(\S+)\s+(\S+)\s+(\S+)', data)
        if not vals:
            return None
        P = np.array([[float(a), float(b), float(c)] for a, b, c in vals])
    else:
        if len(data) < 84:
            return None
        n = int(np.frombuffer(data[80:84], dtype='<u4')[0])
        rec = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        if len(data) < 84 + n * 50:
            return None
        tri = np.frombuffer(data[84:84 + n * 50], dtype=rec)
        P = tri['v'].reshape(-1, 3).astype(float)
    return P.min(0), P.max(0)


def dae_bbox(path):
    """collada 꼭짓점의 경계 상자 (m, Z-up). 실패 시 None"""
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return None
    ns = ''
    m = re.match(r'\{.*\}', root.tag)
    if m:
        ns = m.group(0)
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
        srcs = {s.get('id'): s for s in mesh.findall(q('source'))}
        pts = None
        verts = mesh.find(q('vertices'))
        if verts is not None:
            for inp in verts.findall(q('input')):
                if inp.get('semantic') == 'POSITION':
                    s = srcs.get(inp.get('source', '').lstrip('#'))
                    if s is not None:
                        fa = s.find(q('float_array'))
                        if fa is not None and fa.text:
                            pts = np.array([float(x) for x in fa.text.split()]).reshape(-1, 3)
        if pts is not None and len(pts):
            geoms[g.get('id')] = pts
    if not geoms:
        return None

    allp = []

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
                allp.append(tf_pts(T, geoms[gid]))
        for ch in n.findall(q('node')):
            walk(ch, T)

    scenes = list(root.iter(q('visual_scene')))
    for sc in scenes:
        for n in sc.findall(q('node')):
            walk(n, np.eye(4))
    if not allp:   # 장면 정보가 없으면 꼭짓점 그대로
        allp = list(geoms.values())
    P = np.vstack(allp) * unit
    if up == 'Y_UP':
        P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)
    elif up == 'X_UP':
        P = np.stack([-P[:, 1], P[:, 0], P[:, 2]], axis=1)
    return P.min(0), P.max(0)


# ---------------------------------------------------------------- world 읽기
def geom_bbox(geom, resolver, warn):
    """visual 좌표계에서 (lo, hi, 설명). 모르면 None"""
    if geom is None:
        return None
    b = geom.find('box/size')
    if b is not None:
        s = np.array([float(x) for x in b.text.split()])
        return -s / 2, s / 2, 'box %s' % ' '.join('%.3f' % v for v in s)
    c = geom.find('cylinder')
    if c is not None:
        r = float(c.findtext('radius', '0')); l = float(c.findtext('length', '0'))
        return np.array([-r, -r, -l / 2]), np.array([r, r, l / 2]), 'cyl r%.3f l%.3f' % (r, l)
    s = geom.find('sphere')
    if s is not None:
        r = float(s.findtext('radius', '0'))
        return -np.ones(3) * r, np.ones(3) * r, 'sphere r%.3f' % r
    p = geom.find('plane')
    if p is not None:
        n = np.array([float(x) for x in (p.findtext('normal') or '0 0 1').split()])
        sz = [float(x) for x in (p.findtext('size') or '1 1').split()]
        ax = int(np.argmax(np.abs(n)))
        half = np.zeros(3)
        others = [i for i in range(3) if i != ax]
        half[others[0]], half[others[1]] = sz[0] / 2, sz[1] / 2
        half[ax] = 0.0005
        return -half, half, 'plane %s' % ' '.join('%.3f' % v for v in sz)
    m = geom.find('mesh')
    if m is not None:
        uri = m.findtext('uri', '')
        sc = np.array([float(x) for x in (m.findtext('scale') or '1 1 1').split()])
        path = resolver.resolve(uri)
        bb = None
        if path and path.lower().endswith('.dae'):
            bb = dae_bbox(path)
        elif path and path.lower().endswith('.stl'):
            bb = stl_bbox(path)
        if bb is None:
            warn.append('메시 크기 모름: %s (%s)' % (uri, '파일 없음' if not path else '읽기 실패'))
            return None
        lo, hi = bb[0] * sc, bb[1] * sc
        return np.minimum(lo, hi), np.maximum(lo, hi), 'mesh %s' % os.path.basename(uri)
    return None


def material_name(vis):
    n = vis.findtext('material/script/name')
    if n:
        return n.strip()
    t = vis.findtext('material/pbr/metal/albedo_map')
    return os.path.basename(t.strip()) if t else ''


def scan_models(world_root, resolver, warn):
    """반환: 모델 목록 [{name, T, static, visuals:[{name, T(model 기준), lo, hi, desc, mat}]}]"""
    out = []

    def read_model(el, T_parent, name_prefix, top_name):
        name = el.get('name') or name_prefix
        T_m = T_parent @ pose_T(el.find('pose'))
        mod = {'name': name, 'top': top_name or name, 'T': T_m, 'visuals': []}
        for link in el.findall('link'):
            T_l = pose_T(link.find('pose'))
            for vis in link.findall('visual'):
                T_v = T_l @ pose_T(vis.find('pose'))
                bb = geom_bbox(vis.find('geometry'), resolver, warn)
                mod['visuals'].append({'name': '%s/%s' % (link.get('name'), vis.get('name')),
                                       'T': T_v, 'bb': bb, 'mat': material_name(vis)})
        out.append(mod)
        for sub in el.findall('model'):
            read_model(sub, T_m, sub.get('name'), top_name or name)
        for inc in el.findall('include'):
            read_include(inc, T_m, top_name or name)

    def read_include(inc, T_parent, top_name):
        uri = inc.findtext('uri', '')
        T_i = T_parent @ pose_T(inc.find('pose'))
        name = inc.findtext('name') or os.path.basename(uri.rstrip('/'))
        d = resolver.resolve(uri)
        sdf = None
        if d and os.path.isdir(d):
            for cand in ('model.sdf', 'model-1_7.sdf', 'model-1_6.sdf'):
                if os.path.exists(os.path.join(d, cand)):
                    sdf = os.path.join(d, cand)
                    break
        if sdf is None:
            warn.append('include 모델 파일 못 찾음: %s (%s)' % (name, uri))
            out.append({'name': name, 'top': top_name or name, 'T': T_i, 'visuals': []})
            return
        m = ET.parse(sdf).getroot().find('model')
        if m is None:
            return
        m.set('name', name)
        p = m.find('pose')
        if p is not None:          # <include> 의 pose 가 모델 자체 pose 를 대신함
            m.remove(p)
        read_model(m, T_i, name, top_name)

    world = world_root.find('world') if world_root.tag == 'sdf' else world_root
    for el in world.findall('model'):
        read_model(el, np.eye(4), el.get('name'), None)
    for inc in world.findall('include'):
        read_include(inc, np.eye(4), None)
    return out


def classify(name, rules):
    low = name.lower()
    for item in rules:
        cls, groups = item[0], item[1]
        excl = item[2] if len(item) > 2 else ()
        if any(k in low for k in excl):
            continue
        if any(all(k in low for k in g) for g in groups):
            return cls
    return None


def visual_box_in_model(v):
    lo, hi = v['bb'][0], v['bb'][1]
    P = tf_pts(v['T'], box_corners(lo, hi))
    return P.min(0), P.max(0)


FACE_DEFAULT = {'triangle': [0.01, 0.24, 0.21], 'round': [0.01, 0.224, 0.224]}


def sign_face(mod, warn):
    """표지판 그림 면: (center, size, front) 모델 좌표계. 실패 시 None"""
    vis = [v for v in mod['visuals'] if v['bb'] is not None]
    pf = [v for v in mod['visuals'] if v['bb'] is None and 'print' in v['name'].lower()]
    if pf and not any('print' in v['name'].lower() for v in vis):
        # 그림 메시를 못 읽음 → printed_face 위치에 기본 크기 (make_arena.py: 앞면 = 모델 +x)
        kind = 'triangle' if mod['name'].lower().find('ped') >= 0 or 'bump' in mod['name'].lower() else 'round'
        warn.append('%s: 그림 메시를 못 읽어 기본 크기 %s 사용 → --mesh-root 확인, debug 사진으로 박스 확인 필수'
                    % (mod['name'], FACE_DEFAULT[kind]))
        return pf[0]['T'][:3, 3].copy(), np.array(FACE_DEFAULT[kind]), '+x', pf[0]['name'] + '(기본크기)'
    if not vis:
        return None
    named = [v for v in vis if any(k in v['name'].lower() for k in FACE_KEYS)]
    pool = named or [v for v in vis if not any(k in v['name'].lower() for k in NOT_FACE_KEYS)] or vis
    boxes = [(v,) + visual_box_in_model(v) for v in pool]
    # 얇은 판 중 넓이가 가장 큰 것 (그림 면이 받침판보다 작으면 named 로 이미 골라짐)
    def flat_area(b):
        s = b[2] - b[1]
        srt = np.sort(s)
        return srt[1] * srt[2] if srt[0] < 0.03 else 0.0
    boxes.sort(key=flat_area, reverse=True)
    v, lo, hi = boxes[0]
    size = hi - lo
    center = (hi + lo) / 2
    ax = int(np.argmin(size))
    if size[ax] > 0.05:
        warn.append('%s: 얇은 그림 면을 못 찾음 (가장 얇은 visual %s 두께 %.3f) → 앞면 +x 가정'
                    % (mod['name'], v['name'], size[ax]))
        return center, size, '+x', v['name']
    # 부호: 다른 visual(받침판·기둥) 중심보다 그림 면이 튀어나온 쪽, 없으면 모델 원점 기준
    others = [visual_box_in_model(o) for o in vis if o is not v]
    if others:
        ref = np.mean([(l + h) / 2 for l, h in others], axis=0)[ax]
    else:
        ref = 0.0
    diff = center[ax] - ref
    if abs(diff) < 1e-3:
        warn.append('%s: 앞/뒷면 구분 불가 → %s+ 가정 (디버그 사진에서 뒷면 라벨 여부 확인)'
                    % (mod['name'], 'xyz'[ax]))
        diff = 1.0
    front = ('+' if diff > 0 else '-') + 'xyz'[ax]
    size = size.copy()
    size[ax] = max(size[ax], 0.01)
    return center, size, front, v['name']


def axis_vec(front):
    v = np.zeros(3)
    v['xyz'.index(front[1])] = 1.0 if front[0] == '+' else -1.0
    return v


def r3(a):
    return [round(float(x), 4) for x in a]


# ---------------------------------------------------------------- 메인
def latest_world(pattern):
    hits = glob.glob(os.path.expanduser(pattern))
    return max(hits, key=os.path.getmtime) if hits else None


def build(world_path, mesh_roots, cluster_m=0.8):
    warn = []
    resolver = Resolver(list(mesh_roots) + [os.path.dirname(world_path)] +
                        os.environ.get('GAZEBO_MODEL_PATH', '').split(':'))
    root = ET.parse(world_path).getroot()
    models = scan_models(root, resolver, warn)
    text = open(world_path).read()
    sign_ab = re.search(r'<model name="branch_sign_panel">.*?Kudos/Sign_([AB])\b', text, re.S)
    sign_ab = sign_ab.group(1) if sign_ab else None

    objs = []          # 라벨 대상
    others = []        # 리포트·금지 구역
    parts = {}         # 횡단보도·유도선 조각 (월드 좌표 꼭짓점)
    used = set()
    for mod in models:
        nm = mod['name']
        if nm in KNOWN:
            k = KNOWN[nm]
            T = mod['T']
            c = tf_pts(T, np.array([k['offset']]))[0]
            n = T[:3, :3] @ axis_vec(k['front'])
            objs.append({'class': k['class'], 'model': nm, 'size': k['size'], 'offset': k['offset'],
                         'front': k['front'], 'center': c, 'normal_yaw': math.atan2(n[1], n[0]),
                         'how': '10/29 확인값'})
            used.add(nm)
            continue
        cls = classify(nm, CLASS_RULES)
        if cls is None or cls in ('crosswalk', 'slow_guide_line'):
            # 큰 모델(트랙 바닥 등) 안의 visual 이름도 본다
            for v in mod['visuals']:
                vc = classify(v['name'], CLASS_RULES) if cls is None else cls
                if vc in ('crosswalk', 'slow_guide_line') and v['bb'] is not None:
                    P = tf_pts(mod['T'] @ v['T'], box_corners(*v['bb'][:2]))
                    yaw = yaw_of_R((mod['T'] @ v['T'])[:3, :3])
                    parts.setdefault(vc, []).append({'src': '%s:%s' % (nm, v['name']), 'P': P, 'yaw': yaw})
                    used.add(nm)
            if cls in ('crosswalk', 'slow_guide_line') and nm not in used:
                warn.append('%s: %s 로 보이나 크기를 모름' % (nm, cls))
            if cls is None and nm not in used:
                oc = classify(nm, OTHER_RULES)
                others.append({'kind': oc or '-', 'model': nm, 'xy': mod['T'][:2, 3].copy(),
                               'n_vis': len(mod['visuals'])})
            continue
        used.add(nm)
        T = mod['T']
        if cls == 'limo':
            vis = [v for v in mod['visuals'] if v['bb'] is not None]
            if vis:
                bbs = [visual_box_in_model(v) for v in vis]
                lo = np.min([b[0] for b in bbs], axis=0); hi = np.max([b[1] for b in bbs], axis=0)
                size, off, how = hi - lo, (hi + lo) / 2, 'visual 크기'
                if size.max() > 0.6 or size.min() < 0.05:
                    warn.append('%s: LIMO 크기가 이상함 %s → 기본값 사용' % (nm, r3(size)))
                    size, off, how = np.array(LIMO_DEFAULT['size']), np.array(LIMO_DEFAULT['offset']), '기본값'
            else:
                size, off, how = np.array(LIMO_DEFAULT['size']), np.array(LIMO_DEFAULT['offset']), '기본값(visual 없음)'
            c = tf_pts(T, np.array([off]))[0]
            objs.append({'class': 'limo', 'model': nm, 'size': r3(size), 'offset': r3(off), 'front': '',
                         'center': c, 'normal_yaw': yaw_of_R(T[:3, :3]), 'how': how})
            continue
        face = sign_face(mod, warn)
        if face is None:
            warn.append('%s (%s): visual 크기를 하나도 못 구함 → 대상에서 뺌' % (nm, cls))
            continue
        center, size, front, vname = face
        c = tf_pts(T, np.array([center]))[0]
        n = T[:3, :3] @ axis_vec(front)
        objs.append({'class': cls, 'model': nm, 'size': r3(size), 'offset': r3(center), 'front': front,
                     'center': c, 'normal_yaw': math.atan2(n[1], n[0]), 'how': 'visual %s' % vname})

    # 횡단보도·유도선 조각 묶기
    for cls, plist in parts.items():
        groups = []
        for pt in plist:
            ctr = pt['P'].mean(0)
            for g in groups:
                if min(np.linalg.norm(ctr[:2] - q['P'].mean(0)[:2]) for q in g) < cluster_m:
                    g.append(pt)
                    break
            else:
                groups.append([pt])
        for gi, g in enumerate(groups):
            yaw = g[0]['yaw']
            Rz = rpy_to_R(0, 0, yaw)
            P = np.vstack([q['P'] for q in g])
            ctr0 = P.mean(0)
            L = (P - ctr0) @ Rz          # 묶음 좌표계 (yaw 만)
            lo, hi = L.min(0), L.max(0)
            size = hi - lo
            size[2] = max(size[2], 0.004)
            c = ctr0 + Rz @ ((hi + lo) / 2)
            # 짧은 수평축 = 차가 지나가는 방향
            ax = 0 if size[0] < size[1] else 1
            n = Rz[:, ax]
            objs.append({'class': cls, 'model': '%s_%d' % (cls, gi), 'world_pose': r3([c[0], c[1], c[2], yaw]),
                         'size': r3(size), 'front': '', 'center': c, 'normal_yaw': math.atan2(n[1], n[0]),
                         'along': float(size[ax]), 'across': float(size[1 - ax]),
                         'how': '%d 조각 묶음 (%s ...)' % (len(g), g[0]['src'])})
    return objs, others, warn, sign_ab, models


def targets_yaml(objs, sign, include):
    lines = ['# world_targets.py 가 만든 auto_labeler 대상 — 갈림길 표지판 %s 실행분. 직접 고치지 말고 다시 생성' % sign,
             'classes:']
    lines += ['  - %s   # %d' % (c, i) for i, c in enumerate(CLASSES)]
    lines.append('targets:')
    for o in objs:
        cls = o['class']
        if cls == 'BRANCH':
            cls = 'marker_left' if sign == 'A' else 'marker_right'
        if cls not in include:
            continue
        if 'world_pose' in o:
            lines.append('  - {class: %s, world_pose: %s, size: %s}   # %s' % (cls, o['world_pose'], o['size'], o['how']))
        else:
            fr = (", front: '%s'" % o['front']) if o['front'] else ''
            lines.append('  - {class: %s, gazebo_name: %s, size: %s, offset: %s%s}   # %s'
                         % (cls, o['model'], list(o['size']), list(o['offset']), fr, o['how']))
    return '\n'.join(lines) + '\n'


def viewpoints_yaml(objs, others, include):
    regions, avoid = [], []
    seen_cls = {}
    for o in objs:
        cls = 'branch' if o['class'] == 'BRANCH' else o['class']
        if o['class'] != 'BRANCH' and o['class'] not in include:
            continue
        tx, ty = float(o['center'][0]), float(o['center'][1])
        k = seen_cls.get(cls, 0)
        seen_cls[cls] = k + 1
        name = cls if k == 0 else '%s_%d' % (cls, k)
        if o['class'] == 'limo':
            avoid.append({'xy': [round(tx, 3), round(ty, 3)], 'r': 0.4})
            continue
        if o['class'] in ('crosswalk', 'slow_guide_line'):
            half = o['along'] / 2
            lat = max(0.05, min(0.25, o['across'] / 2 - 0.1))
            for s, nm in ((0.0, name + '_a'), (math.pi, name + '_b')):
                regions.append({'name': nm, 'type': 'facing', 'target': [round(tx, 3), round(ty, 3)],
                                'normal_yaw': round(o['normal_yaw'] + s, 4),
                                'dist': [round(half + 0.25, 3), round(half + 2.0, 3)],
                                'lateral': [-round(lat, 3), round(lat, 3)], 'yaw_jitter': 0.25, 'weight': 1.0})
            continue
        far = 2.5 if o['class'] == 'traffic_light' else 3.0
        w = {'BRANCH': 2.0, 'parking_sign': 1.0, 'traffic_light': 1.5}.get(o['class'], 1.5)
        regions.append({'name': name, 'type': 'facing', 'target': [round(tx, 3), round(ty, 3)],
                        'normal_yaw': round(o['normal_yaw'], 4), 'dist': [0.6, far], 'lateral': [-0.5, 0.5],
                        'yaw_jitter': 0.35, 'weight': w})
        avoid.append({'xy': [round(tx, 3), round(ty, 3)], 'r': 0.3})
    lim = [o for o in objs if o['class'] == 'limo' and 'traffic_car' in o['model']]
    if lim and 'limo' in include:
        xs = [float(o['center'][0]) for o in lim]; ys = [float(o['center'][1]) for o in lim]
        regions.append({'name': 'limo_lanes', 'type': 'box', 'x': [round(min(xs) - 0.8, 2), round(max(xs) + 1.5, 2)],
                        'y': [round(min(ys) - 0.1, 2), round(max(ys) + 0.1, 2)],
                        'yaw': round(math.pi, 4),          # 아래 직선은 -x 로 달림 (10/29 확인)
                        'yaw_jitter': 0.5, 'weight': 1.5})
    for o in others:
        if o['kind'] in ('pedestrian_figure', 'speed_bump', 'cone'):
            avoid.append({'xy': [round(float(o['xy'][0]), 3), round(float(o['xy'][1]), 3)],
                          'r': 0.35 if o['kind'] != 'cone' else 0.2})
    return {'bounds': [ARENA[0] + 0.15, ARENA[1] - 0.15, ARENA[2] + 0.15, ARENA[3] - 0.15],
            'regions': regions, 'avoid': avoid}


def mission_points(objs, include):
    pts = []
    for o in objs:
        cls = o['class']
        if cls == 'BRANCH':
            cls = 'branch_sign'
        elif cls not in include or cls == 'limo':
            continue
        n = o['normal_yaw']
        tx, ty = float(o['center'][0]), float(o['center'][1])
        if cls in ('crosswalk', 'slow_guide_line'):
            half = o['along'] / 2
            dists = [1.5, 1.0, 0.5, 0.25]
            for s in (0.0, math.pi):
                for d in dists:
                    a = n + s
                    x, y = tx + (half + d) * math.cos(a), ty + (half + d) * math.sin(a)
                    pts.append({'name': '%s %s %.2fm' % (o['model'], 'ab'[int(s > 0)], d), 'class': cls,
                                'x': round(x, 3), 'y': round(y, 3), 'yaw': round(math.atan2(ty - y, tx - x), 4),
                                'expect_x': d})
            continue
        for d in ([2.0, 1.2, 0.7] if cls != 'traffic_light' else [2.0, 1.2, 0.8]):
            x, y = tx + d * math.cos(n), ty + d * math.sin(n)
            pts.append({'name': '%s %.1fm' % (o['model'], d), 'class': cls, 'x': round(x, 3), 'y': round(y, 3),
                        'yaw': round(math.atan2(ty - y, tx - x), 4), 'expect_x': d})
    return {'points': pts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--world', default='auto', help='world 파일. auto = /tmp/kudos_arena_*/track_2026.world 중 최신')
    ap.add_argument('--mesh-root', action='append', default=None,
                    help='메시·include 모델을 찾을 폴더 (여러 번 가능). 기본 ~/26v2/KUDOS_Autorace2026_Gazebo')
    ap.add_argument('--out', default='~/yolo_data/v2/config_full')
    ap.add_argument('--classes', default='traffic_light,marker_left,marker_right,parking_sign,pedestrian_sign,'
                                         'slow_sign,bump_sign,crosswalk,limo',
                    help='라벨에 넣을 클래스 (쉼표). 기본은 slow_guide_line 제외')
    a = ap.parse_args()
    world = latest_world('/tmp/kudos_arena_*/track_2026.world') if a.world == 'auto' else os.path.expanduser(a.world)
    if not world or not os.path.exists(world):
        sys.exit('world 파일이 없음. Gazebo(arena/competition.launch.py)를 한 번 띄우거나 --world 로 지정')
    roots = a.mesh_root or ['~/26v2/KUDOS_Autorace2026_Gazebo']
    include = set(c.strip() for c in a.classes.split(',') if c.strip())

    objs, others, warn, sign_ab, models = build(world, roots)
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    for s in ('A', 'B'):
        with open(os.path.join(out, 'targets_full_%s.yaml' % s), 'w') as f:
            f.write(targets_yaml(objs, s, include | {'marker_left' if s == 'A' else 'marker_right'}))
    with open(os.path.join(out, 'viewpoints_full.yaml'), 'w') as f:
        f.write('# world_targets.py 가 만든 viewpoint_sampler 구역 (%s)\n' % world)
        yaml.safe_dump(viewpoints_yaml(objs, others, include), f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)
    with open(os.path.join(out, 'mission_points.yaml'), 'w') as f:
        f.write('# mission_check.py 가 차를 세울 위치 (물체 앞 거리 expect_x)\n')
        yaml.safe_dump(mission_points(objs, include), f, allow_unicode=True, sort_keys=False,
                       default_flow_style=None)

    rep = ['world: %s' % world, '갈림길 표지판 그림: %s' % (sign_ab or '못 읽음'), '',
           '[라벨 대상 %d개]' % len(objs)]
    for o in objs:
        cls = 'marker_left/right' if o['class'] == 'BRANCH' else o['class']
        flag = '' if (o['class'] == 'BRANCH' or o['class'] in include) else '  (이번 라벨에서 제외)'
        rep.append('  %-16s %-24s 위치 (%.2f, %.2f, %.2f) 앞면 %4.0f°  크기 %s  ← %s%s'
                   % (cls, o['model'], o['center'][0], o['center'][1], o['center'][2],
                      math.degrees(o['normal_yaw']), o['size'], o['how'], flag))
    missing = [c for c in ('pedestrian_sign', 'bump_sign', 'slow_sign', 'crosswalk', 'limo')
               if not any(o['class'] == c for o in objs)]
    rep += ['', '[월드에 없는 클래스] %s' % (', '.join(missing) or '없음 (전부 찾음)'), '',
            '[분류 안 된 모델 %d개] (이름에 미션 물체가 숨어 있으면 알려주세요)' % len(others)]
    for o in others:
        rep.append('  %-18s %-30s (%.2f, %.2f) visual %d' % (o['kind'], o['model'], o['xy'][0], o['xy'][1], o['n_vis']))
    uniq = []
    for w in warn:
        if w not in uniq:
            uniq.append(w)
    rep += ['', '[경고 %d]' % len(uniq)] + ['  %s%s' % (w, ' (x%d)' % warn.count(w) if warn.count(w) > 1 else '')
                                            for w in uniq]
    rep += ['', '저장: %s' % out]
    txt = '\n'.join(rep)
    with open(os.path.join(out, 'world_report.txt'), 'w') as f:
        f.write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
