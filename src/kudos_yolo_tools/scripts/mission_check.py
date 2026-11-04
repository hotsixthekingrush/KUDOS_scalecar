#!/usr/bin/env python3
"""
mission_check.py — Gazebo 에서 미션 1~5 YOLO 인식 점검표 (11/4)

로봇을 미션 물체 앞 여러 거리(mission_points.yaml)로 순간이동시키고, 그 자리에서
/perception/mission_objects 를 몇 장 받아 "기대한 클래스가 나왔는지, 거리(x_m)가 맞는지,
신호등 색"을 표로 출력한다. 판단(FSM) 없이 인지 출력만 본다.

준비
  터미널 1: Gazebo  (ros2 launch kudos_arena competition.launch.py branch_sign:=A)
            ※ 차를 움직이는 노드(판단·제어·teleop)는 끄기 — 순간이동한 차가 움직이면 거리가 틀어짐
  터미널 2: YOLO    (ros2 launch kudos_yolo_tools yolo_detector_sim.launch.py weights:=$HOME/.../best.pt)
  터미널 3: python3 world_targets.py   (한 번만. mission_points.yaml 생성)
            ros2 run kudos_yolo_tools mission_check.py --ros-args -p use_sim_time:=true \
                -p points_file:=$HOME/yolo_data/v2/config_full/mission_points.yaml

  -p only:=crosswalk,traffic_light   이 클래스 지점만
  -p frames:=5                        지점마다 받을 결과 수 (기본 5)

판정
  OK    : 기대 클래스가 받은 결과의 60% 이상에서 나옴 + (위치가 있으면) |x_m - 기대| ≤ 허용
  약함  : 나오긴 했지만 60% 미만이거나 거리 오차가 큼
  없음  : 한 번도 안 나옴
  허용 오차: 횡단보도 max(5cm, 10%) — 3cm 정지라 가장 엄격 / 나머지 max(15cm, 15%)
결과는 화면 + ~/yolo_data/v2/mission_check_<시각>.csv (지점마다 바로 저장). 다 돌면 스스로 종료
"""
import csv
import glob
import math
import os
import re
import time

import yaml
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState

from ar_msgs.msg import MissionObject, MissionObjectArray

# mission_points 의 class → 기대하는 MissionObject 상수 이름 (여러 개면 그중 하나)
EXPECT = {
    'traffic_light': ['TRAFFIC_LIGHT'],
    'branch_sign': ['SIGN_A', 'SIGN_B', 'MARKER_LEFT', 'MARKER_RIGHT'],
    'parking_sign': ['PARKING_SIGN'],
    'pedestrian_sign': ['PEDESTRIAN_SIGN'],
    'bump_sign': ['BUMP_SIGN'],
    'slow_sign': ['SLOW_SIGN'],
    'crosswalk': ['CROSSWALK'],
    'slow_guide_line': ['SLOW_GUIDE_LINE'],
}
MISSION = {'traffic_light': 'M1', 'crosswalk': 'M1/M3', 'branch_sign': 'M2', 'pedestrian_sign': 'M3',
           'parking_sign': 'M4', 'bump_sign': 'M5', 'slow_sign': 'M5', 'slow_guide_line': 'M6'}
LIGHT = {0: '모름', 1: '빨강', 2: '노랑', 3: '초록'}


def const_names():
    out = {}
    for k in dir(MissionObject):
        if k.isupper() and not k.startswith('POS_') and not k.startswith('_'):
            v = getattr(MissionObject, k)
            if isinstance(v, int):
                out[v] = k
    return out


def world_branch_sign(pattern='/tmp/kudos_arena_*/track_2026.world'):
    hits = glob.glob(pattern)
    if not hits:
        return None
    text = open(max(hits, key=os.path.getmtime)).read()
    m = re.search(r'<model name="branch_sign_panel">.*?Kudos/Sign_([AB])\b', text, re.S)
    return m.group(1) if m else None


class MissionCheck(Node):
    def __init__(self):
        super().__init__('mission_check')
        p = self.declare_parameter
        p('points_file', os.path.expanduser('~/yolo_data/v2/config_full/mission_points.yaml'))
        p('robot', 'limo')
        p('frames', 5)
        p('settle_sec', 0.6)        # 순간이동 후 이만큼(시뮬 시간) 지난 뒤 찍힌 결과부터 인정
        p('timeout_sec', 25.0)      # 실제 시간
        p('only', '')
        p('topic', '/perception/mission_objects')
        g = lambda n: self.get_parameter(n).value
        with open(os.path.expanduser(g('points_file'))) as f:
            pts = yaml.safe_load(f)['points']
        only = [s.strip() for s in str(g('only')).split(',') if s.strip()]
        self.points = [q for q in pts if not only or q['class'] in only]
        self.robot = g('robot')
        self.frames = int(g('frames'))
        self.settle_ns = int(float(g('settle_sec')) * 1e9)
        self.timeout = float(g('timeout_sec'))
        self.names = const_names()
        self.sign = world_branch_sign()
        if self.sign:
            self.get_logger().info('월드 갈림길 표지판 = %s → 기대 %s' % (self.sign, 'SIGN_' + self.sign))

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.robot_z = None
        self.create_subscription(ModelStates, '/gazebo/model_states', self.states_cb, qos_profile_sensor_data)
        q = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(MissionObjectArray, g('topic'), self.obj_cb, q)
        self.i = -1
        self.phase = 'wait'
        self.buf = []
        self.rows = []
        self.t_move = 0
        self.deadline = 0.0
        self.done = False
        self.t0 = time.monotonic()
        self.seen_models = []
        self.n_msgs = 0
        self.csv_path = os.path.expanduser('~/yolo_data/v2/mission_check_%s.csv' % time.strftime('%m%d_%H%M'))
        # 시뮬 시간이 느리거나 멈춰도 돌도록 벽시계 타이머 (11/4 저녁: 점검표가 안 나온 문제 대응)
        self.create_timer(0.1, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('점검 지점 %d개 (%s)' % (len(self.points), g('points_file')))

    def states_cb(self, msg):
        self.seen_models = list(msg.name)
        if self.robot_z is None and self.robot in msg.name:
            self.robot_z = msg.pose[msg.name.index(self.robot)].position.z

    def obj_cb(self, msg):
        self.n_msgs += 1
        if self.phase != 'collect':
            return
        st = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if st >= self.t_move + self.settle_ns:
            self.buf.append(msg)

    def teleport(self, x, y, yaw):
        s = EntityState()
        s.name = self.robot
        s.reference_frame = 'world'
        s.pose.position.x, s.pose.position.y, s.pose.position.z = float(x), float(y), float(self.robot_z + 0.01)
        s.pose.orientation.z, s.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        req = SetEntityState.Request()
        req.state = s
        self.cli.call_async(req)

    def tick(self):
        if self.phase == 'wait':
            if self.robot_z is None or not self.cli.service_is_ready():
                self.get_logger().info('기다리는 중: 로봇(%s) 위치 %s / set_entity_state %s / YOLO 결과 %d개 받음 / 보이는 모델 %s'
                                       % (self.robot, 'OK' if self.robot_z is not None else '없음',
                                          'OK' if self.cli.service_is_ready() else '없음', self.n_msgs,
                                          self.seen_models[:8]), throttle_duration_sec=5.0)
                if time.monotonic() - self.t0 > 120:
                    self.get_logger().error('2분 동안 준비 안 됨 → 종료. Gazebo/로봇 이름(-p robot:=) 확인')
                    self.finish()
                return
            self.phase = 'next'
        if self.phase == 'next':
            self.i += 1
            if self.i >= len(self.points):
                self.finish()
                self.phase = 'done'
                return
            q = self.points[self.i]
            self.get_logger().info('[%d/%d] %s 로 이동 (지금까지 YOLO 결과 %d개)'
                                   % (self.i + 1, len(self.points), q['name'], self.n_msgs))
            self.teleport(q['x'], q['y'], q['yaw'])
            self.t_move = self.get_clock().now().nanoseconds
            self.buf = []
            self.deadline = time.monotonic() + self.timeout
            self.phase = 'collect'
        elif self.phase == 'collect':
            if len(self.buf) >= self.frames or time.monotonic() > self.deadline:
                self.judge(self.points[self.i], self.buf[:self.frames])
                self.phase = 'next'

    def judge(self, q, msgs):
        cls = q['class']
        want = list(EXPECT.get(cls, []))
        if cls == 'branch_sign' and self.sign:
            want = ['SIGN_' + self.sign, 'MARKER_' + ('LEFT' if self.sign == 'A' else 'RIGHT')]
        n = len(msgs)
        hits, xs, ys, confs, colors, wrong = 0, [], [], [], {}, {}
        for m in msgs:
            got = None
            for o in m.objects:
                nm = self.names.get(int(o.class_id), str(o.class_id))
                if nm in want:
                    if got is None or o.confidence > got.confidence:
                        got = o
                elif nm != 'LIMO':
                    wrong[nm] = wrong.get(nm, 0) + 1
            if got is None:
                continue
            hits += 1
            confs.append(got.confidence)
            if got.position_valid:
                xs.append(got.x_m)
                ys.append(got.y_m)
            if cls == 'traffic_light' and hasattr(got, 'light_color'):
                c = LIGHT.get(int(got.light_color), str(got.light_color))
                colors[c] = colors.get(c, 0) + 1
        ex = float(q['expect_x'])
        tol = max(0.05, 0.10 * ex) if cls == 'crosswalk' else max(0.15, 0.15 * ex)
        mx = sum(xs) / len(xs) if xs else None
        my = sum(ys) / len(ys) if ys else None
        if n == 0:
            verdict = '결과없음'
        elif hits == 0:
            verdict = '없음'
        elif hits / n >= 0.6 and (mx is None or abs(mx - ex) <= tol):
            verdict = 'OK'
        else:
            verdict = '약함'
        if cls == 'crosswalk' and hits and not xs and ex <= 0.3:
            note = '위치 없음(화면 하단 닿음 → 판단이 odom 으로)'
        else:
            note = ''
        row = {'mission': MISSION.get(cls, '-'), 'point': q['name'], 'class': cls, 'verdict': verdict,
               'hit': '%d/%d' % (hits, n), 'conf': '%.2f' % (sum(confs) / len(confs)) if confs else '-',
               'x_m': '%.2f' % mx if mx is not None else '-', 'expect_x': '%.2f' % ex,
               'err_cm': '%+.0f' % ((mx - ex) * 100) if mx is not None else '-',
               'y_m': '%.2f' % my if my is not None else '-',
               'light': ' '.join('%s%d' % kv for kv in colors.items()) or '-',
               'other': ' '.join('%s%d' % kv for kv in wrong.items()) or '-', 'note': note}
        self.rows.append(row)
        new = not os.path.exists(self.csv_path)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, 'a', newline='') as f:     # 지점마다 바로 저장 (중간에 끊겨도 남게)
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        self.get_logger().info('[%s] %-28s %-5s %s conf %s x %s (기대 %s, %s cm) y %s 색 %s 다른것 %s %s'
                               % (row['mission'], row['point'], row['verdict'], row['hit'], row['conf'],
                                  row['x_m'], row['expect_x'], row['err_cm'], row['y_m'], row['light'],
                                  row['other'], note))

    def finish(self):
        P = lambda *a: print(*a, flush=True)
        P('\n================ 미션 인식 점검표 ================')
        P('%-6s %-30s %-6s %-6s %-5s %-6s %-6s %-6s %-12s %s'
          % ('미션', '지점', '판정', '검출', 'conf', 'x_m', '기대', '오차cm', '신호등색', '다른 클래스'))
        for r in self.rows:
            P('%-6s %-30s %-6s %-6s %-5s %-6s %-6s %-6s %-12s %s %s'
              % (r['mission'], r['point'], r['verdict'], r['hit'], r['conf'], r['x_m'], r['expect_x'],
                 r['err_cm'], r['light'], r['other'], r['note']))
        by = {}
        for r in self.rows:
            by.setdefault(r['class'], []).append(r['verdict'])
        P('\n클래스별 OK: ' + ', '.join('%s %d/%d' % (c, v.count('OK'), len(v)) for c, v in by.items()))
        P('YOLO 결과 받은 수: %d | 저장: %s\n' % (self.n_msgs, self.csv_path if self.rows else '(없음)'))
        self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = MissionCheck()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
