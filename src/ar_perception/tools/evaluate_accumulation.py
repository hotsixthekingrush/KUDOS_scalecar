"""Offline replay of actual on_scan, without DDS; truth used ONLY for scoring.

Run from workspace root with ROS environment sourced. See --help.
"""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from collections import Counter

import numpy as np
import yaml
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from ar_perception.lidar_perception import LidarPerception
from ar_perception.point_accumulation import DEFAULTS, PointAccumulator, OdomHistory, transform


class Sink:
    def __init__(self):
        self.msg = None
    def publish(self, msg):
        self.msg = msg


class Logger:
    def info(self, *a, **k):
        pass
    warn = error = debug = info


def offline_node(cls, params):
    node = cls.__new__(cls)
    node.get_parameter = lambda k: SimpleNamespace(value=params[k])
    node.get_logger = lambda: Logger()
    for name in ('obstacles', 'event', 'walls', 'parking_gap', 'debug', 'clusters', 'accumulation_debug'):
        setattr(node, 'pub_' + name, Sink())
    node.latest_odom = None
    node.parking_boundaries = []
    node.parking_frame_id = 0
    node.previous_parking_gap = None
    node.parking_gap_confirm_count = 0
    node.confirmed_parking_gap = None
    node.parking_gap_id_counter = 0
    node.active_parking_gap_id = 0
    node.accumulation_odom = OdomHistory()
    node.point_accumulator = PointAccumulator()
    node.accumulation_epoch = 0
    return node


def defaults(source, config):
    params = dict(DEFAULTS)
    for n in ast.walk(ast.parse(Path(source).read_text())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'declare_parameter' and len(n.args) == 2:
            try:
                params[ast.literal_eval(n.args[0])] = ast.literal_eval(n.args[1])
            except (ValueError, TypeError):
                pass
    params.update(yaml.safe_load(Path(config).read_text())['lidar_perception']['ros__parameters'])
    return params


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def stats(values, errors=True):
    if not values:
        return {}
    a = np.array(values)
    result = {'n': len(a), 'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'max': float(np.max(a)), 'over_5cm_pct': float(100*np.mean(a > .05))}
    if not errors:
        result.pop('over_5cm_pct')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--truth', help='JSON mapping name -> [world x, world y]; only scorer reads it')
    ap.add_argument('--output', required=True)
    ap.add_argument('--config', default='src/ar_perception/config/lidar.yaml')
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('baseline_lidar', args.backup)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = defaults('src/ar_perception/ar_perception/lidar_perception.py', args.config)
    nodes = [offline_node(mod.LidarPerception, dict(params)),
             offline_node(LidarPerception, dict(params, use_point_accumulation=False)),
             offline_node(LidarPerception, dict(params, use_point_accumulation=True))]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in ('/scan', '/odom')}
    truth = json.loads(Path(args.truth).read_text()) if args.truth else {}
    rows, durations, reasons = [], [[], [], []], Counter()
    scans = 0
    invariance = True
    side_effects = True
    unmatched = 0
    scan_steps = []
    prev = None
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(raw, types[topic])
        if topic == '/odom':
            for node in nodes:
                node.odom_callback(msg)
            continue
        scans += 1
        if prev is not None:
            scan_steps.append(stamp(msg)-prev)
        prev = stamp(msg)
        for i, node in enumerate(nodes):
            t = time.perf_counter()
            node.on_scan(msg)
            durations[i].append(1000*(time.perf_counter()-t))
        for key in ('obstacles', 'event', 'walls', 'parking_gap', 'debug', 'clusters'):
            a, b, c = [getattr(n, 'pub_'+key).msg for n in nodes]
            invariance &= a == b
            if key != 'obstacles':
                side_effects &= b == c
        raw_debug = nodes[2].pub_accumulation_debug.msg
        diagnostics = json.loads(raw_debug.data)['candidates'] if raw_debug else []
        reasons.update(d.get('reason', '?') for d in diagnostics)
        by_index = {d['obstacle_index']: d for d in diagnostics if 'obstacle_index' in d}
        base = nodes[1].pub_obstacles.msg.obstacles
        enhanced = nodes[2].pub_obstacles.msg.obstacles
        assert len(base) == len(enhanced)
        pose = nodes[2].accumulation_odom.at(stamp(msg), .06)
        # Same timestamped odom for both outputs; no world data enters estimator.
        for j, (a, b) in enumerate(zip(base, enhanced)):
            if abs(a.radius - .16) > .001:
                assert a == b
                continue
            d = by_index.get(j, {})
            row = dict(scan=scans, stamp=stamp(msg), index=j, baseline=[a.x, a.y], output=[b.x, b.y], diagnostic=d)
            if pose is not None and truth:
                wa, wb = transform([[a.x, a.y], [b.x, b.y]], pose)
                name, dist = min(((name, np.linalg.norm(wa - xy)) for name, xy in truth.items()), key=lambda v:v[1])
                if dist <= .5:
                    row.update(truth_id=name, baseline_error=float(dist),
                               output_error=float(np.linalg.norm(wb-truth[name])))
                else:
                    unmatched += 1
            rows.append(row)
    errors = [r for r in rows if 'baseline_error' in r]
    accepted = [r for r in errors if r['diagnostic'].get('accepted')]
    changes = [r['output_error']-r['baseline_error'] for r in errors]
    summary = {'bag': args.bag, 'scans': scans, 'detections': len(rows), 'unmatched': unmatched,
               'disabled_fields_identical': bool(invariance), 'unchanged_other_outputs': bool(side_effects),
               'baseline': stats([r['baseline_error'] for r in errors]),
               'accumulated': stats([r['output_error'] for r in errors]),
               'accepted_scored': len(accepted), 'reasons': dict(reasons),
               'worsened_over_2cm': sum(v > .02 for v in changes),
               'worsened_over_5cm': sum(v > .05 for v in changes),
               'worst_error_increase_m': max(changes, default=0),
               'worsened_any': sum(v > 1e-6 for v in changes),
               'accepted_baseline': stats([r['baseline_error'] for r in accepted]),
               'accepted_accumulated': stats([r['output_error'] for r in accepted]),
               'callback_ms': [stats(v, errors=False) for v in durations],
               'scan_period_sec': stats(scan_steps, errors=False)}
    Path(args.output).write_text(json.dumps({'summary': summary, 'detections': rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
