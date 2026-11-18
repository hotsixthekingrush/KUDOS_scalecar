"""Optional finite-wall rejection using scan geometry and short odom memory.

No mission/time-window labels enter this module. Coordinates are base_link.
"""
import math
import numpy as np
from ar_perception.point_accumulation import transform

WALL_DEFAULTS = {
    'wall_min_length': 0.65,
    'wall_min_points': 12,
    'wall_fit_distance': 0.025,
    'wall_max_point_gap': 0.25,
    'wall_candidate_distance': 0.035,
    'wall_candidate_fraction': 0.85,
    'wall_track_sec': 0.5,
    'wall_odom_max_age_sec': 0.06,
    'wall_max_range': 4.0,
}


class WallFilter:
    def __init__(self, config=None):
        self.cfg = dict(WALL_DEFAULTS, **(config or {}))
        self.clear()

    def clear(self):
        self.memory = []
        self.walls = []
        self.last_stamp = None
        self.last_pose = None
        self.epoch = None

    def detect(self, points):
        cfg = self.cfg
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[np.linalg.norm(pts, axis=1) <= cfg['wall_max_range']]
        rng = np.random.default_rng(0)
        walls = []
        # Bounded deterministic RANSAC; split support to avoid bridging objects.
        for _ in range(8):
            if len(pts) < cfg['wall_min_points']:
                break
            best = np.zeros(len(pts), dtype=bool)
            for _ in range(100):
                a, b = pts[rng.choice(len(pts), 2, replace=False)]
                d = b-a
                length = np.linalg.norm(d)
                if length < .15:
                    continue
                d /= length
                mask = np.abs((pts-a) @ np.array([-d[1], d[0]])) <= cfg['wall_fit_distance']
                if mask.sum() > best.sum():
                    best = mask
            if best.sum() < cfg['wall_min_points']:
                break
            support = pts[best]
            pts = pts[~best]
            center = support.mean(axis=0)
            _, _, vh = np.linalg.svd(support-center, full_matrices=False)
            d = vh[0]
            support = support[np.argsort((support-center) @ d)]
            gaps = np.diff(support @ d)
            for group in np.split(support, np.where(gaps > cfg['wall_max_point_gap'])[0]+1):
                if len(group) < cfg['wall_min_points']:
                    continue
                c = group.mean(axis=0)
                _, _, vh = np.linalg.svd(group-c, full_matrices=False)
                direction = vh[0]
                t = (group-c) @ direction
                normal = np.array([-direction[1], direction[0]])
                if np.ptp(t) < cfg['wall_min_length']:
                    continue
                if np.max(np.abs((group-c) @ normal)) > cfg['wall_fit_distance'] * 1.5:
                    continue
                walls.append(np.array([c+t.min()*direction, c+t.max()*direction]))
        return walls

    def update(self, stamp, pose, points, epoch=0):
        if (not math.isfinite(stamp) or self.epoch != epoch or
                (self.last_stamp is not None and stamp <= self.last_stamp)):
            self.clear()
        if not math.isfinite(stamp):
            return
        if pose is not None and not np.isfinite(pose).all():
            pose = None
        if pose is not None and self.last_pose is not None and self.last_stamp is not None:
            dt = stamp-self.last_stamp
            dyaw = math.atan2(math.sin(pose[2]-self.last_pose[2]), math.cos(pose[2]-self.last_pose[2]))
            if (np.linalg.norm(np.array(pose[:2])-self.last_pose[:2]) > .1+2*dt or
                    abs(dyaw) > .1+3*dt):
                self.memory = []
        self.epoch = epoch
        self.last_stamp, self.last_pose = stamp, pose
        current = self.detect(points)
        # Missing/stale odometry: current scan only, never stale local geometry.
        if pose is None:
            self.memory = []
        self.memory = [(t, w) for t, w in self.memory if 0 <= stamp-t <= self.cfg['wall_track_sec']]
        self.walls = [(w, 'scan') for w in current]
        if pose is not None:
            self.walls += [(transform(w, pose, inverse=True), 'odom') for _, w in self.memory]
            self.memory += [(stamp, transform(w, pose)) for w in current]
            self.memory = self.memory[-64:]

    def reject(self, cluster):
        pts = np.asarray(cluster, dtype=float).reshape(-1, 2)
        if not len(pts) or not np.isfinite(pts).all():
            return False, 'keep'
        for wall, source in self.walls:
            a, b = wall
            direction = b-a
            length = np.linalg.norm(direction)
            if length < 1e-9:
                continue
            direction /= length
            along = (pts-a) @ direction
            distance = np.abs((pts-a) @ np.array([-direction[1], direction[0]]))
            # Finite extent, no extension beyond endpoints. Use measured surface,
            # not the inferred LIMO center (which is offset from that surface).
            threshold = self.cfg['wall_candidate_distance'] * (0.75 if source == 'odom' else 1.)
            fraction = np.mean((distance <= threshold) & (along >= 0) & (along <= length))
            if fraction >= self.cfg['wall_candidate_fraction']:
                return True, source
        return False, 'keep'
