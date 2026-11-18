"""Bounded, odometry-aligned static rectangle estimation; no map/mission inputs."""
from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


DEFAULTS = {
    'accumulation_window_sec': 1.0,
    'accumulation_max_points': 90,
    'accumulation_min_scans': 3,
    'accumulation_odom_max_age_sec': 0.06,
    'accumulation_track_gap_sec': 0.35,
    'accumulation_match_distance': 0.18,
    'accumulation_max_center_shift': 0.10,
    'accumulation_fit_max_rms': 0.012,
    'accumulation_stability_distance': 0.025,
    'accumulation_budget_ms': 20.0,
    'accumulation_max_fits': 3,
    'accumulation_max_tracks': 24,
}


def transform(points, pose, inverse=False):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    pts = np.asarray(points, dtype=float)
    return (pts - (x, y)) @ rot if inverse else pts @ rot.T + (x, y)


class OdomHistory:
    """Timestamped planar poses; interpolate, never use arbitrarily old odom."""
    def __init__(self):
        self.samples = deque(maxlen=400)
        self.frame = None
        self.epoch = 0

    def add(self, stamp, pose, frame):
        if not math.isfinite(stamp) or not np.isfinite(pose).all():
            self.samples.clear()
            self.epoch += 1
            return
        if self.samples and (stamp < self.samples[-1][0] or frame != self.frame):
            self.samples.clear()
            self.epoch += 1
        self.frame = frame
        if self.samples and stamp == self.samples[-1][0]:
            self.samples.pop()
        self.samples.append((stamp, tuple(pose)))

    def at(self, stamp, max_age):
        if not self.samples:
            return None
        times = np.array([s[0] for s in self.samples])
        i = int(np.searchsorted(times, stamp))
        if 0 < i < len(times):
            t0, p0 = self.samples[i - 1]
            t1, p1 = self.samples[i]
            if max(stamp - t0, t1 - stamp) > max_age:
                return None
            w = (stamp - t0) / (t1 - t0)
            dyaw = math.atan2(math.sin(p1[2] - p0[2]), math.cos(p1[2] - p0[2]))
            return (p0[0] + w * (p1[0] - p0[0]),
                    p0[1] + w * (p1[1] - p0[1]), p0[2] + w * dyaw)
        t, pose = self.samples[min(i, len(times) - 1)]
        return pose if abs(t - stamp) <= max_age else None


@dataclass
class Track:
    ident: int
    center: np.ndarray
    frames: deque = field(default_factory=lambda: deque(maxlen=12))
    proposal: object = None
    last_stamp: float = 0.0
    last_points: object = None


class FitBudgetExceeded(Exception):
    pass


def face_distances(points, sensors, state):
    """Distances to finite, sensor-facing rectangle edges (known dimensions only)."""
    c, s = math.cos(state[2]), math.sin(state[2])
    rot = np.array([[c, -s], [s, c]])
    p = (points - state[:2]) @ rot
    sensor = (sensors - state[:2]) @ rot
    a, b = 0.322 / 2, 0.22 / 2
    dx = np.maximum(np.abs(p[:, 0]) - a, 0)
    dy = np.maximum(np.abs(p[:, 1]) - b, 0)
    distances = np.column_stack((
        np.hypot(p[:, 0] + a, dy), np.hypot(p[:, 0] - a, dy),
        np.hypot(p[:, 1] + b, dx), np.hypot(p[:, 1] - b, dx)))
    visible = np.column_stack((sensor[:, 0] < -a, sensor[:, 0] > a,
                               sensor[:, 1] < -b, sensor[:, 1] > b))
    distances[~visible] = 1.0
    return distances, p


def fit_rectangle(points, sensors, baseline, deadline, max_rms):
    """Search all headings; bounded refinement, two-face/ambiguity checks."""
    def residual(state):
        if time.perf_counter() > deadline:
            raise FitBudgetExceeded()
        return face_distances(points, sensors, state)[0].min(axis=1)

    # Every angle covers both long/short-side interpretations. No scene heading.
    # Vectorized coarse search keeps heading discovery within the scan budget.
    angles = np.linspace(0, math.pi, 24, endpoint=False)
    rots = np.stack([np.array([[math.cos(a), -math.sin(a)],
                               [math.sin(a), math.cos(a)]]) for a in angles])
    local = np.einsum('ni,aij->anj', points, rots)
    lo, hi = np.quantile(local, [0.02, 0.98], axis=1)
    local_centers = np.stack([np.column_stack((u, v))
                             for u in (lo[:, 0]+.161, hi[:, 0]-.161)
                             for v in (lo[:, 1]+.11, hi[:, 1]-.11)], axis=1)
    centers = np.einsum('aki,aji->akj', local_centers, rots).reshape(-1, 2)
    headings = np.repeat(angles, 4)
    valid = np.linalg.norm(centers - baseline, axis=1) <= .16
    centers, headings = centers[valid], headings[valid]
    seeds = []
    if len(centers):
        c, sn = np.cos(headings)[:, None], np.sin(headings)[:, None]
        delta = points[None, :, :] - centers[:, None, :]
        delta_s = sensors[None, :, :] - centers[:, None, :]
        u, v = c*delta[:, :, 0]+sn*delta[:, :, 1], -sn*delta[:, :, 0]+c*delta[:, :, 1]
        su, sv = c*delta_s[:, :, 0]+sn*delta_s[:, :, 1], -sn*delta_s[:, :, 0]+c*delta_s[:, :, 1]
        dx, dy = np.maximum(abs(u)-.161, 0), np.maximum(abs(v)-.11, 0)
        ds = np.stack([np.hypot(u+.161, dy), np.hypot(u-.161, dy),
                       np.hypot(v+.11, dx), np.hypot(v-.11, dx)], axis=2)
        visible = np.stack([su < -.161, su > .161, sv < -.11, sv > .11], axis=2)
        ds[~visible] = 1.
        scores = np.mean(np.minimum(ds.min(axis=2), .08)**2, axis=1)
        seeds = [(float(score), np.r_[center, angle])
                 for score, center, angle in zip(scores, centers, headings)]
    if not seeds:
        return None, 'no_seed'
    seeds.sort(key=lambda v: v[0])
    starts = []
    for _, state in seeds:
        if all(np.linalg.norm(state[:2] - q[:2]) > .025 or
               abs(math.sin(state[2] - q[2])) > .25 for q in starts):
            starts.append(state)
        if len(starts) == 3:
            break
    solutions = []
    for state in starts:
        result = least_squares(residual, state, loss='soft_l1', f_scale=.008,
                               max_nfev=18, ftol=1e-4, xtol=1e-4, gtol=1e-4,
                               bounds=(np.r_[baseline - .16, -2*math.pi],
                                       np.r_[baseline + .16, 3*math.pi]))
        r = residual(result.x)
        rms = float(np.sqrt(np.mean(r*r)))
        if result.success and np.isfinite(result.x).all():
            solutions.append((rms, result.x))
    if not solutions:
        return None, 'fit_failed'
    solutions.sort(key=lambda v: v[0])
    rms, state = solutions[0]
    if rms > max_rms:
        return None, 'fit_residual'
    for score, alternate in solutions[1:]:
        if score <= rms + .003 and np.linalg.norm(alternate[:2] - state[:2]) > .035:
            return None, 'ambiguous_rectangle'
    distances, local = face_distances(points, sensors, state)
    labels = distances.argmin(axis=1)
    good = distances.min(axis=1) <= .02
    supports = []
    for face in range(4):
        subset = local[(labels == face) & good]
        axis = 1 if face < 2 else 0
        if len(subset) >= max(5, int(.12 * len(points))) and np.ptp(subset[:, axis]) >= .065:
            supports.append(face)
    # Need two adjacent faces, not merely a straight wall or a corner blob.
    if not any(a < 2 <= b for a in supports for b in supports):
        return None, 'insufficient_faces'
    return (state, rms), 'fit_ok'


class PointAccumulator:
    def __init__(self, config=None):
        self.cfg = dict(DEFAULTS)
        if config:
            self.cfg.update(config)
        self.tracks = []
        self.next_id = 1
        self.last_stamp = None

    def clear(self):
        self.tracks = []
        self.last_stamp = None

    def process(self, stamp, pose, sensor_offset, candidates):
        """Candidates: (obstacle index, base_link surface points, baseline dict).

        Returns accepted baseline-shaped replacements and per-candidate diagnostics.
        Unmatched/ambiguous tracks are discarded; duplicates never add observations.
        """
        begin = time.perf_counter()
        cfg = self.cfg
        deadline = begin + max(0., min(cfg['accumulation_budget_ms'], 50.)) / 1000
        out, debug = {}, []
        if pose is None:
            self.clear()
            return out, [{'reason': 'odom_unavailable'}]
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.clear()
            return out, [{'reason': 'non_increasing_scan_stamp'}]
        self.last_stamp = stamp
        limit = max(1, min(24, int(cfg['accumulation_max_tracks'])))
        if len(candidates) > limit:
            self.clear()
            return out, [{'reason': 'too_many_candidates'}]
        sensor = transform(np.array([sensor_offset]), pose)[0]
        current = []
        for idx, pts, base in candidates:
            pts = np.asarray(pts, float)
            if len(pts) > 90:
                pts = pts[np.linspace(0, len(pts)-1, 90, dtype=int)]
            current.append((idx, transform(pts, pose),
                            transform(np.array([[base['x'], base['y']]]), pose)[0], base))
        old = [t for t in self.tracks if 0 < stamp - t.last_stamp <= cfg['accumulation_track_gap_sec']]
        costs = np.full((len(current), len(old)), np.inf)
        for i, (_, pts, center, _) in enumerate(current):
            for j, track in enumerate(old):
                if time.perf_counter() > deadline:
                    self.tracks = []
                    return {}, [{'reason': 'association_budget', 'accepted': False}]
                d = float(np.linalg.norm(center - track.center))
                if d > cfg['accumulation_match_distance']:
                    continue
                # Shared observed surface, not just proximity of biased fitted centers.
                da = cKDTree(track.last_points).query(pts)[0]
                db = cKDTree(pts).query(track.last_points)[0]
                if np.mean(da < .045) < .60 or np.mean(db < .045) < .60:
                    continue
                costs[i, j] = d + float(np.median(da) + np.median(db))
        tracks = []
        fits = 0
        for i, (idx, pts, center, base) in enumerate(current):
            diag = {'obstacle_index': idx, 'baseline': [base['x'], base['y']],
                    'method': base['method'], 'accepted': False}
            debug.append(diag)
            finite = np.flatnonzero(np.isfinite(costs[i]))
            ambiguous = any(np.linalg.norm(center - other[2]) < .28
                            for k, other in enumerate(current) if k != i)
            track = None
            if len(finite):
                ranked = finite[np.argsort(costs[i, finite])]
                j = ranked[0]
                competitors = np.flatnonzero(np.isfinite(costs[:, j]))
                ambiguous |= (len(ranked) > 1 and costs[i, ranked[1]] - costs[i, j] < .06)
                ambiguous |= len(competitors) != 1
                if not ambiguous:
                    track = old[j]
            if ambiguous:
                diag['reason'] = 'ambiguous_association'
                continue
            if track is None:
                track = Track(self.next_id, center.copy())
                self.next_id += 1
            track.center = center.copy()
            track.last_points = pts
            track.last_stamp = stamp
            window = max(.01, min(1.0, cfg['accumulation_window_sec']))
            track.frames = deque([f for f in track.frames if stamp - f[0] <= window], maxlen=12)
            track.frames.append((stamp, pts, sensor.copy()))
            # Hard cap on stored points, evenly retain each scan (not only last scan).
            max_points = max(3, min(90, int(cfg['accumulation_max_points'])))
            while len(track.frames) > max_points // 3:
                track.frames.popleft()
            quota = max_points // len(track.frames)
            track.frames = deque([(t, p[np.linspace(0, len(p)-1, min(len(p), quota), dtype=int)], o)
                                  for t, p, o in track.frames], maxlen=12)
            tracks.append(track)
            diag.update(track=track.ident, scans=len(track.frames),
                        points=sum(len(f[1]) for f in track.frames))
            if base['method'] != 'CIRCLE':
                track.proposal = None
                diag['reason'] = 'line_preserved'
                continue
            if len(track.frames) < max(3, int(cfg['accumulation_min_scans'])):
                diag['reason'] = 'warming_up'
                continue
            if fits >= min(6, int(cfg['accumulation_max_fits'])) or time.perf_counter() > deadline:
                track.proposal = None
                diag['reason'] = 'budget'
                continue
            points = np.concatenate([f[1] for f in track.frames])
            sensors = np.concatenate([np.tile(f[2], (len(f[1]), 1)) for f in track.frames])
            if np.linalg.norm(np.ptp(points, axis=0)) > .48:
                track.proposal = None
                diag['reason'] = 'extent_inconsistent'
                continue
            fits += 1
            try:
                fit, reason = fit_rectangle(points, sensors, center, deadline,
                                            cfg['accumulation_fit_max_rms'])
            except FitBudgetExceeded:
                fit, reason = None, 'budget'
            except (ValueError, np.linalg.LinAlgError):
                fit, reason = None, 'fit_exception'
            diag['reason'] = reason
            if fit is None:
                track.proposal = None
                continue
            state, rms = fit
            corrected = transform(np.array([state[:2]]), pose, inverse=True)[0]
            diag.update(proposed=corrected.tolist(), rms=rms,
                        yaw_base=float(state[2] - pose[2]))
            if np.linalg.norm(state[:2] - center) > cfg['accumulation_max_center_shift']:
                track.proposal = None
                diag['reason'] = 'center_shift'
                continue
            # Verify current scan too; old points cannot overrule incompatible new geometry.
            current_rms = float(np.sqrt(np.mean(face_distances(
                pts, np.tile(sensor, (len(pts), 1)), state)[0].min(axis=1)**2)))
            if current_rms > cfg['accumulation_fit_max_rms'] * 1.5:
                track.proposal = None
                diag['reason'] = 'current_scan_mismatch'
                continue
            stable = (track.proposal is not None and
                      np.linalg.norm(state[:2] - track.proposal[:2]) <= cfg['accumulation_stability_distance']
                      and abs(math.sin(state[2] - track.proposal[2])) < math.sin(math.radians(12)))
            track.proposal = state
            if not stable:
                diag['reason'] = 'awaiting_stability'
                continue
            if time.perf_counter() > deadline:
                track.proposal = None
                diag['reason'] = 'budget'
                continue
            out[idx] = dict(base, x=float(corrected[0]), y=float(corrected[1]))
            diag.update(accepted=True, reason='accepted')
        self.tracks = tracks
        elapsed = (time.perf_counter() - begin) * 1000
        for d in debug:
            d['accumulation_ms'] = elapsed
        return out, debug
