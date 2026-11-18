"""Geometry and failure-mode checks; no ROS executor or Gazebo required."""
import math
import time
import numpy as np
from ar_perception.point_accumulation import (
    OdomHistory, PointAccumulator, fit_rectangle, transform)


def surface(center=(1., .4), yaw=.37):
    # Two sensor-facing, perpendicular sides; long side orientation is arbitrary.
    local = np.vstack((np.column_stack((np.full(30, -.161), np.linspace(-.11, .11, 30))),
                       np.column_stack((np.linspace(-.161, .161, 40), np.full(40, -.11)))))
    return transform(local, (*center, yaw))


def candidate(pts, center=(1., .4), method='CIRCLE', index=0):
    return (index, pts, {'x': center[0]+.025, 'y': center[1]+.015,
                         'method': method, 'residual': .02, 'length': .32})


def test_rectangle_unknown_heading():
    p = surface((1., .8))
    fit, reason = fit_rectangle(p, np.zeros_like(p), np.array([1.025, .815]),
                                time.perf_counter()+1., .012)
    assert fit is not None, reason
    assert np.linalg.norm(fit[0][:2] - [1., .8]) < .01
    assert abs(math.sin(fit[0][2]-.37)) < .05


def test_odom_interpolation_stale_and_reset():
    h = OdomHistory()
    h.add(1., (0., 0., math.radians(179)), 'odom')
    h.add(1.1, (.1, 0., math.radians(-179)), 'odom')
    p = h.at(1.05, .06)
    assert abs(p[0]-.05) < 1e-9
    assert abs(abs(p[2])-math.pi) < .001
    assert h.at(2., .06) is None
    h.add(.1, (0., 0., 0.), 'odom')
    assert h.epoch == 1 and len(h.samples) == 1


def test_line_unchanged_and_storage_bound():
    acc = PointAccumulator()
    for k in range(30):
        out, diag = acc.process(k*.05, (0, 0, 0), (0, 0),
                                [candidate(surface(), method='LINE')])
        assert out == {}
        assert sum(len(f[1]) for f in acc.tracks[0].frames) <= 90
        assert all(k*.05-f[0] <= 1. for f in acc.tracks[0].frames)


def test_accumulation_accepts_then_lost_odom_falls_back():
    acc = PointAccumulator({'accumulation_budget_ms': 50.})
    adopted = False
    for k in range(6):
        pose = (.015*k, 0., .01*k)
        pts = transform(surface(), pose, inverse=True)
        center = transform([[1., .4]], pose, inverse=True)[0]
        out, diag = acc.process(1+k*.1, pose, (0, 0), [candidate(pts, center)])
        if k < 2:
            assert not out
        adopted |= bool(out)
        if out:
            world = transform([[out[0]['x'], out[0]['y']]], pose)[0]
            assert np.linalg.norm(world-[1., .4]) < .02
    assert adopted, diag
    out, diag = acc.process(2., None, (0, 0), [candidate(surface())])
    assert not out and not acc.tracks


def test_duplicate_scans_and_ambiguous_vehicles():
    acc = PointAccumulator()
    c = candidate(surface())
    for _ in range(4):
        out, _ = acc.process(1., (0,0,0), (0,0), [c])
        assert not out
    close = candidate(surface((1.1,.4)), center=(1.1,.4), index=1)
    out, diag = acc.process(2., (0,0,0), (0,0), [c, close])
    assert not out and all(d['reason']=='ambiguous_association' for d in diag)


def test_budget_fallback_and_track_break():
    acc = PointAccumulator({'accumulation_budget_ms': 0.})
    for k in range(5):
        out, diag = acc.process(k*.1, (0,0,0), (0,0), [candidate(surface())])
        assert not out
    assert diag[0]['reason'] in ('warming_up', 'budget', 'association_budget')
    acc = PointAccumulator()
    acc.process(.4, (0,0,0), (0,0), [candidate(surface())])
    old = acc.tracks[0].ident
    acc.process(.6, (0,0,0), (0,0), [])
    _, diag = acc.process(.7, (0,0,0), (0,0), [candidate(surface())])
    assert diag[0]['track'] != old and diag[0]['scans'] == 1


def test_single_face_is_not_a_rectangle_center_measurement():
    pts = surface()[:30]
    fit, reason = fit_rectangle(pts, np.zeros_like(pts), np.array([1.025,.415]),
                                time.perf_counter()+1., .012)
    assert fit is None


def test_two_separate_tracks_do_not_mix_and_single_scan_cannot_adopt():
    acc = PointAccumulator({'accumulation_budget_ms': 50.})
    for k in range(4):
        out, diag = acc.process(k*.1, (0,0,0), (0,0), [
            candidate(surface((1., .8)), (1., .8), index=0),
            candidate(surface((2., .8)), (2., .8), index=1)])
        assert len(acc.tracks) == 2
        for track in acc.tracks:
            points = np.concatenate([f[1] for f in track.frames])
            assert np.ptp(points[:, 0]) < .4
            assert len(points) <= 90
        if k < 2:
            assert not out
