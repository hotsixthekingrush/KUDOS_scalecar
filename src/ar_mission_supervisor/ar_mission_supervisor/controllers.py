"""공용 제어 함수."""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def pd_steering(lateral_err, heading_err, prev_lateral, dt, gains, max_steer):
    """횡방향 오차 + 헤딩 오차 -> 조향각(rad)."""
    d_term = 0.0 if dt <= 0.0 else (lateral_err - prev_lateral) / dt
    steer = (gains['kp_lateral'] * lateral_err
             + gains['kd_lateral'] * d_term
             + gains['kp_heading'] * heading_err)
    return clamp(steer, -max_steer, max_steer)


def steering_to_omega(v, steer, wheel_base):
    """애커만 기구학: omega = v * tan(delta) / L"""
    if wheel_base <= 0.0:
        return 0.0
    return v * math.tan(steer) / wheel_base