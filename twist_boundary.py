"""Exact rotating edge grips for the twist experiment; no interior contact changes.

# python twist_square.py --stage run --settings output/trough20_E_smooth_g200_s200/config.json --output output/square_twist --resolution 64 --degrees 360 --twist-seconds 3 --hold-seconds 1 --settle-seconds 1 --grid-size 200 --substeps 200
"""
import warp as wp

from warp_mpm.mpm_data_structure import MPMStateStruct
from warp_mpm.precision import mat33, scalar, vec3


@wp.kernel
def rotating_grips(
    state: MPMStateStruct, rest: wp.array(dtype=vec3), signs: wp.array(dtype=wp.int32),
    offset: int, origin: vec3, angle: scalar, next_angle: scalar, dt: scalar, end: int,
) -> None:
    p = wp.tid()
    sign = scalar(signs[p])
    a, b = sign * angle, sign * next_angle
    r = rest[p] - origin
    current = vec3(r[0], wp.cos(a) * r[1] - wp.sin(a) * r[2],
                   wp.sin(a) * r[1] + wp.cos(a) * r[2])
    target = vec3(r[0], wp.cos(b) * r[1] - wp.sin(b) * r[2],
                  wp.sin(b) * r[1] + wp.cos(b) * r[2])
    position = current
    if end == 1:
        position = target
    state.particle_x[offset + p] = origin + position
    state.particle_v[offset + p] = (target - current) / dt
    c, s = (wp.cos(b - a) - scalar(1.0)) / dt, wp.sin(b - a) / dt
    state.particle_C[offset + p] = mat33(scalar(0.0), scalar(0.0), scalar(0.0),
                                        scalar(0.0), c, -s, scalar(0.0), s, c)
