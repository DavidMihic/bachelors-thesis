"""geometry.py - zajednicke geometrijske funkcije za kmr_iiwa_task.

Kvaternioni su svugdje u (x, y, z, w) redoslijedu, isto kao u ROS-ovom
geometry_msgs/Quaternion. Sve funkcije vracaju liste, ne tuple, da se rezultat
moze bez pretvorbe proslijediti u numpy ili u ROS poruke.

Prije su ove funkcije bile razasute po modulima: quat_rotate_vector u
add_door_collision, quat_mul/quat_conj duplicirani u door_open i open_revolute,
quat_angle_between u handle_approach, a open_sliding je imao vlastitu kopiju
quat_rotate_vector. Osim dupliciranja, to je stvaralo i krizne importe izmedju
cvorova koji jedan s drugim nemaju veze.
"""

import math

import numpy as np


def wrap_pi(a):
    """Kut u (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def math_dist(a, b):
    """Euklidska udaljenost dviju tocaka bilo koje dimenzije."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def quat_rotate_vector(q, v):
    """Rotiraj vektor v kvaternionom q=(x,y,z,w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def quat_mul(q1, q2):
    """Hamiltonov produkt, oba u (x,y,z,w). q2 se primjenjuje kao DODATNA
    LOKALNA rotacija nakon q1 - standardna konvencija za slaganje u okviru
    tijela."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


# Stariji naziv iste operacije, zadrzan da postojeci pozivi rade.
quat_multiply = quat_mul


def quat_conj(q):
    """Konjugat, tj. inverz jedinicnog kvaterniona."""
    x, y, z, w = q
    return [-x, -y, -z, w]


def quat_angle_between(q1, q2):
    """Kut najkrace rotacije izmedju dvije orijentacije, u radijanima.

    Apsolutna vrijednost skalarnog produkta jer q i -q predstavljaju istu
    rotaciju (dvostruko pokrivanje).
    """
    d = abs(sum(a * b for a, b in zip(q1, q2)))
    return 2.0 * math.acos(min(1.0, d))


def quat_z_axis(q):
    """Z-os orijentacije q (treci stupac rotacijske matrice)."""
    x, y, z, w = q
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )


def quat_x_axis(q):
    """X-os orijentacije q (prvi stupac rotacijske matrice)."""
    x, y, z, w = q
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


def rotmat_to_quat(r):
    """Rotacijska matrica (lista triju STUPACA, svaki [x,y,z]) -> kvaternion
    (x,y,z,w). Shepperdova metoda."""
    m = np.array(r).T  # stupci u retke za standardnu formulu
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def tf_compose(pa, qa, pb, qb):
    """Slozi transformacije: A->B pa B->C daje A->C."""
    p = np.array(pa) + np.array(quat_rotate_vector(qa, list(pb)))
    return p, quat_mul(qa, qb)


def tf_inverse(p, q):
    """Inverz transformacije (p, q)."""
    qi = quat_conj(q)
    return -np.array(quat_rotate_vector(qi, list(p))), qi
