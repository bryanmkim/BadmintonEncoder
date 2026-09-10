"""Badminton court geometry and camera helpers shared by the Phase 1 pipeline.

Court coordinates are metres, in the same frame the annotator uses: X across the court (-3.05 left to
+3.05 right as seen from the main camera), Y along it (-6.7 far baseline to +6.7 near baseline, net at
0), Z up.
"""
import numpy as np

HALF_W, HALF_L = 3.05, 6.7
SINGLES_X = 2.59
SHORT_SERVICE_Y = 1.98
LONG_SERVICE_DOUBLES_Y = 5.94
NET_POST_H, NET_MID_H = 1.55, 1.524

# Painted floor lines as (start, end) segments. Lines are 40 mm wide and sit inside the court's
# dimensions; the 20 mm offset to their centre is ignored (about 1 px at 720p).
LINES = {
    "far baseline": ((-HALF_W, -HALF_L), (HALF_W, -HALF_L)),
    "far long service": ((-HALF_W, -LONG_SERVICE_DOUBLES_Y), (HALF_W, -LONG_SERVICE_DOUBLES_Y)),
    "far short service": ((-HALF_W, -SHORT_SERVICE_Y), (HALF_W, -SHORT_SERVICE_Y)),
    "near short service": ((-HALF_W, SHORT_SERVICE_Y), (HALF_W, SHORT_SERVICE_Y)),
    "near long service": ((-HALF_W, LONG_SERVICE_DOUBLES_Y), (HALF_W, LONG_SERVICE_DOUBLES_Y)),
    "near baseline": ((-HALF_W, HALF_L), (HALF_W, HALF_L)),
    "left doubles sideline": ((-HALF_W, -HALF_L), (-HALF_W, HALF_L)),
    "left singles sideline": ((-SINGLES_X, -HALF_L), (-SINGLES_X, HALF_L)),
    "right singles sideline": ((SINGLES_X, -HALF_L), (SINGLES_X, HALF_L)),
    "right doubles sideline": ((HALF_W, -HALF_L), (HALF_W, HALF_L)),
    "far centre line": ((0.0, -HALF_L), (0.0, -SHORT_SERVICE_Y)),
    "near centre line": ((0.0, SHORT_SERVICE_Y), (0.0, HALF_L)),
}
# The full lines those segments lie on, far to near and left to right
HORIZONTAL_Y = [-HALF_L, -LONG_SERVICE_DOUBLES_Y, -SHORT_SERVICE_Y, SHORT_SERVICE_Y, LONG_SERVICE_DOUBLES_Y, HALF_L]
VERTICAL_X = [-HALF_W, -SINGLES_X, 0.0, SINGLES_X, HALF_W]
CORNERS = [(-HALF_W, -HALF_L), (HALF_W, -HALF_L), (HALF_W, HALF_L), (-HALF_W, HALF_L)]  # far-left, far-right, near-right, near-left

NET = [  # 3D polylines: both posts (on the doubles sidelines) and the tape along the top
    [(-HALF_W, 0.0, 0.0), (-HALF_W, 0.0, NET_POST_H)],
    [(HALF_W, 0.0, 0.0), (HALF_W, 0.0, NET_POST_H)],
    [(-HALF_W, 0.0, NET_POST_H), (0.0, 0.0, NET_MID_H), (HALF_W, 0.0, NET_POST_H)],
]


def sample_lines(per_metre=6):
    """Points along every painted line: (names, (N, 2) court metres)."""
    names, pts = [], []
    for name, (a, b) in LINES.items():
        a, b = np.array(a), np.array(b)
        n = max(2, int(np.linalg.norm(b - a) * per_metre))
        pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
        names += [name] * n
    return np.array(names), np.concatenate(pts)


def apply_h(H, pts):
    """Map (N, 2) points through a 3x3 homography."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def to_normalized(pts):
    """Court metres -> the annotator's 0-1 court fractions (x across, y from the far to the near baseline)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.c_[pts[:, 0] / (2 * HALF_W) + 0.5, pts[:, 1] / (2 * HALF_L) + 0.5]


def camera_from_homography(H, width, height):
    """Recover a pinhole camera from the floor homography (court metres -> image pixels).

    Assumes square pixels, no skew and the principal point at the image centre, which leaves the focal
    length as the only intrinsic; it follows from the floor axes being perpendicular and equally scaled.
    Returns K, R, t such that pixel ~ K (R @ P + t) for a court point P = (X, Y, Z).
    """
    cx, cy = width / 2, height / 2
    Hc = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]) @ np.asarray(H, float)
    h1, h2 = Hc[:, 0], Hc[:, 1]
    # r1 . r2 = 0 and |r1| = |r2| are both linear in w = 1 / f^2
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = -np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    w = float(a @ b / (a @ a))
    if w <= 0:
        raise ValueError("homography doesn't correspond to a real camera")
    f = 1 / np.sqrt(w)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    A = np.linalg.inv(K) @ np.asarray(H, float)
    scale = 2 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    if A[2, 2] * scale < 0:  # the court must be in front of the camera
        scale = -scale
    r1, r2, t = scale * A[:, 0], scale * A[:, 1], scale * A[:, 2]
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    R = U @ Vt
    # The court axes are left-handed (X right, Y towards the camera, Z up): pick the floor normal that
    # puts the camera above the floor
    if -(R[:, 2] @ t) < 0:
        R[:, 2] = -R[:, 2]
    return K, R, t


def camera_center(R, t):
    return -R.T @ t


def project(K, R, t, pts3d):
    """Court points (N, 3) in metres -> image pixels (N, 2)."""
    cam = np.asarray(pts3d, float).reshape(-1, 3) @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]
