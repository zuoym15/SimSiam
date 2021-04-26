import numpy as np
import cv2


def apply_4x4(rt, xyz):
    E, F = rt.shape
    assert (E == 4) and (F == 4)
    N, D = xyz.shape
    assert (D == 3)

    ones = np.ones([N, 1], dtype=xyz.dtype)
    xyz1 = np.concatenate([xyz, ones], axis=1)
    xyz1_t = xyz1.T
    xyz2_t = rt @ xyz1_t
    xyz2 = xyz2_t.T[:, :3]
    return xyz2


def merge_rt(r, t):
    D1 = t.shape[0]
    C, D = r.shape
    assert (D1 == 3)
    assert (C == 3) and (D == 3)

    rt = np.eye(4)
    rt[:3, :3] = r
    rt[:3, 3] = t
    return rt


def merge_lrt(l, rt):
    D1 = l.shape[0]
    C, D = rt.shape
    assert (D1 == 3)
    assert (C == 4) and (D == 4)

    rt = rt.reshape(-1)
    lrt = np.concatenate([l, rt], axis=0)
    return lrt


def split_lrt(lrt):
    D = lrt.shape[0]
    assert (D == 19)
    l = lrt[:3]
    rt = lrt[3:].reshape(4, 4)
    return l, rt


def apply_4x4_to_lrt(Y_T_X, lrt_X):
    D = lrt_X.shape[0]
    assert (D == 19)
    E, F = Y_T_X.shape
    assert (E == 4) and (F == 4)

    l, rt_X = split_lrt(lrt_X)
    rt_Y = Y_T_X @ rt_X
    lrt_Y = merge_lrt(l, rt_Y)
    return lrt_Y


def get_xyzlist_from_len(l):
    D = l.shape[0]
    assert (D == 3)
    lx, ly, lz = l
    xs = np.array([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.])
    ys = np.array([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.])
    zs = np.array([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.])
    xyzlist = np.stack([xs, ys, zs], axis=-1) # 8 x 3
    return xyzlist


def get_xyzlist_from_lrt(lrt):
    D = lrt.shape[0]
    assert (D == 19)

    l, rt = split_lrt(lrt)
    xyzlist_obj = get_xyzlist_from_len(l)
    xyzlist_cam = apply_4x4(rt, xyzlist_obj)
    return xyzlist_cam


def split_intrinsics(K):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    return fx, fy, x0, y0


def camera2pixels(xyz, pix_T_cam):
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = np.split(xyz, 3, axis=1)

    EPS = 1e-4
    z = np.clip(z, a_min=EPS, a_max=None)
    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    xy = np.concatenate([x, y], axis=-1)
    return xy


def draw_corners_on_image(img, corners):
    image = img.copy()
    for i in range(corners.shape[0]):
        x, y = corners[i].astype(int)
        cv2.circle(image, (x, y), 1, (255, 0, 0), -1)
    return image
