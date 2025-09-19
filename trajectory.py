import csv
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
# ----------------------------
# Parse (u,v) centers from CSV
# ----------------------------
def get_centers_from_csv(file_path):
    centers = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                xmin = float(row["x1"])
                ymin = float(row["y1"])
                xmax = float(row["x2"])
                ymax = float(row["y2"])
                centers.append([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
            except (KeyError, ValueError, TypeError):
                continue
    return np.asarray(centers, dtype=float)

# ----------------------------
# Simple temporal smoothing
# Grabs a moving average over a window, ignoring NaNs
# ----------------------------
def smooth_moving_average(centers, window=5):
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    if n == 0 or window <= 1:
        return centers.copy()

    smoothed = np.empty_like(centers)
    sums = np.zeros(2, dtype=float)
    counts = np.zeros(2, dtype=float)
    buf = []
    for i in range(n):
        buf.append(centers[i])
        for d in range(2):
            if np.isfinite(centers[i, d]):
                sums[d] += centers[i, d]
                counts[d] += 1
        if len(buf) > window:
            old = buf.pop(0)
            for d in range(2):
                if np.isfinite(old[d]):
                    sums[d] -= old[d]
                    counts[d] -= 1
        smoothed[i] = [sums[d] / counts[d] if counts[d] > 0 else np.nan for d in range(2)]
    return smoothed

# ----------------------------
# Constant-velocity Kalman (Used Chat help here)
# ----------------------------
def smooth_kalman_cv(centers, dt=1.0, process_var=5.0, meas_var=25.0):
    z = np.asarray(centers, dtype=float)
    n = len(z)
    if n == 0:
        return z.copy()

    F = np.array([[1,0,dt,0],
                  [0,1,0,dt],
                  [0,0,1, 0],
                  [0,0,0, 1]], dtype=float)
    H = np.array([[1,0,0,0],
                  [0,1,0,0]], dtype=float)
    Q = process_var * np.array([[dt**4/4, 0,        dt**3/2, 0       ],
                                [0,        dt**4/4, 0,        dt**3/2],
                                [dt**3/2,  0,        dt**2,   0       ],
                                [0,        dt**3/2, 0,        dt**2   ]], dtype=float)
    R = meas_var * np.eye(2)

    # find first finite measurement to initialize
    finite = np.isfinite(z).all(axis=1)
    if not np.any(finite):
        return np.full_like(z, np.nan)

    idx0 = int(np.flatnonzero(finite)[0])
    out = np.full_like(z, np.nan)

    # state [u, v, du, dv]
    x = np.zeros((4,1))
    x[0,0], x[1,0] = z[idx0]        # position from first valid measurement
    P = np.eye(4) * 1e3

    # emit filtered estimate for idx0
    out[idx0] = z[idx0]

    # run forward from idx0+1
    for i in range(idx0+1, n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update if valid
        zi = z[i]
        if np.isfinite(zi).all():
            y = zi.reshape(2,1) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(4) - K @ H) @ P
        out[i] = (H @ x).ravel()
    return out


# ------------------------------------------------------------
# Load xyz frames (H,W,3) from .npz files
# ------------------------------------------------------------
def _to_hw3(arr):
    """
    Normalize various xyz representations to a real-valued (H, W, 3) float array.
    Supported:
      - (H, W, 3) or (H, W, 4) -> take first 3 channels
      - (3, H, W) or (4, H, W) -> moveaxis to (H, W, C), take first 3
      - structured arrays with fields 'x','y','z'
      - masked arrays -> fill masked with NaN
    """
    # Unwrap masked arrays to plain ndarray with NaNs
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(np.nan)

    # Structured dtype with fields
    if getattr(arr, "dtype", None) is not None and arr.dtype.fields:
        flds = arr.dtype.fields
        if all(k in flds for k in ("x", "y", "z")):
            x = np.asarray(arr["x"], dtype=float)
            y = np.asarray(arr["y"], dtype=float)
            z = np.asarray(arr["z"], dtype=float)
            if x.shape == y.shape == z.shape:
                return np.stack([x, y, z], axis=-1)

    a = np.asarray(arr)

    if a.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {a.shape}")

    # (H, W, C)
    if a.shape[2] in (3, 4):
        return a[..., :3].astype(float, copy=False)

    # (C, H, W)
    if a.shape[0] in (3, 4):
        a = np.moveaxis(a, 0, -1)
        return a[..., :3].astype(float, copy=False)

    raise ValueError(f"Could not interpret array with shape {a.shape} as (H,W,3).")

def load_xyz_sequence(folder_or_pattern):
    """
    Loads a sequence of xyz frames and returns:
      xyz_frames: list of arrays (H, W, 3)
      files:      list of file paths
    Supports:
      - Separate keys 'x','y','z' each (H,W)
      - Single key 'xyz', 'points', 'arr_0' containing (H,W,3/4) or (3/4,H,W)
      - Structured arrays with fields x,y,z
      - Masked arrays (mask->NaN)
    """
    if os.path.isdir(folder_or_pattern):
        files = sorted(glob.glob(os.path.join(folder_or_pattern, "*.npz")))
    else:
        files = sorted(glob.glob(folder_or_pattern))
    if not files:
        raise FileNotFoundError(f"No .npz files found for '{folder_or_pattern}'")

    xyz_frames = []
    for f in files:
        # avoid pickle for safety; most .npz won't need it
        data = np.load(f, allow_pickle=False)
        keys = list(data.keys())

        # Case 1: explicit x/y/z arrays
        if all(k in data for k in ("x", "y", "z")):
            x, y, z = data["x"], data["y"], data["z"]
            # Handle masked arrays and enforce float
            if isinstance(x, np.ma.MaskedArray): x = x.filled(np.nan)
            if isinstance(y, np.ma.MaskedArray): y = y.filled(np.nan)
            if isinstance(z, np.ma.MaskedArray): z = z.filled(np.nan)
            if x.shape == y.shape == z.shape and x.ndim == 2:
                arr = np.stack([x, y, z], axis=-1).astype(float, copy=False)
                xyz_frames.append(arr)
                continue

        # Case 2: known single-key names
        arr = None
        for pref in ("xyz", "points", "arr_0"):
            if pref in data:
                try:
                    arr = _to_hw3(data[pref])
                    break
                except Exception:
                    arr = None

        # Case 3: try any key
        if arr is None:
            for k in keys:
                try:
                    arr = _to_hw3(data[k])
                    break
                except Exception:
                    pass

        if arr is None:
            raise ValueError(f"{f}: Could not find/normalize xyz. Keys: {keys}")

        xyz_frames.append(arr)

    return xyz_frames, files



# ------------------------------------------------------------
# Robust 3D sampling around (u,v) per frame
# ------------------------------------------------------------
def bilinear_sample_xyz(xyz, u, v):
    H, W, _ = xyz.shape
    # need a full 2x2 neighborhood
    if not (0 <= u < W-1 and 0 <= v < H-1):
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    x0 = int(np.floor(u)); x1 = x0 + 1
    y0 = int(np.floor(v)); y1 = y0 + 1
    du = u - x0; dv = v - y0

    p00 = xyz[y0, x0]; p10 = xyz[y0, x1]
    p01 = xyz[y1, x0]; p11 = xyz[y1, x1]
    nbrs = np.stack([p00, p10, p01, p11], axis=0)

    # if any neighbor invalid, fall back to median of the valid ones
    if not np.isfinite(nbrs).all():
        finite = np.isfinite(nbrs).all(axis=1)
        return np.nanmedian(nbrs[finite], axis=0) if np.any(finite) else np.array([np.nan, np.nan, np.nan])

    top = (1-du)*p00 + du*p10
    bot = (1-du)*p01 + du*p11
    return (1-dv)*top + dv*bot


def xyz_patch_median(xyz_frames, centers, k=5, require_min=5, use_mahal=False):
    half = k // 2
    out = np.full((len(xyz_frames), 3), np.nan, dtype=float)

    for i, (xyz, (u, v)) in enumerate(zip(xyz_frames, centers)):
        if not np.all(np.isfinite([u, v])): 
            continue
        H, W, _ = xyz.shape
        c = int(round(u)); r = int(round(v))
        r1 = max(0, r - half); r2 = min(H, r + half + 1)
        c1 = max(0, c - half); c2 = min(W, c + half + 1)

        patch = xyz[r1:r2, c1:c2, :].reshape(-1, 3)
        mask = np.isfinite(patch).all(axis=1)
        pts = patch[mask]
        if pts.shape[0] < require_min:
            continue

        if use_mahal and pts.shape[0] >= 8:
            mu = np.mean(pts, axis=0)
            C = np.cov(pts - mu, rowvar=False)
            # regularize covariance for stability
            C.flat[::4] += 1e-6
            try:
                invC = np.linalg.inv(C)
                d2 = np.sum((pts - mu) @ invC * (pts - mu), axis=1)
                keep = d2 < np.percentile(d2, 90)  # keep central 90%
                pts = pts[keep]
            except np.linalg.LinAlgError:
                pass

        out[i] = np.nanmedian(pts, axis=0)
    return out


# ------------------------------------------------------------
# Convenience: full pipeline → per-frame 3D positions
# ------------------------------------------------------------
def centers_to_3d_positions(
    csv_path,
    xyz_folder_or_pattern,
    smoothing="kalman",  # 'kalman' | 'ma' | None
    k=5,
    fps=30.0,
    process_var=3.0,
    meas_var=20.0,
):
    """
    1) Parse centers from CSV
    2) Stabilize (moving average or Kalman)
    3) For each frame, take median XYZ in k×k patch around (u,v)
    Returns: dict with centers_raw, centers_smooth, xyz_positions, xyz_files
    """
    centers_raw = get_centers_from_csv(csv_path)
    if smoothing == "kalman":
        dt = 1.0 / float(fps)
        centers_smooth = smooth_kalman_cv(centers_raw, dt=dt, process_var=process_var, meas_var=meas_var)
    elif smoothing == "ma":
        centers_smooth = smooth_moving_average(centers_raw, window=5)
    else:
        centers_smooth = centers_raw.copy()

    xyz_frames, xyz_files = load_xyz_sequence(xyz_folder_or_pattern)
    if len(xyz_frames) != len(centers_smooth):
        raise ValueError(f"Frame count mismatch: {len(xyz_frames)} xyz vs {len(centers_smooth)} centers")

    xyz_positions = np.full((len(xyz_frames), 3), np.nan, dtype=float)
    for i, (xyz, (u, v)) in enumerate(zip(xyz_frames, centers_smooth)):
        if not np.all(np.isfinite([u, v])):
            continue
        # try bilinear at subpixel (u, v)
        p = bilinear_sample_xyz(xyz, u, v)
        if np.isfinite(p).all():
            xyz_positions[i] = p
            continue
        # fallback: robust k×k median around the (rounded) pixel
        xyz_positions[i] = xyz_patch_median([xyz], np.array([[u, v]]), k=k)[0]


    return {
        "centers_raw": centers_raw,
        "centers_smooth": centers_smooth,
        "xyz_positions": xyz_positions,  # meters, camera coords (+X forward, +Y right, +Z up)
        "xyz_files": xyz_files
    }
# ----------------------------
# Phase 2 — pull 3D (already done by centers_to_3d_positions)
# ----------------------------
# XYZ = (T,3) from your centers_to_3d_positions()
# Camera axes: +X forward, +Y right, +Z up

def compute_ground_vectors_from_xyz(XYZ):
    """
    From per-frame (X,Y,Z) in camera coords:
      - horizontal vector camera→light: h_t = (X_t, Y_t)
      - camera position in pole-centric frame: p_t^{pole} = (-X_t, -Y_t)
    Returns:
      h_xy: (T,2), p_pole: (T,2)
    """
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ.ndim != 2 or XYZ.shape[1] != 3:
        raise ValueError("XYZ must be shape (T,3)")
    h_xy = XYZ[:, :2].copy()            # (X, Y)
    p_pole = -h_xy                      # (-X, -Y)
    return h_xy, p_pole

# ----------------------------
# Phase 3 — ego pose on the ground plane
# ----------------------------
def align_to_world_from_sequence(p_pole, y_left=True, use_frames=15):
    p_pole = np.asarray(p_pole, dtype=float)
    mask = np.isfinite(p_pole).all(axis=1)
    if not np.any(mask): 
        T = p_pole.shape[0]
        nanxy = np.full((T,2), np.nan)
        return nanxy, nanxy, np.nan, None

    idxs = np.flatnonzero(mask)[:use_frames]  # early window for heading
    angles = np.arctan2(p_pole[idxs,1], p_pole[idxs,0])
    # robust center
    theta0 = np.median(angles)

    c, s = np.cos(-theta0), np.sin(-theta0)
    R = np.array([[c, -s],[s, c]], dtype=float)
    p_world = np.full_like(p_pole, np.nan)
    p_world[mask] = (R @ p_pole[mask].T).T

    p_bev = p_world.copy()
    if y_left: 
        p_bev[:,1] = -p_bev[:,1]

    return p_world, p_bev, float(theta0), int(idxs[0])


def xyz_to_bev(XYZ, y_left=True):
    """
    Convenience wrapper: XYZ -> (p_world, p_bev, theta0, idx0)
    """
    _, p_pole = compute_ground_vectors_from_xyz(XYZ)
    return align_to_world_from_sequence(p_pole, y_left=y_left)





def plot_bev_trajectory(p_bev, save_path="out/trajectory.png", connect=True):
    """
    Plot BEV scatter of the ego trajectory relative to the traffic light at (0,0).
    X axis = forward (meters), Y axis = left (meters).
    - Marks (0,0) as the traffic light
    - Plots points in time order; optionally connects with a thin line
    - Marks start and end points
    - Equal aspect ratio, grid, labeled axes
    - Saves to save_path
    """
    p_bev = np.asarray(p_bev, dtype=float)
    if p_bev.ndim != 2 or p_bev.shape[1] != 2:
        raise ValueError("p_bev must be shape (T,2)")

    # keep only finite points, in time order
    mask = np.isfinite(p_bev).all(axis=1)
    xy = p_bev[mask]
    if xy.size == 0:
        raise ValueError("No finite (x,y) points to plot.")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if connect and len(xy) >= 2:
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1, label="Ego trajectory")

    ax.scatter(xy[:, 0], xy[:, 1], s=12)

    # Mark start (red X) and end (green dot)
    ax.scatter([xy[0, 0]], [xy[0, 1]], color='red', marker='x', s=80, label="Start")
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], color='green', s=80, label="End")

    # mark traffic light at (0,0)
    ax.scatter([0.0], [0.0], s=80, marker='*', color='black', label="Traffic light (origin)")

    ax.set_xlabel("Forward (X, m)")
    ax.set_ylabel("Lateral (Y, m)")
    ax.set_title("Static Example: Ego Only")
    ax.legend(fontsize=5)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved BEV trajectory → {save_path}")



result = centers_to_3d_positions(
    csv_path="dataset/bbox_light.csv",
    xyz_folder_or_pattern="dataset/xyz/*.npz",  # or "xyz" directory
    smoothing="kalman",  # or "ma" or None
    k=5,
    fps=30.0
)
XYZ = result["xyz_positions"]  # shape (T, 3)
p_world, p_bev, theta0, idx0 = xyz_to_bev(XYZ, y_left=True)
plot_bev_trajectory(p_bev, save_path="trajectory.png", connect=True)
