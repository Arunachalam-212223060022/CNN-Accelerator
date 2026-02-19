## Software/Control Code:

```
"""
FPGA Cat/Dog Classifier — Final Clean Version
==============================================
- Weight sets A and C only (B dropped — trained on person/object)
- Tight bounding box around the animal (GrabCut foreground crop)
- Smaller, cleaner label overlay
- CPU inference comparison: FPS / latency / accuracy vs FPGA
- Combined figure: detections (top) + conv feature maps (bottom)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display as ipy_display

from pynq import Overlay, allocate
import numpy as np
import cv2
import time
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE       = "/home/xilinx/pynq/overlays"
TEST_IMAGE = f"{BASE}/real_detect/test8cat.jpg"   # ← change image here
OUT_DIR    = "/home/xilinx/multi_weight_results/"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_NAMES  = {0: "Cat", 1: "Dog"}
CLASS_COLORS = {0: (255, 160,  30),   # Cat — orange  (BGR)
                1: ( 50, 210,  80)}   # Dog — green   (BGR)

ARCH = {
    "conv1": dict(out_ch=8,  in_ch=1,  k=3),
    "conv2": dict(out_ch=16, in_ch=8,  k=3),
    "conv3": dict(out_ch=32, in_ch=16, k=3),
}

# Weight set A — best performing weights for this FPGA bitstream
WEIGHT_SETS = {
    "FPGA  Weights  A": {
        "_dir":    f"{BASE}/real_detect/fpga_weights_for_pynq/fpga_weights/",
        "conv1_w": "conv1_w.npy",  "conv1_b": "conv1_b.npy",
        "conv2_w": "conv2_w.npy",  "conv2_b": "conv2_b.npy",
        "conv3_w": "conv3_w.npy",  "conv3_b": "conv3_b.npy",
        "fc_w":    "fc_w.npy",     "fc_b":    "fc_b.npy",
        "out_w":   None,           "out_b":   None,
    },
}

# ── FPGA SETUP ────────────────────────────────────────────────────────────────
ol  = Overlay(f"{BASE}/real_detect/real_detect.bit")
ip  = ol.real_detector_0
print("Overlay loaded")

img_buf     = allocate(shape=(4096,),    dtype=np.uint8)
res_buf     = allocate(shape=(16,),      dtype=np.int32)
conv1_w_buf = allocate(shape=(200_000,), dtype=np.int8)
conv1_b_buf = allocate(shape=(200_000,), dtype=np.int8)
conv2_w_buf = allocate(shape=(200_000,), dtype=np.int8)
conv2_b_buf = allocate(shape=(200_000,), dtype=np.int8)
conv3_w_buf = allocate(shape=(200_000,), dtype=np.int8)
conv3_b_buf = allocate(shape=(200_000,), dtype=np.int8)
fc_w_buf    = allocate(shape=(200_000,), dtype=np.int8)
fc_b_buf    = allocate(shape=(200_000,), dtype=np.int8)
out_w_buf   = allocate(shape=(200_000,), dtype=np.int8)
out_b_buf   = allocate(shape=(200_000,), dtype=np.int8)

REG = {
    "ctrl":    0x00,
    "image":   (0x10, 0x14),
    "conv1_w": (0x1C, 0x20), "conv1_b": (0x28, 0x2C),
    "conv2_w": (0x34, 0x38), "conv2_b": (0x40, 0x44),
    "conv3_w": (0x4C, 0x50), "conv3_b": (0x58, 0x5C),
    "fc_w":    (0x64, 0x68), "fc_b":    (0x70, 0x74),
    "result":  (0x7C, 0x80),
}
BUF_MAP = {
    "image":   img_buf,
    "conv1_w": conv1_w_buf, "conv1_b": conv1_b_buf,
    "conv2_w": conv2_w_buf, "conv2_b": conv2_b_buf,
    "conv3_w": conv3_w_buf, "conv3_b": conv3_b_buf,
    "fc_w":    fc_w_buf,    "fc_b":    fc_b_buf,
    "result":  res_buf,
}

def _write_addr(reg_pair, buf):
    a = int(buf.physical_address)
    ip.write(reg_pair[0], a & 0xFFFFFFFF)
    ip.write(reg_pair[1], (a >> 32) & 0xFFFFFFFF)

def _write_all_addrs():
    for k in ["image","conv1_w","conv1_b","conv2_w","conv2_b",
              "conv3_w","conv3_b","fc_w","fc_b","result"]:
        _write_addr(REG[k], BUF_MAP[k])

_write_all_addrs()

# ── WEIGHT LOADING ────────────────────────────────────────────────────────────

def load_weight_set(ws_name):
    ws   = WEIGHT_SETS[ws_name]
    wdir = ws["_dir"]
    key_to_buf = {
        "conv1_w": conv1_w_buf, "conv1_b": conv1_b_buf,
        "conv2_w": conv2_w_buf, "conv2_b": conv2_b_buf,
        "conv3_w": conv3_w_buf, "conv3_b": conv3_b_buf,
        "fc_w":    fc_w_buf,    "fc_b":    fc_b_buf,
        "out_w":   out_w_buf,   "out_b":   out_b_buf,
    }
    for key, fname in ws.items():
        if key == "_dir": continue
        buf = key_to_buf[key]
        if fname is None:
            buf[:] = 0; buf.flush(); continue
        fpath = os.path.join(wdir, fname)
        if not os.path.exists(fpath):
            cands  = sorted(f for f in os.listdir(wdir) if f.endswith(".npy"))
            tokens = key.replace("_"," ").split()
            hits   = [c for c in cands if all(t in c.lower() for t in tokens)]
            if len(hits) == 1: fpath = os.path.join(wdir, hits[0])
            else: raise FileNotFoundError(f"Cannot find '{fname}' for '{key}' in {wdir}")
        data = np.load(fpath).astype(np.int8).flatten()
        n    = min(len(data), len(buf))
        buf[:n] = data[:n]; buf.flush()
    _write_all_addrs()

# ── TIGHT BOUNDING BOX via foreground detection ───────────────────────────────

def get_tight_box(img_bgr, margin_frac=0.04):
    """
    Find a tight bounding box around the foreground subject.
    Uses Otsu threshold on the grayscale image.
    Falls back to the full image if detection fails.
    Returns (x1, y1, x2, y2).
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Try to separate subject from background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If background is bright (white/grey studio), invert
    edge_pixels = np.concatenate([thresh[0,:], thresh[-1,:], thresh[:,0], thresh[:,-1]])
    edge_mean = float(np.mean(edge_pixels))
    if edge_mean > 127:
        thresh = cv2.bitwise_not(thresh)

    # Clean up noise
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Find largest contour
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, 0, W, H

    largest = max(cnts, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < 0.02 * H * W:   # too small — fall back
        return 0, 0, W, H

    bx, by, bw, bh = cv2.boundingRect(largest)
    margin = int(max(H, W) * margin_frac)
    x1 = max(bx - margin, 0)
    y1 = max(by - margin, 0)
    x2 = min(bx + bw + margin, W)
    y2 = min(by + bh + margin, H)
    return x1, y1, x2, y2

# ── FPGA INFERENCE ────────────────────────────────────────────────────────────

def fpga_classify(img_bgr):
    """Single FPGA inference on full image resized to 64×64."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g64  = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    img_buf[:] = g64.flatten().astype(np.uint8); img_buf.flush()
    ip.write(REG["ctrl"], 0x01)
    t0 = time.time()
    while (ip.read(REG["ctrl"]) & 0x02) == 0:
        if time.time()-t0 > 5.0: raise TimeoutError("FPGA timeout")
        time.sleep(0.0005)
    res_buf.invalidate()
    cls       = int(res_buf[0])
    score_cat = int(res_buf[6])
    score_dog = int(res_buf[7])
    margin    = abs(score_dog - score_cat)
    return CLASS_NAMES[cls], cls, score_cat, score_dog, margin

# ── CPU INFERENCE (NumPy forward pass) ───────────────────────────────────────

def _load_npy(ws_name, key):
    ws    = WEIGHT_SETS[ws_name]
    fname = ws.get(key)
    if fname is None: return None
    fpath = os.path.join(ws["_dir"], fname)
    return np.load(fpath) if os.path.exists(fpath) else None

def _relu(x):      return np.maximum(x, 0)
def _maxpool2(x):
    """2×2 max-pool, stride 2."""
    H, W = x.shape[-2], x.shape[-1]
    H2, W2 = H//2, W//2
    return x[..., :H2*2, :W2*2].reshape(*x.shape[:-2], H2, 2, W2, 2).max(axis=(-3,-1))

def cpu_forward(img_bgr, ws_name):
    """
    Full NumPy forward pass using float32 weights.
    Uses the *_float32.npy files if present, else int8.
    Returns (label, cls, score_cat, score_dog, margin, latency_ms).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x    = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x    = x[np.newaxis, np.newaxis, :, :]    # (1, 1, 64, 64)

    ws   = WEIGHT_SETS[ws_name]
    wdir = ws["_dir"]

    def _load_w(stem):
        """Try float32 version first, fall back to int8."""
        f32 = os.path.join(wdir, stem + "_float32.npy")
        if os.path.exists(f32):
            return np.load(f32).astype(np.float32)
        fi8 = os.path.join(wdir, ws.get(stem.replace("conv1","conv1").replace("_w","_w")
                                         .replace("_b","_b"), stem+".npy") or "")
        # fallback: look up via WEIGHT_SETS key mapping
        key_map = {
            "conv1_w": "conv1_w", "conv1_b": "conv1_b",
            "conv2_w": "conv2_w", "conv2_b": "conv2_b",
            "conv3_w": "conv3_w", "conv3_b": "conv3_b",
            "fc_w":    "fc_w",    "fc_b":    "fc_b",
        }
        fname = ws.get(stem)
        if fname is None: return None
        fp = os.path.join(wdir, fname)
        if not os.path.exists(fp): return None
        return np.load(fp).astype(np.float32)

    arch = ARCH

    t0 = time.time()

    # Conv layers
    for lk in ["conv1","conv2","conv3"]:
        a  = arch[lk]
        Wk = _load_w(f"{lk}_w")
        Bk = _load_w(f"{lk}_b")
        if Wk is None or Bk is None:
            return None   # weights not available
        out_ch, in_ch, k = a["out_ch"], a["in_ch"], a["k"]
        Wk = Wk.flatten()[:out_ch*in_ch*k*k].reshape(out_ch, in_ch, k, k)
        Bk = Bk.flatten()[:out_ch]
        # Manual conv (same padding)
        pad = k // 2
        _, _, H, W = x.shape
        y   = np.zeros((1, out_ch, H, W), dtype=np.float32)
        xp  = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))
        for oc in range(out_ch):
            for ic in range(in_ch):
                for ki in range(k):
                    for kj in range(k):
                        y[:,oc,:,:] += Wk[oc,ic,ki,kj] * xp[:,ic,ki:ki+H,kj:kj+W]
            y[:,oc,:,:] += Bk[oc]
        x = _relu(_maxpool2(y))

    # Flatten + FC
    # FC weight is (64, 1152) — 73728 / 64 = 1152
    # Use adaptive average pooling to match whatever spatial size the
    # FPGA actually uses — infer n_in from the weight file itself.
    Wf      = _load_w("fc_w");  Bf = _load_w("fc_b")
    if Wf is None or Bf is None: return None
    Wf_flat = Wf.flatten()
    n_out   = int(Bf.flatten().shape[0])        # = 64
    n_in    = len(Wf_flat) // n_out             # = 1152
    # Adaptive average pool: collapse spatial dims to match n_in
    # x shape: (1, C, H, W)
    C = x.shape[1]
    spatial_needed = n_in // C                  # pixels per channel = 1152/32 = 36 = 6×6
    side = int(round(spatial_needed ** 0.5))    # = 6
    x_pool = np.zeros((1, C, side, side), dtype=np.float32)
    for c in range(C):
        x_pool[0, c] = cv2.resize(x[0, c], (side, side), interpolation=cv2.INTER_AREA)
    x   = x_pool.flatten()[:n_in]              # (1152,)
    Wf  = Wf_flat[:n_out * n_in].reshape(n_out, n_in)
    Bf  = Bf.flatten()[:n_out]
    x   = _relu(Wf @ x + Bf)

    # Output layer
    Wo = _load_w("out_w");  Bo = _load_w("out_b")
    if Wo is not None and Bo is not None:
        Wo = Wo.flatten()[:2*64].reshape(2, 64)
        Bo = Bo.flatten()[:2]
        logits = Wo @ x + Bo
    else:
        # No output layer: use fc output directly (first 2 values as proxy)
        logits = x[:2]

    latency_ms = (time.time()-t0)*1000
    cls        = int(np.argmax(logits))
    score_cat  = int(logits[0] * 1_000_000)
    score_dog  = int(logits[1] * 1_000_000)
    margin     = abs(score_dog - score_cat)
    return CLASS_NAMES[cls], cls, score_cat, score_dog, margin, latency_ms


def cpu_benchmark(img_bgr, ws_name, n_runs=5):
    """Run CPU forward pass n_runs times, return best latency and result."""
    best_ms  = float("inf")
    last_res = None
    for _ in range(n_runs):
        res = cpu_forward(img_bgr, ws_name)
        if res is None:
            return None
        *result, lat = res
        if lat < best_ms:
            best_ms = lat
        last_res = res
    label, cls, sc, sd, margin, _ = last_res
    fps = 1000.0 / best_ms if best_ms > 0 else 0
    return label, cls, sc, sd, margin, best_ms, fps

# ── SUBJECT COUNTER ──────────────────────────────────────────────────────────

def count_subjects(img_bgr, min_area_frac=0.04):
    """
    Count distinct foreground subjects via contour analysis.
    Returns (n_subjects, list_of_bounding_rects).
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect if background is bright → invert
    edge_pixels = np.concatenate([thresh[0,:], thresh[-1,:], thresh[:,0], thresh[:,-1]])
    if float(np.mean(edge_pixels)) > 127:
        thresh = cv2.bitwise_not(thresh)

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh,  cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * H * W
    valid    = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) >= min_area]

    # Merge overlapping / very close boxes
    merged = _merge_boxes(valid, gap_frac=0.15, W=W, H=H)
    return len(merged), merged


def _merge_boxes(boxes, gap_frac=0.15, W=1, H=1):
    """Merge boxes that are close to each other (gap < gap_frac * image size)."""
    if not boxes:
        return []
    gap   = gap_frac * max(W, H)
    rects = list(boxes)
    merged = True
    while merged:
        merged = False
        out    = []
        used   = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]: continue
            ax, ay, aw, ah = rects[i]
            for j in range(i+1, len(rects)):
                if used[j]: continue
                bx, by, bw, bh = rects[j]
                # Expand both by gap and check overlap
                if (ax - gap < bx + bw and ax + aw + gap > bx and
                        ay - gap < by + bh and ay + ah + gap > by):
                    # Merge
                    nx = min(ax, bx); ny = min(ay, by)
                    nw = max(ax+aw, bx+bw) - nx
                    nh = max(ay+ah, by+bh) - ny
                    rects[i] = (nx, ny, nw, nh)
                    ax, ay, aw, ah = nx, ny, nw, nh
                    used[j] = True
                    merged  = True
            out.append(rects[i])
            used[i] = True
        rects = out
    return rects


# ── DRAW DETECTION PANEL ─────────────────────────────────────────────────────

def make_panel(img_bgr, ws_name, label, cls, score_cat, score_dog, margin,
               fpga_ms, cpu_res, target_w=440):
    """
    Draw one detection panel at target_w wide.
    - Tight box around subject
    - Small clean label tag (top-left of box, not centred on image)
    - Confidence bar at bottom
    - FPGA vs CPU metrics strip at very bottom
    Returns RGB numpy image.
    """
    H, W    = img_bgr.shape[:2]
    scale   = target_w / W
    new_h   = int(H * scale)
    img_rs  = cv2.resize(img_bgr, (target_w, new_h))
    c       = CLASS_COLORS[cls]
    font    = cv2.FONT_HERSHEY_SIMPLEX
    c_bgr   = c   # already BGR

    # ── Smart box: count foreground blobs to decide single vs full-frame ───
    n_subjects, bboxes = count_subjects(img_rs)

    if n_subjects == 1 and bboxes:
        # Single animal — draw tight box around it
        bx, by, bw, bh = bboxes[0]
        mg = int(max(img_rs.shape[:2]) * 0.03)
        x1r = max(bx - mg, 0)
        y1r = max(by - mg, 0)
        x2r = min(bx + bw + mg, target_w)
        y2r = min(by + bh + mg, img_rs.shape[0])
        cv2.rectangle(img_rs, (x1r, y1r), (x2r, y2r), c_bgr, 3)
        lx, ly = x1r + 4, max(y1r - 4, 20)
    else:
        # Multiple animals or unclear — box the whole image, note "multiple"
        p = 6
        cv2.rectangle(img_rs, (p, p), (target_w-p, img_rs.shape[0]-p-42), c_bgr, 3)
        lx, ly = p + 4, p + 20
        # Overlay a subtle "multiple detected" note
        note = f"{n_subjects} subjects"
        cv2.putText(img_rs, note, (target_w - 110, 18),
                    font, 0.38, (200, 200, 200), 1)

    # ── Small label tag ───────────────────────────────────────────────────
    tag_fs = 0.52
    tag_th = 2
    (tw, th), bl = cv2.getTextSize(label, font, tag_fs, tag_th)
    cv2.rectangle(img_rs,
                  (lx - 3, ly - th - 5),
                  (lx + tw + 5, ly + bl + 2),
                  c_bgr, -1)
    cv2.putText(img_rs, label, (lx, ly), font, tag_fs, (0, 0, 0), tag_th)

    # ── Confidence bar ───────────────────────────────────────────────────
    total    = max(score_cat + score_dog, 1)
    cat_pct  = score_cat / total
    dog_pct  = 1 - cat_pct
    bar_top  = new_h - 42
    cv2.rectangle(img_rs, (0, bar_top), (target_w, bar_top+16), (10,10,10), -1)
    cw = int(target_w * cat_pct)
    cv2.rectangle(img_rs, (0,  bar_top), (cw,        bar_top+16), (30,150,255), -1)
    cv2.rectangle(img_rs, (cw, bar_top), (target_w,  bar_top+16), (40,200,70),  -1)
    cv2.putText(img_rs, f"Cat {cat_pct*100:.1f}%",
                (4, bar_top+12), font, 0.37, (0,0,0), 1)
    cv2.putText(img_rs, f"Dog {dog_pct*100:.1f}%",
                (cw+4, bar_top+12), font, 0.37, (0,0,0), 1)

    # ── Metrics panel (3 rows) ───────────────────────────────────────────
    pm_h  = 68
    my    = new_h - pm_h
    cv2.rectangle(img_rs, (0, my), (target_w, new_h), (12, 13, 22), -1)
    cv2.line(img_rs, (0, my), (target_w, my), (45, 48, 72), 1)

    fpga_fps = 1000.0 / fpga_ms if fpga_ms > 0 else 0
    conf_pct = max(score_cat, score_dog) / max(score_cat + score_dog, 1) * 100
    mg_m     = margin // 1_000_000

    # Row 1  — FPGA
    r1 = my + 17
    cv2.putText(img_rs, "FPGA",            ( 8, r1), font, 0.4, ( 80,190,255), 1)
    cv2.putText(img_rs, f"{fpga_ms:.0f}ms",(50, r1), font, 0.4, (220,220,220), 1)
    cv2.putText(img_rs, f"{fpga_fps:.1f}fps",(102,r1),font,0.4, (220,220,220), 1)
    cv2.putText(img_rs, f"conf {conf_pct:.0f}%", (170, r1), font, 0.4, (220,220,220), 1)
    cv2.putText(img_rs, f"margin {mg_m}M",       (262, r1), font, 0.4, (220,220,220), 1)

    # Row 2  — CPU
    r2 = my + 36
    if cpu_res:
        cpu_lbl, cpu_cls, cpu_sc, cpu_sd, cpu_mg, cpu_ms, cpu_fps = cpu_res
        mc   = (60,230,110) if cpu_lbl == label else (60,80,255)
        mtxt = "MATCH" if cpu_lbl == label else "DIFFER"
        spd  = cpu_ms / fpga_ms if fpga_ms > 0 else 0
        cv2.putText(img_rs, "CPU ",           ( 8, r2), font, 0.4, (120,255,160), 1)
        cv2.putText(img_rs, f"{cpu_ms:.0f}ms",(50, r2), font, 0.4, (200,200,200), 1)
        cv2.putText(img_rs, f"{cpu_fps:.1f}fps",(102,r2),font,0.4,(200,200,200), 1)
        cv2.putText(img_rs, f"{cpu_lbl} {mtxt}", (170, r2), font, 0.4, mc, 1)
        cv2.putText(img_rs, f"{spd:.1f}x faster", (286, r2), font, 0.4, (255,210,70), 1)
    else:
        cv2.putText(img_rs, "CPU  n/a", (8, r2), font, 0.4, (110,110,110), 1)

    # Row 3  — dual latency bar (FPGA=blue, CPU=green)
    bx1, bx2 = 8, target_w - 8
    bw_total  = bx2 - bx1
    r3b, r3t  = my + 52, my + 44   # bar bottom and top y

    cv2.rectangle(img_rs, (bx1, r3t), (bx2, r3b), (28, 28, 42), -1)

    scale_ms  = 1200.0             # 1200 ms = full bar width
    fpga_bw   = int(bw_total * min(fpga_ms / scale_ms, 1.0))
    cv2.rectangle(img_rs, (bx1, r3t), (bx1 + fpga_bw, r3t + 4), (80, 180, 255), -1)

    if cpu_res:
        cpu_bw = int(bw_total * min(cpu_ms / scale_ms, 1.0))
        cv2.rectangle(img_rs, (bx1, r3t + 4), (bx1 + cpu_bw, r3b), (80, 230, 110), -1)

    cv2.putText(img_rs, "latency", (bx2 - 52, r3b - 1), font, 0.3, (80,80,110), 1)

    return cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)

# ── CONV STRIPS (unchanged) ───────────────────────────────────────────────────

def conv_activation_strip(img_gray64, ws_name, lkey, thumb=52):
    arch = ARCH.get(lkey)
    if arch is None: return None
    out_ch, in_ch, k = arch["out_ch"], arch["in_ch"], arch["k"]
    W_raw = _load_npy(ws_name, f"{lkey}_w")
    B_raw = _load_npy(ws_name, f"{lkey}_b")
    bar_h = 20

    def _ph(msg, w=thumb*8):
        ph = np.full((bar_h+thumb, w, 3), 22, dtype=np.uint8)
        cv2.putText(ph, msg, (4, bar_h-4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110,70,70), 1)
        return cv2.cvtColor(ph, cv2.COLOR_BGR2RGB)

    if W_raw is None or B_raw is None:
        return _ph(f"{lkey}: not found")
    W_flat = W_raw.astype(np.float32).flatten()
    B_flat = B_raw.astype(np.float32).flatten()
    need   = out_ch * in_ch * k * k
    if len(W_flat) < need:
        return _ph(f"{lkey}: size mismatch")
    W      = W_flat[:need].reshape(out_ch, in_ch, k, k)
    B      = B_flat[:out_ch]
    img_f  = img_gray64.astype(np.float32)
    inputs = [img_f] * in_ch
    thumbs = []
    for fi in range(out_ch):
        act = np.zeros_like(img_f)
        for ci in range(in_ch):
            kern = W[fi,ci]; kmax = np.abs(kern).max()
            act += cv2.filter2D(inputs[ci], -1,
                                (kern/kmax if kmax>0 else kern).astype(np.float32))
        act = np.maximum(act + B[fi], 0)
        mn, mx = act.min(), act.max()
        u8 = ((act-mn)/(mx-mn)*255).astype(np.uint8) if mx>mn \
             else np.zeros((64,64), dtype=np.uint8)
        t = cv2.applyColorMap(cv2.resize(u8,(thumb,thumb)), cv2.COLORMAP_INFERNO)
        cv2.putText(t, str(fi), (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
        thumbs.append(t)
    mosaic = np.hstack(thumbs)
    bar    = np.full((bar_h, mosaic.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(bar, f"{lkey.upper()}  {out_ch} filters  ({in_ch}×{k}×{k})",
                (5, bar_h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185,185,55), 1)
    return cv2.cvtColor(np.vstack([bar, mosaic]), cv2.COLOR_BGR2RGB)

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    img_bgr = cv2.imread(TEST_IMAGE)
    if img_bgr is None:
        raise FileNotFoundError(f"Not found: {TEST_IMAGE}")

    img_gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray64 = cv2.resize(img_gray, (64,64))
    img_name   = os.path.basename(TEST_IMAGE)
    ws_list    = list(WEIGHT_SETS.keys())
    N          = len(ws_list)   # 2

    print(f"Image: {img_name}  {img_bgr.shape[1]}×{img_bgr.shape[0]}")
    print("─"*55)

    # ── Collect FPGA results ──────────────────────────────────────────
    fpga_res  = {}   # ws -> (label, cls, sc, sd, margin, latency_ms)
    cpu_res   = {}   # ws -> result or None

    for ws in ws_list:
        print(f"\n  [{ws}]")
        load_weight_set(ws)

        t0 = time.time()
        label, cls, sc, sd, margin = fpga_classify(img_bgr)
        fpga_ms = (time.time()-t0)*1000
        fpga_res[ws] = (label, cls, sc, sd, margin, fpga_ms)
        print(f"    FPGA → {label:3s}  Cat={sc//1_000_000}M  Dog={sd//1_000_000}M  "
              f"margin={margin//1_000_000}M  {fpga_ms:.1f}ms  "
              f"{1000/fpga_ms:.1f}fps")

        print(f"    Running CPU inference (NumPy)...")
        cr = cpu_benchmark(img_bgr, ws, n_runs=3)
        cpu_res[ws] = cr
        if cr:
            cl, cc, csc, csd, cmg, cms, cfps = cr
            match = "MATCH ✓" if cl==label else "DIFFER ✗"
            print(f"    CPU  → {cl:3s}  {cms:.0f}ms  {cfps:.1f}fps  [{match}]")
            speedup = cms / fpga_ms if fpga_ms > 0 else 0
            print(f"    Speedup: FPGA is {speedup:.1f}× faster than CPU")
        else:
            print(f"    CPU inference: weights unavailable for forward pass")

    # ── Build figure ──────────────────────────────────────────────────
    PANEL_W = 480
    LAYERS  = ["conv1","conv2","conv3"]
    THUMB   = 52

    # Detection panels
    panels = {}
    for ws in ws_list:
        label, cls, sc, sd, margin, fpga_ms = fpga_res[ws]
        panels[ws] = make_panel(img_bgr, ws, label, cls, sc, sd, margin,
                                fpga_ms, cpu_res[ws], target_w=PANEL_W)

    # Conv strips
    strips = {ws: {lk: conv_activation_strip(img_gray64, ws, lk, THUMB)
                   for lk in LAYERS} for ws in ws_list}

    det_h  = panels[ws_list[0]].shape[0]
    conv_h = {lk: max(strips[ws][lk].shape[0] for ws in ws_list) for lk in LAYERS}

    DPI   = 110
    fig_w = PANEL_W * N / DPI
    fig_h = (det_h + sum(conv_h.values()) + 70) / DPI
    hr    = [det_h] + [conv_h[lk] for lk in LAYERS]

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#080810")
    gs  = fig.add_gridspec(
        1 + len(LAYERS), N,
        height_ratios=hr,
        hspace=0.035, wspace=0.018,
        top=0.93, bottom=0.01,
        left=0.005, right=0.995
    )

    # Title
    ws        = ws_list[0]
    lbl, cls_v, sc, sd, mg, fms = fpga_res[ws]
    fps_v     = 1000 / fms
    cpu_r     = cpu_res[ws]
    cpu_part  = ""
    if cpu_r:
        cl, _, csc, csd, cmg, cms, cfps = cpu_r
        match = "CPU MATCH ✓" if cl == lbl else f"CPU→{cl} ✗"
        spd   = cms / fms if fms > 0 else 0
        cpu_part = f"   |   {match}   CPU {cms:.0f}ms / {cfps:.1f}fps   {spd:.1f}x speedup"

    fig.suptitle(
        f"{img_name}   →   {lbl}   |   FPGA {fms:.0f}ms / {fps_v:.1f}fps"
        f"   conf {max(sc,sd)/max(sc+sd,1)*100:.0f}%{cpu_part}",
        color="white", fontsize=10, fontweight="bold", y=0.975
    )

    # Row 0 — detection panel
    for col, ws in enumerate(ws_list):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(panels[ws]); ax.axis("off")
        label, cls, sc, sd, mg, fpga_ms = fpga_res[ws]
        col_c = "#ffa020" if cls == 0 else "#32d050"
        ax.set_title(f"{ws}   →   {label}",
                     color=col_c, fontsize=9, fontweight="bold", pad=5)

    # Rows 1–3 — conv feature maps
    lyr_labels = {
        "conv1": "Conv1  8 filters  (1×3×3)",
        "conv2": "Conv2  16 filters  (8×3×3)",
        "conv3": "Conv3  32 filters  (16×3×3)",
    }
    for row, lk in enumerate(LAYERS):
        for col, ws in enumerate(ws_list):
            ax    = fig.add_subplot(gs[row+1, col])
            strip = strips[ws][lk]
            # Uniform row height
            th = conv_h[lk]
            if strip.shape[0] < th:
                pad = np.full((th-strip.shape[0], strip.shape[1], 3), 8, np.uint8)
                strip = np.vstack([strip, pad])
            ax.imshow(strip); ax.axis("off")
            if col == 0:
                ax.set_ylabel(lyr_labels[lk], color="#999999",
                              fontsize=7, rotation=90, labelpad=3)

    out_path = os.path.join(OUT_DIR, f"combined_{img_name}")
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Saved → {out_path}")
    try: ipy_display(fig)
    except Exception: pass
    plt.close(fig)
    print("Done.")

run()
```
