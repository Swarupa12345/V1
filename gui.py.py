#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLOB DETECTION PIPELINE  v3 (with Canny → Dilation → Mask → Inpaint)
"""

import json
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage
from skimage.feature import blob_log

# ── HELPERS ────────────────────────────────────────────────────────────────────
def _ordered_step_name(step_id: int, tag: str) -> str:
    return f"{step_id:02d}_{tag}.png"


def _clamp_roi(points, shape):
    h, w = shape
    return [[min(max(x, 0), w - 1), min(max(y, 0), h - 1)] for x, y in points]


# ── COLOUR PALETTE ─────────────────────────────────────────────────────────────
C = {
    "bg_deep":   "#1e2026",
    "bg_base":   "#282a36",
    "bg_raised": "#353b45",
    "bg_hover":  "#434758",
    "border":    "#ffffff",
    "border_hi": "#ffffff",
    "cyan":      "#00b4d8",
    "cyan_dim":  "#0077a8",
    "cyan_glow": "#48cae4",
    "amber":     "#f0a500",
    "amber_dim": "#7a5200",
    "green":     "#3fb950",
    "red":       "#f85149",
    "purple":    "#bc8cff",
    "text_hi":   "#f8f8f2",
    "text_mid":  "#d6deeb",
    "text_lo":   "#a3a7b4"
}

# ── DEFAULT PARAMETERS ─────────────────────────────────────────────────────────
DEFAULTS = dict(
    PAD=93, S_MIN=47, S_MAX=93,
    THRESHOLD=0.40, OVERLAP=0.20,
    CLIP_LIMIT=3.0, TILE_SIZE=8,
    ALPHA=2.0, EPS=1e-6, NUM_SIGMA=10,
    CANNY_VAR=1.0, CANNY_MAXERR=0.01,
    CANNY_LO=50, CANNY_HI=150, CANNY_AP=3, CANNY_L2=False,
    DIL_SHAPE="Ellipse", DIL_K=3, DIL_I=1,
    MASK_CONN="Cross", MASK_ERODE=0, MASK_DILATE=0,
    MASK_OVERLAY=1, MASK_INVERT=0,
    INPAINT_METHOD="TELEA", INPAINT_R=5,
)

MAX_STEP = 18

# ── STEP DESCRIPTIONS ──────────────────────────────────────────────────────────
STEPS = [
    {"id": 0,  "tag": "ORIG",   "title": "Original Image",
     "desc": "Raw loaded grayscale image — no processing applied yet.",
     "params": []},
    {"id": 1,  "tag": "ROI",    "title": "ROI Selection",
     "desc": "Draw a rectangle by click-drag.",
     "params": []},
    {"id": 2,  "tag": "ROT",    "title": "Rotate / Flip",
     "desc": "Rotate the image in 90° steps or flip H / V.",
     "params": []},
    {"id": 3,  "tag": "INV",    "title": "Invert",
     "desc": "Pixel values inverted (255 − I).",
     "params": []},
    {"id": 4,  "tag": "CLAHE",  "title": "CLAHE",
     "desc": "Contrast-Limited Adaptive Histogram Equalisation",
     "params": [
         dict(key="CLIP_LIMIT", label="Clip Limit", unit="",   mn=0.5, mx=50.0, res=0.1, fmt="{:.1f}"),
         dict(key="TILE_SIZE",  label="Tile Grid Size", unit="px", mn=2,   mx=60,   res=1,   fmt="{:.0f}"),
     ]},
    {"id": 5,  "tag": "CANNY",  "title": "Canny Edge",
     "desc": "Variance-based smoothing + Canny edge detection.",
     "params": [
         dict(key="CANNY_VAR",    label="Variance (σ²)",          mn=0.01,  mx=80.0,  res=0.01, fmt="{:.2f}"),
         dict(key="CANNY_MAXERR", label="Max error (kernel cutoff)", mn=0.001, mx=0.5,   res=0.001, fmt="{:.3f}"),
         dict(key="CANNY_LO",    label="Lower threshold",         mn=0,     mx=10000, res=1,    fmt="{:.0f}"),
         dict(key="CANNY_HI",    label="Upper threshold",         mn=0,     mx=15000, res=1,    fmt="{:.0f}"),
         dict(key="CANNY_AP",    label="Aperture (3/5/7)",        mn=3,     mx=7,     res=2,    fmt="{:.0f}"),
         dict(key="CANNY_L2",    label="L2 gradient",             type="check"),
     ]},
    {"id": 6,  "tag": "DIL",    "title": "Dilation",
     "desc": "Thickens edges to close small gaps before hole-filling.",
     "params": [
         dict(key="DIL_SHAPE", label="Kernel shape", type="radio",
              options=[("Rect", "Rect"), ("Ellipse", "Ellipse"), ("Cross", "Cross")]),
         dict(key="DIL_K",     label="Kernel size (odd)", mn=1, mx=31, res=2, fmt="{:.0f}"),
         dict(key="DIL_I",     label="Iterations",        mn=1, mx=20, res=1, fmt="{:.0f}"),
     ]},
    {"id": 7,  "tag": "MASK",   "title": "Mask — Binary Fill Hole",
     "desc": "Fills every enclosed region automatically.",
     "params": [
         dict(key="MASK_CONN",   label="Connectivity", type="radio",
              options=[("Cross (4-conn)", "Cross"), ("Square (8-conn)", "Square")]),
         dict(key="MASK_ERODE",  label="Erode iterations",  mn=0, mx=20, res=1, fmt="{:.0f}"),
         dict(key="MASK_DILATE", label="Dilate iterations", mn=0, mx=20, res=1, fmt="{:.0f}"),
         dict(key="MASK_OVERLAY", label="Show overlay (green)", type="check"),
         dict(key="MASK_INVERT", label="Invert mask",           type="check"),
     ]},
    {"id": 8,  "tag": "INPAINT", "title": "Inpaint",
     "desc": "Fill the masked region (Fast-Marching = TELEA or Navier-Stokes).",
     "params": [
         dict(key="INPAINT_METHOD", label="Method", type="radio",
              options=[("Fast Marching (TELEA)", "TELEA"), ("Navier-Stokes (NS)", "NS")]),
         dict(key="INPAINT_R", label="Inpaint radius (px)", mn=1, mx=500, res=1, fmt="{:.0f}"),
     ]},
    {"id": 9,  "tag": "PAD",    "title": "Padding",
     "desc": "Adds a white border to prevent edge.",
     "params": [dict(key="PAD", label="Pad Size", unit="px", mn=10, mx=200, res=1, fmt="{:.0f}")]},
    {"id": 10, "tag": "INV2",   "title": "Invert (again)",
     "desc": "Second inversion — restores original polarity after padding.",
     "params": []},
    {"id": 11, "tag": "CLAHE2", "title": "CLAHE (again)",
     "desc": "Second CLAHE pass on the padded image.",
     "params": [
         dict(key="CLIP_LIMIT", label="Clip Limit", unit="",   mn=0.5, mx=50.0, res=0.1, fmt="{:.1f}"),
         dict(key="TILE_SIZE",  label="Tile Grid Size", unit="px", mn=2,   mx=60,   res=1,   fmt="{:.0f}"),
     ]},
    {"id": 12, "tag": "INT",    "title": "Integral Image",
     "desc": "Summed-area table enables O(1)",
     "params": []},
    {"id": 13, "tag": "BOX",    "title": "Box Filtering",
     "desc": "Mean-intensity box filters at every scale",
     "params": [
         dict(key="S_MIN", label="S MIN", unit="px", mn=5,  mx=150, res=2, fmt="{:.0f}"),
         dict(key="S_MAX", label="S MAX", unit="px", mn=10, mx=300, res=2, fmt="{:.0f}"),
     ]},
    {"id": 14, "tag": "BLOB",   "title": "Blob Filter",
     "desc": "Directional gradient filter amplifying blob",
     "params": [dict(key="ALPHA", label="Alpha", unit="", mn=0.5, mx=50.0, res=0.5, fmt="{:.1f}")]},
    {"id": 15, "tag": "SEL",    "title": "Scale Selection",
     "desc": "Per-pixel maximum response across all scales",
     "params": []},
    {"id": 16, "tag": "NORM",   "title": "Normalise",
     "desc": "Response map rescaled to [0, 1]",
     "params": []},
    {"id": 17, "tag": "LOG",    "title": "LoG Detection",
     "desc": "scikit-image blob_log locates blobs",
     "params": [
         dict(key="THRESHOLD", label="Threshold", unit="", mn=0.05, mx=1.0,  res=0.05, fmt="{:.2f}"),
         dict(key="OVERLAP",   label="Overlap",   unit="", mn=0.0,  mx=1.0,  res=0.05, fmt="{:.2f}"),
         dict(key="NUM_SIGMA", label="Num Sigma", unit="", mn=3,    mx=30,   res=1,    fmt="{:.0f}"),
     ]},
    {"id": 18, "tag": "OUT",    "title": "Final Detections",
     "desc": "Blobs overlaid on processed image. Red circle = boundary",
     "params": []},
]


# ── STEP RUNNERS ───────────────────────────────────────────────────────────────
def run_step0(state, p):
    return state["img0"]


def run_step1(state, p):
    img = state["img0"]
    rect = state.get("roi_rect")
    vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if rect:
        (x1, y1), (x2, y2) = rect
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        roi_img = masked[y1:y2, x1:x2]
    else:
        roi_img = img.copy()
    state["roi_image"] = roi_img
    return vis


def run_step2(state, p):
    img = state.get("roi_image", state["img0"]).copy()
    rot = state.get("rotation", 0) % 360
    fh  = state.get("flip_h", False)
    fv  = state.get("flip_v", False)
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if fh:
        img = cv2.flip(img, 1)
    if fv:
        img = cv2.flip(img, 0)
    state["processed"] = img
    return img


def run_step3_inv(state, p):
    src = state.get("processed", state["img0"])
    inv = 255 - src
    state["inverted"] = inv
    return inv


def run_step4_clahe(state, p):
    clahe = cv2.createCLAHE(
        clipLimit=float(p["CLIP_LIMIT"]),
        tileGridSize=(int(p["TILE_SIZE"]),) * 2,
    )
    img = state.get("inverted")
    out = clahe.apply(img).astype(np.uint8)
    state["clahe"] = out
    return out


def run_step5(state, p):
    gray = state["clahe"] if "processed" in state else state["img0"]
    variance = max(1e-4, p["CANNY_VAR"])
    sigma = math.sqrt(variance)
    max_error = max(1e-6, p["CANNY_MAXERR"])
    half = sigma * math.sqrt(-2.0 * math.log(max_error))
    ksize = 2 * int(math.ceil(half)) + 1
    ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    ksize = min(ksize, 31)
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    lo = int(p["CANNY_LO"])
    hi = int(p["CANNY_HI"])
    ap = int(p["CANNY_AP"])
    ap = ap if ap % 2 == 1 else ap + 1
    ap = max(3, min(7, ap))
    l2 = bool(p["CANNY_L2"])
    edges = cv2.Canny(smoothed, lo, hi, apertureSize=ap, L2gradient=l2)
    state["edges"] = edges
    return edges


def run_step6(state, p):
    if state.get("edges") is None:
        return None
    shape_map = {"Rect": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
    shape = shape_map.get(p["DIL_SHAPE"], cv2.MORPH_ELLIPSE)
    k = int(p["DIL_K"])
    k = k if k % 2 == 1 else k + 1
    k = max(1, k)
    iters = int(p["DIL_I"])
    kern = cv2.getStructuringElement(shape, (k, k))
    dilated = cv2.dilate(state["edges"], kern, iterations=iters)
    state["dilated"] = dilated
    return dilated


def run_step7(state, p):
    if state.get("dilated") is None:
        return None
    binary = state["dilated"] > 0
    struct = (np.ones((3, 3), dtype=bool) if p["MASK_CONN"] == "Square"
              else ndimage.generate_binary_structure(2, 1))
    filled = (ndimage.binary_fill_holes(binary, structure=struct).astype(np.uint8) * 255)
    en = int(p["MASK_ERODE"])
    dn = int(p["MASK_DILATE"])
    if en > 0:
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled = cv2.erode(filled, ke, iterations=en)
    if dn > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled = cv2.dilate(filled, kd, iterations=dn)
    if p["MASK_INVERT"]:
        filled = cv2.bitwise_not(filled)
    state["mask"] = filled
    if p["MASK_OVERLAY"]:
        disp = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)
        disp[state["dilated"] > 0] = [80, 220, 130]
        return disp
    else:
        return filled


def run_step8(state, p):
    if state.get("mask") is None:
        return None
    img = state.get("processed", state["img0"])
    method = cv2.INPAINT_TELEA if p["INPAINT_METHOD"] == "TELEA" else cv2.INPAINT_NS
    radius = int(p["INPAINT_R"])
    inpainted = cv2.inpaint(img, state["mask"], radius, method)
    state["inpainted"] = inpainted
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def run_step9(state, p):
    src = state.get("inpainted", state["processed"])
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    pad = int(p["PAD"])
    padded = cv2.copyMakeBorder(src, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    state["padded"] = padded
    state["PAD"] = pad
    return padded


def run_step10_inv(state, p):
    src = state.get("padded")
    inv = 255 - src
    state["inverted2"] = inv
    return inv


def run_step11_clahe(state, p):
    clahe = cv2.createCLAHE(
        clipLimit=float(p["CLIP_LIMIT"]),
        tileGridSize=(int(p["TILE_SIZE"]),) * 2,
    )
    img = state.get("inverted2")
    out = clahe.apply(img).astype(np.uint8)
    state["clahe2"] = out
    return out


def run_step12(state, p):
    integral = cv2.integral(state["clahe2"])
    state["integral"] = integral
    return cv2.normalize(integral, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def run_step13(state, p):
    s_min = int(p["S_MIN"]); s_max = int(p["S_MAX"])
    if s_min >= s_max:
        s_min = max(3, s_max - 2)
    integral = state["integral"]
    H, W = state["padded"].shape
    scales = list(range(s_min, s_max + 1, 2)) or [s_min]
    box_images = {}
    for S in scales:
        half = S // 2
        if half + 1 > H - half or half + 1 > W - half:
            continue
        A  = integral[half+1:H-half+1, half+1:W-half+1]
        B  = integral[half+1:H-half+1, :W-S]
        C2 = integral[:H-S,            half+1:W-half+1]
        D  = integral[:H-S,            :W-S]
        r  = min(A.shape[0], B.shape[0], C2.shape[0], D.shape[0])
        c  = min(A.shape[1], B.shape[1], C2.shape[1], D.shape[1])
        m  = (A[:r,:c] - B[:r,:c] - C2[:r,:c] + D[:r,:c]) / (S * S)
        pb = np.zeros((H, W), dtype=np.float32)
        pb[half:half+r, half:half+c] = m
        box_images[S] = pb
    state["box_images"] = box_images
    state["scales"] = scales
    if not box_images:
        return state["padded"]
    Sm = scales[len(scales) // 2]
    return cv2.normalize(box_images[Sm], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def run_step14(state, p):
    alpha = float(p["ALPHA"]); eps = 1e-6
    box_images = state.get("box_images", {})
    scales = state.get("scales", [])
    if not box_images:
        return state["padded"]
    H, W = state["padded"].shape
    blob_maps = {}
    for S in scales:
        r0=box_images[S][S:-S,S:-S];   r4=box_images[S][S:-S,:-2*S];  r8=box_images[S][S:-S,2*S:]
        r2=box_images[S][:-2*S,S:-S];  r6=box_images[S][2*S:,S:-S]
        r1=box_images[S][:-2*S,:-2*S]; r5=box_images[S][2*S:,2*S:]
        r3=box_images[S][:-2*S,2*S:];  r7=box_images[S][2*S:,:-2*S]
        mr=min(r0.shape[0],r4.shape[0],r8.shape[0],r2.shape[0],r6.shape[0])
        mc=min(r0.shape[1],r4.shape[1],r8.shape[1],r2.shape[1],r6.shape[1])
        r0,r4,r8,r2,r6 = (x[:mr,:mc] for x in (r0,r4,r8,r2,r6))
        r1,r5,r3,r7    = (x[:mr,:mc] for x in (r1,r5,r3,r7))
        gh  = r0 - np.maximum(r4, r8); gv  = r0 - np.maximum(r2, r6)
        gld = r0 - np.maximum(r1, r5); grd = r0 - np.maximum(r3, r7)
        L1  = (np.minimum(np.abs(gh), np.abs(gv)) /
               (np.maximum(np.abs(gh), np.abs(gv)) + eps)) ** alpha
        L2  = (np.minimum(np.abs(gld), np.abs(grd)) /
               (np.maximum(np.abs(gld), np.abs(grd)) + eps)) ** alpha
        f   = np.zeros((H, W), dtype=np.float32)
        f[S:S+mr, S:S+mc] = (gh+gv)*L1 + (gld+grd)*L2
        blob_maps[S] = f
    state["blob_maps"] = blob_maps
    if not blob_maps:
        return state["padded"]
    Sm = scales[len(scales) // 2]
    bm = blob_maps.get(Sm, list(blob_maps.values())[0])
    return cv2.normalize(bm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def run_step15(state, p):
    blob_maps = state.get("blob_maps", {}); scales = state.get("scales", [])
    if not blob_maps:
        return state["padded"]
    H, W = state["padded"].shape
    fr = np.zeros((H, W), dtype=np.float32)
    for S in scales:
        if S in blob_maps:
            mask = blob_maps[S] > fr
            fr[mask] = blob_maps[S][mask]
    state["final_response"] = fr
    return cv2.normalize(fr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def run_step16(state, p):
    fr = state.get("final_response", state["padded"].astype(np.float32))
    bn = fr / (fr.max() + 1e-6)
    state["blob_norm"] = bn
    return (bn * 255).astype(np.uint8)


def run_step17(state, p):
    bn = state.get("blob_norm")
    if bn is None:
        return state["padded"]
    blobs = blob_log(
        bn,
        min_sigma=(int(p["S_MIN"]) / 2) / np.sqrt(2),
        max_sigma=(int(p["S_MAX"]) / 2) / np.sqrt(2),
        num_sigma=int(p["NUM_SIGMA"]),
        threshold=float(p["THRESHOLD"]),
        overlap=float(p["OVERLAP"]),
    )
    state["blobs"] = blobs
    vis = cv2.cvtColor(state["padded"].astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for y, x, sigma in blobs:
        r = int(sigma * np.sqrt(2))
        cv2.circle(vis, (int(x), int(y)), r, (0, 0, 255), 2)
        cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), 3)
    return vis


def run_step18(state, p):
    blobs = state.get("blobs", np.empty((0, 3)))
    pad   = state.get("PAD", 0)
    base  = state.get("processed", state["img0"])
    if base.ndim == 3:
        vis = base.copy()
    else:
        vis = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for y, x, sigma in blobs:
        r  = int(sigma * np.sqrt(2))
        xi = int(x - pad); yi = int(y - pad)
        if 0 <= xi < vis.shape[1] and 0 <= yi < vis.shape[0]:
            cv2.circle(vis, (xi, yi), r, (0, 0, 255), 2)
            cv2.circle(vis, (xi, yi), 2, (0, 255, 0), 3)
    return vis


STEP_RUNNERS = [
    run_step0, run_step1, run_step2, run_step3_inv, run_step4_clahe,
    run_step5, run_step6, run_step7, run_step8, run_step9,
    run_step10_inv, run_step11_clahe, run_step12, run_step13, run_step14,
    run_step15, run_step16, run_step17, run_step18,
]


# ── PARAMETER ROW ─────────────────────────────────────────────────────────────
class ParamRow(tk.Frame):
    def __init__(self, master, meta, param_vars, on_change, **kw):
        super().__init__(master, bg=C["bg_base"], **kw)
        self.meta = meta; self.key = meta["key"]
        self.res  = meta["res"]; self.fmt = meta["fmt"]
        self.var  = param_vars[self.key]; self.on_change = on_change

        cb = tk.Frame(self, bg=C["border_hi"]); cb.pack(fill="x", padx=12, pady=6)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)

        hdr = tk.Frame(card, bg=C["bg_raised"]); hdr.pack(fill="x", padx=10, pady=(9, 3))
        pill = tk.Frame(hdr, bg=C["cyan"]); pill.pack(side="left")
        tk.Label(pill, text=f"  {meta['label'].upper()}  ",
                 bg=C["cyan"], fg=C["bg_deep"], font=("Ubuntu", 8, "bold"), pady=2).pack()
        unit = meta.get("unit", "")
        if unit:
            tk.Label(hdr, text=f" {unit}", bg=C["bg_raised"], fg=C["text_lo"],
                     font=("Ubuntu", 8)).pack(side="left")
        hint = f"RANGE  {meta['mn']:{meta['fmt']}} – {meta['mx']:{meta['fmt']}}  |  STEP {meta['res']:{meta['fmt']}}"
        tk.Label(hdr, text=hint, bg=C["bg_raised"], fg=C["text_lo"],
                 font=("DejaVu Sans Mono", 8)).pack(side="right")

        srow = tk.Frame(card, bg=C["bg_raised"]); srow.pack(fill="x", padx=10, pady=(2, 2))
        tk.Button(srow, text="▼", width=3, bg=C["bg_hover"], fg=C["red"],
                  activebackground=C["border_hi"], activeforeground=C["red"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 8, "bold"),
                  command=self._step_down).pack(side="left", padx=(0, 4))
        self.scale = tk.Scale(srow, from_=meta["mn"], to=meta["mx"], resolution=meta["res"],
                              orient="horizontal", showvalue=0, variable=self.var,
                              bg=C["bg_raised"], fg=C["cyan"], troughcolor=C["bg_hover"],
                              highlightthickness=1, highlightbackground=C["border_hi"],
                              highlightcolor=C["cyan"], sliderrelief="raised",
                              sliderlength=18, width=12, activebackground=C["cyan_glow"],
                              command=self._on_slider)
        self.scale.pack(side="left", fill="x", expand=True)
        tk.Button(srow, text="▲", width=3, bg=C["bg_hover"], fg=C["green"],
                  activebackground=C["border_hi"], activeforeground=C["green"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 8, "bold"),
                  command=self._step_up).pack(side="left", padx=(4, 0))

        vrow = tk.Frame(card, bg=C["bg_raised"]); vrow.pack(fill="x", padx=10, pady=(0, 10))
        ew = tk.Frame(vrow, bg=C["amber_dim"]); ew.pack(side="left")
        self.entry_var = tk.StringVar(value=self._fmt(self.var.get()))
        self.entry = tk.Entry(ew, textvariable=self.entry_var, width=10, justify="center",
                              bg=C["bg_deep"], fg=C["amber"], insertbackground=C["amber"],
                              relief="flat", bd=4, font=("DejaVu Sans Mono", 11, "bold"))
        self.entry.pack()
        tk.Label(vrow, text="   PRESS  ENTER  TO APPLY", bg=C["bg_raised"],
                 fg=C["text_lo"], font=("Ubuntu", 8)).pack(side="left")
        self.entry.bind("<Return>",   self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)
        self._trace_id = self.var.trace_add("write", self._sync_entry)

    def _fmt(self, v): return f"{v:{self.fmt}}"
    def _on_slider(self, _=None):
        self.entry_var.set(self._fmt(self.var.get())); self.on_change()
    def _on_entry(self, _=None):
        try:
            v = float(self.entry_var.get().strip())
            v = max(self.meta["mn"], min(self.meta["mx"], v))
            self.var.trace_remove("write", self._trace_id)
            self.var.set(round(v, 8))
            self._trace_id = self.var.trace_add("write", self._sync_entry)
            self.entry_var.set(self._fmt(self.var.get()))
            self.scale.set(self.var.get()); self.on_change()
        except ValueError:
            self.entry_var.set(self._fmt(self.var.get()))
    def _step_up(self):
        v = min(self.meta["mx"], self.var.get() + self.res)
        self.var.set(round(v, 8)); self.on_change()
    def _step_down(self):
        v = max(self.meta["mn"], self.var.get() - self.res)
        self.var.set(round(v, 8)); self.on_change()
    def _sync_entry(self, *_):
        self.entry_var.set(self._fmt(self.var.get()))


# ── PARAMETER CONTROL ─────────────────────────────────────────────────────────
class ParamControl(tk.Frame):
    def __init__(self, master, meta, pvars, on_change, **kw):
        super().__init__(master, bg=C["bg_raised"], **kw)
        self.meta = meta
        self.key  = meta["key"]
        self.pvars = pvars
        self.on_change = on_change

        if meta.get("type") == "check":
            var = pvars[self.key]
            tk.Checkbutton(
                self, text=meta["label"], variable=var,
                bg=C["bg_raised"], fg=C["text_mid"],
                selectcolor=C["bg_base"],
                command=self._trigger
            ).pack(anchor="w", padx=10, pady=4)
            return

        if meta.get("type") == "radio":
            var = pvars[self.key]
            tk.Label(self, text=meta["label"], bg=C["bg_raised"],
                     fg=C["text_mid"]).pack(anchor="w", padx=10, pady=(6, 2))
            for txt, val in meta["options"]:
                tk.Radiobutton(
                    self, text=txt, variable=var, value=val,
                    bg=C["bg_raised"], fg=C["text_mid"],
                    selectcolor=C["bg_base"],
                    command=self._trigger
                ).pack(anchor="w", padx=20, pady=1)
            if all(k in meta for k in ("mn", "mx", "res")):
                self._add_range_label()
            return

        var = pvars[self.key]
        mn, mx, step = meta["mn"], meta["mx"], meta["res"]
        fmt = meta.get("fmt", "{:.2f}")

        tk.Label(self, text=meta["label"], bg=C["bg_raised"],
                 fg=C["text_mid"]).pack(anchor="w", padx=10, pady=(6, 2))

        row = tk.Frame(self, bg=C["bg_raised"])
        row.pack(fill="x", padx=10, pady=2)

        self.scale = tk.Scale(row, from_=mn, to=mx, resolution=step,
                             orient="horizontal", showvalue=0, variable=var,
                             bg=C["bg_raised"], fg=C["cyan"],
                             troughcolor=C["bg_hover"],
                             highlightthickness=1, highlightbackground=C["border_hi"],
                             highlightcolor=C["cyan"], sliderrelief="raised",
                             sliderlength=18, width=12,
                             activebackground=C["cyan_glow"],
                             command=lambda _: self._trigger())
        self.scale.pack(side="left", fill="x", expand=True)

        tk.Button(row, text="▼", width=3, bg=C["bg_hover"], fg=C["red"],
                  activebackground=C["border_hi"], activeforeground=C["red"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 8, "bold"),
                  command=self._step_down).pack(side="left", padx=(0, 4))

        tk.Button(row, text="▲", width=3, bg=C["bg_hover"], fg=C["green"],
                  activebackground=C["border_hi"], activeforeground=C["green"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 8, "bold"),
                  command=self._step_up).pack(side="left", padx=(4, 0))

        entry_var = tk.StringVar(value=fmt.format(var.get()))
        entry = tk.Entry(
            row, textvariable=entry_var, width=8, justify="center",
            bg=C["bg_deep"], fg=C["amber"], insertbackground=C["amber"],
            relief="flat", bd=4, font=("DejaVu Sans Mono", 11, "bold")
        )
        entry.pack(side="right", padx=4)

        def sync_entry(*_):
            entry_var.set(fmt.format(var.get()))
            self._trigger()

        var.trace_add("write", sync_entry)

        def on_entry(event=None):
            try:
                v = float(entry_var.get())
                v = max(mn, min(mx, v))
                var.set(round(v, 8))
            except ValueError:
                entry_var.set(fmt.format(var.get()))

        entry.bind("<Return>", on_entry)
        entry.bind("<FocusOut>", on_entry)
        self._add_range_label()

    def _add_range_label(self):
        mn = self.meta.get("mn")
        mx = self.meta.get("mx")
        step = self.meta.get("res")
        if mn is None or mx is None or step is None:
            return
        fmt = self.meta.get("fmt", "{:.2f}")
        txt = f"range: {fmt.format(mn)} – {fmt.format(mx)}  (step {fmt.format(step)})"
        tk.Label(self, text=txt, bg=C["bg_raised"], fg=C["text_lo"],
                 font=("DejaVu Sans Mono", 8)).pack(anchor="w", padx=12, pady=(2, 4))

    def _trigger(self):
        self.on_change()

    def _step_up(self):
        v = min(self.meta["mx"], self.pvars[self.key].get() + self.meta["res"])
        self.pvars[self.key].set(round(v, 8))
        self._trigger()

    def _step_down(self):
        v = max(self.meta["mn"], self.pvars[self.key].get() - self.meta["res"])
        self.pvars[self.key].set(round(v, 8))
        self._trigger()


# ── MAIN GUI CLASS ─────────────────────────────────────────────────────────────
class BlobGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DEFECT DETECTION")
        self.configure(bg=C["bg_deep"])
        self.resizable(True, True)
        self.minsize(1020, 680)

        self.img0        = None
        self.state       = {}
        self.step_cache  = {}
        self.current     = 0
        self._current_pil = None
        self.pvars = {k: self._make_tk_var(v) for k, v in DEFAULTS.items()}
        self.logged_in   = False
        self.config_path: str | None = None
        self.roi_rect    = None
        self._temp_rect_id = None
        self.rotation    = 0
        self.flip_h      = False
        self.flip_v      = False
        self._img_scale  = 1.0
        self._img_offset = (0, 0)
        self._msg_queue  = queue.Queue()
        self._poll_queue()

        self._ctrl_down  = False
        self._ab_pressed = set()
        self.bind_all("<KeyPress>",   self._on_key_press)
        self.bind_all("<KeyRelease>", self._on_key_release)

        self._last_params = {}
        self._stop_event  = threading.Event()
        self._batch_thread = None

        self._build_global_topbar()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.tab1 = tk.Frame(self.nb, bg=C["bg_deep"])
        self.tab2 = tk.Frame(self.nb, bg=C["bg_deep"])

        self.nb.add(self.tab2, text="DEFECT DETECTION")
        self.folder_path   = tk.StringVar()
        self.output_folder = tk.StringVar()

        self._build_tab1(self.tab1)
        self._build_tab2(self.tab2)
        self._init_progress_ui(self.tab2)

        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.isfile(logo_path):
            logo_img = Image.open(logo_path).convert("RGBA")
            logo_img = logo_img.resize((35, 35), Image.LANCZOS)
            self._corner_photo = ImageTk.PhotoImage(logo_img)
            self._corner_lbl = tk.Label(self, image=self._corner_photo,
                                        bg=C["bg_deep"], borderwidth=0)
            self._corner_lbl.place(relx=1.0, rely=0.0, anchor="ne", x=-9, y=6)

        self._watermark = tk.Label(
            self, text="Designed & Developed by DAIV,RCI",
            bg=C["bg_base"], fg=C["text_mid"], font=("Bold", 15), pady=6)
        self._watermark.pack(side="bottom", fill="x")

    # ── PROGRESS UI ────────────────────────────────────────────────────────────
    def _init_progress_ui(self, parent=None):
        if parent is None:
            parent = self
        self.progress_frame = tk.Frame(parent, bg=C["bg_base"])
        self.progress_label = tk.Label(
            self.progress_frame, text="0 %", bg=C["bg_base"], fg=C["text_mid"],
            font=("Ubuntu", 9))
        self.progress_label.pack(side="left", padx=4)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=4)
        self.progress_frame.pack(fill="x", padx=20, pady=(12, 0))
        self.progress_frame.pack_forget()
        self.status_msg = tk.Label(parent, text="", bg=C["bg_base"],
                                   fg=C["text_mid"], font=("Ubuntu", 9))
        self.status_msg.pack(fill="x", padx=20, pady=(4, 12))

    # ── KEYBOARD SHORTCUTS ─────────────────────────────────────────────────────
    def _clear_cache_from(self, start_id: int) -> None:
        for sid in range(start_id, MAX_STEP + 1):
            self.step_cache.pop(sid, None)

    def _on_key_press(self, event):
        ks = event.keysym
        if ks in ("Control_L", "Control_R"):
            self._ctrl_down = True
            self._ab_pressed.clear()
            return
        if self._ctrl_down and ks.lower() in ("a", "b"):
            self._ab_pressed.add(ks.lower())
            if {"a", "b"}.issubset(self._ab_pressed):
                self._show_login_dialog()
                self._ab_pressed.clear()

    def _on_key_release(self, event):
        if event.keysym in ("Control_L", "Control_R"):
            self._ctrl_down = False
            self._ab_pressed.clear()

    def _make_tk_var(self, value):
        if isinstance(value, bool):
            return tk.BooleanVar(value=value)
        if isinstance(value, int):
            return tk.IntVar(value=value)
        if isinstance(value, float):
            return tk.DoubleVar(value=value)
        return tk.StringVar(value=str(value))

    def _sf(self, parent, label, mn, mx, init, step, command, fmt="{:.2f}", var=None):
        if var is None:
            var = tk.DoubleVar(value=init)
        else:
            var.set(init)
        row = tk.Frame(parent, bg=C["bg_raised"])
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text=label, bg=C["bg_raised"], fg=C["text_mid"]).pack(side="left")
        scale = tk.Scale(row, from_=mn, to=mx, resolution=step, orient="horizontal",
                         showvalue=0, variable=var, bg=C["bg_raised"], fg=C["cyan"],
                         troughcolor=C["bg_hover"], highlightthickness=1,
                         highlightbackground=C["border_hi"], highlightcolor=C["cyan"],
                         sliderrelief="raised", sliderlength=18, width=12,
                         activebackground=C["cyan_glow"], command=lambda _: command())
        scale.pack(side="left", fill="x", expand=True, padx=4)
        entry_var = tk.StringVar(value=fmt.format(var.get()))
        entry = tk.Entry(row, textvariable=entry_var, width=8, justify="center",
                         bg=C["bg_deep"], fg=C["amber"], insertbackground=C["amber"],
                         relief="flat", bd=4, font=("DejaVu Sans Mono", 11, "bold"))
        entry.pack(side="right")

        def sync_entry(*_):
            entry_var.set(fmt.format(var.get()))
            command()

        var.trace_add("write", sync_entry)

        def on_entry(event=None):
            try:
                v = float(entry_var.get())
                v = max(mn, min(mx, v))
                var.set(round(v, 8))
            except ValueError:
                entry_var.set(fmt.format(var.get()))

        entry.bind("<Return>", on_entry)
        entry.bind("<FocusOut>", on_entry)
        return var

    # ── GLOBAL TOP BAR ─────────────────────────────────────────────────────────
    def _build_global_topbar(self):
        bar  = tk.Frame(self, bg=C["bg_base"]); bar.pack(fill="x")
        left = tk.Frame(bar, bg=C["bg_base"]);  left.pack(side="left", padx=16, pady=10)

        self.login_btn = tk.Button(
            left, text="Login",
            command=self._show_login_dialog,
            bg=C["amber"], fg=C["bg_deep"],
            activebackground=C["amber_dim"],
            relief="flat", cursor="hand2",
            font=("Ubuntu", 10, "bold"))

        right = tk.Frame(bar, bg=C["bg_base"]); right.pack(side="right", padx=16, pady=8)
        badge = tk.Frame(right, bg=C["bg_raised"],
                         highlightbackground=C["border_hi"], highlightthickness=1)
        badge.pack(side="left", padx=(0, 14))
        self._global_blob_lbl = tk.Label(badge, text="", bg=C["bg_raised"], fg=C["cyan"],
                                         font=("DejaVu Sans Mono", 12, "bold"), pady=3)
        self._global_blob_lbl.pack()
        self.blob_count_lbl = tk.Label(
            bg=C["bg_raised"], fg=C["cyan"],
            font=("DejaVu Sans Mono", 12, "bold"))
        self.blob_count_lbl.master = badge
        self.blob_count_lbl.pack(side="left")

    # ── LOGIN DIALOG ───────────────────────────────────────────────────────────
    def _show_login_dialog(self):
        dlg = tk.Toplevel(self); dlg.title("Login")
        dlg.transient(self); dlg.grab_set()
        dlg.configure(bg=C["bg_base"]); dlg.resizable(False, False)
        pad = dict(padx=12, pady=8)
        tk.Label(dlg, text="Username:", bg=C["bg_base"], fg=C["text_hi"],
                 font=("Ubuntu", 10)).grid(row=0, column=0, sticky="e", **pad)
        user_e = tk.Entry(dlg, bg=C["bg_deep"], fg=C["text_hi"],
                          insertbackground=C["text_hi"])
        user_e.grid(row=0, column=1, **pad)
        tk.Label(dlg, text="Password:", bg=C["bg_base"], fg=C["text_hi"],
                 font=("Ubuntu", 10)).grid(row=1, column=0, sticky="e", **pad)
        pass_e = tk.Entry(dlg, show="*", bg=C["bg_deep"], fg=C["text_hi"],
                          insertbackground=C["text_hi"])
        pass_e.grid(row=1, column=1, **pad)

        def attempt():
            if user_e.get().strip() == "admin" and pass_e.get().strip() == "admin":
                self.logged_in = True; self._login_success(); dlg.destroy()
            else:
                messagebox.showerror("Login failed", "Invalid credentials.", parent=dlg)

        tk.Button(dlg, text="Login", command=attempt,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 10, "bold")).grid(row=2, column=0, columnspan=2, pady=12)
        dlg.wait_window()

    def _login_success(self):
        if not any(self.nb.tab(i, "text") == "SINGLE IMAGE"
                   for i in range(self.nb.index("end"))):
            self.nb.insert(0, self.tab1, text="SINGLE IMAGE")
        self.login_btn.config(text="Logout", command=self._logout)
        self.login_btn.pack(side="left", padx=(0, 12))
        self._log("--Login successful--")

    def _logout(self):
        self.logged_in = False
        for i in range(self.nb.index("end")):
            if self.nb.tab(i, "text") == "SINGLE IMAGE":
                self.nb.forget(i); break
        self.login_btn.pack_forget()
        self.login_btn.config(text="Login", command=self._show_login_dialog)
        self._log("Logged out — only batch tab visible.")

    # ── LOGGING ────────────────────────────────────────────────────────────────
    def _log(self, msg):
        print(msg)

    def _poll_queue(self):
        try:
            while True:
                item = self._msg_queue.get_nowait()
                if isinstance(item, tuple):
                    typ = item[0]
                    if typ == "PROGRESS_START":
                        total = item[1]
                        self.progress_bar["maximum"] = total
                        self.progress_bar["value"] = 0
                        self.progress_label.config(text="0%")
                        self.progress_frame.pack(fill="x", padx=20, pady=(12, 0))
                    elif typ == "PROGRESS_STEP":
                        current = item[1]
                        self.progress_bar["value"] = current
                        pct = int((current / self.progress_bar["maximum"]) * 100)
                        self.progress_label.config(text=f"{pct}%")
                        self.status_msg.config(
                            text=f"Processing {current}/{self.progress_bar['maximum']}")
                    elif typ == "PROGRESS_DONE":
                        self.after(500, self.progress_frame.pack_forget)
                        self.status_msg.config(text="")
                    elif typ == "SHOW_DIALOG":
                        _, title, body = item
                        messagebox.showinfo(title.upper(), body.upper())
                    else:
                        self._log(f"UNKNOWN QUEUE ITEM: {item}")
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ── HELPERS ────────────────────────────────────────────────────────────────
    def _hsep(self, parent=None):
        target = self if parent is None else parent
        tk.Frame(target, bg=C["border"], height=1).pack(fill="x")

    def _get_params(self):
        out = {}
        for k, var in self.pvars.items():
            val = var.get()
            if isinstance(DEFAULTS.get(k), bool):
                out[k] = bool(val)
            elif isinstance(DEFAULTS.get(k), int):
                out[k] = int(val)
            elif isinstance(DEFAULTS.get(k), float):
                out[k] = float(val)
            else:
                out[k] = val
        return out

    # ── TAB 1 — SINGLE IMAGE ──────────────────────────────────────────────────
    def _build_tab1(self, parent):
        self._build_titlebar(parent); self._hsep(parent)
        self._build_info_bar(parent); self._hsep(parent)
        self._build_content(parent);  self._hsep(parent)
        self._build_navbar(parent)
        self.after(50, self._draw_empty_canvas)

    def _build_titlebar(self, parent):
        bar = tk.Frame(parent, bg=C["bg_base"]); bar.pack(fill="x")
        left = tk.Frame(bar, bg=C["bg_base"]); left.pack(side="left", padx=16, pady=10)
        tk.Frame(left, bg=C["cyan"], width=4, height=26).pack(side="left")
        tk.Label(left, text="Defect Detection", bg=C["bg_base"], fg=C["text_hi"],
                 font=("Ubuntu", 14, "bold")).pack(side="left")

        right = tk.Frame(bar, bg=C["bg_base"]); right.pack(side="right", padx=16, pady=8)
        tk.Button(right, text="▲  Load Image",
                  command=self._upload_image,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold"), padx=8, pady=5).pack(side="right")
        tk.Button(right, text="Save Config",
                  command=self._save_config,
                  bg=C["amber"], fg=C["bg_deep"], activebackground=C["amber_dim"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold")).pack(side="right")
        tk.Button(right, text="Load Config",
                  command=self._load_config,
                  bg=C["amber"], fg=C["bg_deep"], activebackground=C["amber_dim"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold")).pack(side="right")

    def _build_info_bar(self, parent):
        bar  = tk.Frame(parent, bg=C["bg_base"]); bar.pack(fill="x")
        left = tk.Frame(bar, bg=C["bg_base"])
        left.pack(side="left", padx=16, fill="x", expand=True)
        self.step_tag_lbl = tk.Label(left, text="", bg=C["bg_base"], fg=C["purple"],
                                     font=("DejaVu Sans Mono", 9, "bold"))
        self.step_tag_lbl.pack(side="left")
        self.step_title_lbl = tk.Label(left, text="", bg=C["bg_base"], fg=C["text_hi"],
                                       font=("Ubuntu", 10, "bold"))
        self.step_title_lbl.pack(side="left", padx=(4, 16))
        self.step_desc_lbl = tk.Label(left, text="", bg=C["bg_base"], fg=C["text_mid"],
                                      font=("Ubuntu", 9), anchor="w", justify="left")
        self.step_desc_lbl.pack(side="left", fill="x", expand=True)

        right = tk.Frame(bar, bg=C["bg_base"]); right.pack(side="right", padx=16, pady=6)
        self.step_btns = []
        for i, s in enumerate(STEPS):
            col = tk.Frame(right, bg=C["bg_base"]); col.pack(side="left", padx=2)
            b = tk.Button(col, text=str(i), width=3,
                          bg=C["bg_raised"], fg=C["text_lo"],
                          activebackground=C["bg_hover"], activeforeground=C["text_hi"],
                          relief="flat", bd=0, cursor="hand2",
                          font=("Ubuntu", 8, "bold"),
                          command=lambda idx=i: self._jump_to(idx))
            b.pack()
            tk.Label(col, text=s["tag"], bg=C["bg_base"], fg=C["text_lo"],
                     font=("Ubuntu", 6)).pack()
            self.step_btns.append(b)

    def _build_content(self, parent):
        content = tk.Frame(parent, bg=C["bg_deep"]); content.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(content, bg=C["bg_deep"], highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self.canvas.bind("<Configure>", self._redraw_canvas)
        self.canvas.bind("<ButtonPress-1>",   self._roi_start_drag)
        self.canvas.bind("<B1-Motion>",       self._roi_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._roi_end_drag)

        tk.Frame(content, bg=C["border"], width=1).pack(side="left", fill="y")

        right = tk.Frame(content, bg=C["bg_base"], width=390)
        right.pack(side="right", fill="y"); right.pack_propagate(False)

        ph = tk.Frame(right, bg=C["bg_base"]); ph.pack(fill="x", padx=14, pady=(10, 6))
        tk.Frame(ph, bg=C["cyan"], width=3, height=14).pack(side="left", padx=(0, 8))
        tk.Label(ph, text="Parameters", bg=C["bg_base"], fg=C["text_hi"],
                 font=("Ubuntu", 9, "bold")).pack(side="left")
        tk.Frame(right, bg=C["border"]).pack(fill="x")

        self.param_canvas = tk.Canvas(right, bg=C["bg_base"], highlightthickness=0)
        sb = tk.Scrollbar(right, orient="vertical", command=self.param_canvas.yview,
                          bg=C["bg_raised"], troughcolor=C["bg_base"], width=10)
        self.param_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.param_canvas.pack(side="left", fill="both", expand=True)
        self.param_frame = tk.Frame(self.param_canvas, bg=C["bg_base"])
        self._pw = self.param_canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        self.param_frame.bind("<Configure>",
            lambda e: self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all")))
        self.param_canvas.bind("<Configure>",
            lambda e: self.param_canvas.itemconfig(self._pw, width=e.width))

        def _bind(e):
            self.param_canvas.bind_all("<MouseWheel>",
                lambda ev: self.param_canvas.yview_scroll(int(-1*(ev.delta/120)), "units"))
            self.param_canvas.bind_all("<Button-4>",
                lambda ev: self.param_canvas.yview_scroll(-1, "units"))
            self.param_canvas.bind_all("<Button-5>",
                lambda ev: self.param_canvas.yview_scroll(1, "units"))

        def _unbind(e):
            self.param_canvas.unbind_all("<MouseWheel>")
            self.param_canvas.unbind_all("<Button-4>")
            self.param_canvas.unbind_all("<Button-5>")

        self.param_canvas.bind("<Enter>", _bind)
        self.param_canvas.bind("<Leave>", _unbind)

    def _build_navbar(self, parent):
        nav = tk.Frame(parent, bg=C["bg_base"]); nav.pack(fill="x", side="bottom")
        row = tk.Frame(nav, bg=C["bg_base"]); row.pack(fill="x", padx=14, pady=10)

        self.prev_btn = tk.Button(row, text="◀   Previous", command=self._go_prev,
                                  bg=C["bg_raised"], fg=C["text_mid"],
                                  activebackground=C["bg_hover"], activeforeground=C["text_hi"],
                                  relief="flat", bd=0, cursor="hand2",
                                  font=("Ubuntu", 9, "bold"), padx=16, pady=8, state="disabled")
        self.prev_btn.pack(side="left")

        ctr = tk.Frame(row, bg=C["bg_base"]); ctr.pack(side="left", expand=True)
        self.step_ind_lbl = tk.Label(ctr, text="", bg=C["bg_raised"], fg=C["text_mid"],
                                     font=("Ubuntu", 9))
        self.step_ind_lbl.pack()
        dot_row = tk.Frame(ctr, bg=C["bg_base"]); dot_row.pack(pady=(5, 0))
        self.dots = []
        for _ in range(MAX_STEP + 1):
            d = tk.Canvas(dot_row, width=10, height=10, bg=C["bg_base"], highlightthickness=0)
            d.pack(side="left", padx=2)
            d.create_oval(1, 1, 9, 9, fill=C["bg_hover"], outline="")
            self.dots.append(d)

        self.next_btn = tk.Button(row, text="Next   ▶", command=self._go_next,
                                  bg=C["cyan"], fg=C["bg_deep"],
                                  activebackground=C["cyan_glow"], activeforeground=C["bg_deep"],
                                  relief="flat", bd=0, cursor="hand2",
                                  font=("Ubuntu", 9, "bold"), padx=16, pady=8, state="disabled")
        self.next_btn.pack(side="right")
        self._refresh_step_ui()

    # ── TAB 2 — BATCH PROCESS ─────────────────────────────────────────────────
    def _build_tab2(self, parent):
        pad = dict(padx=20, pady=12)

        row1 = tk.Frame(parent, bg=C["bg_base"])
        row1.pack(fill="x", **pad)
        tk.Button(row1, text="Images Folder", command=self._select_folder,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 15, "bold"), width=30).grid(row=0, column=0, sticky="ew")
        tk.Entry(row1, textvariable=self.folder_path, state="readonly",
                 bg=C["bg_deep"], fg=C["red"], insertbackground=C["text_hi"],
                 relief="flat", font=("Ubuntu", 15), width=80).grid(row=0, column=1,
                                                                     sticky="ew", padx=6)
        tk.Button(row1, text="Browse", command=self._select_folder,
                  bg=C["bg_base"], fg=C["cyan"], activebackground=C["bg_hover"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 9, "bold")).grid(row=0, column=2, sticky="ew")
        row1.columnconfigure(0, weight=0)
        row1.columnconfigure(1, weight=0)
        row1.columnconfigure(2, weight=0)

        row2 = tk.Frame(parent, bg=C["bg_base"])
        row2.pack(fill="x", **pad)
        tk.Button(row2, text="Output Folder", command=self._select_output_folder,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 15, "bold"), width=30).grid(row=0, column=0, sticky="ew")
        tk.Entry(row2, textvariable=self.output_folder, state="readonly",
                 bg=C["bg_deep"], fg=C["red"], insertbackground=C["text_hi"],
                 relief="flat", font=("Ubuntu", 15), width=80).grid(row=0, column=1,
                                                                     sticky="ew", padx=6)
        tk.Button(row2, text="Browse", command=self._select_output_folder,
                  bg=C["bg_base"], fg=C["cyan"], activebackground=C["bg_hover"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 9, "bold")).grid(row=0, column=2, sticky="ew")
        row2.columnconfigure(0, weight=0)
        row2.columnconfigure(1, weight=0)
        row2.columnconfigure(2, weight=0)

        bot_btns = tk.Frame(parent, bg=C["bg_base"])
        bot_btns.pack(fill="x", **pad)

        self.start_btn = tk.Button(bot_btns, text="Start",
                                   command=self._process_folder,
                                   bg=C["green"], fg=C["bg_deep"], activebackground=C["green"],
                                   relief="flat", cursor="hand2",
                                   font=("Ubuntu", 15, "bold"))
        self.start_btn.grid(row=0, column=0, sticky="ew", pady=8)

        self.stop_btn = tk.Button(bot_btns, text="Stop",
                                  command=self._stop_process,
                                  bg=C["red"], fg=C["bg_deep"], activebackground="#c0392b",
                                  relief="flat", cursor="hand2",
                                  font=("Ubuntu", 15, "bold"))
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        tk.Button(bot_btns, text="Clear",
                  command=self._clear_process,
                  bg=C["amber"], fg=C["bg_deep"], activebackground=C["amber_dim"],
                  relief="flat", cursor="hand2",
                  font=("Ubuntu", 15, "bold")).grid(row=0, column=2, sticky="ew", pady=8)

        self.viz_btn = tk.Button(bot_btns, text="Visualize",
                                 command=self._launch_visualizer,
                                 bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                                 relief="flat", cursor="hand2",
                                 font=("Ubuntu", 15, "bold"))
        self.viz_btn.grid(row=0, column=3, sticky="ew", padx=8, pady=8)

        bot_btns.columnconfigure(0, weight=1)
        bot_btns.columnconfigure(1, weight=1)
        bot_btns.columnconfigure(2, weight=1)
        bot_btns.columnconfigure(3, weight=1)

    # ── VISUALIZER LAUNCH ─────────────────────────────────────────────────────
    def _launch_visualizer(self):
        """
        Launch visualize.py from the same folder as gui.py.
        It will auto-detect results/final and results/initial_images by itself.
        """
        import subprocess, sys
        script_dir = os.path.dirname(os.path.abspath(__file__))
        viz_script = os.path.join(script_dir, "visualize.py")

        if not os.path.isfile(viz_script):
            messagebox.showerror("Not found",
                f"visualize.py not found in:\n{script_dir}")
            return

        # Compute the summary first (optional — shows counts in status bar)
        base = self.output_folder.get() or self.folder_path.get()
        if base and os.path.isdir(base):
            final_folder = os.path.join(base, "results", "final")
            if os.path.isdir(final_folder):
                total, green = self._count_images_and_green(final_folder)
                self.status_msg.config(
                    text=f"Total: {total}   Defected: {green}   Non-defect: {total - green}")

        def _run():
            try:
                subprocess.Popen([sys.executable, viz_script])
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Launch error", str(e)))

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _count_images_and_green(folder: str) -> tuple[int, int]:
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        total = 0; green = 0
        for fn in os.listdir(folder):
            if not fn.lower().endswith(exts):
                continue
            total += 1
            try:
                img = Image.open(os.path.join(folder, fn)).convert("RGB")
                arr = np.array(img)
                if np.any((arr[:, :, 1] > 200) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 80)):
                    green += 1
            except Exception:
                pass
        return total, green

    # ── FOLDER HELPERS ─────────────────────────────────────────────────────────
    def _select_folder(self):
        path = filedialog.askdirectory(title="Select folder with images")
        if path:
            self.folder_path.set(path)

    def _select_output_folder(self):
        path = filedialog.askdirectory(title="Select folder for results")
        if path:
            self.output_folder.set(path)

    # ── BATCH PROCESSING ───────────────────────────────────────────────────────
    def _process_folder(self):
        folder = self.folder_path.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No folder", "Select a folder containing images first.")
            return
        if self.progress_bar is None:
            self._init_progress_ui(self.tab2)
        self._stop_event.clear()
        self._batch_thread = threading.Thread(target=self._run_batch, daemon=True)
        self._batch_thread.start()
        self._set_process_buttons(running=True)

    def _clear_process(self):
        if self.progress_bar is not None:
            self.progress_bar["value"] = 0
        if self.progress_label is not None:
            self.progress_label.config(text="0 %")
        if self.status_msg is not None:
            self.status_msg.config(text="")
        messagebox.showinfo("Clear", "Progress cleared.")

    def _stop_process(self):
        if self._batch_thread and self._batch_thread.is_alive():
            self._stop_event.set()
            self._set_process_buttons(running=False)
            self._msg_queue.put("BATCH PROCESSING STOPPED BY USER")
        else:
            messagebox.showinfo("Stop Process", "No batch job is running.")

    def _set_process_buttons(self, running: bool):
        if hasattr(self, "start_btn"):
            self.start_btn.config(state="disabled" if running else "normal")
        if hasattr(self, "stop_btn"):
            self.stop_btn.config(state="normal" if running else "disabled")

    def _run_batch(self):
        self._msg_queue.put("BATCH PROCESSING STARTED")
        folder   = self.folder_path.get()
        out_root = (os.path.join(self.output_folder.get(), "results")
                    if self.output_folder.get() and os.path.isdir(self.output_folder.get())
                    else os.path.join(folder, "results"))

        final_root   = os.path.join(out_root, "final")
        initial_root = os.path.join(out_root, "initial_images")
        os.makedirs(final_root, exist_ok=True)
        os.makedirs(initial_root, exist_ok=True)

        intermediate_root = None
        if self.logged_in:
            intermediate_root = os.path.join(out_root, "intermediate")
            os.makedirs(intermediate_root, exist_ok=True)

        exts  = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
        total = len(files)
        if total == 0:
            self._msg_queue.put("NO SUPPORTED IMAGES FOUND"); return
        self._msg_queue.put(f"FOUND {total} IMAGE{'S' if total != 1 else ''}")

        params = self._get_params()
        self._msg_queue.put(("PROGRESS_START", total))

        for idx, fname in enumerate(files, 1):
            if self._stop_event.is_set():
                self._msg_queue.put("BATCH PROCESSING ABORTED")
                break

            self._msg_queue.put(f"PROCESSING {idx}/{total}: {fname}")
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._msg_queue.put(f"SKIPPED (UNREADABLE): {fname}"); continue

            batch_state = {
                "img0": img,
                "roi_rect": self.roi_rect,
                "rotation": self.rotation,
                "flip_h":   self.flip_h,
                "flip_v":   self.flip_v,
            }

            cache = {}
            for sid in range(len(STEP_RUNNERS)):
                try:
                    cache[sid] = STEP_RUNNERS[sid](batch_state, params)
                except Exception as e:
                    self._msg_queue.put(f"  step {sid} error: {e}")
                    cache[sid] = None

            name_no_ext = os.path.splitext(fname)[0]

            if cache.get(2) is not None:
                cv2.imwrite(os.path.join(initial_root, f"{name_no_ext}.png"), cache[2])

            if intermediate_root:
                img_dir = os.path.join(intermediate_root, name_no_ext)
                os.makedirs(img_dir, exist_ok=True)
                for step in STEPS[:-1]:
                    sid = step["id"]; tag = step["tag"]
                    out_img = cache.get(sid)
                    if out_img is not None:
                        cv2.imwrite(os.path.join(img_dir, _ordered_step_name(sid, tag)), out_img)

            final_img = cache.get(MAX_STEP)
            if final_img is not None:
                fp = os.path.join(final_root, f"{name_no_ext}.png")
                cv2.imwrite(fp, final_img)
                self._msg_queue.put(f"SAVED: {fp}")

            self._msg_queue.put(("PROGRESS_STEP", idx))

        if not self._stop_event.is_set():
            self._msg_queue.put("BATCH PROCESSING COMPLETED")
            self._msg_queue.put(("SHOW_DIALOG", "DONE", "ALL IMAGES PROCESSED"))
        self._msg_queue.put(("PROGRESS_DONE", None))
        self.after(0, lambda: self._set_process_buttons(running=False))

    # ── SINGLE IMAGE — CONFIG ─────────────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Load Error", "Cannot read selected file."); return
        self.img0 = img
        self.roi_rect = None
        self.rotation = 0; self.flip_h = False; self.flip_v = False
        self.state = {"img0": img, "roi_rect": None, "rotation": 0,
                      "flip_h": False, "flip_v": False}
        self.step_cache   = {}
        self.current      = 0
        self._current_pil = None
        self._refresh_step_ui()
        self._run_from(0)
        self._update_nav_buttons()

    def _save_config(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save configuration")
        if not path: return
        try:
            data = self._get_params()
            data["roi_rect"] = self.roi_rect
            data["rotation"] = self.rotation
            data["flip_h"]   = self.flip_h
            data["flip_v"]   = self.flip_v
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Configuration saved", f"Saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _load_config(self, path: str | None = None):
        if path is None:
            path = self.config_path
        if path is None:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(base_dir, "gui", "inputs", "config.json")
        if not os.path.isfile(path):
            messagebox.showerror("Load error", f"Config file not found:\n{path}")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        for k, v in data.items():
            if k not in self.pvars:
                continue
            var = self.pvars[k]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(v))
            elif isinstance(var, tk.IntVar):
                var.set(int(v))
            elif isinstance(var, tk.DoubleVar):
                var.set(float(v))
            else:
                var.set(v)
        self.roi_rect = data.get("roi_rect")
        self.rotation = int(data.get("rotation", 0)) % 360
        self.flip_h   = bool(data.get("flip_h", False))
        self.flip_v   = bool(data.get("flip_v", False))
        self.state.update({"roi_rect": self.roi_rect, "rotation": self.rotation,
                           "flip_h": self.flip_h, "flip_v": self.flip_v})
        if self.img0 is not None:
            start_step = 2 if self.current >= 2 else self.current
            self._run_from(start_step)
        self._update_roi_status()
        self._update_rot_lbl()

    # ── ROI PANEL ─────────────────────────────────────────────────────────────
    def _build_roi_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr_row = tk.Frame(card, bg=C["bg_raised"]); hdr_row.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr_row, bg=C["purple"]); pill.pack(side="left")
        tk.Label(pill, text="  ROI RECTANGLE  ", bg=C["purple"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        self._roi_status_lbl = tk.Label(card, text="", bg=C["bg_raised"],
                                        fg=C["amber"], font=("Ubuntu", 9),
                                        wraplength=330, justify="left")
        self._roi_status_lbl.pack(fill="x", padx=10, pady=4)
        self._update_roi_status()
        tk.Label(card,
                 text="• Click-drag on the image to draw a rectangle\n"
                      "• Release mouse button to finalize\n"
                      "• Use the buttons below to clear or save",
                 bg=C["bg_raised"], fg=C["text_lo"],
                 font=("Ubuntu", 8), justify="left").pack(anchor="w", padx=10, pady=(0, 8))
        sep = tk.Frame(card, bg=C["border"], height=1); sep.pack(fill="x")
        btn_frame = tk.Frame(card, bg=C["bg_raised"]); btn_frame.pack(fill="x", padx=10, pady=10)
        tk.Button(btn_frame, text="✖  Clear ROI", command=self._roi_clear,
                  bg=C["red"], fg=C["bg_deep"], activebackground="#c0392b",
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold"), pady=7).pack(fill="x", pady=(0, 6))
        tk.Button(btn_frame, text="💾  Save ROI to Config", command=self._roi_save_coords,
                  bg=C["amber"], fg=C["bg_deep"], activebackground=C["amber_dim"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold"), pady=7).pack(fill="x")

    # ── ROTATE / FLIP PANEL ───────────────────────────────────────────────────
    def _build_rotate_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr_row = tk.Frame(card, bg=C["bg_raised"]); hdr_row.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr_row, bg=C["cyan"]); pill.pack(side="left")
        tk.Label(pill, text="  ROTATE / FLIP  ", bg=C["cyan"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        self._rot_lbl = tk.Label(card,
            text=f"Rotation: {self.rotation}°   |   H-Flip: {'ON' if self.flip_h else 'off'}   |   V-Flip: {'ON' if self.flip_v else 'off'}",
            bg=C["bg_raised"], fg=C["amber"], font=("DejaVu Sans Mono", 9))
        self._rot_lbl.pack(padx=10, pady=6)
        sep = tk.Frame(card, bg=C["border"], height=1); sep.pack(fill="x")
        btn_frame = tk.Frame(card, bg=C["bg_raised"]); btn_frame.pack(fill="x", padx=10, pady=10)
        rot_row = tk.Frame(btn_frame, bg=C["bg_raised"]); rot_row.pack(fill="x", pady=(0, 6))
        tk.Button(rot_row, text="↺  Left 90°", command=self._rotate_left,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 10, "bold"), pady=8).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(rot_row, text="↻  Right 90°", command=self._rotate_right,
                  bg=C["cyan"], fg=C["bg_deep"], activebackground=C["cyan_glow"],
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 10, "bold"), pady=8).pack(side="left", fill="x", expand=True, padx=(4, 0))
        flip_row = tk.Frame(btn_frame, bg=C["bg_raised"]); flip_row.pack(fill="x", pady=(0, 6))
        tk.Button(flip_row, text="⇔  Flip Horizontal", command=self._flip_h_toggle,
                  bg=C["purple"], fg=C["bg_deep"], activebackground="#a070dd",
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 10, "bold"), pady=8).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(flip_row, text="⇕  Flip Vertical", command=self._flip_v_toggle,
                  bg=C["purple"], fg=C["bg_deep"], activebackground="#a070dd",
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 10, "bold"), pady=8).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Button(btn_frame, text="⟳  Reset All", command=self._reset_rotation,
                  bg=C["red"], fg=C["bg_deep"], activebackground="#c0392b",
                  relief="flat", bd=0, cursor="hand2",
                  font=("Ubuntu", 9, "bold"), pady=7).pack(fill="x")

    # ── CANNY / DILATION / MASK / INPAINT PANELS ──────────────────────────────
    def _build_canny_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr = tk.Frame(card, bg=C["bg_raised"]); hdr.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr, bg=C["cyan"]); pill.pack(side="left")
        tk.Label(pill, text="  CANNY EDGE  ", bg=C["cyan"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        for meta in STEPS[5]["params"]:
            ParamControl(card, meta, self.pvars, on_change=self._param_changed).pack(fill="x", pady=4)

    def _build_dilation_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr = tk.Frame(card, bg=C["bg_raised"]); hdr.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr, bg=C["purple"]); pill.pack(side="left")
        tk.Label(pill, text="  DILATION  ", bg=C["purple"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        for meta in STEPS[6]["params"]:
            ParamControl(card, meta, self.pvars, on_change=self._param_changed).pack(fill="x", pady=4)

    def _build_mask_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr = tk.Frame(card, bg=C["bg_raised"]); hdr.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr, bg=C["amber"]); pill.pack(side="left")
        tk.Label(pill, text="  MASK — BINARY FILL HOLE  ", bg=C["amber"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        for meta in STEPS[7]["params"]:
            ParamControl(card, meta, self.pvars, on_change=self._param_changed).pack(fill="x", pady=4)

    def _build_inpaint_panel(self):
        cb = tk.Frame(self.param_frame, bg=C["border_hi"])
        cb.pack(fill="x", padx=12, pady=10)
        card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
        hdr = tk.Frame(card, bg=C["bg_raised"]); hdr.pack(fill="x", padx=10, pady=(10, 4))
        pill = tk.Frame(hdr, bg=C["green"]); pill.pack(side="left")
        tk.Label(pill, text="  INPAINT  ", bg=C["green"], fg=C["bg_deep"],
                 font=("Ubuntu", 8, "bold"), pady=2).pack()
        for meta in STEPS[8]["params"]:
            ParamControl(card, meta, self.pvars, on_change=self._param_changed).pack(fill="x", pady=4)

    # ── ROI DRAG HANDLERS ─────────────────────────────────────────────────────
    def _roi_start_drag(self, event):
        if self.current != 1 or self.img0 is None:
            return
        ox, oy = self._img_offset
        scale = self._img_scale if self._img_scale > 0 else 1.0
        self._drag_start_img = (int((event.x - ox) / scale), int((event.y - oy) / scale))
        if self._temp_rect_id:
            self.canvas.delete(self._temp_rect_id)
            self._temp_rect_id = None

    def _roi_drag_motion(self, event):
        if self.current != 1 or self.img0 is None or not hasattr(self, "_drag_start_img"):
            return
        ox, oy = self._img_offset
        scale = self._img_scale if self._img_scale > 0 else 1.0
        cur = (int((event.x - ox) / scale), int((event.y - oy) / scale))
        if self._temp_rect_id:
            self.canvas.delete(self._temp_rect_id)
        x0 = self._drag_start_img[0] * scale + ox
        y0 = self._drag_start_img[1] * scale + oy
        x1 = cur[0] * scale + ox
        y1 = cur[1] * scale + oy
        self._temp_rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline="red", width=2, dash=(4, 2))

    def _roi_end_drag(self, event):
        if self.current != 1 or self.img0 is None or not hasattr(self, "_drag_start_img"):
            return
        ox, oy = self._img_offset
        scale = self._img_scale if self._img_scale > 0 else 1.0
        end = (int((event.x - ox) / scale), int((event.y - oy) / scale))
        x1, y1 = self._drag_start_img
        x2, y2 = end
        x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
        self.roi_rect = [(x1, y1), (x2, y2)]
        self.state["roi_rect"] = self.roi_rect
        if self._temp_rect_id:
            self.canvas.delete(self._temp_rect_id)
            self._temp_rect_id = None
        del self._drag_start_img
        self._run_from(1)
        self._update_roi_status()

    def _update_roi_status(self):
        if self._roi_status_lbl is None:
            return
        if self.roi_rect:
            (x1, y1), (x2, y2) = self.roi_rect
            msg = f"Rectangle: ({x1}, {y1}) – ({x2}, {y2})"
        else:
            msg = "Drag on the image to select a rectangular ROI"
        self._roi_status_lbl.config(text=msg)

    def _roi_clear(self):
        self.roi_rect = None
        self.state["roi_rect"] = None
        if self.img0 is not None and self.current == 1:
            self._run_from(1)
        self._update_roi_status()

    def _roi_save_coords(self):
        self._save_config()

    # ── ROTATION / FLIP ───────────────────────────────────────────────────────
    def _rotate_left(self):
        self.rotation = (self.rotation - 90) % 360
        self.state["rotation"] = self.rotation
        self._run_from(2); self._update_rot_lbl()

    def _rotate_right(self):
        self.rotation = (self.rotation + 90) % 360
        self.state["rotation"] = self.rotation
        self._run_from(2); self._update_rot_lbl()

    def _flip_h_toggle(self):
        self.flip_h = not self.flip_h
        self.state["flip_h"] = self.flip_h
        self._run_from(2); self._update_rot_lbl()

    def _flip_v_toggle(self):
        self.flip_v = not self.flip_v
        self.state["flip_v"] = self.flip_v
        self._run_from(2); self._update_rot_lbl()

    def _reset_rotation(self):
        self.rotation = 0; self.flip_h = False; self.flip_v = False
        self.state.update({"rotation": 0, "flip_h": False, "flip_v": False})
        self._run_from(2); self._update_rot_lbl()

    def _update_rot_lbl(self):
        if self._rot_lbl is None:
            return
        fh = "ON" if self.flip_h else "off"
        fv = "ON" if self.flip_v else "off"
        self._rot_lbl.config(text=f"Rotation: {self.rotation}°   |   H-Flip: {fh}   |   V-Flip: {fv}")

    # ── PARAMETER CHANGE HANDLER ──────────────────────────────────────────────
    PARAM_TO_STEP = {
        "CANNY_VAR": 5, "CANNY_MAXERR": 5, "CANNY_LO": 5, "CANNY_HI": 5,
        "CANNY_AP": 5, "CANNY_L2": 5,
        "DIL_SHAPE": 6, "DIL_K": 6, "DIL_I": 6,
        "MASK_CONN": 7, "MASK_ERODE": 7, "MASK_DILATE": 7, "MASK_OVERLAY": 7, "MASK_INVERT": 7,
        "INPAINT_METHOD": 8, "INPAINT_R": 8,
        "PAD": 9,
        "CLIP_LIMIT": 4, "TILE_SIZE": 4,
        "INT": 12,
        "S_MIN": 13, "S_MAX": 13,
        "ALPHA": 14, "EPS": 14,
        "SEL": 15, "NORM": 16,
        "THRESHOLD": 17, "OVERLAP": 17, "NUM_SIGMA": 17,
    }

    def _param_changed(self, *_):
        if self.img0 is None:
            return
        params = self._get_params()
        changed = [k for k, v in params.items() if self._last_params.get(k) != v]
        earliest = (min(self.PARAM_TO_STEP.get(k, self.current) for k in changed)
                    if changed else self.current)
        self._last_params = params.copy()
        self._run_from(earliest)

    # ── PIPELINE EXECUTION ─────────────────────────────────────────────────────
    def _run_from(self, from_step: int) -> None:
        params = self._get_params()
        self._clear_cache_from(from_step)
        for sid in range(from_step, self.current + 1):
            try:
                self.step_cache[sid] = STEP_RUNNERS[sid](self.state, params)
            except Exception as e:
                self.step_cache[sid] = None
                print(f"[Step {sid}] {e}")
        self._display_current()
        self._update_nav_buttons()

    def _display_current(self):
        arr = self.step_cache.get(self.current)
        if arr is None:
            self._draw_empty_canvas(); return
        if arr.ndim == 3:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr_rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self._current_pil = Image.fromarray(arr_rgb)
        self._redraw_canvas()
        blobs = self.state.get("blobs")
        if blobs is not None and self.current >= 16:
            self.blob_count_lbl.config(text=f"  {len(blobs)}  ")

    # ── CANVAS DRAWING ─────────────────────────────────────────────────────────
    def _redraw_canvas(self, _=None):
        if self._current_pil is None: return
        cw = max(self.canvas.winfo_width(),  1)
        ch = max(self.canvas.winfo_height(), 1)
        orig_w, orig_h = self._current_pil.size
        img = self._current_pil.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        iw, ih = img.size
        self._img_scale  = iw / orig_w if orig_w > 0 else 1.0
        ox = (cw - iw) // 2; oy = (ch - ih) // 2
        self._img_offset = (ox, oy)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_rectangle(ox+4, oy+4, ox+iw+4, oy+ih+4, fill="#000000", outline="")
        self.canvas.create_rectangle(ox-1, oy-1, ox+iw+1, oy+ih+1,
                                     outline=C["border_hi"], fill="")
        self.canvas.create_image(cw//2, ch//2, anchor="center", image=self._tk_img)

    def _draw_empty_canvas(self):
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 400)
        ch = max(self.canvas.winfo_height(), 300)
        self.canvas.delete("all")
        cx, cy = cw // 2, ch // 2
        self.canvas.create_rectangle(cx-80, cy-60, cx+80, cy+60,
                                     outline=C["border_hi"], dash=(4, 4), width=1)
        self.canvas.create_text(cx, cy-14, text="▲", fill=C["border_hi"], font=("Ubuntu", 24))
        self.canvas.create_text(cx, cy+16, text="Load image to begin",
                                fill=C["border_hi"], font=("Ubuntu", 10, "bold"))
        self.canvas.create_text(cx, cy-24, text="PNG  ·  JPG  ·  BMP  ·  TIFF",
                                fill=C["text_lo"], font=("Ubuntu", 8))

    # ── NAVIGATION ────────────────────────────────────────────────────────────
    def _jump_to(self, idx):
        if self.img0 is None: return
        old = self.current; self.current = idx
        self._run_from(old + 1 if idx > old else idx)
        self._refresh_step_ui(); self._update_nav_buttons()

    def _go_next(self):
        if self.current < MAX_STEP:
            self.current += 1; self._run_from(self.current)
            self._refresh_step_ui(); self._update_nav_buttons()

    def _go_prev(self):
        if self.current > 0:
            self.current -= 1; self._display_current()
            self._refresh_step_ui(); self._update_nav_buttons()

    # ── UI REFRESH ─────────────────────────────────────────────────────────────
    def _refresh_step_ui(self):
        step = STEPS[self.current]
        self.step_tag_lbl.config(
            text=f"[{step['tag']}]  STEP {self.current:02d}/{MAX_STEP:02d}  ·  ")
        self.step_title_lbl.config(text=step["title"].upper())
        self.step_desc_lbl.config(text=step["desc"])
        self.step_ind_lbl.config(
            text=f"STEP {self.current} OF {MAX_STEP}   ·   {step['title']}")
        self.canvas.config(cursor="crosshair" if self.current == 1 else "")

        for i, d in enumerate(self.dots):
            d.delete("all")
            if i < self.current:
                d.create_oval(1, 1, 9, 9, fill=C["cyan_dim"], outline="")
            elif i == self.current:
                d.create_oval(1, 1, 9, 9, fill=C["cyan"], outline="")
                d.create_oval(3, 3, 7, 7, fill=C["bg_deep"], outline="")
            else:
                d.create_oval(1, 1, 9, 9, fill=C["bg_hover"], outline="")

        for i, b in enumerate(self.step_btns):
            if i == self.current:
                b.config(bg=C["cyan"], fg=C["bg_deep"])
            elif i < self.current:
                b.config(bg=C["bg_raised"], fg=C["green"])
            else:
                b.config(bg=C["bg_raised"], fg=C["text_lo"])

        for w in self.param_frame.winfo_children():
            w.destroy()
        self._roi_status_lbl = None
        self._rot_lbl        = None

        if self.current == 1:
            self._build_roi_panel()
        elif self.current == 2:
            self._build_rotate_panel()
        elif self.current == 5:
            self._build_canny_panel()
        elif self.current == 6:
            self._build_dilation_panel()
        elif self.current == 7:
            self._build_mask_panel()
        elif self.current == 8:
            self._build_inpaint_panel()
        elif step["params"]:
            for meta in step["params"]:
                ParamControl(self.param_frame, meta, self.pvars,
                             on_change=self._param_changed).pack(fill="x")
        else:
            cb = tk.Frame(self.param_frame, bg=C["border_hi"])
            cb.pack(fill="x", padx=12, pady=14)
            card = tk.Frame(cb, bg=C["bg_raised"]); card.pack(fill="x", padx=1, pady=1)
            tk.Label(card, text="VIEW ONLY", bg=C["bg_raised"], fg=C["text_lo"],
                     font=("Ubuntu", 8, "bold"), pady=6).pack()
            tk.Label(card, text="No adjustable parameters.\nResult updates automatically.",
                     bg=C["bg_raised"], fg=C["text_lo"], font=("Ubuntu", 9),
                     justify="center", pady=10).pack()

        self.param_canvas.yview_moveto(0)

    # ── NAV BUTTON STATE ──────────────────────────────────────────────────────
    def _update_nav_buttons(self):
        has = self.img0 is not None
        self.prev_btn.config(
            state="normal" if has and self.current > 0 else "disabled",
            bg=C["bg_raised"] if has and self.current > 0 else C["bg_deep"],
            fg=C["text_mid"] if has and self.current > 0 else C["text_lo"])
        self.next_btn.config(
            state="normal" if has and self.current < MAX_STEP else "disabled",
            bg=C["cyan"]   if has and self.current < MAX_STEP else C["bg_deep"],
            fg=C["bg_deep"] if has and self.current < MAX_STEP else C["text_lo"])

    def _on_tab_changed(self, event=None):
        pass

    def destroy(self):
        super().destroy()


# ── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BlobGUI()
    app.config_path = "config.json"
    app._load_config()
    app.mainloop()
