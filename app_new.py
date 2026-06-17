#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =========================================================
# app.py  —  DRDL Aerospace AI Platform  v9.7  (UPGRADED)
# =========================================================
# CHANGES FROM v9.6:
#   1. All fonts changed to Arial
#   2. Fixed invisible buttons in Tab 1 & Tab 2 —
#      bottom action-bar now uses sg.Frame with fixed height
#      and explicit background so it is never clipped by
#      the scrollable Column above it.
#   3. Professional UI: consistent padding, refined colours,
#      rounded borders via Frame relief, labelled separators.
#   4. sec_hdr / sec_hdr_rows layout rules unchanged (correct).
# =========================================================

import os
import sys
import time
import threading
import queue
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

try:
    import PySimpleGUI as sg
except ImportError:
    print("ERROR: pip install PySimpleGUI==4.60.5")
    sys.exit(1)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from predictor import aerodynamic_prediction as _raw_pred, get_top_features, ENSEMBLE_MODE
from optimizer import run_optimization, PARAM_NAMES
from envelope  import alpha_sweep, mach_sweep, altitude_sweep

# =========================================================
# INFERENCE CACHE  (LRU-style, max 5000 entries)
# =========================================================
_pred_cache = {}
_CACHE_MAX  = 5000
_CACHE_RND  = 3

# =========================================================
# RESULT NORMALISATION
# ---------------------------------------------------------
# "TypeError: list indices must be integers or slices, not str"
# happens whenever code does result['CL'] but result is actually
# a list/tuple (e.g. predictor.py / optimizer.py / envelope.py
# returning positional values instead of a dict for some inputs).
# Every place in this file that indexes a result with a string
# key now goes through a dict first, so the GUI can never crash
# on a type mismatch coming from those modules — worst case a
# field shows as N/A instead of popping an error dialog.
# =========================================================
def _as_dict(obj, keys):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        return {k: (obj[i] if i < len(obj) else None)
                for i, k in enumerate(keys)}
    out = {}
    for k in keys:
        out[k] = getattr(obj, k, None)
    return out

_RESULT_KEYS = ['CL', 'CD', 'XCP', 'XCP_D', 'metrics',
                'detailed_metrics', 'mode', 'elapsed_ms']

def aerodynamic_prediction(params: dict) -> dict:
    key = tuple(round(float(params[p]), _CACHE_RND) for p in PARAMS)
    if key in _pred_cache:
        return _pred_cache[key]
    raw = _raw_pred(params)
    result = _as_dict(raw, _RESULT_KEYS)
    if len(_pred_cache) >= _CACHE_MAX:
        for k in list(_pred_cache)[:400]:
            del _pred_cache[k]
    _pred_cache[key] = result
    return result

# =========================================================
# COLOUR PALETTE
# =========================================================
C_BG    = '#0B0F1A'
C_PANEL = '#111827'
C_INP   = '#1C2333'
C_DARK  = '#07090F'
C_BLUE  = '#3B82F6'
C_CYAN  = '#06B6D4'
C_GREEN = '#10B981'
C_AMBER = '#F59E0B'
C_RED   = '#EF4444'
C_WHITE = '#F1F5F9'
C_DIM   = '#94A3B8'
C_BDR   = '#1E293B'
C_PURP  = '#8B5CF6'
C_HDR   = '#0F172A'
C_INDG  = '#6366F1'
C_ROSE  = '#FB7185'
C_BAR   = '#0D1424'   # action-bar background (slightly different from C_BG)

# =========================================================
# FONTS
# ---------------------------------------------------------
# 'Arial' is often NOT installed on Linux. When Tk can't find
# a requested family it silently substitutes whatever generic
# font is configured system-wide, which is what produced the
# "weird"/inconsistent font look. We now ask Tk what is
# actually installed and pick the closest clean sans-serif
# match, so every Text/Input/Button in the app renders with
# the same real font instead of a random fallback.
# =========================================================
def _detect_font():
    candidates = ['Segoe UI', 'Helvetica Neue', 'Helvetica',
                  'Arial', 'DejaVu Sans', 'Verdana', 'Liberation Sans']
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        available = set(_root.tk.call('font', 'families'))
        _root.destroy()
        for c in candidates:
            if c in available:
                return c
    except Exception:
        pass
    return 'Helvetica'   # Tk always has this as a built-in alias

_F = _detect_font()
F_TITLE  = (_F, 13, 'bold')
F_SUB    = (_F, 10)
F_SEC    = (_F, 11, 'bold')
F_LBL    = (_F, 11)
F_INP    = (_F, 11, 'bold')
F_OUT    = (_F, 12, 'bold')
F_BTN    = (_F, 11, 'bold')
F_STS    = (_F, 11, 'bold')
F_TBL    = (_F, 11)
F_TOP5   = (_F, 11)
F_CHROME = (_F, 10, 'bold')
F_ARROW  = (_F, 15, 'bold')
F_PLTLBL = (_F, 11, 'bold')
F_TABTXT = (_F, 12, 'bold')

# =========================================================
# PARAMETER DEFINITIONS
# =========================================================
DEFAULTS = {
    'nose_len'   : 300,  'body_len'   : 2700, 'wing_le'    : 1500,
    'root_chord' : 200,  'tip_chord'  : 150,  'semi_span'  : 1000,
    'root_th'    : 20,   'tip_th'     : 5,    'wing_sweep' : 2.86,
    'tail_le'    : 2870, 'root_chord1': 120,  'tip_chord1' : 60,
    'semi_span1' : 100,  'root_th1'   : 15,   'tip_th1'    : 5,
    'mach'       : 0.2,  'alpha'      : 2,    'alt'        : 0,
}
LABELS = {
    'nose_len'   : 'Nose Length (mm)',
    'body_len'   : 'Body Length (mm)',
    'wing_le'    : 'Wing LE (mm)',
    'root_chord' : 'Root Chord (mm)',
    'tip_chord'  : 'Tip Chord (mm)',
    'semi_span'  : 'Semi-Span (mm)',
    'root_th'    : 'Root Thickness',
    'tip_th'     : 'Tip Thickness',
    'wing_sweep' : 'Wing Sweep (deg)',
    'tail_le'    : 'Tail LE (mm)',
    'root_chord1': 'Tail Root Chord',
    'tip_chord1' : 'Tail Tip Chord',
    'semi_span1' : 'Tail Semi-Span',
    'root_th1'   : 'Tail Root Thickness',
    'tip_th1'    : 'Tail Tip Thickness',
    'mach'       : 'Mach Number',
    'alpha'      : 'Alpha (deg)',
    'alt'        : 'Altitude (m)',
}
PARAMS = list(DEFAULTS.keys())

# optimizer.py exposes its own PARAM_NAMES ordering. If it ever
# drifts out of sync with this file's PARAMS list (same names,
# different order — or a missing/extra one), every bound sent to
# run_optimization gets silently mapped to the wrong parameter,
# which is a common source of bizarre/garbage optimizer results
# or downstream type errors. We use PARAM_NAMES (the optimizer's
# own source of truth) whenever it's a valid permutation of PARAMS,
# and fall back to PARAMS with a console warning otherwise.
if set(PARAM_NAMES) == set(PARAMS) and len(PARAM_NAMES) == len(PARAMS):
    _BOUNDS_ORDER = list(PARAM_NAMES)
else:
    print('[WARN] optimizer.PARAM_NAMES does not match this app\'s '
          'PARAMS list — falling back to local order. Bounds may '
          'still be mismatched if optimizer.py expects PARAM_NAMES '
          'order internally.')
    print('  optimizer.PARAM_NAMES:', PARAM_NAMES)
    print('  app.PARAMS           :', PARAMS)
    _BOUNDS_ORDER = PARAMS

BOUNDS = {
    'nose_len'   : (120,  360),  'body_len'   : (2400, 3000),
    'wing_le'    : (1000, 2000), 'root_chord' : (150,  250),
    'tip_chord'  : (110,  190),  'semi_span'  : (600,  1500),
    'root_th'    : (15,   25),   'tip_th'     : (5,    11),
    'wing_sweep' : (0.0,  70.0), 'tail_le'    : (2830, 2910),
    'root_chord1': (80,   160),  'tip_chord1' : (30,   90),
    'semi_span1' : (60,   140),  'root_th1'   : (15,   21),
    'tip_th1'    : (5,    11),   'mach'       : (0.2,  0.8),
    'alpha'      : (0,    20),   'alt'        : (0,    6000),
}

# =========================================================
# THEME
# =========================================================
sg.theme_add_new('DRDL', {
    'BACKGROUND'    : C_BG,
    'TEXT'          : C_WHITE,
    'INPUT'         : C_INP,
    'TEXT_INPUT'    : C_WHITE,
    'SCROLL'        : C_BLUE,
    'BUTTON'        : (C_WHITE, C_BLUE),
    'PROGRESS'      : (C_CYAN, '#0A1020'),
    'BORDER'        : 1,
    'SLIDER_DEPTH'  : 0,
    'PROGRESS_DEPTH': 0,
})
sg.theme('DRDL')

# =========================================================
# LOGIN
# =========================================================
_VALID_USERS = {'drdl2026': 'aero1234'}

def show_login() -> bool:
    ly = [
        [sg.Text('', background_color=C_BG, pad=(0, 10))],
        [sg.Text('DRDL AEROSPACE AI PLATFORM',
                 font=(_F, 15, 'bold'), text_color=C_CYAN,
                 background_color=C_BG, justification='center',
                 expand_x=True)],
        [sg.Text('Secure Access  |  v9.7',
                 font=(_F, 11), text_color=C_DIM,
                 background_color=C_BG, justification='center',
                 expand_x=True, pad=(0, (0, 18)))],
        [sg.Text('Username', size=(10, 1), font=(_F, 11),
                 text_color=C_DIM, background_color=C_BG),
         sg.Input('', key='LG_USER', size=(24, 1),
                  font=(_F, 12, 'bold'), background_color=C_INP,
                  text_color=C_WHITE, border_width=1, focus=True)],
        [sg.Text('', size=(0, 1), background_color=C_BG, pad=(0, 3))],
        [sg.Text('Password', size=(10, 1), font=(_F, 11),
                 text_color=C_DIM, background_color=C_BG),
         sg.Input('', key='LG_PASS', size=(24, 1),
                  font=(_F, 12, 'bold'), background_color=C_INP,
                  text_color=C_WHITE, border_width=1, password_char='*')],
        [sg.Text('', key='LG_ERR', size=(40, 1), font=(_F, 11),
                 text_color=C_RED, background_color=C_BG,
                 pad=(0, (8, 6)))],
        [sg.Column([[
            sg.Button('  LOGIN  ', key='LG_OK',
                      font=(_F, 12, 'bold'),
                      button_color=(C_WHITE, C_BLUE),
                      border_width=2, bind_return_key=True,
                      mouseover_colors=(C_WHITE, '#2563EB'), pad=(10, 5)),
            sg.Button('  EXIT  ', key='LG_EXIT',
                      font=(_F, 12, 'bold'),
                      button_color=(C_WHITE, '#7F1D1D'),
                      border_width=2,
                      mouseover_colors=(C_WHITE, '#B91C1C'), pad=(10, 5)),
        ]], background_color=C_BG, justification='center', expand_x=True)],
        [sg.Text('', background_color=C_BG, pad=(0, 10))],
    ]
    win = sg.Window('DRDL Login', ly, size=(460, 330), finalize=True,
                    background_color=C_BG,
                    element_justification='center',
                    margins=(22, 12), keep_on_top=True)
    attempts = 0
    while True:
        ev, vals = win.read(timeout=100)
        if ev in (sg.WIN_CLOSED, 'LG_EXIT', None):
            win.close()
            return False
        if ev == 'LG_OK':
            u = vals.get('LG_USER', '').strip()
            p = vals.get('LG_PASS', '').strip()
            if _VALID_USERS.get(u) == p:
                win.close()
                return True
            attempts += 1
            win['LG_ERR'].update(
                f'✗  Invalid credentials  (attempt {attempts})')
            win['LG_PASS'].update('')

if not show_login():
    sys.exit(0)

# =========================================================
# RUNTIME STATE
# =========================================================
_model_rdy  = True
_opt_run    = False
_flt_run    = False
_is_max     = True
pred_q      = queue.Queue()
opt_log_q   = queue.Queue()

_opt_figs   = []
_opt_idx    = 0
_opt_agg    = None

_env_figs   = []
_env_idx    = 0
_env_agg    = None

_t_start_pred = '--:--:--'
_t_start_opt  = '--:--:--'
_t_start_env  = '--:--:--'

_OPT_TITLES = [
    'Fitness Evolution',
    'Aero Metrics per Gen',
    'CL/CD vs XCP Scatter',
]
_ENV_TITLES = [
    'Alpha Sweep (CL / CD / XCP)',
    'Mach Sweep (CL / CD / CL/CD)',
    'Altitude Sweep (CL / CD / CL/CD)',
]

_SWEEP_POOL = ThreadPoolExecutor(max_workers=4)

# =========================================================
# HELPERS
# =========================================================
def sf(values, key, default=0.0):
    try:    return float(values[key])
    except: return default

def _ts():
    return datetime.now().strftime('%H:%M:%S')

# ─────────────────────────────────────────────────────────
# sec_hdr / sec_hdr_rows — LAYOUT RULES
# ─────────────────────────────────────────────────────────
# sec_hdr(text)      → returns [sg.Text(...)]     — already IS a valid row.
#   Use it BARE inside a plain list-of-rows:
#       layout = [ sec_hdr('TITLE'), [el, el], [el] ]
#
# sec_hdr_rows(text) → returns [[sg.Text(...)]]   — list containing one row.
#   Use it with + concatenation so the + operator works on list-of-rows:
#       layout = sec_hdr_rows('TITLE') + [[el,el], [el]]
# ─────────────────────────────────────────────────────────
def sec_hdr(text, color=C_CYAN):
    """Returns ONE ROW as a flat list: [sg.Text(...)].
    Use bare (no extra []) inside a plain list-of-rows literal."""
    return [sg.Text(f'▌ {text}', font=F_SEC, text_color=color,
                    background_color=C_PANEL,
                    pad=(6, (10, 3)), expand_x=True)]

def sec_hdr_rows(text, color=C_CYAN):
    """Returns LIST-OF-ROWS: [[sg.Text(...)]].
    Use with + concatenation on other list-of-rows."""
    return [sec_hdr(text, color)]

def lbl(text, w=26):
    return sg.Text(text, size=(w, 1), font=F_LBL,
                   text_color=C_DIM, background_color=C_PANEL,
                   pad=(4, 3))

def inp(key, value='', w=12):
    return sg.Input(str(value), key=key, size=(w, 1), font=F_INP,
                    background_color=C_INP, text_color=C_WHITE,
                    border_width=1, pad=(4, 3))

def out_field(key, w=20, color=C_CYAN):
    return sg.Input('---', key=key, size=(w, 1), font=F_OUT,
                    text_color=color, background_color=C_BDR,
                    readonly=True, border_width=0,
                    disabled_readonly_background_color=C_BDR,
                    disabled_readonly_text_color=color, pad=(4, 3))

def action_btn(text, key, bg=C_BLUE, w=20):
    return sg.Button(text, key=key, size=(w, 1), font=F_BTN,
                     button_color=(C_WHITE, bg),
                     border_width=2,
                     mouseover_colors=(C_WHITE, '#2563EB'),
                     pad=(6, 6))

def prog_row(bk, pk, mk):
    return [
        sg.ProgressBar(100, orientation='h', size=(24, 18), key=bk,
                       bar_color=(C_CYAN, '#0A1020'),
                       expand_x=True, pad=(4, 4)),
        sg.Text(' 0%', key=pk, size=(5, 1), font=F_INP,
                text_color=C_CYAN, background_color=C_BAR),
        sg.Text('', key=mk, size=(22, 1), font=F_LBL,
                text_color=C_AMBER, background_color=C_BAR,
                expand_x=True),
    ]

def set_prog(bk, pk, mk, pct, msg=''):
    pct = max(0, min(100, int(pct)))
    window[bk].update(pct)
    window[pk].update(f'{pct:3d}%')
    window[mk].update(msg)

def set_status(msg, elapsed=None, color=C_BLUE):
    window['STS'].update(msg)
    window['STS'].update(text_color=color)
    window['STS_T'].update(
        f'Time: {elapsed:.3f}s' if elapsed is not None else '')
    window.refresh()

def con_clear(key):
    window[key].update('', disabled=False)
    window[key].update(disabled=True)

def con_append(key, text):
    el = window[key]
    el.update(disabled=False)
    el.print(text)
    el.update(disabled=True)

# =========================================================
# ACTION BAR BUILDER
# ---------------------------------------------------------
# ROOT CAUSE OF INVISIBLE BUTTONS IN TAB 1 & TAB 2:
#   The action bar was the *last row* inside the same Column
#   that was `scrollable=True`. When the column content was
#   taller than the viewport the bar scrolled out of view
#   and, because it shared C_BG background, appeared gone.
#
# FIX: action bar is built as a *separate* sg.Frame element
#   placed OUTSIDE and BELOW the scrollable content column,
#   at the tab's top-level layout row.  It always stays
#   pinned at the bottom of the tab, fully visible.
# =========================================================
def action_bar(key_prefix, buttons, bar_bg=C_BAR):
    """
    Returns a single-row layout (list of elements) suitable
    for a dedicated non-scrolling row at the bottom of a tab.

    buttons : list of (label, key, bg_colour, width) tuples
    """
    btn_els = []
    for (txt, k, bg, w) in buttons:
        btn_els.append(action_btn(txt, k, bg, w))

    prog_key  = f'PB_{key_prefix}'
    pct_key   = f'PP_{key_prefix}'
    msg_key   = f'PM_{key_prefix}'

    row = (
        btn_els +
        [sg.Push(background_color=bar_bg),
         sg.Text('Progress:', font=F_LBL,
                 text_color=C_DIM, background_color=bar_bg)] +
        prog_row(prog_key, pct_key, msg_key)
    )

    return sg.Frame(
        '', [row],
        background_color=bar_bg,
        border_width=0,
        relief=sg.RELIEF_FLAT,
        expand_x=True,
        pad=(0, 0),
    )

# =========================================================
# MATPLOTLIB DARK STYLE
# =========================================================
def _mpl_style():
    plt.rcParams.update({
        'figure.facecolor' : C_BG,
        'axes.facecolor'   : C_INP,
        'axes.edgecolor'   : C_BDR,
        'axes.labelcolor'  : C_DIM,
        'axes.titlecolor'  : C_CYAN,
        'xtick.color'      : C_DIM,
        'ytick.color'      : C_DIM,
        'grid.color'       : C_BDR,
        'grid.linewidth'   : 0.6,
        'grid.linestyle'   : '--',
        'text.color'       : C_WHITE,
        'font.family'      : _F,
        'axes.titlesize'   : 10,
        'axes.labelsize'   : 9,
        'xtick.labelsize'  : 8,
        'ytick.labelsize'  : 8,
        'legend.fontsize'  : 8,
        'legend.facecolor' : C_PANEL,
        'legend.edgecolor' : C_BDR,
    })

# =========================================================
# CANVAS EMBED
# =========================================================
def _embed_fig(fig, canvas_key):
    cv = window[canvas_key].TKCanvas
    for ch in cv.winfo_children():
        ch.destroy()
    agg = FigureCanvasTkAgg(fig, master=cv)
    agg.draw()
    agg.get_tk_widget().pack(side='top', fill='both', expand=True)
    agg.mpl_connect('button_press_event',
                    lambda e: (_open_zoom(fig) if e.dblclick else None))
    return agg

def _open_zoom(fig):
    _mpl_style()
    fig2 = plt.figure(figsize=(14, 6))
    n_ax = len(fig.axes)
    for i, ax_src in enumerate(fig.axes):
        ax2 = fig2.add_subplot(1, n_ax, i + 1)
        for line in ax_src.get_lines():
            ax2.plot(line.get_xdata(), line.get_ydata(),
                     color=line.get_color(), lw=line.get_linewidth() + 0.5,
                     marker=line.get_marker(),
                     ms=line.get_markersize() + 2,
                     label=line.get_label())
        for coll in ax_src.collections:
            try:
                offs = coll.get_offsets()
                if len(offs):
                    fc = coll.get_facecolor()
                    ax2.scatter(offs[:, 0], offs[:, 1],
                                color=fc[0], s=40, alpha=0.8)
            except Exception:
                pass
        ax2.set_title(ax_src.get_title(), color=C_CYAN, fontsize=10)
        ax2.set_xlabel(ax_src.get_xlabel(), color=C_DIM)
        ax2.set_ylabel(ax_src.get_ylabel(), color=C_DIM)
        ax2.set_facecolor(C_INP)
        ax2.grid(True, lw=0.5, ls='--', color=C_BDR)
        if any(ln.get_label() and not ln.get_label().startswith('_')
               for ln in ax_src.get_lines()):
            ax2.legend()
    fig2.patch.set_facecolor(C_BG)
    fig2.suptitle('ZOOM  (close window to return)',
                  color=C_AMBER, fontsize=11)
    fig2.tight_layout()
    plt.show(block=False)

def _save_fig(fig, out_dir, filename):
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename),
                    dpi=120, bbox_inches='tight', facecolor=C_BG)
    except Exception as ex:
        print(f'[WARN] save {filename}: {ex}')

# =========================================================
# PLOT CONSOLE SHOW HELPERS
# =========================================================
def _show_opt_plot(idx):
    global _opt_idx, _opt_agg
    if not _opt_figs:
        return
    idx = max(0, min(idx, len(_opt_figs) - 1))
    _opt_idx = idx
    _opt_agg = _embed_fig(_opt_figs[idx], 'CANVAS_OPT')
    title = (_OPT_TITLES[idx] if idx < len(_OPT_TITLES)
             else f'Plot {idx + 1}')
    window['OPT_PLT_LBL'].update(
        f'{idx + 1} / {len(_opt_figs)}  —  {title}')
    window.refresh()

def _show_env_plot(idx):
    global _env_idx, _env_agg
    if not _env_figs:
        return
    idx = max(0, min(idx, len(_env_figs) - 1))
    _env_idx = idx
    _env_agg = _embed_fig(_env_figs[idx], 'CANVAS_ENV')
    title = (_ENV_TITLES[idx] if idx < len(_ENV_TITLES)
             else f'Plot {idx + 1}')
    window['ENV_PLT_LBL'].update(
        f'{idx + 1} / {len(_env_figs)}  —  {title}')
    window.refresh()

# =========================================================
# PLOT CONSOLE WIDGET BUILDER
# Returns list-of-rows — safe for + concatenation.
# =========================================================
def _plot_console(canvas_key, lbl_key, prev_key, next_key,
                  hdr_text, hdr_color, canvas_h=290):
    return (
        sec_hdr_rows(hdr_text, hdr_color) +
        [[sg.Text('', key=lbl_key, font=F_PLTLBL,
                  text_color=hdr_color, background_color=C_PANEL,
                  size=(54, 1), pad=(6, 2))]] +
        [[sg.Canvas(key=canvas_key, size=(820, canvas_h),
                    background_color=C_BG,
                    expand_x=True, pad=(4, 4))]] +
        [[
            sg.Text('', expand_x=True, background_color=C_PANEL),
            sg.Button('◄', key=prev_key, size=(4, 1), font=F_ARROW,
                      button_color=(C_WHITE, '#1E3A5F'),
                      mouseover_colors=(C_WHITE, '#4F46E5'),
                      border_width=2, pad=(4, 4),
                      tooltip='Previous plot'),
            sg.Text('', size=(2, 1), background_color=C_PANEL),
            sg.Button('►', key=next_key, size=(4, 1), font=F_ARROW,
                      button_color=(C_WHITE, '#1E3A5F'),
                      mouseover_colors=(C_WHITE, '#4F46E5'),
                      border_width=2, pad=(4, 4),
                      tooltip='Next plot'),
            sg.Text('  Double-click plot to zoom',
                    font=(_F, 10), text_color=C_DIM,
                    background_color=C_PANEL),
            sg.Text('', expand_x=True, background_color=C_PANEL),
        ]]
    )

# =========================================================
# SCROLLABLE COLUMN HEIGHT
# ---------------------------------------------------------
# ROOT CAUSE OF THE TAB 2 "MISSING" RUN/ABORT/CLEAR BUTTONS:
#   sg.Column(scrollable=True) only actually scrolls (i.e. caps
#   its own height and gives you a scrollbar) if you also give
#   it an explicit size=(w, h). Without a height, the column
#   just grows to fit ALL of its content. Tab 2's right-hand
#   column stacks an 22-line + 12-line Multiline plus a 300px
#   plot canvas — that's taller than most screens. Because the
#   action bar sits in a row *below* that column, the column's
#   real height pushes the action bar below the bottom of the
#   window, so the RUN OPTIMIZER / ABORT / CLEAR buttons are
#   never drawn on screen at all (it isn't a colour/contrast
#   problem — the row is simply off-window).
#
# FIX: give every scrollable Column an explicit height so it
#   scrolls internally instead of stretching the tab. The
#   action bar row then always has room and stays visible.
#   Tune _COL_H if you're on a much larger/smaller monitor.
# =========================================================
_COL_H = 560

# =========================================================
# ── TAB 1 : PREDICTION
# =========================================================
_geo  = PARAMS[:9]
_tail = PARAMS[9:15]
_flt  = PARAMS[15:]

# GEO_COL — uses sec_hdr_rows() + concatenation: correct
GEO_COL = (
    sec_hdr_rows('GEOMETRY PARAMETERS', C_BLUE) +
    [[lbl(LABELS[p]), inp(p, DEFAULTS[p])] for p in _geo] +
    sec_hdr_rows('FLIGHT CONDITIONS', C_AMBER) +
    [[lbl(LABELS[p]), inp(p, DEFAULTS[p])] for p in _flt] +
    sec_hdr_rows('TAIL PARAMETERS', C_GREEN) +
    [[lbl(LABELS[p]), inp(p, DEFAULTS[p])] for p in _tail]
)

# OUT_COL — plain list literal. sec_hdr() used BARE (no wrapping []).
OUT_COL = [
    sec_hdr('AERODYNAMIC OUTPUTS', C_CYAN),
    [lbl('Lift Coefficient  CL',   28), out_field('CL_OUT',  18, C_GREEN)],
    [lbl('Drag Coefficient  CD',   28), out_field('CD_OUT',  18, C_RED)],
    [lbl('Centre of Pressure XCP', 28), out_field('XCP_OUT', 18, C_BLUE)],
    [lbl('XCP/D (dataset rows)',   28), out_field('XCPD_OUT', 18, C_PURP),
     sg.Text('(dataset rows only)', font=(_F, 10),
             text_color=C_DIM, background_color=C_PANEL)],
    [lbl('Lift-to-Drag  CL/CD',   28), out_field('LD_OUT',  18, C_AMBER)],
    [lbl('Computation Time',       28), out_field('TIME_P',  30)],
    [lbl('Run Start / End Time',   28), out_field('TIME_STAMP_P', 30)],
    sec_hdr('MODEL ACCURACY', C_GREEN),
    [lbl('CL  -- MAE / RMSE / R2', 28), out_field('MET_CL',  36)],
    [lbl('CD  -- MAE / RMSE / R2', 28), out_field('MET_CD',  36)],
    [lbl('XCP -- MAE / RMSE / R2', 28), out_field('MET_XCP', 36)],
    # hidden stubs
    [sg.Input('', key='SRC_OUT',   visible=False)],
    [sg.Input('', key='MODE_OUT',  visible=False)],
    [sg.Multiline('', key='PRED_CON', visible=False, disabled=True)],
    [sg.Multiline('', key='TOP5_OUT', visible=False, disabled=True)],
]

# ── Tab 1 action bar (separate, non-scrolling row) ─────
_pred_bar = action_bar('P', [
    ('▶  ESTIMATE (F5)', 'Estimate',   '#065F46', 22),
    ('↺  RESET',         'Reset_Pred', '#1E3A5F', 14),
])

prediction_tab = [
    # Main two-column area — scrollable columns, never clips buttons
    [sg.Column(GEO_COL,  background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True,
               pad=(8, 6)),
     sg.VSeparator(color=C_BDR, pad=(3, 0)),
     sg.Column(OUT_COL,  background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True, pad=(8, 6))],
    # Action bar — pinned below, always visible
    [_pred_bar],
]

# =========================================================
# ── TAB 2 : OPTIMIZER
# =========================================================
bounds_rows = [[
    sg.Text('Parameter',   size=(28, 1), font=(_F, 12, 'bold'),
            text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
    sg.Text('Lower Bound', size=(14, 1), font=(_F, 12, 'bold'),
            text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
    sg.Text('Upper Bound', size=(14, 1), font=(_F, 12, 'bold'),
            text_color=C_AMBER, background_color=C_PANEL, pad=(4, 3)),
]]
for p in PARAMS:
    lo, hi = BOUNDS[p]
    bounds_rows.append([lbl(LABELS[p], 28),
                        inp(f'{p}_LOW',  lo, 11),
                        inp(f'{p}_HIGH', hi, 11)])

OPT_LEFT = (
    sec_hdr_rows('PARAMETER SEARCH BOUNDS', C_AMBER) +
    bounds_rows +
    [
        sec_hdr('OUTPUT CONSTRAINTS', C_RED),
        [sg.Text(
            'Penalty: infeasible if constraint violated (fitness → −∞).',
            font=(_F, 11), text_color=C_DIM,
            background_color=C_PANEL, pad=(6, 4))],
        [lbl('CL  Min',  12), inp('CL_MIN',  '-3.723'),
         lbl('CL  Max',  12), inp('CL_MAX',  '15.2213')],
        [lbl('CD  Min',  12), inp('CD_MIN',  '-1.187'),
         lbl('CD  Max',  12), inp('CD_MAX',  '5.7352')],
        [lbl('XCP Min',  12), inp('XCP_MIN', '-12.3114'),
         lbl('XCP Max',  12), inp('XCP_MAX', '-3.5322')],
        sec_hdr('OPTIMIZATION SETTINGS', C_BLUE),
        [lbl('Max Generations',    24), inp('MAXITER', '50', 8),
         lbl('Population Size',    18), inp('POPSIZE', '10', 8)],
        [lbl('Max Gene-Swap Steps', 24), inp('ITERMAX', '5',  8),
         sg.Text('(Algorithm 1 max-crossover)',
                 font=(_F, 11), text_color=C_DIM,
                 background_color=C_PANEL)],
        [sg.Input('de_output', key='OUT_DIR', visible=False)],
    ]
)

_TOP5_INFO = (
    'HOW TOP-5 ARE SELECTED:\n'
    '  Fitness = 0.95*(CL/CD) + 0.05*0.2464*(1/|XCP-(-5.15)|),  CD >= 0.12.\n'
    '  Best-of-generation -> hall-of-fame -> sorted, de-duplicated, top 5 kept.\n'
)

OPT_RIGHT = (
    [
        sec_hdr('OPTIMAL RESULT', C_CYAN),
        [lbl('Best CL',              28), out_field('OPT_CL',        18, C_GREEN)],
        [lbl('Best CD',              28), out_field('OPT_CD',        18, C_RED)],
        [lbl('Best XCP',             28), out_field('OPT_XCP',       18, C_BLUE)],
        [lbl('Best XCP/D',           28), out_field('OPT_XCPD',      18, C_PURP)],
        [lbl('Max CL/CD',            28), out_field('OPT_LD',        18, C_AMBER)],
        [lbl('Composite Fitness',    28), out_field('OPT_FIT',       18, C_CYAN)],
        [lbl('Elapsed Time',         28), out_field('OPT_TIME',      24)],
        [lbl('Run Start / End Time', 28), out_field('OPT_TIMESTAMP', 30)],
        [sg.Input('', key='OPT_MODE', visible=False)],
        sec_hdr('LIVE OPTIMIZATION LOG', C_GREEN),
        [sg.Multiline('', key='OPT_LOG', size=(56, 5),
                      font=F_TBL, background_color=C_DARK,
                      text_color=C_GREEN, autoscroll=True,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
    ] +
    sec_hdr_rows('BEST GEOMETRY (all 18 parameters)', C_AMBER) +
    [
        [sg.Multiline('', key='OPT_GEO', size=(56, 22),
                      font=F_TBL, background_color=C_PANEL,
                      text_color=C_AMBER, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
    ] +
    sec_hdr_rows('TOP-5 BEST PARAMETER SETS', C_PURP) +
    [
        [sg.Text(_TOP5_INFO, font=(_F, 10),
                 text_color=C_DIM, background_color=C_PANEL,
                 pad=(6, 3))],
        [sg.Multiline('', key='TOP5_OPT', size=(56, 12),
                      font=F_TOP5, background_color=C_PANEL,
                      text_color=C_PURP, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
    ] +
    _plot_console('CANVAS_OPT', 'OPT_PLT_LBL',
                  'OPT_PREV', 'OPT_NEXT',
                  'OPTIMISATION PLOTS  (◄ / ► to browse,  double-click = zoom)',
                  C_INDG, canvas_h=300)
)

# ── Tab 2 action bar (separate, non-scrolling row) ─────
_opt_bar = action_bar('O', [
    ('▶  RUN OPTIMIZER', 'Run_Opt',   '#065F46', 22),
    ('✕  ABORT',         'Abort_Opt', '#7F1D1D', 12),
    ('↺  CLEAR',         'Clear_Opt', '#1E3A5F', 12),
])

optimization_tab = [
    [sg.Column(OPT_LEFT,  background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True,
               pad=(8, 6)),
     sg.VSeparator(color=C_BDR, pad=(3, 0)),
     sg.Column(OPT_RIGHT, background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True,
               pad=(8, 6))],
    # Action bar — pinned below, always visible
    [_opt_bar],
]

# =========================================================
# ── TAB 3 : FLIGHT ENVELOPE
# =========================================================
def sweep_frame(title, mk, lk, sk, mv, lv, sv):
    return sg.Frame(title, [[
        sg.Text('Min',  size=(3, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(mk, mv, 7),
        sg.Text('Max',  size=(3, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(lk, lv, 7),
        sg.Text('Step', size=(4, 1), font=F_LBL,
                text_color=C_DIM, background_color=C_PANEL),
        inp(sk, sv, 7),
    ]], font=F_SEC, title_color=C_CYAN,
       background_color=C_PANEL, border_width=1,
       relief=sg.RELIEF_FLAT, expand_x=True, pad=(6, 5))

ENV_BASE_ROWS = (
    sec_hdr_rows('BASE PARAMETERS  (fixed during sweep)', C_GREEN) +
    [[lbl(LABELS[p], 28), inp(f'E_{p}', DEFAULTS[p])] for p in PARAMS]
)

ENV_LEFT = (
    [
        sec_hdr('GEOMETRY SOURCE', C_AMBER),
        [sg.Checkbox(
            '  Use optimal geometry from optimizer (loads best_geometry.csv)',
            key='USE_OPT_GEO', default=False,
            font=F_LBL, text_color=C_AMBER,
            background_color=C_PANEL,
            checkbox_color=C_AMBER,
            enable_events=True, pad=(8, 6))],
        [sg.Text('', key='OPT_GEO_STATUS',
                 font=(_F, 11), text_color=C_GREEN,
                 background_color=C_PANEL,
                 size=(58, 1), pad=(10, 4))],
        sec_hdr('SWEEP RANGES', C_BLUE),
        [sweep_frame('ALPHA SWEEP (deg)',
                     'ALPHA_MIN', 'ALPHA_MAX', 'ALPHA_STP',
                     '0', '20', '2')],
        [sweep_frame('MACH NUMBER SWEEP',
                     'MACH_MIN', 'MACH_MAX', 'MACH_STP',
                     '0.2', '0.8', '0.1')],
        [sweep_frame('ALTITUDE SWEEP (m)',
                     'ALT_MIN', 'ALT_MAX', 'ALT_STP',
                     '0', '6000', '1000')],
    ] + ENV_BASE_ROWS
)

ENV_RIGHT = (
    [
        sec_hdr('ALPHA SWEEP RESULTS', C_BLUE),
        [lbl('Run Start / End Time', 28), out_field('ENV_TIMESTAMP', 30)],
        [sg.Multiline('', key='ENV_ALPHA', size=(66, 7),
                      font=F_TBL, background_color=C_PANEL,
                      text_color=C_CYAN, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
        sec_hdr('MACH SWEEP RESULTS', C_GREEN),
        [sg.Multiline('', key='ENV_MACH', size=(66, 7),
                      font=F_TBL, background_color=C_PANEL,
                      text_color=C_GREEN, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
        sec_hdr('ALTITUDE SWEEP RESULTS', C_AMBER),
        [sg.Multiline('', key='ENV_ALT', size=(66, 7),
                      font=F_TBL, background_color=C_PANEL,
                      text_color=C_AMBER, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
        sec_hdr('SUMMARY STATISTICS', C_RED),
        [sg.Multiline('', key='ENV_SUM', size=(66, 8),
                      font=F_TBL, background_color=C_PANEL,
                      text_color=C_WHITE, autoscroll=False,
                      border_width=1, expand_x=True,
                      disabled=True, pad=(6, 4))],
    ] +
    _plot_console('CANVAS_ENV', 'ENV_PLT_LBL',
                  'ENV_PREV', 'ENV_NEXT',
                  'SWEEP PLOTS  (◄ / ► to browse,  double-click = zoom)',
                  C_CYAN, canvas_h=300) +
    [
        [sg.Text(
            '  All 3 plots saved as PNG files to the output directory.',
            font=(_F, 10), text_color=C_DIM,
            background_color=C_PANEL, pad=(8, 5))],
    ]
)

# ── Tab 3 action bar ────────────────────────────────────
_env_bar = action_bar('E', [
    ('▶  RUN SWEEPS',  'Run_Env',   '#065F46', 18),
    ('↺  CLEAR',       'Clear_Env', '#1E3A5F', 12),
    ('⬇  EXPORT CSV', 'Export_Env', '#374151', 15),
])

flight_tab = [
    [sg.Column(ENV_LEFT,  background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True,
               pad=(8, 6)),
     sg.VSeparator(color=C_BDR, pad=(3, 0)),
     sg.Column(ENV_RIGHT, background_color=C_PANEL,
               expand_x=True, expand_y=True, size=(None, _COL_H),
               scrollable=True, vertical_scroll_only=True,
               pad=(8, 6))],
    [_env_bar],
]

# =========================================================
# MAIN LAYOUT
# =========================================================
_mode_label = 'ENSEMBLE' if ENSEMBLE_MODE else 'XGBoost'

HEADER_ROW = [
    sg.Text(
        '  OPTIMAL AERODYNAMIC CONFIGURATION DESIGN — AEROSPACE VEHICLES',
        font=F_TITLE, text_color=C_CYAN, background_color=C_HDR,
        justification='left', expand_x=True, pad=(10, 6)),
    sg.Button(' _ ', key='W_MIN',  size=(3, 1), font=F_CHROME,
              button_color=(C_WHITE, '#334155'),
              mouseover_colors=(C_WHITE, '#475569'),
              border_width=0, pad=(3, 2), tooltip='Minimise'),
    sg.Button('[ ]', key='W_MAX', size=(3, 1), font=F_CHROME,
              button_color=(C_WHITE, '#334155'),
              mouseover_colors=(C_WHITE, '#475569'),
              border_width=0, pad=(3, 2), tooltip='Maximise/Restore'),
    sg.Button(' X ', key='Exit',  size=(3, 1), font=F_CHROME,
              button_color=(C_WHITE, '#B91C1C'),
              mouseover_colors=(C_WHITE, '#DC2626'),
              border_width=0, pad=(3, 2), tooltip='Close'),
    sg.Text('  ', background_color=C_HDR),
]

SUB_HDR_ROW = [
    sg.Text(
        f'  XGBoost Engine  |  DE Optimizer  |  Flight Envelope  '
        f'|  Mode: {_mode_label}  |  v9.7',
        font=F_SUB, text_color=C_DIM, background_color=C_HDR,
        justification='left', expand_x=True, pad=(10, 2)),
]

TAB_KW = dict(expand_x=True, expand_y=True,
              pad=(0, 0), background_color=C_BG)

tab_group = sg.TabGroup([[
    sg.Tab('  PREDICTION  ',      prediction_tab,   **TAB_KW),
    sg.Tab('  OPTIMIZER   ',      optimization_tab, **TAB_KW),
    sg.Tab('  FLIGHT ENVELOPE ',  flight_tab,       **TAB_KW),
]], tab_location='top', font=F_TABTXT,
    selected_title_color=C_WHITE,
    title_color=C_DIM,
    selected_background_color='#1D4ED8',
    background_color=C_DARK,
    tab_background_color=C_DARK,
    border_width=0,
    expand_x=True, expand_y=True,
    key='TABS')

STATUS_ROW = [
    sg.Text('●', font=(_F, 13), text_color=C_GREEN,
            background_color=C_DARK, pad=(10, 4)),
    sg.Text('INITIALISING...', key='STS', font=F_STS,
            text_color=C_AMBER, background_color=C_DARK, size=(58, 1)),
    sg.Push(background_color=C_DARK),
    sg.Text(f'[{_mode_label}]', font=(_F, 11, 'bold'),
            text_color=C_PURP, background_color=C_DARK, pad=(6, 4)),
    sg.Text('', key='STS_T', font=(_F, 11),
            text_color=C_CYAN, background_color=C_DARK,
            size=(18, 1), justification='right'),
]

layout = [
    [sg.Column([HEADER_ROW, SUB_HDR_ROW],
               background_color=C_HDR,
               expand_x=True, pad=(0, 0))],
    [tab_group],
    [sg.Column([STATUS_ROW], background_color=C_DARK,
               expand_x=True, pad=(0, 0))],
]

window = sg.Window(
    'DRDL Aerospace Platform v9.7', layout,
    size=(1400, 860), finalize=True,
    resizable=True, element_justification='left',
    background_color=C_BG, margins=(0, 0),
    return_keyboard_events=True,
    use_custom_titlebar=False,
)

try:
    window.maximize()
    _is_max = True
except Exception:
    _is_max = False

# =========================================================
# BACKGROUND MODEL PRELOADER + CACHE WARM-UP
# =========================================================
def _preload():
    global _model_rdy
    t0 = time.perf_counter()
    aerodynamic_prediction(DEFAULTS)
    for mach in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for alpha in [0, 2, 4, 6, 8, 10]:
            p = dict(DEFAULTS, mach=mach, alpha=alpha)
            aerodynamic_prediction(p)
    elapsed = time.perf_counter() - t0
    _model_rdy = True
    window.write_event_value('MODEL_READY', elapsed)

threading.Thread(target=_preload, daemon=True).start()
set_status('Loading XGBoost model + CSV dataset...', color=C_AMBER)
_startup_pct = 0

# =========================================================
# PREDICTION WORKER
# =========================================================
def _pred_worker(params):
    try:
        t0 = time.perf_counter()
        result = aerodynamic_prediction(params)
        elapsed = time.perf_counter() - t0
        pred_q.put({'ok': True, 'result': result, 'elapsed': elapsed})
    except Exception as e:
        pred_q.put({'ok': False, 'error': str(e)})

def render_prediction(payload):
    result  = payload['result']
    elapsed = payload['elapsed']
    cl      = result.get('CL',  0.0)
    cd      = result.get('CD',  0.0)
    xcp     = result.get('XCP', 0.0)
    xcpd    = result.get('XCP_D', None)
    met     = result.get('metrics', {})
    det     = result.get('detailed_metrics', {})
    mode    = result.get('mode', 'xgboost')
    ems     = result.get('elapsed_ms', elapsed * 1000)
    ld      = cl / cd if abs(cd) > 1e-9 else float('inf')

    window['CL_OUT' ].update(f'{cl:.4f}')
    window['CD_OUT' ].update(f'{cd:.4f}')
    window['XCP_OUT'].update(f'{xcp:.4f}')
    window['LD_OUT' ].update(f'{ld:.4f}')
    window['XCPD_OUT'].update(
        f'{xcpd:.6f}' if xcpd is not None else 'N/A')
    window['TIME_P'].update(
        f'{ems:.2f} ms pred / {elapsed * 1000:.2f} ms total')
    window['TIME_STAMP_P'].update(
        f'Start {_t_start_pred}  |  End {_ts()}')
    window['MODE_OUT'].update(
        'ENSEMBLE' if mode == 'ensemble' else 'XGBOOST')
    window['SRC_OUT'].update('XGBOOST MODEL')

    def fmt(col):
        m = (det.get(col) if det and col in det else met) or {}
        try:
            return (f"MAE={float(m.get('MAE', 0)):.4f}  "
                    f"RMSE={float(m.get('RMSE', 0)):.4f}  "
                    f"R2={float(m.get('R2', 0)):.4f}")
        except:
            return 'N/A'

    window['MET_CL' ].update(fmt('CL')      if det else fmt('avg'))
    window['MET_CD' ].update(fmt('CD')      if det else fmt('avg'))
    window['MET_XCP'].update(fmt('X-C.P.') if det else fmt('avg'))

    xs = f'{xcpd:.6f}' if xcpd is not None else 'N/A'
    set_prog('PB_P', 'PP_P', 'PM_P', 100, 'Complete')
    set_status(
        f'OK  {mode.upper()} | CL={cl:.4f}  CD={cd:.4f}  '
        f'XCP={xcp:.4f}  XCP/D={xs[:8]}',
        elapsed, C_GREEN)

# =========================================================
# OPTIMIZER WORKER
# =========================================================
def _opt_worker(bounds, maxiter, popsize, itermax,
                constraints, out_dir):
    global _opt_run
    try:
        def _log(msg):
            opt_log_q.put(msg)
            try:
                gen = int(msg.split()[1])
            except Exception:
                gen = 0
            pct = min(98, int(gen / maxiter * 100)) if maxiter > 0 else 50
            window.write_event_value('OPT_PROG', (pct, msg))

        result, history, elapsed = run_optimization(
            bounds=bounds, maxiter=maxiter, popsize=popsize,
            itermax=itermax, constraints=constraints,
            out_dir=out_dir if out_dir.strip() else None,
            log_callback=_log)
        window.write_event_value('OPT_DONE', (result, history, elapsed))
    except Exception as e:
        window.write_event_value('OPT_ERR', str(e))
    finally:
        _opt_run = False

def render_optimization(result, history, elapsed):
    global _opt_figs, _opt_idx

    _hist_keys = ['generation', 'fitness', 'avg_fitness',
                  'CL', 'CD', 'XCP', 'CLCD']
    history = [_as_dict(h, _hist_keys) for h in (history or [])]

    best_x   = result.x
    best_prm = {p: round(float(v), 4)
                for p, v in zip(_BOUNDS_ORDER, best_x)}
    t0 = time.perf_counter()
    best_r   = aerodynamic_prediction(best_prm)
    call_ms  = (time.perf_counter() - t0) * 1000

    cl   = best_r['CL']
    cd   = best_r['CD']
    xcp  = best_r['XCP']
    xcpd = best_r.get('XCP_D', None)
    mode = result.mode if hasattr(result, 'mode') else 'xgboost'
    ld   = cl / cd if abs(cd) > 1e-9 else 0.0
    comp_fit = -float(result.fun)
    mode_txt = ('ENSEMBLE (XGB+RF+GB)'
                if mode == 'ensemble' else 'XGBOOST ONLY')
    xs = f'{xcpd:.6f}' if xcpd is not None else 'N/A'

    window['OPT_CL'       ].update(f'{cl:.6f}')
    window['OPT_CD'       ].update(f'{cd:.6f}')
    window['OPT_XCP'      ].update(f'{xcp:.6f}')
    window['OPT_XCPD'     ].update(xs)
    window['OPT_LD'       ].update(f'{ld:.6f}')
    window['OPT_FIT'      ].update(f'{comp_fit:.6f}')
    window['OPT_TIME'     ].update(
        f'{elapsed:.4f} s  ({call_ms:.1f} ms/call)')
    window['OPT_TIMESTAMP'].update(
        f'Start {_t_start_opt}  |  End {_ts()}')
    window['OPT_MODE'     ].update(mode_txt)

    geo = [
        '  OPTIMAL GEOMETRY -- 18 PARAMETERS',
        '  ' + '-' * 56,
        f'  {"Parameter":<32}  {"Value":>12}',
        '  ' + '-' * 56,
    ]
    for p, v in best_prm.items():
        geo.append(f'  {LABELS[p]:<32}  {v:>12.4f}')
    geo += [
        '  ' + '-' * 56,
        f'  {"Best CL":<32}  {cl:>12.6f}',
        f'  {"Best CD":<32}  {cd:>12.6f}',
        f'  {"Best XCP (calibres)":<32}  {xcp:>12.6f}',
        f'  {"Best XCP/D":<32}  {xs:>12}',
        f'  {"Best CL/CD":<32}  {ld:>12.6f}',
        f'  {"Composite fitness":<32}  {comp_fit:>12.6f}',
        f'  {"Mode":<32}  {mode_txt}',
        '  ' + '-' * 56,
    ]
    con_clear('OPT_GEO')
    con_append('OPT_GEO', '\n'.join(geo))

    top5 = [_as_dict(s, ['rank', 'fitness', 'CL', 'CD', 'XCP', 'CLCD'])
            for s in getattr(result, 'top5_solutions', [])]
    con_clear('TOP5_OPT')
    if top5:
        hdr = (f'  {"RANK":>4}  {"FITNESS":>12}  {"CL":>10}  '
               f'{"CD":>10}  {"XCP":>10}  {"CL/CD":>10}')
        sep = '  ' + '-' * len(hdr.strip())
        lines = [
            '  TOP-5 BEST PARAMETER SETS  (after optimization)',
            sep, hdr, sep
        ]
        for sol in top5:
            lines.append(
                f'  {sol["rank"]:>4}  {sol["fitness"]:>12.6f}  '
                f'{sol["CL"]:>10.4f}  {sol["CD"]:>10.4f}  '
                f'{sol["XCP"]:>10.4f}  {sol["CLCD"]:>10.4f}')
        lines.append(sep)
        lines.append(
            '  Full 18-parameter geometry for Rank #1 is shown above.')
        con_append('TOP5_OPT', '\n'.join(lines))
    else:
        con_append('TOP5_OPT',
                   '  (not available — optimizer did not return '
                   'top5_solutions)')

    con_clear('OPT_LOG')
    con_append('OPT_LOG',
               f'Completed {len(history)} generations | '
               f'fitness={comp_fit:.6f} | {elapsed:.3f} s')

    _mpl_style()
    plt.close('all')
    _opt_figs.clear()
    _opt_idx = 0

    if history:
        gens = [h['generation'] for h in history]

        fig_a, ax_a = plt.subplots(figsize=(9, 4))
        fig_a.patch.set_facecolor(C_BG)
        bfv = [h['fitness']     for h in history]
        afv = [h['avg_fitness'] for h in history]
        ax_a.plot(gens, bfv, '-o', color=C_GREEN,  lw=2,   ms=4,
                  label='Best fitness')
        ax_a.plot(gens, afv, '-s', color=C_AMBER,  lw=1.5, ms=3,
                  label='Avg fitness', alpha=0.8)
        ax_a.fill_between(gens, bfv, alpha=0.08, color=C_GREEN)
        ax_a.axhline(bfv[-1], color=C_CYAN, lw=1, ls='--',
                     label=f'Final: {bfv[-1]:.4f}')
        ax_a.set_xlabel('Generation')
        ax_a.set_ylabel('Fitness')
        ax_a.set_title(f'DE Fitness per Generation  [{mode_txt}]')
        ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_a.legend(); ax_a.grid(True)
        fig_a.tight_layout()
        _opt_figs.append(fig_a)

        fig_b, ax_b = plt.subplots(figsize=(9, 4))
        fig_b.patch.set_facecolor(C_BG)
        ax_b.plot(gens, [h['CL']   for h in history],
                  '-o', color='steelblue', lw=2, ms=4, label='CL')
        ax_b.plot(gens, [h['CD']   for h in history],
                  '-s', color='crimson',   lw=2, ms=4, label='CD')
        ax_b.plot(gens, [h['CLCD'] for h in history],
                  '-^', color='darkgreen', lw=2, ms=4, label='CL/CD')
        ax_b.plot(gens, [h['XCP']  for h in history],
                  '-D', color='purple',    lw=2, ms=4, label='XCP')
        ax_b.set_xlabel('Generation')
        ax_b.set_ylabel('Value')
        ax_b.set_title('Aerodynamic Metrics per Generation')
        ax_b.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_b.legend(); ax_b.grid(True)
        fig_b.tight_layout()
        _opt_figs.append(fig_b)

        perf_df = getattr(result, 'perf_df', None)
        fig_c, ax_c = plt.subplots(figsize=(9, 4))
        fig_c.patch.set_facecolor(C_BG)
        if (perf_df is not None
                and 'CL/CD_pred' in perf_df.columns):
            ax_c.scatter(perf_df['CL/CD_pred'], perf_df['XCP_pred'],
                         c='steelblue', alpha=0.7,
                         edgecolor='k', s=28)
            ax_c.set_xlabel('CL/CD')
            ax_c.set_ylabel('XCP')
            ax_c.set_title('Optimised Geometry Performance')
        else:
            ax_c.scatter([h['CLCD'] for h in history],
                         [h['XCP']  for h in history],
                         c='steelblue', alpha=0.7,
                         edgecolor='k', s=28)
            ax_c.set_xlabel('CL/CD (best/gen)')
            ax_c.set_ylabel('XCP (best/gen)')
            ax_c.set_title('CL/CD vs XCP per generation')
        ax_c.scatter([ld], [xcp], c=C_AMBER, s=100, zorder=5,
                     edgecolor='white', lw=1.5,
                     label=f'Optimal  XCP={xcp:.3f}')
        ax_c.axvline(ld,  color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax_c.axhline(xcp, color=C_CYAN, lw=0.8, ls='--', alpha=0.6)
        ax_c.legend(); ax_c.grid(True)
        fig_c.tight_layout()
        _opt_figs.append(fig_c)

        _show_opt_plot(0)
        out_dir = 'de_output'
        _save_fig(fig_a, out_dir, 'opt_fitness_evolution.png')
        _save_fig(fig_b, out_dir, 'opt_metrics_per_gen.png')
        _save_fig(fig_c, out_dir, 'opt_clcd_vs_xcp.png')

    set_prog('PB_O', 'PP_O', 'PM_O', 100, 'Complete')
    set_status(
        f'OK  Optimization | CL/CD={ld:.4f} | XCP={xcp:.4f} | '
        f'fitness={comp_fit:.4f} | {mode_txt}',
        elapsed, C_GREEN)

# =========================================================
# FLIGHT ENVELOPE HELPERS
# =========================================================
def _load_optimal_base(out_dir):
    import pandas as pd
    feat_to_param = {
        'nose length' : 'nose_len',   'body_length' : 'body_len',
        'wing LE'     : 'wing_le',    'root chord'  : 'root_chord',
        'tip chord'   : 'tip_chord',  'semi-span'   : 'semi_span',
        'root th'     : 'root_th',    'tip th'      : 'tip_th',
        'wing sweep'  : 'wing_sweep', 'tail LE'     : 'tail_le',
        'root chord.1': 'root_chord1','tip chord.1' : 'tip_chord1',
        'semi-span.1' : 'semi_span1', 'root th.1'   : 'root_th1',
        'tip th.1'    : 'tip_th1',
    }
    gp = os.path.join(out_dir, 'best_geometry.csv')
    if not os.path.exists(gp):
        raise FileNotFoundError(
            f"best_geometry.csv not found in '{out_dir}'.\n"
            "Run Optimizer first.")
    df  = pd.read_csv(gp)
    row = df.iloc[0]
    base = dict(DEFAULTS)
    for feat, param in feat_to_param.items():
        if feat in row.index:
            base[param] = float(row[feat])
        elif param in row.index:
            base[param] = float(row[param])
    return base, gp

def _table(rows, var_key, var_label):
    hdr = (f'  {var_label:>10}  {"CL":>10}  {"CD":>10}  '
           f'{"XCP":>12}  {"XCP/D":>12}  {"CL/CD":>10}')
    sep = '  ' + '-' * (len(hdr) - 2)
    lines = [sep, hdr, sep]
    for r in rows:
        v    = r[var_key]
        cl   = r['CL']
        cd   = r['CD']
        xcp  = r['XCP']
        ld   = cl / cd if abs(cd) > 1e-9 else 0.0
        xcpd = r.get('XCP_D')
        xs   = f'{xcpd:.4f}' if xcpd is not None else '  N/A  '
        lines.append(
            f'  {v:>10.3f}  {cl:>10.4f}  {cd:>10.4f}  '
            f'{xcp:>12.4f}  {xs:>12}  {ld:>10.4f}')
    lines.append(sep)
    return '\n'.join(lines)

def _stats(rows, var_key, label):
    cls  = [r['CL']  for r in rows]
    cds  = [r['CD']  for r in rows]
    xcps = [r['XCP'] for r in rows]
    lds  = [r['CL'] / r['CD']
            for r in rows if abs(r['CD']) > 1e-9]
    n = len(rows)
    if n == 0:
        return f'  {label}  (0 points — no data)'
    return '\n'.join([
        f'  {label}  ({n} points)',
        f'    {"":>4}  {"Min":>8}  {"Max":>8}  {"Mean":>8}',
        f'    {"CL":>4}  {min(cls):>8.4f}  {max(cls):>8.4f}'
        f'  {sum(cls)/n:>8.4f}',
        f'    {"CD":>4}  {min(cds):>8.4f}  {max(cds):>8.4f}'
        f'  {sum(cds)/n:>8.4f}',
        f'    {"XCP":>4}  {min(xcps):>8.4f}  {max(xcps):>8.4f}'
        f'  {sum(xcps)/n:>8.4f}',
        f'    {"L/D":>4}  {min(lds):>8.4f}  {max(lds):>8.4f}'
        f'  {sum(lds)/len(lds):>8.4f}',
    ])

def _make_sweep_fig(rows, var_key, var_label, title, metrics_cfg):
    _mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle(title, color=C_CYAN, fontsize=10, fontweight='bold')
    xs = [r[var_key] for r in rows]
    for ax, (yk, tit, col) in zip(axes, metrics_cfg):
        ys = ([r['CL'] / r['CD'] if abs(r['CD']) > 1e-9 else 0.0
               for r in rows]
              if yk == 'CLCD' else [r[yk] for r in rows])
        ax.plot(xs, ys, '-o', color=col, lw=2, ms=4,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.fill_between(xs, ys, alpha=0.12, color=col)
        ax.set_title(tit, color=C_CYAN, fontsize=9)
        ax.set_xlabel(var_label, color=C_DIM, fontsize=8)
        ax.set_ylabel(yk if yk != 'CLCD' else 'CL/CD',
                      color=C_DIM, fontsize=8)
        ax.grid(True, lw=0.5, ls='--', color=C_BDR)
        ax.set_facecolor(C_INP)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

_ALPHA_CFG = [('CL',   'CL vs Alpha',        C_BLUE),
              ('CD',   'CD vs Alpha',         C_RED),
              ('XCP',  'XCP vs Alpha',        C_AMBER)]
_MACH_CFG  = [('CL',   'CL vs Mach',         C_GREEN),
              ('CD',   'CD vs Mach',          C_RED),
              ('CLCD', 'CL/CD vs Mach',       C_AMBER)]
_ALT_CFG   = [('CL',   'CL vs Altitude',     C_CYAN),
              ('CD',   'CD vs Altitude',      C_RED),
              ('CLCD', 'CL/CD vs Altitude',   C_AMBER)]

def render_flight(ar, mr, lr, elapsed,
                  geom_label='user-typed base params',
                  base_params=None, out_dir='de_output'):
    global _env_figs, _env_idx

    n = len(ar) + len(mr) + len(lr)
    window['ENV_TIMESTAMP'].update(
        f'Start {_t_start_env}  |  End {_ts()}')
    con_clear('ENV_ALPHA')
    con_append('ENV_ALPHA', _table(ar, 'alpha', 'Alpha(deg)'))
    con_clear('ENV_MACH')
    con_append('ENV_MACH',  _table(mr, 'mach',  'Mach'))
    con_clear('ENV_ALT')
    con_append('ENV_ALT',   _table(lr, 'alt',   'Alt(m)'))
    con_clear('ENV_SUM')
    summary_parts = [
        f'  Geometry source    : {geom_label}',
        f'  Total simulations  : {n}   Elapsed: {elapsed:.3f} s',
    ]
    if n > 0:
        summary_parts.append(
            f'  Avg time/sim       : {elapsed/n*1000:.2f} ms')
    summary_parts += [
        '',
        _stats(ar, 'alpha', 'ALPHA SWEEP'),
        '',
        _stats(mr, 'mach',  'MACH  SWEEP'),
        '',
        _stats(lr, 'alt',   'ALTITUDE SWEEP'),
    ]
    con_append('ENV_SUM', '\n'.join(summary_parts))

    _mpl_style()
    plt.close('all')
    _env_figs.clear()
    _env_idx = 0

    fig1 = _make_sweep_fig(ar, 'alpha', 'Alpha (deg)',
                           f'Alpha Sweep [{geom_label}]', _ALPHA_CFG)
    _env_figs.append(fig1)
    _save_fig(fig1, out_dir, 'sweep_alpha.png')

    fig2 = _make_sweep_fig(mr, 'mach', 'Mach',
                           f'Mach Sweep [{geom_label}]', _MACH_CFG)
    _env_figs.append(fig2)
    _save_fig(fig2, out_dir, 'sweep_mach.png')

    fig3 = _make_sweep_fig(lr, 'alt', 'Altitude (m)',
                           f'Altitude Sweep [{geom_label}]', _ALT_CFG)
    _env_figs.append(fig3)
    _save_fig(fig3, out_dir, 'sweep_altitude.png')

    _show_env_plot(0)

    set_prog('PB_E', 'PP_E', 'PM_E', 100, 'Sweeps complete')
    set_status(
        (f'OK  Flight envelope | {n} sims | '
         f'{elapsed*1000/n:.1f} ms/sim | {elapsed:.3f} s | '
         f'Plots saved to {out_dir}') if n > 0 else
        f'OK  Flight envelope | {elapsed:.3f} s',
        elapsed, C_GREEN)

# =========================================================
# EXPORT CSV
# =========================================================
_last_sweep = {'ar': [], 'mr': [], 'lr': [], 'label': ''}

def export_envelope_csv():
    ar = _last_sweep['ar']
    mr = _last_sweep['mr']
    lr = _last_sweep['lr']
    if not (ar or mr or lr):
        sg.popup_quick_message('Run the sweeps first before exporting.')
        return
    path = sg.popup_get_file(
        'Save sweep results as CSV', save_as=True,
        default_extension='.csv',
        file_types=(('CSV Files', '*.csv'),))
    if not path:
        return
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Sweep', 'Variable', 'Value',
                    'CL', 'CD', 'XCP', 'XCP_D', 'CL_CD'])
        for rows, sname, vk in [
                (ar, 'Alpha', 'alpha'),
                (mr, 'Mach',  'mach'),
                (lr, 'Altitude', 'alt')]:
            for r in rows:
                cl  = r['CL']
                cd  = r['CD']
                xcp = r['XCP']
                ld  = cl / cd if abs(cd) > 1e-9 else ''
                w.writerow([sname, vk, r[vk],
                             cl, cd, xcp,
                             r.get('XCP_D', ''), ld])
    sg.popup_quick_message(f'Exported to:\n{path}')

# =========================================================
# RESET / CLEAR HELPERS
# =========================================================
def reset_pred():
    for p in PARAMS:
        window[p].update(str(DEFAULTS[p]))
    for k in ['CL_OUT', 'CD_OUT', 'XCP_OUT', 'XCPD_OUT', 'LD_OUT',
              'TIME_P', 'MET_CL', 'MET_CD', 'MET_XCP']:
        window[k].update('---')
    window['TIME_STAMP_P'].update('---')
    set_prog('PB_P', 'PP_P', 'PM_P', 0, '')
    set_status('Parameters reset to defaults.')

def clear_opt():
    global _opt_figs, _opt_idx
    for k in ['OPT_CL', 'OPT_CD', 'OPT_XCP', 'OPT_XCPD',
              'OPT_LD', 'OPT_FIT', 'OPT_TIME']:
        window[k].update('---')
    window['OPT_TIMESTAMP'].update('---')
    con_clear('OPT_GEO')
    con_clear('TOP5_OPT')
    con_clear('OPT_LOG')
    _opt_figs.clear()
    _opt_idx = 0
    window['OPT_PLT_LBL'].update('')
    try:
        cv = window['CANVAS_OPT'].TKCanvas
        for ch in cv.winfo_children():
            ch.destroy()
    except Exception:
        pass
    set_prog('PB_O', 'PP_O', 'PM_O', 0, '')
    set_status('Optimization results cleared.')

def clear_env():
    global _env_figs, _env_idx
    for k in ['ENV_ALPHA', 'ENV_MACH', 'ENV_ALT', 'ENV_SUM']:
        con_clear(k)
    window['ENV_TIMESTAMP'].update('---')
    _last_sweep.update({'ar': [], 'mr': [], 'lr': [], 'label': ''})
    _env_figs.clear()
    _env_idx = 0
    window['ENV_PLT_LBL'].update('')
    try:
        cv = window['CANVAS_ENV'].TKCanvas
        for ch in cv.winfo_children():
            ch.destroy()
    except Exception:
        pass
    set_prog('PB_E', 'PP_E', 'PM_E', 0, '')
    set_status('Flight envelope cleared.')

# =========================================================
# PARALLEL SWEEP WORKER
# =========================================================
_SWEEP_ROW_KEYS = ['CL', 'CD', 'XCP', 'XCP_D']

def _normalize_rows(rows, var_key):
    keys = [var_key] + _SWEEP_ROW_KEYS
    return [_as_dict(r, keys) for r in (rows or [])]

def _run_sweeps_parallel(base, ac, mc, lc):
    f_alpha = _SWEEP_POOL.submit(alpha_sweep,    base, *ac)
    f_mach  = _SWEEP_POOL.submit(mach_sweep,     base, *mc)
    f_alt   = _SWEEP_POOL.submit(altitude_sweep, base, *lc)
    ar = _normalize_rows(f_alpha.result(), 'alpha')
    mr = _normalize_rows(f_mach.result(),  'mach')
    lr = _normalize_rows(f_alt.result(),   'alt')
    return ar, mr, lr

# =========================================================
# MAIN EVENT LOOP
# =========================================================
while True:
    event, values = window.read(timeout=300)

    if event in (sg.WINDOW_CLOSED, 'Exit', None):
        break

    if event == sg.TIMEOUT_EVENT:
        if not _model_rdy:
            _startup_pct = min(_startup_pct + 2, 92)
            set_prog('PB_P', 'PP_P', 'PM_P', _startup_pct,
                     'Loading CSV + training XGBoost model...')
        try:
            while True:
                line = opt_log_q.get_nowait()
                con_append('OPT_LOG', line)
        except queue.Empty:
            pass
        continue

    if event == 'WIN_CFG':
        continue

    if event == 'MODEL_READY':
        elapsed = values['MODEL_READY']
        _model_rdy = True
        set_prog('PB_P', 'PP_P', 'PM_P', 100, 'Model ready')
        set_status(
            f'OK  READY | Loaded in {elapsed:.2f} s | '
            f'{"ENSEMBLE" if ENSEMBLE_MODE else "XGBoost-only"} | '
            f'Press F5 or ESTIMATE',
            color=C_GREEN)
        continue

    if event in ('F5:116', 'F5:65474', 'F5'):
        event = 'Estimate'

    if event == 'W_MAX':
        if _is_max:
            try:
                window.normal()
            except Exception:
                pass
            _is_max = False
        else:
            try:
                window.maximize()
            except Exception:
                pass
            _is_max = True
        continue

    if event == 'W_MIN':
        try:
            window.minimize()
        except Exception:
            pass
        continue

    if event == 'USE_OPT_GEO':
        checked = values.get('USE_OPT_GEO', False)
        if checked:
            odv = (values.get('OUT_DIR', 'de_output').strip()
                   or 'de_output')
            gp = os.path.join(odv, 'best_geometry.csv')
            window['OPT_GEO_STATUS'].update(
                f'✓  Will load: {gp}' if os.path.exists(gp)
                else f'!  Not found: {gp}  (run Optimizer first)')
        else:
            window['OPT_GEO_STATUS'].update('')
        continue

    if event == 'OPT_PREV':
        _show_opt_plot(_opt_idx - 1)
        continue
    if event == 'OPT_NEXT':
        _show_opt_plot(_opt_idx + 1)
        continue

    if event == 'ENV_PREV':
        _show_env_plot(_env_idx - 1)
        continue
    if event == 'ENV_NEXT':
        _show_env_plot(_env_idx + 1)
        continue

    try:
        msg = pred_q.get_nowait()
    except queue.Empty:
        msg = None
    if msg is not None:
        if msg['ok']:
            try:
                render_prediction(msg)
            except Exception as e:
                set_prog('PB_P', 'PP_P', 'PM_P', 0, '')
                set_status(f'ERROR  Displaying prediction: {e}',
                          color=C_RED)
                sg.popup_error(
                    f'Error displaying prediction results:\n{e}')
        else:
            set_prog('PB_P', 'PP_P', 'PM_P', 0, '')
            set_status(
                f'ERROR  Prediction: {msg["error"]}', color=C_RED)
            sg.popup_error(f'Prediction Error:\n{msg["error"]}')

    if event == 'OPT_PROG':
        pct, msg = values['OPT_PROG']
        set_prog('PB_O', 'PP_O', 'PM_O', pct, msg)
        continue
    if event == 'OPT_DONE':
        result, hist, elapsed = values['OPT_DONE']
        try:
            render_optimization(result, hist, elapsed)
        except Exception as e:
            set_status(f'ERROR  Displaying optimizer results: {e}',
                      color=C_RED)
            sg.popup_error(
                f'Error displaying optimization results:\n{e}')
            set_prog('PB_O', 'PP_O', 'PM_O', 0, 'error')
        continue
    if event == 'OPT_ERR':
        e = values['OPT_ERR']
        set_status(f'ERROR  Optimizer: {e}', color=C_RED)
        sg.popup_error(f'Optimization Error:\n{e}')
        set_prog('PB_O', 'PP_O', 'PM_O', 0, 'error')
        _opt_run = False
        continue

    if event == 'ENV_DONE':
        pl = values['ENV_DONE']
        ar, mr, lr, elapsed = pl[:4]
        glbl = pl[4] if len(pl) > 4 else 'user params'
        bp   = pl[5] if len(pl) > 5 else None
        odv  = pl[6] if len(pl) > 6 else 'de_output'
        _last_sweep.update({'ar': ar, 'mr': mr,
                            'lr': lr, 'label': glbl})
        try:
            render_flight(ar, mr, lr, elapsed, glbl,
                          base_params=bp, out_dir=odv)
        except Exception as e:
            set_status(f'ERROR  Displaying envelope results: {e}',
                      color=C_RED)
            sg.popup_error(
                f'Error displaying flight envelope results:\n{e}')
            set_prog('PB_E', 'PP_E', 'PM_E', 0, 'error')
        continue
    if event == 'ENV_ERR':
        e = values['ENV_ERR']
        set_status(f'ERROR  Envelope: {e}', color=C_RED)
        sg.popup_error(f'Flight Envelope Error:\n{e}')
        set_prog('PB_E', 'PP_E', 'PM_E', 0, 'error')
        _flt_run = False
        continue

    # ── Tab 1: Prediction ─────────────────────────────────
    if event == 'Estimate':
        if not _model_rdy:
            sg.popup_quick_message(
                'Model still loading -- please wait...')
            continue
        params = {p: sf(values, p) for p in PARAMS}
        _t_start_pred = _ts()
        window['TIME_STAMP_P'].update(
            f'Start {_t_start_pred}  |  End --:--:--')
        set_prog('PB_P', 'PP_P', 'PM_P', 15, 'Running prediction...')
        set_status('Running aerodynamic prediction...', color=C_AMBER)
        threading.Thread(target=_pred_worker,
                         args=(params,), daemon=True).start()

    elif event == 'Reset_Pred':
        reset_pred()

    # ── Tab 2: Optimization ───────────────────────────────
    elif event == 'Run_Opt':
        if not _model_rdy:
            sg.popup_quick_message(
                'Model still loading -- please wait...')
            continue
        if _opt_run:
            sg.popup_quick_message(
                'Optimization is already running!')
            continue
        _opt_run = True
        _t_start_opt = _ts()
        window['OPT_TIMESTAMP'].update(
            f'Start {_t_start_opt}  |  End --:--:--')
        bounds = [(sf(values, f'{p}_LOW'),
                   sf(values, f'{p}_HIGH'))
                  for p in _BOUNDS_ORDER]
        constraints = {
            'CL' : (sf(values, 'CL_MIN'),  sf(values, 'CL_MAX')),
            'CD' : (sf(values, 'CD_MIN'),  sf(values, 'CD_MAX')),
            'XCP': (sf(values, 'XCP_MIN'), sf(values, 'XCP_MAX')),
        }
        maxiter = int(sf(values, 'MAXITER', 50))
        popsize = int(sf(values, 'POPSIZE', 10))
        itermax = int(sf(values, 'ITERMAX', 5))
        out_dir = values.get('OUT_DIR', 'de_output').strip()
        con_clear('OPT_GEO')
        con_clear('TOP5_OPT')
        con_clear('OPT_LOG')
        set_prog('PB_O', 'PP_O', 'PM_O', 2,
                 'Initialising custom DE...')
        mh = ('ENSEMBLE (XGB+RF+GB)'
              if ENSEMBLE_MODE else 'XGBOOST ONLY')
        con_append('OPT_LOG',
                   '=' * 58 + '\n'
                   f'  CUSTOM DE v9.7  |  {mh}\n'
                   f'  Generations:{maxiter}  Pop:{popsize}'
                   f'  GeneSwap:{itermax}\n'
                   f'  Output: {out_dir or "(none)"}\n'
                   + '-' * 58)
        set_status(f'Running Custom DE [{mh}]...', color=C_AMBER)
        threading.Thread(
            target=_opt_worker,
            args=(bounds, maxiter, popsize,
                  itermax, constraints, out_dir),
            daemon=True).start()

    elif event == 'Abort_Opt':
        _opt_run = False
        set_status('Optimization aborted.', color=C_RED)
        set_prog('PB_O', 'PP_O', 'PM_O', 0, 'aborted')

    elif event == 'Clear_Opt':
        clear_opt()

    # ── Tab 3: Flight Envelope ────────────────────────────
    elif event == 'Run_Env':
        if not _model_rdy:
            sg.popup_quick_message(
                'Model still loading -- please wait...')
            continue
        if _flt_run:
            sg.popup_quick_message('Sweep already running!')
            continue
        _flt_run = True
        _t_start_env = _ts()
        window['ENV_TIMESTAMP'].update(
            f'Start {_t_start_env}  |  End --:--:--')
        use_opt = values.get('USE_OPT_GEO', False)
        odv = (values.get('OUT_DIR', 'de_output').strip()
               or 'de_output')
        if use_opt:
            try:
                base, lp = _load_optimal_base(odv)
                glbl = f'optimal ({os.path.basename(lp)})'
            except FileNotFoundError as ex:
                sg.popup_error(str(ex))
                _flt_run = False
                continue
        else:
            base = {p: sf(values, f'E_{p}', DEFAULTS[p])
                    for p in PARAMS}
            glbl = 'user-typed base params'

        ac = (sf(values, 'ALPHA_MIN'),
              sf(values, 'ALPHA_MAX'),
              sf(values, 'ALPHA_STP'))
        mc = (sf(values, 'MACH_MIN'),
              sf(values, 'MACH_MAX'),
              sf(values, 'MACH_STP'))
        lc = (sf(values, 'ALT_MIN'),
              sf(values, 'ALT_MAX'),
              sf(values, 'ALT_STP'))

        for k in ['ENV_ALPHA', 'ENV_MACH', 'ENV_ALT', 'ENV_SUM']:
            con_clear(k)
        set_prog('PB_E', 'PP_E', 'PM_E', 5,
                 f'Starting sweeps [{glbl}]...')
        set_status(
            f'Running flight envelope sweeps [{glbl}]...',
            color=C_AMBER)

        def _env_worker(base, ac, mc, lc, label, odir):
            global _flt_run
            try:
                t0 = time.perf_counter()
                ar, mr, lr = _run_sweeps_parallel(base, ac, mc, lc)
                elapsed = time.perf_counter() - t0
                window.write_event_value(
                    'ENV_DONE',
                    (ar, mr, lr, elapsed, label, base, odir))
            except Exception as e:
                window.write_event_value('ENV_ERR', str(e))
            finally:
                _flt_run = False

        threading.Thread(
            target=_env_worker,
            args=(base, ac, mc, lc, glbl, odv),
            daemon=True).start()

    elif event == 'Clear_Env':
        clear_env()

    elif event == 'Export_Env':
        export_envelope_csv()

# Clean shutdown
try:
    _SWEEP_POOL.shutdown(wait=False)
except Exception:
    pass

window.close()
