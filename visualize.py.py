import os
import sys
import math
import subprocess
import re
import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

# Constants and defaults
WINDOW_W, WINDOW_H = 1280, 720
GAP_PIXELS = 20
CYLINDER_RADIUS = 2.5
IMAGE_WIDTH = IMAGE_HEIGHT = 1.2
HORIZONTAL_GAP = VERTICAL_GAP = 0
BORDER_WIDTH = 6
ZOOM_DISTANCE = 1.7
FPS = 60
BLUE_BORDER = (0.0, 0.4, 1.0)
RED_BORDER = (1.0, 0.0, 0.0)
DRAG_THRESHOLD = 5

# Supported image extensions
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Helper functions
def _core_id(name: str) -> str:
    """
    Return the complete stem of a file (lowerâ€‘cased) and use it as the
    identifier for matching final â†” initial images.

    Example:
        "1"   -> "1"
        "1aa" -> "1aa"
        "1ab" -> "1ab"
    """
    return name.lower()

def detect_green_dots(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return np.any((arr[:, :, 1] > 200) & (arr[:, :, 0] < 80) & (arr[:, :, 2] < 80))

# Main script
FINAL_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
INITIAL_FOLDER = sys.argv[2] if len(sys.argv) > 2 else ""

final_files = sorted([os.path.join(FINAL_FOLDER, f)
                      for f in os.listdir(FINAL_FOLDER) if f.lower().endswith(exts)])

print(f"Total images with at least one green dot: {sum(detect_green_dots(f) for f in final_files)}")

# ----------------------------------------------------------------------
# Build a map: core identifier â†’ path of the *initial* image (if it exists)
# ----------------------------------------------------------------------
initial_map = {}
if INITIAL_FOLDER and os.path.isdir(INITIAL_FOLDER):
    init_files = [os.path.join(INITIAL_FOLDER, f)
        for f in os.listdir(INITIAL_FOLDER) if f.lower().endswith(exts)]
    for p in init_files:
        stem = os.path.splitext(os.path.basename(p))[0]
        key = _core_id(stem)
        initial_map[key] = p

if not final_files:
    sys.exit("NO FINAL IMAGES FOUND IN: " + FINAL_FOLDER)

def detect_green_dot(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return bool(np.any((arr[:,:,1]>200)&(arr[:,:,0]<80)&(arr[:,:,2]<80)))

ideal_cols = (2 * math.pi * CYLINDER_RADIUS) / IMAGE_WIDTH
IMAGES_PER_ROW = max(1, int(round(ideal_cols)))
angle_step_rad = (2 * math.pi) / IMAGES_PER_ROW
ROWS = math.ceil(len(final_files) / IMAGES_PER_ROW)
vertical_step = IMAGE_HEIGHT + VERTICAL_GAP
total_width    = ROWS * vertical_step

def load_texture(path):
    img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    data = img.tobytes()
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, data)
    return tex

panels = []

def build_panels(textures, green_flags, init_textures):
    """
    Populate the global ``panels`` list.
    Each panel now also stores the texture of its matching initial image
    (or ``None`` if no initial image exists).
    """
    panels.clear()
    idx = 0
    for row in range(ROWS):
        x = -total_width/2 + row*vertical_step + IMAGE_WIDTH/2
        for col in range(IMAGES_PER_ROW):
            if idx >= len(final_files):
                break
            theta = col*angle_step_rad
            y = CYLINDER_RADIUS*math.cos(theta)
            z = CYLINDER_RADIUS*math.sin(theta)
            ln = math.sqrt(y*y+z*z)

            final_path = final_files[idx]
            final_stem = os.path.splitext(os.path.basename(final_path))[0]   # full name (e.g. "2a")
            core = _core_id(final_stem)

            panels.append({
                "center": (x, y, z),
                "radial": (0.0, y/ln, z/ln),
                "tex": textures[idx],
                "init_tex": init_textures.get(initial_map.get(core)),
                "final_stem": final_stem,
                "row": row,
                "col": col,
                "has_green_dot": green_flags[idx],
                "final_path": final_path,
                "initial_path": initial_map.get(core),
                "filename": core,
                "flip_h": False,
                "flip_v": False,
                "rot": 0
            })
            idx += 1

def draw_panel(panel, selected=False):
    cx, cy, cz = panel["center"]
    rx, ry, rz = panel["radial"]
    hw = IMAGE_WIDTH/2
    hh = IMAGE_HEIGHT/2
    ty = -rz
    tz = ry
    corners = [(cx-hw, cy-ty*hh, cz-tz*hh),
               (cx+hw, cy-ty*hh, cz-tz*hh),
               (cx+hw, cy+ty*hh, cz+tz*hh),
               (cx-hw, cy+ty*hh, cz+tz*hh)]
    tex_coords = [(0,0),(1,0),(1,1),(0,1)]

    if selected:
        if panel["flip_h"]: tex_coords = [(1-u,v) for u,v in tex_coords]
        if panel["flip_v"]: tex_coords = [(u,1-v) for u,v in tex_coords]
        rot = panel["rot"] % 360
        if rot == 90:   tex_coords = [(v,1-u) for u,v in tex_coords]
        elif rot == 180: tex_coords = [(1-u,1-v) for u,v in tex_coords]
        elif rot == 270: tex_coords = [(1-v,u) for u,v in tex_coords]

    glBindTexture(GL_TEXTURE_2D, panel["tex"])
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    for (x,y,z), (u,v) in zip(corners, tex_coords):
        glTexCoord2f(u, v)
        glVertex3f(x, y, z)
    glEnd()

    # draw coloured border when the panel faces the camera
    mv = glGetDoublev(GL_MODELVIEW_MATRIX)
    eye_x = -(mv[0][0]*mv[3][0] + mv[0][1]*mv[3][1] + mv[0][2]*mv[3][2])
    eye_y = -(mv[1][0]*mv[3][0] + mv[1][1]*mv[3][1] + mv[1][2]*mv[3][2])
    eye_z = -(mv[2][0]*mv[3][0] + mv[2][1]*mv[3][1] + mv[2][2]*mv[3][2])
    view_vec = (eye_x-cx, eye_y-cy, eye_z-cz)
    dot = view_vec[0]*rx + view_vec[1]*ry + view_vec[2]*rz
    if dot > 0 and (panel["has_green_dot"] or selected):
        glDisable(GL_DEPTH_TEST)
        glLineWidth(BORDER_WIDTH)
        glColor3f(*(RED_BORDER if panel["has_green_dot"] and not selected else BLUE_BORDER))
        ox2, oy2, oz2 = rx*0.02, ry*0.02, rz*0.02
        glBegin(GL_LINE_LOOP)
        for cc in corners:
            glVertex3f(cc[0]+ox2, cc[1]+oy2, cc[2]+oz2)
        glEnd()
        glEnable(GL_DEPTH_TEST)

# ----------------------------------------------------------------------
# Global state (used by the main loop)
# ----------------------------------------------------------------------
selected_idx = 0          # currently selected panel
zoomed = False           # zoomâ€‘in flag
view_mode = "3d"          # "3d" = cylinder view, "pair" = sideâ€‘byâ€‘side view

def move_next_col():
    global selected_idx
    p = panels[selected_idx]
    rl = [i for i,pp in enumerate(panels) if pp["row"]==p["row"]]
    selected_idx = rl[(rl.index(selected_idx)+1) % len(rl)]

def move_prev_col():
    global selected_idx
    p = panels[selected_idx]
    rl = [i for i,pp in enumerate(panels) if pp["row"]==p["row"]]
    selected_idx = rl[(rl.index(selected_idx)-1) % len(rl)]

def move_next_row():
    global selected_idx
    p = panels[selected_idx]
    m = [i for i,q in enumerate(panels) if q["row"]==p["row"]+1 and q["col"]==p["col"]]
    if m: selected_idx = m[0]

def move_prev_row():
    global selected_idx
    p = panels[selected_idx]
    m = [i for i,q in enumerate(panels) if q["row"]==p["row"]-1 and q["col"]==p["col"]]
    if m: selected_idx = m[0]

cam_yaw = 0
cam_pitch = 0
cam_dist = 12
drag_active = False
drag_last = (0,0)

def apply_camera():
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    if zoomed:
        p = panels[selected_idx]
        cx, cy, cz = p["center"]
        rx, ry, rz = p["radial"]
        ex = cx + rx*ZOOM_DISTANCE
        ey = cy + ry*ZOOM_DISTANCE
        ez = cz + rz*ZOOM_DISTANCE
        gluLookAt(ex, ey, ez, cx, cy, cz, 0,0,1)
    else:
        yr = math.radians(cam_yaw)
        pr = math.radians(cam_pitch)
        ex = cam_dist*math.cos(pr)*math.sin(yr)
        ey = cam_dist*math.sin(pr)
        ez = cam_dist*math.cos(pr)*math.cos(yr)
        gluLookAt(ex, ey, ez, 0,0,0, 0,1,0)

def get_ray(mx,my):
    vp = glGetIntegerv(GL_VIEWPORT)
    wy = vp[3] - my
    mv = glGetDoublev(GL_MODELVIEW_MATRIX)
    pj = glGetDoublev(GL_PROJECTION_MATRIX)
    near = gluUnProject(mx, wy, 0.0, mv, pj, vp)
    far  = gluUnProject(mx, wy, 1.0, mv, pj, vp)
    d = (far[0]-near[0], far[1]-near[1], far[2]-near[2])
    l = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    return near, (d[0]/l, d[1]/l, d[2]/l)

def hit_panel(ro, rd, panel):
    cx, cy, cz = panel["center"]
    nx, ny, nz = panel["radial"]
    denom = nx*rd[0] + ny*rd[1] + nz*rd[2]
    if abs(denom) < 1e-6: return False
    t = ((cx-ro[0])*nx + (cy-ro[1])*ny + (cz-ro[2])*nz) / denom
    if t < 0: return False
    ix = ro[0] + t*rd[0]
    iy = ro[1] + t*rd[1]
    iz = ro[2] + t*rd[2]
    dx, dy, dz = ix-cx, iy-cy, iz-cz
    # local axes (u = -nz, v = 1,0,0) â€“ same as original script
    u = (0, -nz, ny)
    v = (1, 0, 0)
    lx = dx*u[0] + dy*u[1] + dz*u[2]
    ly = dx*v[0] + dy*v[1] + dz*v[2]
    return (-IMAGE_WIDTH/2 <= lx <= IMAGE_WIDTH/2) and (-IMAGE_HEIGHT/2 <= ly <= IMAGE_HEIGHT/2)

def main():
    global drag_active, drag_last, cam_yaw, cam_pitch, cam_dist
    global zoomed, selected_idx, view_mode   # <-- needed because we modify them

    pygame.init()
    pygame.font.init()

    font = pygame.font.SysFont("Arial", 30, bold=True)

    pygame.display.set_mode(
        (WINDOW_W, WINDOW_H),
        DOUBLEBUF | OPENGL | RESIZABLE
    )

    pygame.display.set_caption(
        "3D Viewer â€“ " + os.path.basename(FINAL_FOLDER)
    )

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)

    glClearColor(0.08,0.08,0.1,1)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(
        45.0,
        WINDOW_W / WINDOW_H,
        0.1,
        200.0
    )

    textures = [load_texture(f) for f in final_files]

    green_flags = [
        detect_green_dot(f)
        for f in final_files
    ]

    init_textures = {}

    for key, path in initial_map.items():

        try:
            init_textures[path] = load_texture(path)

        except Exception:
            init_textures[path] = None

    # Build panels â€“ each panel now also holds the texture of its initial image
    build_panels(textures, green_flags, init_textures)

    clock = pygame.time.Clock()
    potential_click = False
    sw, sh = WINDOW_W, WINDOW_H

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:
                sw, sh = event.w, event.h
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()
                gluPerspective(45.0, sw/sh, 0.1, 200.0)
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT:   move_next_row()
                elif event.key == K_LEFT:  move_prev_row()
                elif event.key == K_UP:    move_prev_col()
                elif event.key == K_DOWN:  move_next_col()
                elif event.key == K_RETURN:
                    # ---- SHOW FINAL + INITIAL sideâ€‘byâ€‘side in the viewer ----
                    view_mode = "pair"
                elif event.key in (K_r, K_ESCAPE):
                    # Return to normal 3â€‘D view
                    view_mode = "3d"
                    zoomed = False
                if view_mode == "3d":
                    if event.key == K_q:
                        panels[selected_idx]["rot"] = (panels[selected_idx]["rot"] - 90) % 360
                    elif event.key == K_e:
                        panels[selected_idx]["rot"] = (panels[selected_idx]["rot"] + 90) % 360
                    elif event.key == K_h:
                        panels[selected_idx]["flip_h"] = not panels[selected_idx]["flip_h"]
                    elif event.key == K_v:
                        panels[selected_idx]["flip_v"] = not panels[selected_idx]["flip_v"]
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1 and not zoomed:
                    drag_active = True
                    drag_last = event.pos
                    potential_click = True
                elif event.button == 4 and not zoomed:
                    cam_dist = max(3.0, cam_dist - 0.5)
                elif event.button == 5 and not zoomed:
                    cam_dist = min(30.0, cam_dist + 0.5)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    if potential_click and not zoomed:
                        ro, rd = get_ray(*event.pos)
                        hit, min_d = None, float("inf")
                        for i, p in enumerate(panels):
                            if hit_panel(ro, rd, p):
                                cx, cy, cz = p["center"]
                                d = math.sqrt((cx-ro[0])**2 + (cy-ro[1])**2 + (cz-ro[2])**2)
                                if d < min_d:
                                    min_d = d
                                    hit = i
                        if hit is not None:
                            selected_idx = hit
                            zoomed = False
                    drag_active = False
                    potential_click = False
            elif event.type == MOUSEMOTION:
                if drag_active and not zoomed:
                    dx = event.pos[0] - drag_last[0]
                    dy = event.pos[1] - drag_last[1]
                    if potential_click and (abs(dx) > DRAG_THRESHOLD or abs(dy) > DRAG_THRESHOLD):
                        potential_click = False
                    if not potential_click:
                        cam_yaw   -= dx * 0.4
                        cam_pitch = max(-80, min(80, cam_pitch + dy * 0.3))
                    drag_last = event.pos

        # ------------------------------------------------------------------
        # RENDERING
        # ------------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if view_mode == "3d":
            # ----- restore perspective projection (needed after leaving pair view) -----
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, sw / sh, 0.1, 200.0)
            glMatrixMode(GL_MODELVIEW)

            # ----- normal 3â€‘D cylinder view -----
            apply_camera()
            for i, panel in enumerate(panels):

                draw_panel(
                    panel,
                    selected=(i == selected_idx)
                )

        else:

            # -------------------------------------------------
            #  Pair view â€“ now **initial** on the left, **final** on the right
            # -------------------------------------------------
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, sw, 0, sh, -1, 1)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            p = panels[selected_idx]
            panel_w = (sw - GAP_PIXELS) // 2

            # ---------- LEFT IMAGE : INITIAL ----------
            if p.get("init_tex"):
                glBindTexture(GL_TEXTURE_2D, p["init_tex"])
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(0,          0)
                glTexCoord2f(1, 0); glVertex2f(panel_w,    0)
                glTexCoord2f(1, 1); glVertex2f(panel_w,    sh)
                glTexCoord2f(0, 1); glVertex2f(0,          sh)
                glEnd()
            else:
                glDisable(GL_TEXTURE_2D)
                glColor3f(0.2, 0.2, 0.2)          # placeholder colour
                glBegin(GL_QUADS)
                glVertex2f(0,          0)
                glVertex2f(panel_w,    0)
                glVertex2f(panel_w,    sh)
                glVertex2f(0,          sh)
                glEnd()

            # ---------- RIGHT IMAGE : FINAL ----------
            right_x0 = panel_w + GAP_PIXELS
            right_x1 = right_x0 + panel_w

            glBindTexture(GL_TEXTURE_2D, p["tex"])
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0); glVertex2f(right_x0, 0)
            glTexCoord2f(1, 0); glVertex2f(right_x1, 0)
            glTexCoord2f(1, 1); glVertex2f(right_x1, sh)
            glTexCoord2f(0, 1); glVertex2f(right_x0, sh)
            glEnd()

            # ---------- LABELS ----------
            MARGIN = 30                     # distance from the image border
            BLACK  = (255,0,0)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) 

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glWindowPos2d(panel_w - MARGIN, sh - MARGIN)
            init_name = os.path.splitext(os.path.basename(p["initial_path"]))[0] if p.get("initial_path") else ""
            init_text = f"{init_name}"
            init_surface = font.render(init_text, True, BLACK, None).convert_alpha()
            init_data = pygame.image.tostring(init_surface, "RGBA", True)
            glDisable(GL_DEPTH_TEST)
            glDrawPixels(init_surface.get_width(), init_surface.get_height(),
            GL_RGBA, GL_UNSIGNED_BYTE, init_data)
            glEnable(GL_DEPTH_TEST)

            # RIGHT label â€“ FINAL (topâ€‘right of the right image)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glWindowPos2d(right_x1 - MARGIN, sh - MARGIN)
            final_text = f"{p['final_stem']}"
            final_surface = font.render(final_text, True, BLACK, None).convert_alpha()
            final_data = pygame.image.tostring(final_surface, "RGBA", True)
            glDrawPixels(final_surface.get_width(), final_surface.get_height(),
            GL_RGBA, GL_UNSIGNED_BYTE, final_data)
            glEnable(GL_DEPTH_TEST)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
