"""
GPU decoding for FUL v4 to v8 files (OpenGL 3.3 core, via PyOpenGL).

The work is split by what each processor is good at:

  CPU  entropy decoding and dequantisation. Each symbol depends on the one
       before, which suits a CPU core (tiles still run on all cores).
  GPU  everything that touches every pixel independently:
         * inverse wavelet: one full-screen shader pass per lifting step,
           down the columns then along the rows, level by level. A single
           general shader applies any integer lifting step, so the same
           pipeline runs v4's tiled 5/3 and v5's whole-image 5/3 or 9/7
         * YCoCg -> RGB colour conversion
         * grain regeneration from a per-pixel position hash

Only the compact coefficients cross from CPU to GPU. In a real viewer the
final pass would draw straight to the screen; here it is read back so the
lab can check it against the CPU decoder pixel for pixel.

Context: on a desktop, Qt creates an offscreen OpenGL context on the real GPU.
On a headless Linux machine set FUL_GPU_EGL=1 to use EGL instead (used for
testing with Mesa's software renderer).
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

USE_EGL = os.environ.get("FUL_GPU_EGL") == "1"
if USE_EGL:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from OpenGL import GL  # noqa: E402

import codec_v4  # noqa: E402
import codec_v5  # noqa: E402
import codec_v6  # noqa: E402
import codec_v7  # noqa: E402
import codec_v8  # noqa: E402

VERTEX = """
#version 330 core
void main() {
    // one triangle that covers the whole viewport
    vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

COMMON = """
#version 330 core
// floor(a / 2^s) for any sign, matching the CPU decoders
int fds(int a, int s) { return a >= 0 ? (a >> s) : -((-a + (1 << s) - 1) >> s); }
int fd2(int a) { return fds(a, 1); }
"""

LIFT = COMMON + """
uniform isampler2D uSrc;
uniform ivec2 uImg;     // image size
uniform int uTile;      // tile size (larger than the image for whole-image transforms)
uniform int uLevel;     // undoing this level: region = tile size halved uLevel times
uniform int uAxis;      // 0 = along rows (x), 1 = down columns (y)
uniform int uMode;      // 0 = first step, reads lows|highs and interleaves (even update)
                        // 1 = odd update, 2 = even update (interleaved input)
                        // 3 = copy, shifting left by uShift (expands precision)
uniform ivec4 uStep;    // sign, multiplier, rounding, shift of this lifting step
uniform int uShift;
uniform isampler2D uDir;  // v6: per 16x16 block, column shift in half pixels per row
uniform int uHasDir;
out ivec4 oVal;

ivec2 gBase; ivec2 gLp; int gW;
ivec4 at(int k) {
    return texelFetch(uSrc, gBase + (uAxis == 0 ? ivec2(k, gLp.y) : ivec2(gLp.x, k)), 0);
}
// row k of the column pass, sampled dh half-pixels sideways (clamped)
ivec4 atd(int k, int dh) {
    if ((dh & 1) == 0) {
        int c = clamp(gLp.x + fds(dh, 1), 0, gW - 1);
        return texelFetch(uSrc, gBase + ivec2(c, k), 0);
    }
    int c0 = clamp(gLp.x + fds(dh - 1, 1), 0, gW - 1);
    int c1 = clamp(gLp.x + fds(dh + 1, 1), 0, gW - 1);
    ivec4 t = texelFetch(uSrc, gBase + ivec2(c0, k), 0) + texelFetch(uSrc, gBase + ivec2(c1, k), 0);
    return ivec4(fds(t.x, 1), fds(t.y, 1), fds(t.z, 1), fds(t.w, 1));
}
ivec4 nb(int k, int dh) { return uAxis == 0 ? at(k) : atd(k, dh); }
ivec4 lift(ivec4 l, ivec4 r) {
    ivec4 t = uStep.y * (l + r) + uStep.z;
    return uStep.x * ivec4(fds(t.x, uStep.w), fds(t.y, uStep.w), fds(t.z, uStep.w), fds(t.w, uStep.w));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec4 v = texelFetch(uSrc, p, 0);
    if (uMode == 3) { oVal = ivec4(v.xyz << uShift, 0); return; }
    ivec2 tile = p / uTile;
    gBase = tile * uTile;
    gLp = p - gBase;
    ivec2 d = min(ivec2(uTile), uImg - gBase);
    for (int i = 0; i < uLevel; i++) d = (d + 1) / 2;
    if (gLp.x >= d.x || gLp.y >= d.y) { oVal = v; return; }
    gW = d.x;
    int n = uAxis == 0 ? d.x : d.y;
    int i = uAxis == 0 ? gLp.x : gLp.y;
    if (n <= 1) { oVal = v; return; }
    int nh = n / 2;
    int nl = n - nh;
    int dh = (uAxis == 1 && uHasDir == 1) ? texelFetch(uDir, ivec2(gLp.x >> 4, i >> 4), 0).r : 0;
    if (uMode == 0) {
        int m = i / 2;
        if ((i & 1) == 0) {
            ivec4 l = m > 0 ? nb(nl + m - 1, -dh) : nb(nl, -dh);
            ivec4 r = m < nh ? nb(nl + m, dh) : nb(nl + nh - 1, dh);
            oVal = at(m) - lift(l, r);
        } else {
            oVal = at(nl + m);
        }
    } else if (uMode == 1) {
        if ((i & 1) == 1) {
            ivec4 r = (i + 1 < n) ? nb(i + 1, dh) : nb(i - 1, dh);
            oVal = v - lift(nb(i - 1, -dh), r);
        } else {
            oVal = v;
        }
    } else {
        if ((i & 1) == 0) {
            ivec4 l = i > 0 ? nb(i - 1, -dh) : nb(1, -dh);
            ivec4 r = (i + 1 < n) ? nb(i + 1, dh) : nb(i - 1, dh);
            oVal = v - lift(l, r);
        } else {
            oVal = v;
        }
    }
}
"""

COLOUR = COMMON + """
uniform isampler2D uSrc;
uniform sampler2D uGrain;   // per grain cell: strength x8 in 4 brightness ranges
uniform int uCell;
uniform int uHasGrain;
uniform float uGrainAmount;
uniform int uShift;         // undo pre-scale + extra precision
out vec4 oColour;

uint hash32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16; return x;
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec4 v = texelFetch(uSrc, p, 0);
    int half_ = (1 << uShift) >> 1;
    int Y = fds(v.x + half_, uShift), co = fds(v.y + half_, uShift), cg = fds(v.z + half_, uShift);
    if (uHasGrain == 1 && uGrainAmount > 0.0) {
        vec4 gs = texelFetch(uGrain, p / uCell, 0);
        float f = float(Y - 32) / 64.0;
        float sig;
        if (f <= 0.0) sig = gs.x;
        else if (f >= 3.0) sig = gs.w;
        else {
            int i0 = int(f);
            float a = i0 == 0 ? gs.x : (i0 == 1 ? gs.y : gs.z);
            float b = i0 == 0 ? gs.y : (i0 == 1 ? gs.z : gs.w);
            sig = a + (b - a) * (f - float(i0));
        }
        uint hv = hash32((uint(p.y) << 16) ^ uint(p.x) ^ 0x5F3759DFu);
        float g = (float((hv & 255u) + ((hv >> 8) & 255u) + ((hv >> 16) & 255u)
                         + (hv >> 24)) - 510.0) / 147.8;
        Y += int(floor(g * sig * uGrainAmount / 8.0 + 0.5));
    }
    int t = Y - fd2(cg);
    int g2 = cg + t;
    int b = t - fd2(co);
    int r = b + co;
    oColour = vec4(clamp(vec3(r, g2, b), 0.0, 255.0) / 255.0, 1.0);
}
"""


FILTER = COMMON + """
// v8 post-filter: 5x5 edge-aware (bilateral) filter, integer, same tables as the CPU
uniform sampler2D uImg;
uniform isampler2D uMap;     // strength per 32x32 block, 0 = off
uniform isampler2D uRange;   // brightness-similarity weights: 256 x 5 strengths
uniform ivec2 uSize;
out vec4 oColour;
const int SP[25] = int[25](""" + "1, 5, 9, 5, 1, 5, 24, 39, 24, 5, 9, 39, 64, 39, 9, 5, 24, 39, 24, 5, 1, 5, 9, 5, 1" + """);
ivec3 px(ivec2 q) { return ivec3(round(texelFetch(uImg, q, 0).rgb * 255.0)); }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int k = texelFetch(uMap, p >> 5, 0).r;
    vec4 c0 = texelFetch(uImg, p, 0);
    if (k == 0) { oColour = c0; return; }
    ivec3 cc = px(p);
    int yc = (77 * cc.r + 150 * cc.g + 29 * cc.b) >> 8;
    ivec3 n = ivec3(0);
    int den = 0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            ivec3 v = px(clamp(p + ivec2(dx, dy), ivec2(0), uSize - 1));
            int yn = (77 * v.r + 150 * v.g + 29 * v.b) >> 8;
            int wgt = SP[(dy + 2) * 5 + dx + 2] * texelFetch(uRange, ivec2(abs(yn - yc), k), 0).r;
            n += wgt * v;
            den += wgt;
        }
    }
    ivec3 r = (n + den / 2) / den;
    oColour = vec4(vec3(r) / 255.0, 1.0);
}
"""


# --------------------------------------------------------------------------
# Context creation
# --------------------------------------------------------------------------
class _QtContext:
    def __init__(self):
        from PyQt6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat
        self.app = QGuiApplication.instance() or QGuiApplication([])
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            raise RuntimeError("Qt couldn't create an OpenGL 3.3 context")
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        self.make_current()

    def make_current(self):
        if not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("Couldn't activate the OpenGL context")


class _EglContext:
    def __init__(self):
        from OpenGL import EGL
        self.EGL = EGL
        self.dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        major, minor = EGL.EGLint(), EGL.EGLint()
        if not EGL.eglInitialize(self.dpy, ctypes.pointer(major), ctypes.pointer(minor)):
            raise RuntimeError("EGL initialisation failed")
        attrs = (EGL.EGLint * 5)(EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
                                 EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT, EGL.EGL_NONE)
        cfg, n = EGL.EGLConfig(), EGL.EGLint()
        EGL.eglChooseConfig(self.dpy, attrs, ctypes.pointer(cfg), 1, ctypes.pointer(n))
        if n.value < 1:
            raise RuntimeError("No suitable EGL configuration")
        self.surf = EGL.eglCreatePbufferSurface(
            self.dpy, cfg, (EGL.EGLint * 5)(EGL.EGL_WIDTH, 16, EGL.EGL_HEIGHT, 16, EGL.EGL_NONE))
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        cattrs = (EGL.EGLint * 7)(EGL.EGL_CONTEXT_MAJOR_VERSION, 3,
                                  EGL.EGL_CONTEXT_MINOR_VERSION, 3,
                                  EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                  EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL.EGL_NONE)
        self.ctx = EGL.eglCreateContext(self.dpy, cfg, EGL.EGL_NO_CONTEXT, cattrs)
        self.make_current()

    def make_current(self):
        if not self.EGL.eglMakeCurrent(self.dpy, self.surf, self.surf, self.ctx):
            raise RuntimeError("Couldn't activate the EGL context")


def _compile(vertex: str, fragment: str) -> int:
    prog = GL.glCreateProgram()
    for kind, src in ((GL.GL_VERTEX_SHADER, vertex), (GL.GL_FRAGMENT_SHADER, fragment)):
        sh = GL.glCreateShader(kind)
        GL.glShaderSource(sh, src)
        GL.glCompileShader(sh)
        if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
            raise RuntimeError("Shader compile failed:\n" + GL.glGetShaderInfoLog(sh).decode())
        GL.glAttachShader(prog, sh)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError("Shader link failed:\n" + GL.glGetProgramInfoLog(prog).decode())
    return prog


# --------------------------------------------------------------------------
# The decoder
# --------------------------------------------------------------------------
class GpuDecoder:
    def __init__(self):
        self.context = _EglContext() if USE_EGL else _QtContext()
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode()
        self.lift = _compile(VERTEX, LIFT)
        self.colour = _compile(VERTEX, COLOUR)
        self.filter = _compile(VERTEX, FILTER)
        self.vao = GL.glGenVertexArrays(1)
        self._size = None
        self._uniforms = {}

    def _u(self, prog, name):
        key = (prog, name)
        if key not in self._uniforms:
            self._uniforms[key] = GL.glGetUniformLocation(prog, name)
        return self._uniforms[key]

    def _texture(self, internal, w, h, fmt, typ, data=None):
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, typ, data)
        return tex

    def _fbo(self, tex):
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, tex, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer incomplete")
        return fbo

    def _ensure_size(self, w, h):
        if self._size == (w, h):
            return
        if self._size is not None:
            GL.glDeleteTextures(list(self.tex) + [self.out_tex, self.upload_tex, self.out2_tex])
            fbos = list(self.fbo) + [self.out_fbo, self.out2_fbo]
            GL.glDeleteFramebuffers(len(fbos), fbos)
        # compact 16-bit upload; 32-bit working space stays on the GPU
        self.upload_tex = self._texture(GL.GL_RGB16I, w, h, GL.GL_RGB_INTEGER, GL.GL_SHORT)
        self.tex = [self._texture(GL.GL_RGBA32I, w, h, GL.GL_RGBA_INTEGER, GL.GL_INT)
                    for _ in range(2)]
        self.fbo = [self._fbo(t) for t in self.tex]
        self.out_tex = self._texture(GL.GL_RGBA8, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        self.out_fbo = self._fbo(self.out_tex)
        self.out2_tex = self._texture(GL.GL_RGBA8, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        self.out2_fbo = self._fbo(self.out2_tex)
        self._size = (w, h)

    def _prepare(self, data: bytes):
        """CPU stage for either version. Returns coefficients as (H, W, 3)
        int16 plus everything the GPU passes need."""
        version = data[4] if len(data) > 4 else 0
        if version == 4:
            coef, grain_sig, info = codec_v4.decode_coefficients(data)
            return dict(coef=coef, grain=grain_sig, w=info["width"], h=info["height"],
                        tile=info["tile_size"], levels=info["levels"],
                        steps=codec_v5.STEPS_53, extra=0, shift=0,
                        cell=info["tile_size"], rows_first=False, dirs=[])
        if version == 5:
            coef, grain_sig, info, _ = codec_v5.decode_coefficients(data)
            if np.abs(coef).max() > 32767:
                raise ValueError("Coefficients exceed 16 bits; use the CPU decoder")
            packed = np.ascontiguousarray(coef.transpose(1, 2, 0)).astype(np.int16)
            w, h = info["width"], info["height"]
            return dict(coef=packed, grain=grain_sig, w=w, h=h, tile=max(w, h),
                        levels=info["levels"],
                        steps=codec_v5.STEPS_97 if info["wavelet"] == "9/7" else codec_v5.STEPS_53,
                        extra=info["extra"], shift=info["prescale"] + info["extra"],
                        cell=codec_v5.GRAIN_CELL, rows_first=False, dirs=[])
        if version in (6, 7, 8):
            codec = {6: codec_v6, 7: codec_v7, 8: codec_v8}[version]
            coef, grain_sig, info, _, dirmaps = codec.decode_coefficients(data)
            if np.abs(coef).max() > 32767:
                raise ValueError("Coefficients exceed 16 bits; use the CPU decoder")
            packed = np.ascontiguousarray(coef.transpose(1, 2, 0)).astype(np.int16)
            w, h = info["width"], info["height"]
            return dict(coef=packed, grain=grain_sig, w=w, h=h, tile=max(w, h),
                        levels=info["levels"],
                        steps=codec_v6.STEPS_97 if info["wavelet"] == "9/7" else codec_v6.STEPS_53,
                        extra=info["extra"], shift=info["prescale"] + info["extra"],
                        cell=codec_v6.GRAIN_CELL, rows_first=True,
                        dirs=[np.ascontiguousarray(codec_v6.DIRS[m].astype(np.int32)) for m in dirmaps],
                        filter=(codec_v8.filter_map(data, info).astype(np.int32)
                                if version == 8 and info.get("filter") else None))
        raise ValueError(f"GPU decoding supports FUL v4 to v8, not version {version}")

    def _pass(self, src_tex, dst):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo[dst])
        GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

    def decode(self, data: bytes, grain: float = 1.0):
        """Decode a FUL v4 or v5 file. Returns (rgb array, timings in ms)."""
        self.context.make_current()
        t0 = time.perf_counter()
        job = self._prepare(data)
        t1 = time.perf_counter()

        w, h = job["w"], job["h"]
        self._ensure_size(w, h)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.upload_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB_INTEGER,
                           GL.GL_SHORT, np.ascontiguousarray(job["coef"]))
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        gsig = job["grain"]
        if gsig is not None:
            gs = np.ascontiguousarray(gsig.astype(np.float32))
            gtex = self._texture(GL.GL_RGBA32F, gs.shape[1], gs.shape[0],
                                 GL.GL_RGBA, GL.GL_FLOAT, gs)
        else:
            gtex = self._texture(GL.GL_RGBA32F, 1, 1, GL.GL_RGBA, GL.GL_FLOAT,
                                 np.zeros(4, np.float32))
        GL.glFinish()
        t2 = time.perf_counter()

        GL.glViewport(0, 0, w, h)
        GL.glBindVertexArray(self.vao)
        GL.glUseProgram(self.lift)
        u = lambda name: self._u(self.lift, name)
        GL.glUniform2i(u("uImg"), w, h)
        GL.glUniform1i(u("uTile"), job["tile"])
        GL.glUniform1i(u("uSrc"), 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        # 1) widen the 16-bit upload to 32 bits, adding the extra precision
        GL.glUniform1i(u("uMode"), 3)
        GL.glUniform1i(u("uShift"), job["extra"])
        self._pass(self.upload_tex, 0)
        src, passes = 0, 1
        # 2) undo the lifting steps in reverse, level by level (v4/v5: columns
        #    then rows; v6: rows then columns, with per-block directions)
        steps = job["steps"]
        GL.glUniform1i(u("uDir"), 2)
        dir_tex = [self._texture(GL.GL_R32I, m.shape[1], m.shape[0], GL.GL_RED_INTEGER,
                                 GL.GL_INT, m) for m in job["dirs"]]
        GL.glActiveTexture(GL.GL_TEXTURE0)
        for level in range(job["levels"] - 1, -1, -1):
            GL.glUniform1i(u("uLevel"), level)
            has_dir = level < len(dir_tex)
            if has_dir:
                GL.glActiveTexture(GL.GL_TEXTURE2)
                GL.glBindTexture(GL.GL_TEXTURE_2D, dir_tex[level])
                GL.glActiveTexture(GL.GL_TEXTURE0)
            for axis in ((0, 1) if job["rows_first"] else (1, 0)):
                GL.glUniform1i(u("uAxis"), axis)
                GL.glUniform1i(u("uHasDir"), 1 if (has_dir and axis == 1) else 0)
                for k in range(steps.shape[0] - 1, -1, -1):
                    odd, sg, q, r, sh = (int(v) for v in steps[k])
                    first = k == steps.shape[0] - 1
                    GL.glUniform1i(u("uMode"), 0 if first else (1 if odd else 2))
                    GL.glUniform4i(u("uStep"), sg, q, r, sh)
                    self._pass(self.tex[src], 1 - src)
                    src = 1 - src
                    passes += 1

        # 3) colour conversion and grain, into an 8-bit image
        GL.glUseProgram(self.colour)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.out_fbo)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex[src])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, gtex)
        c = lambda name: self._u(self.colour, name)
        GL.glUniform1i(c("uSrc"), 0)
        GL.glUniform1i(c("uGrain"), 1)
        GL.glUniform1i(c("uCell"), job["cell"])
        GL.glUniform1i(c("uShift"), job["shift"])
        GL.glUniform1i(c("uHasGrain"), 1 if gsig is not None else 0)
        GL.glUniform1f(c("uGrainAmount"), float(grain))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        filter_tex = []
        fmap = job.get("filter")
        if fmap is not None:
            # 4) v8 post-filter, reading the colour pass's 8-bit output
            GL.glUseProgram(self.filter)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.out2_fbo)
            mtex = self._texture(GL.GL_R32I, fmap.shape[1], fmap.shape[0], GL.GL_RED_INTEGER,
                                 GL.GL_INT, np.ascontiguousarray(fmap))
            rlut = np.ascontiguousarray(codec_v8.RANGE_LUT.astype(np.int32))
            rtex = self._texture(GL.GL_R32I, rlut.shape[1], rlut.shape[0], GL.GL_RED_INTEGER, GL.GL_INT, rlut)
            filter_tex = [mtex, rtex]
            for unit, tex, name in ((0, self.out_tex, "uImg"), (1, mtex, "uMap"), (2, rtex, "uRange")):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glUniform1i(self._u(self.filter, name), unit)
            GL.glUniform2i(self._u(self.filter, "uSize"), w, h)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            passes += 1
        GL.glFinish()
        t3 = time.perf_counter()

        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.out2_fbo if fmap is not None else self.out_fbo)
        raw = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3].copy()
        t4 = time.perf_counter()
        GL.glDeleteTextures([gtex] + dir_tex + filter_tex)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        return rgb, {"cpu_ms": (t1 - t0) * 1e3, "upload_ms": (t2 - t1) * 1e3,
                     "gpu_ms": (t3 - t2) * 1e3, "readback_ms": (t4 - t3) * 1e3,
                     "total_ms": (t4 - t0) * 1e3, "passes": passes + 1}


_decoder: GpuDecoder | None = None
_error: str | None = None


def get_decoder() -> GpuDecoder | None:
    """Shared decoder, created on first use. None if no GPU is usable."""
    global _decoder, _error
    if _decoder is None and _error is None:
        try:
            _decoder = GpuDecoder()
        except Exception as exc:
            _error = f"{type(exc).__name__}: {exc}"
    return _decoder


def last_error() -> str | None:
    return _error


if __name__ == "__main__":
    import codec_v1

    dec = get_decoder()
    if dec is None:
        raise SystemExit(f"No usable GPU: {last_error()}")
    print(f"Renderer: {dec.renderer}")
    rng = np.random.default_rng(9)
    images = [codec_v1.make_test_image()] + [
        rng.integers(0, 256, shape, dtype=np.uint8)
        for shape in [(1, 1, 3), (37, 53, 3), (300, 260, 3), (513, 771, 3)]]
    for img in images:
        for q in (100, 70, 30):
            for tile in (64, 256, 1024):
                data = codec_v4.encode(img, q, tile_size=tile, denoise=0.0, grain=False)
                assert np.array_equal(dec.decode(data)[0], codec_v4.decode(data)), ("v4", img.shape, q, tile)
            data = codec_v5.encode(img, q, denoise=0.0, grain=False)
            assert np.array_equal(dec.decode(data)[0], codec_v5.decode(data)), ("v5", img.shape, q)
            data = codec_v6.encode(img, q, denoise=0.0, grain=False)
            assert np.array_equal(dec.decode(data)[0], codec_v6.decode(data)), ("v6", img.shape, q)
            data = codec_v7.encode(img, q, denoise=0.0, grain=False)
            assert np.array_equal(dec.decode(data)[0], codec_v7.decode(data)), ("v7", img.shape, q)
            data = codec_v8.encode(img, q, denoise=0.0, grain=False)
            assert np.array_equal(dec.decode(data)[0], codec_v8.decode(data)), ("v8", img.shape, q)
    print("GPU output is bit-identical to the CPU decoders: v4 (tiled 5/3), "
          "v5 (whole-image 5/3 and 9/7), v6 to v8 (directional 9/7; v8 with its post-filter), odd sizes, lossless and lossy.")
    img = codec_v1.make_test_image()
    for version, codec in ((4, codec_v4), (5, codec_v5), (6, codec_v6), (7, codec_v7), (8, codec_v8)):
        data = codec.encode(img, 75, denoise=1.0, grain=True)
        gpu, t = dec.decode(data)
        differ = np.count_nonzero(np.any(gpu != codec.decode(data), axis=2))
        print(f"v{version} with grain: {differ} pixels differ; "
              f"{t['passes']} passes; " + ", ".join(f"{k} {v:.1f}" for k, v in t.items() if k.endswith("ms")))
