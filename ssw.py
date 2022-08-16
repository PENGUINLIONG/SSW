from code import interact
from turtle import color
import taichi as ti
import numpy as np
import PIL.Image as Image

ti.init(arch=ti.vulkan, offline_cache=False)

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)
ivec2 = ti.types.vector(2, ti.i32)
ivec3 = ti.types.vector(3, ti.i32)
ivec4 = ti.types.vector(4, ti.i32)

W, H = (512, 512)
RES = (W, H)
def set_res(res2):
    global W, H
    res = tuple(res2)
    W = res[0]
    H = res[1]

image2D = ti.types.rw_texture(
    num_dimensions=2, num_channels=1, channel_format=ti.f32, lod=0)
sampler2D = ti.types.texture(num_dimensions=2)

tex = ti.Texture(dtype=ti.f32, num_channels=1, arr_shape=RES)
imgTex = ti.Texture(dtype=ti.u8, num_channels=4, arr_shape=RES)

poisson_noise_data = np.zeros(shape=RES, dtype=np.float32)
for i in range(30):
    poisson_noise_data += np.random.random(W * H).reshape(RES) / 30
poissonTex = ti.Texture(dtype=ti.u8, num_channels=4, arr_shape=RES)
poissonTex.from_image(Image.fromarray(poisson_noise_data, mode="L"))

@ti.func
def get_uv(x, y, w, h) -> vec2:
    u = (x + 0.5) / w
    v = (y + 0.5) / h
    return vec2(u, v)

@ti.func
def get_pos(x, y, w, h) -> vec2:
    return get_uv(x, y, w, h) * 2.0 - 1.0

@ti.func
def get_duv(w, h) -> vec2:
    return 1.0 / vec2(w, h)

@ti.kernel
def view_uv(mainTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        pos = ivec2(x, y)
        mainTex.store(pos, vec4(uv.xy, 0, 1))

d1Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 2)
d2Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 4)
d3Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 8)
d4Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 16)
u4Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 16)
u3Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 8)
u2Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 4)
u1Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=ivec2(RES) / 2)

@ti.kernel
def downsample(dTex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        duv = get_duv(w, h)

        d1 = dTex.sample_lod(uv + vec2(duv[0], 0), 0.0)
        d2 = dTex.sample_lod(uv - vec2(duv[0], 0), 0.0)
        d3 = dTex.sample_lod(uv + vec2(0, duv[1]), 0.0)
        d4 = dTex.sample_lod(uv - vec2(0, duv[1]), 0.0)
        d = (d1 + d2 + d3 + d4) * 0.25

        outTex.store(ivec2(x, y), vec4(d))

@ti.kernel
def upsample(dTex: sampler2D, uTex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        duv = get_duv(w, h)

        d1 = dTex.sample_lod(uv + vec2(duv[0], duv[1]), 0.0)
        d2 = dTex.sample_lod(uv + vec2(-duv[0], duv[1]), 0.0)
        d3 = dTex.sample_lod(uv + vec2(duv[0], -duv[1]), 0.0)
        d4 = dTex.sample_lod(uv + vec2(-duv[0], -duv[1]), 0.0)
        d = (d1 + d2 + d3 + d4) * 0.25
        u = uTex.sample_lod(uv, 0.0)

        outTex.store(ivec2(x, y), vec4(d*0.2 + u))

@ti.kernel
def copy(inTex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        outTex.store(ivec2(x, y), inTex.sample_lod(uv, 0.0))

@ti.kernel
def bloom_sum(d1Tex: sampler2D, d2Tex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32, intensity: ti.template()):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)

        d1 = d1Tex.sample_lod(uv, 0.0)
        d2 = d2Tex.sample_lod(uv, 0.0)
        d = d1 + d2 * intensity

        outTex.store(ivec2(x, y), d)

def bloom(inTex, outTex, bloom_intensity):
    set_res(d1Tex.shape)
    downsample(inTex, d1Tex, W, H)

    set_res(d2Tex.shape)
    downsample(d1Tex, d2Tex, W, H)

    set_res(d3Tex.shape)
    downsample(d2Tex, d3Tex, W, H)

    set_res(d4Tex.shape)
    downsample(d3Tex, d4Tex, W, H)

    set_res(u4Tex.shape)
    upsample(d4Tex, d3Tex, u4Tex, W, H)

    set_res(u3Tex.shape)
    upsample(u4Tex, d2Tex, u3Tex, W, H)

    set_res(u2Tex.shape)
    upsample(u3Tex, d1Tex, u2Tex, W, H)

    set_res(u1Tex.shape)
    upsample(u2Tex, tex, u1Tex, W, H)

    bloom_sum(inTex, u1Tex, outTex, outTex.shape[0], outTex.shape[1], bloom_intensity)

@ti.kernel
def sum5(d1Tex: sampler2D, d2Tex: sampler2D, d3Tex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)

        d1 = d1Tex.sample_lod(uv, 0.0) * 1
        d2 = d2Tex.sample_lod(uv, 0.0) * 2
        d3 = d3Tex.sample_lod(uv, 0.0) * 4
        d = (d1 + d2 + d3) / 7

        outTex.store(ivec2(x, y), d)


def blur(inTex, outTex):
    set_res(d1Tex.shape)
    downsample(inTex, d1Tex, W, H)

    set_res(d2Tex.shape)
    downsample(d1Tex, d2Tex, W, H)

    set_res(outTex.shape)
    sum5(inTex, d1Tex, d2Tex, outTex, W, H)


blurTex = ti.Texture(dtype=ti.f32, num_channels=1, arr_shape=RES)


cutTex = ti.Texture(dtype=ti.f32, num_channels=1, arr_shape=RES)


@ti.kernel
def cut(blurTex: sampler2D, cutTex: image2D, w: ti.i32, h: ti.i32, cut_thresh: ti.f32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)

        value = 0.0
        if blurTex.sample_lod(uv, 0.0)[0] > cut_thresh:
            value = 1.0

        cutTex.store(ivec2(x, y), vec4(value))


normalTex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)

@ti.kernel
def normal(cutTex: sampler2D, normalTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        duv = get_duv(w, h)

        d00 = cutTex.sample_lod(uv, 0.0)[0]
        d10 = cutTex.sample_lod(uv + vec2(duv[0], 0), 0.0)[0]
        d01 = cutTex.sample_lod(uv + vec2(0, duv[1]), 0.0)[0]

        dx = d10 - d00
        dy = d01 - d00

        normalTex.store(ivec2(x, y), vec4(dx, dy, 0, 0) * 0.5 + 0.5)

diffuse1Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)
diffuse2Tex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)

@ti.kernel
def diffuse(normalTex: sampler2D, diffuseTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        duv = get_duv(w, h)

        dn0 = normalTex.sample_lod(uv + vec2(duv[0], 1), 0.0) * 2.0 - 1.0
        dp0 = normalTex.sample_lod(uv + vec2(-duv[0], 1), 0.0) * 2.0 - 1.0
        d0n = normalTex.sample_lod(uv + vec2(0, duv[1]), 0.0) * 2.0 - 1.0
        d0p = normalTex.sample_lod(uv + vec2(0, -duv[1]), 0.0) * 2.0 - 1.0

        dxy = (dn0 + dp0 + d0n + d0p)

        diffuseTex.store(ivec2(x, y), vec4(dxy[0], dxy[1], 0, 0) * 0.5 + 0.5)


maskedTex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)

@ti.kernel
def mask(maskTex: sampler2D, normalTex: sampler2D, maskedTex: image2D, w: ti.i32, h: ti.i32):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)

        msk = maskTex.sample_lod(uv, 0.0)[0]
        normal = normalTex.sample_lod(uv, 0.0) * 2.0 - 1.0

        maskedTex.store(ivec2(x, y), msk * normal * 0.5 + 0.5)

bgTex = ti.Texture(dtype=ti.u8, num_channels=4, arr_shape=RES)
bgTex.from_image(Image.open("assets/penguinliong.jpg").resize(RES).transpose(Image.FLIP_TOP_BOTTOM))
composeTex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)

@ti.func
def clamp(x, mn, mx):
    return ti.max(ti.min(x, mx), mn)

@ti.func
def saturate(x):
    return clamp(x, 0.0, 1.0)

@ti.func
def normalize(x):
    return x / ti.sqrt(saturate(x.dot(x)))


@ti.func
def fresnel(amount, alpha, normal, view):
    return ti.pow(1.0 - clamp(normalize(normal).dot(normalize(view)), 0.0, 1.0), amount) * alpha

@ti.func
def dimming(normal, view):
    return normalize(normal).dot(normalize(view))

@ti.func
def lerp(x, y, s):
    return (1 - s) * x + s * y

@ti.kernel
def compose(maskTex: sampler2D, bgTex: sampler2D, normalTex: sampler2D, poissonTex: sampler2D, outTex: image2D, w: ti.i32, h: ti.i32, fresnel_amount: ti.template(), fresnel_alpha: ti.template(), dimming_amount: ti.template(), normal_z: ti.template(), liquid_color_r: ti.template(), liquid_color_g: ti.template(), liquid_color_b: ti.template(), light_exponent: ti.template(), light_intensity: ti.template()):
    for x, y in ti.ndrange(w, h):
        uv = get_uv(x, y, w, h)
        duv = get_duv(w, h)

        msk = maskTex.sample_lod(uv, 0.0)[0]
        normal = normalTex.sample_lod(uv, 0.0) * 2.0 - 1.0
        normall = vec2(normal[0], normal[1])
        # `duv` to add a bit of tiny bit of refraction to the perpendicular view.
        bg = bgTex.sample_lod(uv - normall * 0.5 + duv, 0.0)
        poisson = saturate(poissonTex.sample_lod(uv, 0.0)[0] * 0.2 + 0.8)

        normal2 = normalize(vec3(normall, normal_z))
        view = normalize(vec3(0.02, 0.02, 1))
        light = normalize(vec3(-1,-1,0.7))

        f = fresnel(fresnel_amount, fresnel_alpha, normal2, view)

        # A bit of dimming because water somehow absorbs light energy.
        d = clamp(dimming(normal2, view) * dimming_amount, 0.0, 1.0)
        liquid_color = vec4(liquid_color_r, liquid_color_g, liquid_color_b, 1.0)

        # Specular by a directional light.
        h = ti.pow(saturate(normalize((light + view) * 0.5).dot(normal2)), light_exponent) * light_intensity

        color_mask = lerp(vec4(1.0), liquid_color, msk)

        color = bg * color_mask + (h - f - d) * msk * poisson
        outTex.store(ivec2(x, y), color)


finalTex = ti.Texture(dtype=ti.f32, num_channels=4, arr_shape=RES)


window = ti.ui.Window("X", RES)
canvas = window.get_canvas()
gui = window.get_gui()

cut_thresh = 0.35
fresnel_amount = 63.56
fresnel_alpha = 1.949
dimming_amount = 0.01
normal_z = 0.924
liquid_color = (207/255, 228/255, 244/255)
bloom_intensity = 2.0
light_exponent = 70.0
light_intensity = 0.2

k = 0
while window.running:
    with gui.sub_window("Screen Space Liquid", 0.1, 0.1, 0.4, 0.8) as w:
        cut_thresh = w.slider_float("Cut Threshold", cut_thresh, 0.0, 1.0)
        fresnel_amount = w.slider_float("Fresnel Amount", fresnel_amount, 0.0, 100.0)
        fresnel_alpha = w.slider_float("Fresnel Alpha", fresnel_alpha, 0.0, 10.0)
        dimming_amount = w.slider_float("Dimming Amount", dimming_amount, 0.0, 1.0)
        normal_z = w.slider_float("Normal Z", normal_z, 0.0, 1.0)
        liquid_color = w.color_edit_3("Liquid Color", liquid_color)
        bloom_intensity = w.slider_float("Bloom Intensity", bloom_intensity, 0.0, 5.0)
        light_exponent = w.slider_float("Light Exponent", light_exponent, 0.0, 100.0)
        light_intensity = w.slider_float("Light Intensity", light_intensity, 0.0, 1.0)

    #set_res(tex.shape)
    #noise(tex, W, H)
    imgTex.from_image(Image.open(f"assets/mpm88-output/k{k%240:03}.png").resize(RES).transpose(Image.TRANSPOSE))

    #blur(tex, blurTex)
    blur(imgTex, blurTex)
    ti.sync()

    set_res(cutTex.shape)
    cut(blurTex, cutTex, W, H, cut_thresh)

    set_res(normalTex.shape)
    normal(cutTex, normalTex, W, H)

    set_res(diffuse1Tex.shape)
    diffuse(normalTex, diffuse1Tex, W, H)


    blur(diffuse1Tex, diffuse2Tex)
    ti.sync()

    set_res(maskedTex.shape)
    mask(cutTex, diffuse2Tex, maskedTex, W, H)

    set_res(composeTex.shape)
    compose(cutTex, bgTex, maskedTex, poissonTex, composeTex, W, H, fresnel_amount, fresnel_alpha, dimming_amount, normal_z, liquid_color[0], liquid_color[1], liquid_color[2], light_exponent, light_intensity)
    ti.sync()

    bloom(composeTex, finalTex, bloom_intensity)
    ti.sync()

    canvas.set_image(finalTex)
    window.show()
    k += 1
