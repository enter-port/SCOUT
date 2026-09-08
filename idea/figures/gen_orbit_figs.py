# -*- coding: utf-8 -*-
"""orbit 两相控制三联示意图 v3（按用户意见重画）:
① atypical: 只画两个箭头 —— 我们走的方向 / cost 也增大但不走的方向
② orbit 无限制(fb_clamp=none): 爬坡 → 壳上扶κ+切向巡行; 远壳点一根无界拉力长箭
③ orbit 带限制(fb_clamp=soft): 同场景; 远壳点拉力封顶短箭(虚线=无限制版对比) + 噪声限带
所有箭头位置均从景观几何解析计算(梯度场/等值线射线求交), 非手摆坐标.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
})

# ---- 配色(静 mute 学术风) ----
INK    = "#2B3038"   # 正文
MUTED  = "#82888F"   # 次要
SLATE  = "#41638C"   # phase-1 爬坡(冷蓝)
AMBER  = "#D98A2B"   # phase-2 切向巡行(琥珀)
CRIM   = "#C24850"   # κ 环 / 拉力反馈 / cap(绛红)
GHOST  = "#9BA0A6"   # 对比虚线
BG     = "#FFFFFF"
RAMP   = LinearSegmentedColormap.from_list(
    "tealsoft", ["#F7FAF9", "#E3EFEC", "#CBE2DC", "#AFD3CB", "#93C3B9"])

KAPPA, DELTA, LAM = 2.5, 0.25, 0.5
AU, AV, TH = 2.35, 1.32, 0.5
C, S = np.cos(TH), np.sin(TH)
XL, YL = (-5.6, 5.6), (-4.3, 4.3)
WB = dict(fc="white", ec="none", alpha=0.72, pad=1.8)   # 暗区标签白底

def f_xy(x, y):
    u = C * x + S * y
    v = -S * x + C * y
    return (u / AU) ** 2 + (v / AV) ** 2

def grad_f(x, y):
    u = C * x + S * y
    v = -S * x + C * y
    dfx = 2 * u / AU**2 * C - 2 * v / AV**2 * S
    dfy = 2 * u / AU**2 * S + 2 * v / AV**2 * C
    return np.array([dfx, dfy])

def ghat(p):
    g = grad_f(*p)
    return g / np.hypot(*g)

def ray_hit(d, level):
    """从原点沿单位方向 d 走到 f=level 的距离(景观对射线是二次型,解析解)."""
    ud = C * d[0] + S * d[1]
    vd = -S * d[0] + C * d[1]
    q = (ud / AU) ** 2 + (vd / AV) ** 2
    return np.sqrt(level / q)

def ring_p(phi_deg, level=KAPPA):
    d = np.array([np.cos(np.radians(phi_deg)), np.sin(np.radians(phi_deg))])
    return ray_hit(d, level) * d

def climb_path(p0, stop, step=0.035, max_it=4000):
    """梯度上升数值积分, 返回路径点列."""
    p = np.asarray(p0, float)
    path = [p.copy()]
    for _ in range(max_it):
        g = grad_f(*p)
        n = np.hypot(*g)
        if n < 1e-9:
            break
        p = p + step * g / n
        path.append(p.copy())
        if f_xy(*p) >= stop:
            break
    return np.array(path)

def arrow(ax, p, d, color, scale, lw=2.2, ms=13, zorder=8, alpha=1.0, ls="-"):
    d = np.asarray(d, float)
    n = np.hypot(*d)
    d = d / n * scale
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(np.asarray(p) + d),
                                 arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, zorder=zorder, alpha=alpha,
                                 linestyle=ls, shrinkA=0, shrinkB=0))

def base_ax(ax, band=True):
    xs = np.linspace(*XL, 560); ys = np.linspace(*YL, 460)
    X, Y = np.meshgrid(xs, ys)
    F = f_xy(X, Y)
    ax.set_facecolor(BG)
    ax.contourf(X, Y, F, levels=np.linspace(0, 4.6, 17), cmap=RAMP, zorder=0)
    if band:
        ax.contourf(X, Y, F, levels=[KAPPA - DELTA, KAPPA + DELTA],
                    colors=[CRIM], alpha=0.07, zorder=1)
        for lev in (KAPPA - DELTA, KAPPA + DELTA):
            pts = [ring_p(a, lev) for a in np.linspace(0, 360, 361)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CRIM,
                    lw=0.8, ls=(0, (2, 3)), alpha=0.55, zorder=3)
    pts = [ring_p(a) for a in np.linspace(0, 360, 721)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CRIM, lw=2.4,
            zorder=4, solid_capstyle="round")
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.axis("off")

def a0_dot(ax):
    ax.plot(0, 0, "o", ms=6.5, mfc=INK, mec="white", mew=1.1, zorder=9)
    ax.text(-0.72, -0.22, r"$a^0$", fontsize=11, color=INK, zorder=9)

def formula_box(ax, lines, xy, fontsize=10.0):
    ax.text(*xy, "\n".join(lines), fontsize=fontsize, color=INK, va="top",
            ha="left", zorder=12, linespacing=1.75,
            bbox=dict(boxstyle="round,pad=0.5", fc="#FAFAF6", ec="#DDDED6", lw=0.9))

# ================= 图 ① atypical: 两个箭头 =================
def fig1(tag="", title=True):
    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=200)
    fig.patch.set_facecolor(BG)
    base_ax(ax, band=False)
    a0_dot(ax)

    # 箭头1: 我们走的 —— 从 a⁰ 近旁沿 ∇f 梯度上升(积分真实路径), 到 κ 环封顶为止
    path = climb_path((0.30, -0.18), KAPPA)
    ax.plot(path[:, 0], path[:, 1], color=CRIM, lw=2.4, alpha=0.75, zorder=6)
    arrow(ax, path[-6], path[-1] - path[-6], CRIM, 0.12, lw=3.2, ms=24, zorder=9)
    ax.annotate("① 我们走的:沿 $\\nabla f$ 爬 cost\n（最陡方向,到 $\\kappa$ 封顶为止）",
                xy=tuple(path[int(len(path) * 0.50)]), xytext=(-2.35, -1.85),
                fontsize=11.5, color=CRIM, ha="left", zorder=10, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=CRIM, lw=0.9, alpha=0.6))

    # 箭头2: cost 也增大但 argmax 不走的方向(虚线, 止于半山腰)
    d2 = np.array([np.cos(np.radians(97)), np.sin(np.radians(97))])
    p2s = 0.45 * d2
    p2e = ray_hit(d2, 1.35) * d2
    arrow(ax, p2s, d2, GHOST, np.hypot(*(p2e - p2s)), lw=2.4, ms=17,
          zorder=8, ls=(0, (5, 3)))
    ax.text(-0.28, 2.52, "② cost 也增大,但 argmax 梯度上升不走",
            fontsize=11.5, color=MUTED, ha="left", zorder=10, bbox=WB)

    # κ 环标注(左下外侧)
    pl = ring_p(200)
    ax.text(pl[0] - 0.25, pl[1] - 0.42, r"$\kappa$ 封顶环 $\{f=\kappa\}$",
            fontsize=11, color=CRIM, ha="right", zorder=10, bbox=WB)

    formula_box(ax, [
        r"cost$(a\,|\,\bar s)=\min\left(\mathrm{KL}(q_\phi(z|\bar s,a)\,\|\,q_\phi(z|\bar s,a^0)),\ \kappa\right)$",
        r"注入: $x_t\leftarrow x_t+\eta\sqrt{1-\bar\alpha_t}\,\nabla\,\mathrm{cost}$"
        r"$\quad(\eta=3.0,\ \kappa=2.5)$",
        r"argmax 控制:只爬一条最陡棱线,其余上坡方向全部放弃"],
        (-5.45, 4.05))
    if title:
        ax.set_title("① atypical（entropy cost）:把动作推离 DP 意图,到 κ 封顶",
                     fontsize=14, color=INK, pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"idea/figures/fig1_atypical{tag}.png", facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

# ============ 图 ②/③ 共用: 两相几何 ============
CLIMB_START = (0.40, -1.50)   # 壳内出发
FB_PHI = 80                   # 法向反馈演示箭所在的 xy 方位角(巡航弧之后)

def phase_geometry(ax):
    """爬坡路径(蓝) + 壳上巡行(琥珀) + 一根法向反馈演示箭(红)."""
    # phase 1: 梯度上升到 κ−δ 交接
    path = climb_path(CLIMB_START, KAPPA - DELTA)
    ax.plot(path[:, 0], path[:, 1], color=SLATE, lw=1.9, alpha=0.8, zorder=5)
    ax.plot(*CLIMB_START, "o", ms=5, mfc=SLATE, mec="white", mew=0.9, zorder=9)
    for frac in (0.28, 0.62, 0.93):
        i = int(len(path) * frac)
        p, q = path[i], path[min(i + 4, len(path) - 1)]
        arrow(ax, p, q - p, SLATE, 0.52, lw=2.3, ms=15, zorder=8)
    p_end = path[-1]
    phi0 = np.degrees(np.arctan2(p_end[1], p_end[0]))

    # phase 2: 沿 κ 环逆时针一段弧, 轨迹线+箭头沿真实切向, 向外偏 0.2 不压环线
    off = 0.20
    arc = np.linspace(phi0 + 18, phi0 + 105, 80)
    cruise = np.array([ring_p(a) + ghat(ring_p(a)) * off for a in arc])
    ax.plot(cruise[:, 0], cruise[:, 1], color=AMBER, lw=2.5, alpha=0.95, zorder=6)
    for a in (phi0 + 32, phi0 + 62, phi0 + 92):
        p = ring_p(a) + ghat(ring_p(a)) * off
        g = ghat(ring_p(a))
        t = np.array([-g[1], g[0]])          # 逆时针切向
        arrow(ax, p - t * 0.36, t * 0.72, AMBER, 0.72, lw=2.7, ms=18, zorder=8)

    # 法向反馈演示箭: 壳外一点被拉回 κ(f<κ 时同式反向推出)
    g = ghat(ring_p(FB_PHI))
    p_out = ring_p(FB_PHI, KAPPA + 0.40)
    arrow(ax, p_out, -g, CRIM, 0.50, lw=2.0, ms=13, zorder=8)
    return path

def far_point():
    """左轴线上壳外远点(f≈5.2) + 指向壳的法向单位向量."""
    p = ring_p(180, 5.2)
    return p, -ghat(p)

# ================= 图 ② orbit 无限制 =================
def fig2(tag="", title=True):
    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=200)
    fig.patch.set_facecolor(BG)
    base_ax(ax)
    a0_dot(ax)
    path = phase_geometry(ax)

    ax.annotate("Phase 1（$f<\\kappa-\\delta$）:照常爬坡 = atypical",
                xy=CLIMB_START, xytext=(-2.1, -2.95),
                fontsize=11, color=SLATE, ha="left", zorder=10, bbox=WB,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.9, alpha=0.6))
    ax.text(0.9, 3.35, "Phase 2（$f\\geq\\kappa-\\delta$）:法向反馈扶 κ + 切向噪声巡行",
            fontsize=11, color=AMBER, ha="left", zorder=10, bbox=WB)
    ax.text(0.82, 2.42, "法向反馈:$f>\\kappa$ 拉回（$f<\\kappa$ 推出）",
            fontsize=9.5, color=CRIM, ha="left", zorder=10, bbox=WB)

    p_far, din = far_point()
    arrow(ax, p_far, din, CRIM, 1.34, lw=2.8, ms=20, zorder=9)
    ax.annotate("残差 $(f-\\kappa)$ 不封顶:拉力 $\\propto|f-\\kappa|$ 无上界\n"
                "远壳行一步猛拽 → 链后期 fb 过热、jerk 爆、救回归零",
                xy=tuple(p_far + din * 0.25), xytext=(-5.45, -3.92),
                fontsize=10.5, color=CRIM, ha="left",
                arrowprops=dict(arrowstyle="->", color=CRIM, lw=1.1), zorder=10)

    pl = ring_p(222)
    ax.text(pl[0] - 0.50, pl[1] - 0.20, r"$\kappa$ 壳", fontsize=10.5,
            color=CRIM, ha="right", zorder=10, bbox=WB)
    formula_box(ax, [
        r"Phase 2 注入（替换爬坡, 逐行二选一）:",
        r"$\Delta x_t=-\lambda(f-\kappa)\,g/\|g\|^2+\sigma\sqrt{1-\bar\alpha_t}\,\xi_\perp,"
        r"\quad g=\nabla f,\ \ \xi_\perp=\xi-(\xi\!\cdot\!\hat g)\hat g$",
        r"法向两侧同指壳（交接无缺口、无双份剂量）;$\lambda$ 无量纲"],
        (-5.45, 4.05))
    if title:
        ax.set_title("② orbit 两相控制（无限制 fb_clamp=none）:爬坡 → 壳上约束动力学",
                     fontsize=14, color=INK, pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"idea/figures/fig2_orbit_none{tag}.png", facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

# ================= 图 ③ orbit 带限制 =================
def fig3(tag="", title=True):
    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=200)
    fig.patch.set_facecolor(BG)
    base_ax(ax)
    a0_dot(ax)
    path = phase_geometry(ax)

    ax.annotate("Phase 1（$f<\\kappa-\\delta$）:照常爬坡 = atypical",
                xy=CLIMB_START, xytext=(-2.1, -2.95),
                fontsize=11, color=SLATE, ha="left", zorder=10, bbox=WB,
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.9, alpha=0.6))
    ax.text(0.9, 3.35, "Phase 2（$f\\geq\\kappa-\\delta$）:法向反馈扶 κ + 切向噪声巡行",
            fontsize=11, color=AMBER, ha="left", zorder=10, bbox=WB)
    ax.text(0.82, 2.42, "法向反馈:$f>\\kappa$ 拉回（$f<\\kappa$ 推出）",
            fontsize=9.5, color=CRIM, ha="left", zorder=10, bbox=WB)

    p_far, din = far_point()
    # 虚线 ghost = 无限制版会拉的长度; 实线短箭 = soft 封顶后的每步拉力
    arrow(ax, p_far, din, GHOST, 1.34, lw=1.8, ms=14, zorder=8,
          ls=(0, (5, 3)), alpha=0.95)
    arrow(ax, p_far, din, CRIM, 0.55, lw=2.8, ms=20, zorder=9)
    ax.annotate("fb soft-clamp:$(f-\\kappa)\\to\\delta\\tanh((f-\\kappa)/\\delta)$\n"
                "每步拉力封顶 $\\leq\\lambda\\delta/\\|g\\|$（虚线=无限制版拉距,多步缓回）",
                xy=tuple(p_far + din * 0.30), xytext=(-5.45, -3.92),
                fontsize=10.5, color=CRIM, ha="left",
                arrowprops=dict(arrowstyle="->", color=CRIM, lw=1.1), zorder=10)

    # 噪声限带标注: 指向左上弧的带边
    pb = ring_p(150, KAPPA + DELTA)
    ax.annotate("切向噪声限带 $[\\kappa-\\delta,\\ \\kappa+\\delta]$（带外置零,RNG 流不变）",
                xy=tuple(pb), xytext=(-5.45, 1.60), fontsize=10.5, color=MUTED,
                ha="left", zorder=10,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0),
                bbox=WB)

    pl = ring_p(222)
    ax.text(pl[0] - 0.50, pl[1] - 0.20, r"$\kappa$ 壳", fontsize=10.5,
            color=CRIM, ha="right", zorder=10, bbox=WB)
    formula_box(ax, [
        r"Phase 2 注入（fb_clamp=soft）:",
        r"$\Delta x_t=-\lambda\,\delta\tanh\left(\frac{f-\kappa}{\delta}\right)\,"
        r"g/\|g\|^2+\sigma\,0.5^{\,(r-1)}\,(1-\bar\alpha_t)^{p/2}\,\xi_\perp$",
        r"拉力饱和 + 噪声限带 + 轮次调度（$\sigma$ 每轮减半, $p{=}2$）:"
        r"jerk 回 atypical 档"],
        (-5.45, 4.05))
    if title:
        ax.set_title("③ orbit 两相控制（带限制 fb_clamp=soft）:拉力封顶 + 噪声限带",
                     fontsize=14, color=INK, pad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"idea/figures/fig3_orbit_soft{tag}.png", facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3"]
    tags = [""] if "qa" in which else ["", "_slide"]
    if "qa" in which:
        which = [w for w in which if w != "qa"]
    for w in which:
        for tg in tags:
            {"1": fig1, "2": fig2, "3": fig3}[w](tg)
    print("done")
