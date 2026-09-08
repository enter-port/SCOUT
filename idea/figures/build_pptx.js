const pptxgen = require("pptxgenjs");
const fs = require("fs");

const W = 13.33, H = 7.5, M = 0.5;
const BG = "FFFFFF", PRIMARY = "2B3038", ACCENT = "C24850", TEXT = "1F2430", MUTED = "6B7280";
const FONT = "微软雅黑";

// 从 PNG IHDR 读宽高, 宽高比永不漂移
function pngAr(p) {
  const b = fs.readFileSync(p);
  return b.readUInt32BE(16) / b.readUInt32BE(20);
}

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "SCOUT";
pres.title = "orbit 两相控制示意图";

const slides = [
  {
    n: 1,
    img: "idea/figures/fig1_atypical_slide.png",
    head: "① atypical（entropy cost）：把动作推离 DP 意图，到 κ 封顶",
    take1: "cost = min(KL(q(z|s̄,a) ‖ q(z|s̄,a⁰)), κ)：以 DP 自身意图 a⁰ 为参照，引导只走到 κ 封顶为止（双信任域）。",
    take2: "argmax 实现：梯度上升只爬一条最陡棱线，其余上坡方向全部放弃 —— 这正是 orbit 要解决的问题。",
  },
  {
    n: 2,
    img: "idea/figures/fig2_orbit_none_slide.png",
    head: "② orbit 两相控制（无限制，fb_clamp=none）：爬坡 → 壳上约束动力学",
    take1: "phase 1 逐字节 = atypical；phase 2 换约束动力学：法向 Newton 扶 κ + 切向巡行给方向覆盖。",
    take2: "但残差无界：远壳行拉力 ∝ |f−κ| → 链后期 VIB 共适应、KL 右移后 jerk 爆，救回归零（fb 0.55/0.61 vs 健康 0.24–0.33）。",
  },
  {
    n: 3,
    img: "idea/figures/fig3_orbit_soft_slide.png",
    head: "③ orbit 两相控制（带限制，fb_clamp=soft）：拉力封顶 + 噪声限带",
    take1: "fb soft-clamp 把远壳拉力封顶 λδ/‖g‖；切向噪声限带 [κ−δ, κ+δ]（带外置零，RNG 流位同）。",
    take2: "配 σ 轮衰减 0.5^(r−1) 与 noise_anneal p=2 → 最终固定参数组一组跨任务免标定（square 0.80→0.96、can 0.97→0.98）。",
  },
];

for (const s of slides) {
  const slide = pres.addSlide();
  slide.background = { color: BG };

  slide.addText(`SCOUT · orbit 两相控制示意`, {
    x: M, y: 0.22, w: 6.0, h: 0.32, margin: 0,
    fontFace: FONT, fontSize: 11, color: MUTED, charSpacing: 2,
  });
  slide.addText(`${s.n} / 3`, {
    x: W - M - 1.2, y: 0.22, w: 1.2, h: 0.32, margin: 0, align: "right",
    fontFace: FONT, fontSize: 11, color: MUTED,
  });

  slide.addText(s.head, {
    x: M, y: 0.52, w: W - 2 * M, h: 0.52, margin: 0,
    fontFace: FONT, fontSize: 21, bold: true, color: PRIMARY,
  });

  const maxH = 4.92;
  const ar = pngAr(s.img);
  const imgW = Math.min(maxH * ar, W - 2 * M);
  const imgH = imgW / ar;
  const imgX = (W - imgW) / 2;
  const imgY = 1.18 + (maxH - imgH) / 2;
  slide.addImage({ path: s.img, x: imgX, y: imgY, w: imgW, h: imgH });

  slide.addText([
    { text: s.take1, options: { bold: true, color: TEXT, breakLine: true } },
    { text: s.take2, options: { color: MUTED } },
  ], {
    x: M, y: 6.28, w: W - 2 * M, h: 0.92, margin: 0,
    fontFace: FONT, fontSize: 12.5, valign: "top", paraSpaceAfter: 4,
  });
}

pres.writeFile({ fileName: "idea/figures/orbit_two_phase_schematic.pptx" })
  .then(() => console.log("pptx done"));
