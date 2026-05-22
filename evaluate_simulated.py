"""
模拟数据定量评价
============================================================
生成合成条纹图案（含线性/环形条纹），添加高斯/散斑噪声，
对比五种去噪方法的 MAE、PSNR(dB)、SSIM、方向误差(°)。

方法：
  1. 含噪图像（基线）
  2. 高斯滤波 (σ = 2)
  3. 中值滤波 (5×5)
  4. NL-Means (h = 15, 搜索 21×21, 块 7×7)
  5. 本文 CED 方法
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import structural_similarity as calc_ssim
import cv2
import sys
import os
import io
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ced_denoising import ced_denoise, compute_structure_tensor


class _Quiet:
    """Suppress stdout during CED batch calls"""
    def __enter__(self):
        self._out = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *a):
        sys.stdout = self._out


# ================================================================
# 数据生成 — 多种条纹类型
# ================================================================

def generate_linear_fringe(H, W, freq, angle, phase):
    """线性条纹: I = 0.5*(1 + cos(2π·f·(x·cosθ + y·sinθ) + φ))"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    xr = x * np.cos(angle) + y * np.sin(angle)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * freq * xr + phase))


def generate_circular_fringe(H, W, freq, phase):
    """环形条纹: I = 0.5*(1 + cos(2π·f·r + φ))"""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return 0.5 * (1.0 + np.cos(2.0 * np.pi * freq * r + phase))


def add_noise(img, ntype, level):
    if ntype == "gaussian":
        return np.clip(img + np.random.randn(*img.shape) * level, 0, 1)
    else:
        return np.clip(img * (1.0 + np.random.randn(*img.shape) * level), 0, 1)


# ================================================================
# 评价指标
# ================================================================

def direction_error(clean, test, sg=1.0, rho=1.0):
    """
    方向误差 (°) — 基于结构张量主方向的角度偏差，
    在梯度幅度大的区域（条纹区域）加权。
    """
    J11c, J12c, J22c = compute_structure_tensor(clean, sg, rho)
    J11t, J12t, J22t = compute_structure_tensor(test, sg, rho)

    tc = 0.5 * np.arctan2(2.0 * J12c, J11c - J22c)
    tt = 0.5 * np.arctan2(2.0 * J12t, J11t - J22t)

    diff = np.abs(tc - tt)
    diff = np.minimum(diff, np.pi - diff)

    # 用 clean 图像的相干性作为权重（条纹区权重大，平坦区权重小）
    coherence = np.sqrt((J11c - J22c) ** 2 + 4 * J12c ** 2)
    w = coherence / (np.mean(coherence) + 1e-10)

    return np.mean(diff * w) * 180.0 / np.pi


def evaluate(clean, denoised):
    mae = np.mean(np.abs(clean - denoised)) * 255.0
    mse = np.mean((clean - denoised) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-20 else 100.0
    ssim = calc_ssim(clean, np.clip(denoised, 0, 1), data_range=1.0)
    derr = direction_error(clean, denoised)
    return mae, psnr, ssim, derr


# ================================================================
# 主程序
# ================================================================

def main():
    np.random.seed(42)
    H, W = 256, 256

    # --- 构建测试配置 ---
    configs = []

    # 线性条纹 + 高斯噪声（各种频率、角度、噪声等级）
    for freq in [0.03, 0.05, 0.07, 0.04]:
        for ang in [0, np.pi / 6, np.pi / 3, np.pi / 2]:
            ph = np.random.uniform(0, 2 * np.pi)
            for lv in [0.08, 0.12, 0.18, 0.25]:
                configs.append(("linear", "gaussian", lv, freq, ang, ph))

    # 环形条纹 + 高斯噪声
    for freq in [0.03, 0.05, 0.04]:
        ph = np.random.uniform(0, 2 * np.pi)
        for lv in [0.08, 0.12, 0.18, 0.25]:
            configs.append(("circular", "gaussian", lv, freq, 0, ph))

    N = len(configs)
    print(f"测试样本数: {N}")

    methods = ["含噪图像", "高斯滤波(σ=2)", "中值滤波(5×5)",
               "NL-Means", "本文CED"]
    keys = ["mae", "psnr", "ssim", "dir"]
    R = {m: {k: [] for k in keys} for m in methods}

    t0 = time.time()

    for i, (ptype, ntype, lv, freq, ang, ph) in enumerate(configs):
        if ptype == "linear":
            clean = generate_linear_fringe(H, W, freq, ang, ph)
        else:
            clean = generate_circular_fringe(H, W, freq, ph)
        noisy = add_noise(clean, ntype, lv)

        # 1. 含噪图像
        for k, v in zip(keys, evaluate(clean, noisy)):
            R["含噪图像"][k].append(v)

        # 2. 高斯滤波 σ=2
        dg = np.clip(gaussian_filter(noisy, 2.0), 0, 1)
        for k, v in zip(keys, evaluate(clean, dg)):
            R["高斯滤波(σ=2)"][k].append(v)

        # 3. 中值滤波 5×5
        dm = median_filter(noisy, size=5, mode="nearest")
        for k, v in zip(keys, evaluate(clean, dm)):
            R["中值滤波(5×5)"][k].append(v)

        # 4. NL-Means (h=15, 搜索 21×21, 块 7×7)
        #    h 自适应调整至噪声水平以保证公平对比
        u8 = (noisy * 255).astype(np.uint8)
        _diff = noisy.astype(np.float64) - median_filter(noisy, size=3)
        _sigma_est = np.median(np.abs(_diff)) * 1.4826 * 255
        _h = max(15, int(_sigma_est))
        dn = cv2.fastNlMeansDenoising(u8, None, _h, 7, 21)
        dn = dn.astype(np.float64) / 255.0
        for k, v in zip(keys, evaluate(clean, dn)):
            R["NL-Means"][k].append(v)

        # 5. 本文 CED
        with _Quiet():
            dc = ced_denoise(noisy)
        res = evaluate(clean, dc)
        for k, v in zip(keys, res):
            R["本文CED"][k].append(v)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[{i+1}/{N}] {ptype} {ntype}(σ={lv:.2f}) "
                  f"CED: MAE={res[0]:.1f} PSNR={res[1]:.1f}dB "
                  f"SSIM={res[2]:.3f} DirErr={res[3]:.1f}° | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s")

    # === 输出表格 ===
    hdr = "{:<35} {:>14} {:>14} {:>14} {:>14}"
    print("\n" + "=" * 90)
    print("表 3: 模拟数据上的定量评价（平均值 ± 标准差）")
    print("=" * 90)
    print(hdr.format("方法", "MAE↓", "PSNR(dB)↑", "SSIM↑", "方向误差(°)↓"))
    print("-" * 90)

    for m in methods:
        vals = []
        for k in keys:
            mu, sd = np.mean(R[m][k]), np.std(R[m][k])
            if k == "ssim":
                vals.append(f"{mu:.2f}±{sd:.2f}")
            else:
                vals.append(f"{mu:.1f}±{sd:.1f}")
        print(hdr.format(m, *vals))

    print("=" * 90)

    # Markdown
    print("\n--- Markdown ---\n")
    print("| 方法 | MAE↓ | PSNR(dB)↑ | SSIM↑ | 方向误差 (°)↓ |")
    print("|------|------|-----------|-------|-------------|")
    for m in methods:
        vals = []
        for k in keys:
            mu, sd = np.mean(R[m][k]), np.std(R[m][k])
            if k == "ssim":
                vals.append(f"{mu:.2f}±{sd:.2f}")
            else:
                vals.append(f"{mu:.1f}±{sd:.1f}")
        print(f"| {m} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")


if __name__ == "__main__":
    main()
