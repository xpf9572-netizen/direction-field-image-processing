"""
相干增强扩散 (Coherence-Enhanced Diffusion, CED) — 自适应感知终极版 v4
=========================================================================
物理思想：
  高斯噪声（加性）和散斑噪声（乘性）的物理本质水火不容：
    加性高斯：方差恒定，与均值无关，Var ≈ const
    乘性散斑：方差 ∝ 均值²，Var ≈ σ_s² · μ²

  强行用同一套参数处理两种噪声，必然顾此失彼。

  终极方案：赋予 PDE "感知物理场" 的能力——
    1. 噪声类型感知器：统计局部 Var 与 μ² 的相关性，自动鉴别加性/乘性场
    2. 双态演化分流：高斯路径关闭同态映射，散斑路径开启同态映射 + 大ρ
    3. Weickert 动态扩散率：λ₁=α, λ₂=α+(1-α)·exp(-C/k)
       强噪声区退化为各向同性热传导，条纹清晰区高度非等向扩散

底层保留：
  - Neumann 零通量边界条件（幽灵像元 + 统一中心差分）
  - Numba JIT 加速核心迭代
  - 守恒型半点通量差分格式
"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
import matplotlib.pyplot as plt
import time

try:
    from numba import jit as _jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def _jit(nopython=True, fastmath=False, cache=False):
        def decorator(func):
            return func
        return decorator


# ================================================================
# PDE 求解器 — Numba JIT 加速版 / NumPy 降级版
# ================================================================

@_jit(nopython=True, fastmath=True, cache=True)
def _pde_step_numba(u, D11, D12, D22, dt):
    H, W = u.shape
    u_pad = np.empty((H + 2, W + 2), dtype=np.float64)
    u_pad[1:H+1, 1:W+1] = u
    u_pad[0, 1:W+1] = u[0, :];    u_pad[H+1, 1:W+1] = u[H-1, :]
    u_pad[1:H+1, 0] = u[:, 0];    u_pad[1:H+1, W+1] = u[:, W-1]
    u_pad[0, 0] = u[0, 0];         u_pad[0, W+1] = u[0, W-1]
    u_pad[H+1, 0] = u[H-1, 0];    u_pad[H+1, W+1] = u[H-1, W-1]

    D11_pad = np.empty((H + 2, W + 2), dtype=np.float64)
    D11_pad[1:H+1, 1:W+1] = D11
    D11_pad[0, 1:W+1] = D11[0, :];   D11_pad[H+1, 1:W+1] = D11[H-1, :]
    D11_pad[1:H+1, 0] = D11[:, 0];    D11_pad[1:H+1, W+1] = D11[:, W-1]
    D11_pad[0, 0] = D11[0, 0];        D11_pad[0, W+1] = D11[0, W-1]
    D11_pad[H+1, 0] = D11[H-1, 0];   D11_pad[H+1, W+1] = D11[H-1, W-1]

    D12_pad = np.empty((H + 2, W + 2), dtype=np.float64)
    D12_pad[1:H+1, 1:W+1] = D12
    D12_pad[0, 1:W+1] = D12[0, :];   D12_pad[H+1, 1:W+1] = D12[H-1, :]
    D12_pad[1:H+1, 0] = D12[:, 0];    D12_pad[1:H+1, W+1] = D12[:, W-1]
    D12_pad[0, 0] = D12[0, 0];        D12_pad[0, W+1] = D12[0, W-1]
    D12_pad[H+1, 0] = D12[H-1, 0];   D12_pad[H+1, W+1] = D12[H-1, W-1]

    D22_pad = np.empty((H + 2, W + 2), dtype=np.float64)
    D22_pad[1:H+1, 1:W+1] = D22
    D22_pad[0, 1:W+1] = D22[0, :];   D22_pad[H+1, 1:W+1] = D22[H-1, :]
    D22_pad[1:H+1, 0] = D22[:, 0];    D22_pad[1:H+1, W+1] = D22[:, W-1]
    D22_pad[0, 0] = D22[0, 0];        D22_pad[0, W+1] = D22[0, W-1]
    D22_pad[H+1, 0] = D22[H-1, 0];   D22_pad[H+1, W+1] = D22[H-1, W-1]

    u_new = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        ip = i + 1
        for j in range(W):
            jp = j + 1
            D11_r = 0.5 * (D11_pad[ip, jp] + D11_pad[ip, jp+1])
            D12_r = 0.5 * (D12_pad[ip, jp] + D12_pad[ip, jp+1])
            ux_r = u_pad[ip, jp+1] - u_pad[ip, jp]
            uy_r = 0.25 * (u_pad[ip+1, jp] - u_pad[ip-1, jp] + u_pad[ip+1, jp+1] - u_pad[ip-1, jp+1])
            Fx_r = D11_r * ux_r + D12_r * uy_r

            D11_l = 0.5 * (D11_pad[ip, jp-1] + D11_pad[ip, jp])
            D12_l = 0.5 * (D12_pad[ip, jp-1] + D12_pad[ip, jp])
            ux_l = u_pad[ip, jp] - u_pad[ip, jp-1]
            uy_l = 0.25 * (u_pad[ip+1, jp-1] - u_pad[ip-1, jp-1] + u_pad[ip+1, jp] - u_pad[ip-1, jp])
            Fx_l = D11_l * ux_l + D12_l * uy_l

            D12_d = 0.5 * (D12_pad[ip, jp] + D12_pad[ip+1, jp])
            D22_d = 0.5 * (D22_pad[ip, jp] + D22_pad[ip+1, jp])
            uy_d = u_pad[ip+1, jp] - u_pad[ip, jp]
            ux_d = 0.25 * (u_pad[ip, jp+1] - u_pad[ip, jp-1] + u_pad[ip+1, jp+1] - u_pad[ip+1, jp-1])
            Fy_d = D12_d * ux_d + D22_d * uy_d

            D12_u = 0.5 * (D12_pad[ip-1, jp] + D12_pad[ip, jp])
            D22_u = 0.5 * (D22_pad[ip-1, jp] + D22_pad[ip, jp])
            uy_u = u_pad[ip, jp] - u_pad[ip-1, jp]
            ux_u = 0.25 * (u_pad[ip-1, jp+1] - u_pad[ip-1, jp-1] + u_pad[ip, jp+1] - u_pad[ip, jp-1])
            Fy_u = D12_u * ux_u + D22_u * uy_u

            div_F = (Fx_r - Fx_l) + (Fy_d - Fy_u)
            u_new[i, j] = u[i, j] + dt * div_F
    return u_new


def _pde_step_numpy(u, D11, D12, D22, dt):
    H, W = u.shape
    u_pad = np.pad(u, pad_width=1, mode='edge')
    D11_pad = np.pad(D11, pad_width=1, mode='edge')
    D12_pad = np.pad(D12, pad_width=1, mode='edge')
    D22_pad = np.pad(D22, pad_width=1, mode='edge')

    D11_e = 0.5 * (D11_pad[1:H+1, 1:W+1] + D11_pad[1:H+1, 2:W+2])
    D12_e = 0.5 * (D12_pad[1:H+1, 1:W+1] + D12_pad[1:H+1, 2:W+2])
    D11_w = 0.5 * (D11_pad[1:H+1, 0:W] + D11_pad[1:H+1, 1:W+1])
    D12_w = 0.5 * (D12_pad[1:H+1, 0:W] + D12_pad[1:H+1, 1:W+1])
    D12_s = 0.5 * (D12_pad[1:H+1, 1:W+1] + D12_pad[2:H+2, 1:W+1])
    D22_s = 0.5 * (D22_pad[1:H+1, 1:W+1] + D22_pad[2:H+2, 1:W+1])
    D12_n = 0.5 * (D12_pad[0:H, 1:W+1] + D12_pad[1:H+1, 1:W+1])
    D22_n = 0.5 * (D22_pad[0:H, 1:W+1] + D22_pad[1:H+1, 1:W+1])

    ux_e = u_pad[1:H+1, 2:W+2] - u_pad[1:H+1, 1:W+1]
    uy_e = 0.25 * (u_pad[2:H+2, 2:W+2] - u_pad[0:H, 2:W+2] + u_pad[2:H+2, 1:W+1] - u_pad[0:H, 1:W+1])
    ux_w = u_pad[1:H+1, 1:W+1] - u_pad[1:H+1, 0:W]
    uy_w = 0.25 * (u_pad[2:H+2, 1:W+1] - u_pad[0:H, 1:W+1] + u_pad[2:H+2, 0:W] - u_pad[0:H, 0:W])
    uy_s = u_pad[2:H+2, 1:W+1] - u_pad[1:H+1, 1:W+1]
    ux_s = 0.25 * (u_pad[2:H+2, 2:W+2] - u_pad[2:H+2, 0:W] + u_pad[1:H+1, 2:W+2] - u_pad[1:H+1, 0:W])
    uy_n = u_pad[1:H+1, 1:W+1] - u_pad[0:H, 1:W+1]
    ux_n = 0.25 * (u_pad[0:H, 2:W+2] - u_pad[0:H, 0:W] + u_pad[1:H+1, 2:W+2] - u_pad[1:H+1, 0:W])

    Fx_e = D11_e * ux_e + D12_e * uy_e
    Fx_w = D11_w * ux_w + D12_w * uy_w
    Fy_s = D12_s * ux_s + D22_s * uy_s
    Fy_n = D12_n * ux_n + D22_n * uy_n

    div_F = (Fx_e - Fx_w) + (Fy_s - Fy_n)
    return u + dt * div_F


_pde_step = _pde_step_numba if HAS_NUMBA else _pde_step_numpy


# ================================================================
# 1. 噪声类型物理感知器 (Noise Type Sensor)
# ================================================================

def classify_noise_type(image, block_size=32, threshold=0.3):
    """
    物理场统计特性鉴别器 — 自动判断加性/乘性噪声

    物理原理：
        加性高斯噪声：局部方差与局部均值无关，Var ≈ σ² (常数)
        乘性散斑噪声：局部方差与局部均值的平方成正比，Var ≈ σ_s² · μ²

        通过在无重叠宏观图块上计算局部均值 μ 和局部方差 Var，
        然后拟合 Var = a + b · μ² 的线性回归斜率 b：
          - b ≈ 0 且 Var 接近常数 → 加性高斯噪声
          - b 显著大于 0 → 乘性散斑噪声

        判别指标：用 Pearson 相关系数 corr(Var, μ²) 衡量线性关联程度，
        corr > threshold 判定为乘性噪声场。

    参数：
        image:       输入图像 [0,1]
        block_size:  宏观图块大小（默认 32）
        threshold:   相关系数阈值（默认 0.3，>0.3 判定为乘性）

    返回：
        is_multiplicative: 布尔值，True=乘性散斑，False=加性高斯
        slope:            Var~μ² 回归斜率（物理指标）
        correlation:      Pearson相关系数
    """
    H, W = image.shape
    u = image.astype(np.float64)

    mu_list = []
    var_list = []

    # 在无重叠宏观图块上计算局部统计量
    # 物理意义：图块足够大（32×32）以包含多个散斑周期，
    # 使得局部统计量能反映噪声的物理本质
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = u[i:i+block_size, j:j+block_size]
            block_mean = np.mean(block)
            block_var = np.var(block)
            # 排除极端均值块（纯黑或纯白区域统计量不可靠）
            if 0.05 < block_mean < 0.95:
                mu_list.append(block_mean)
                var_list.append(block_var)

    mu_arr = np.array(mu_list)
    var_arr = np.array(var_list)
    mu_sq_arr = mu_arr ** 2

    # Pearson 相关系数 corr(Var, μ²)
    # 物理意义：加性噪声下 Var 与 μ² 无关 → corr ≈ 0
    #           乘性噪声下 Var ∝ μ² → corr >> 0
    if len(mu_list) < 5:
        return False, 0.0, 0.0

    correlation = np.corrcoef(var_arr, mu_sq_arr)[0, 1]

    # 线性回归 Var = a + b · μ²，物理含义：b 是等效乘性噪声系数
    A = np.column_stack([np.ones_like(mu_sq_arr), mu_sq_arr])
    result = np.linalg.lstsq(A, var_arr, rcond=None)
    slope = result[0][1]

    is_multiplicative = correlation > threshold

    label = 'speckle(multiplicative)' if is_multiplicative else 'gaussian(additive)'
    print(f"  [Noise Sensor] Var~mu2 slope={slope:.6f} | corr={correlation:.4f} | verdict={label} (thresh={threshold})")

    return is_multiplicative, slope, correlation


# ================================================================
# 结构张量计算 — 解耦尺度参数 + 中值滤波预处理
# ================================================================

def compute_structure_tensor(image, sigma_grad, rho, median_window=3):
    """
    计算结构张量 — 含中值滤波预处理

    v4 新增：在计算梯度之前对对数域信号执行中值滤波，
    物理剔除极值孤立散斑点（散斑噪声的盐椒特征），
    保护梯度方向估计不受极值散斑扰乱。
    """
    image_filtered = median_filter(image, size=median_window, mode='nearest')

    Ix = gaussian_filter(image_filtered, sigma=sigma_grad, order=(0, 1), mode='nearest')
    Iy = gaussian_filter(image_filtered, sigma=sigma_grad, order=(1, 0), mode='nearest')

    J11_raw = Ix ** 2
    J12_raw = Ix * Iy
    J22_raw = Iy ** 2

    J11 = gaussian_filter(J11_raw, sigma=rho, mode='nearest')
    J12 = gaussian_filter(J12_raw, sigma=rho, mode='nearest')
    J22 = gaussian_filter(J22_raw, sigma=rho, mode='nearest')

    return J11, J12, J22


# ================================================================
# 3. Weickert 相干性动态扩散张量
# ================================================================

def build_diffusion_tensor(J11, J12, J22, alpha=0.01, C=1e-8):
    """
    构建扩散张量 D — Weickert 相干性动态特征值版

    物理意义：
        废弃固定 λ₁/λ₂ 配置，引入 Weickert 相干性度量 k：
          k = (J₁₁ - J₂₂)² + 4·J₁₂²    （结构张量的不变量）

        动态特征值：
          λ₁ = α                         （法线方向：背景扩散率，各向同性退火）
          λ₂ = α + (1 - α)·exp(-C / (k + ε))  （切线方向：随相干度自适应）

        物理解释：
          - 散斑破损区（k≈0）：λ₂ ≈ α，退化为各向同性热传导 ≈ 热方程
            对纯噪声区域做均匀平滑，不会沿错误方向产生伪条纹
          - 条纹清晰区（k≫0）：λ₂ → 1，高度非等向扩散
            沿条纹切线方向充分平滑噪声，法线方向几乎不扩散
    """
    trace = J11 + J22
    diff = np.sqrt(np.maximum((J11 - J22) ** 2 + 4 * J12 ** 2, 0))
    mu1 = (trace + diff) / 2
    mu2 = (trace - diff) / 2
    denom = np.maximum(mu1 - mu2, 1e-10)

    k = (J11 - J22) ** 2 + 4 * J12 ** 2

    lambda1 = alpha
    lambda2 = alpha + (1 - alpha) * np.exp(-C / (k + 1e-10))

    D11 = lambda2 + (lambda1 - lambda2) * (J11 - mu2) / denom
    D12 = (lambda1 - lambda2) * J12 / denom
    D22 = lambda2 + (lambda1 - lambda2) * (J22 - mu2) / denom

    return D11, D12, D22


# ================================================================
# Lee 滤波器（保留作为可选前处理）
# ================================================================

def lee_filter(image, window_size=7, noise_var=None):
    """Lee 滤波器 — 乘性散斑噪声的 MMSE 估计器"""
    u = image.astype(np.float64)
    local_mean = uniform_filter(u, size=window_size, mode='nearest')
    local_mean_sq = uniform_filter(u ** 2, size=window_size, mode='nearest')
    local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)

    if noise_var is None:
        flat_var = local_var.ravel()
        threshold = np.percentile(flat_var, 5)
        homogeneous_mask = local_var <= threshold
        noise_var = np.mean(local_var[homogeneous_mask]) if np.any(homogeneous_mask) else 0.01

    k = np.clip((local_var - noise_var) / (local_var + 1e-10), 0, 1)
    filtered = local_mean + k * (u - local_mean)
    return np.clip(filtered, 0, 1)


# ================================================================
# 主函数 — 自适应感知双态演化 CED
# ================================================================

def ced_denoise(image, sigma_grad=1.0, rho=8.0,
                num_iter=100, dt=0.2,
                alpha=0.01, C=1e-6,
                use_homomorphic=None, median_window=3,
                force_noise_type=None, noise_threshold=0.3,
                block_size=32,
                pre_smooth_sigma=1.5,
                use_lee_prefilter=True,
                fringe_mean_prior=0.5):
    """
    相干增强扩散 (CED) 主函数 — 自适应感知双态演化版 v5

    v5 关键改进（散斑路径）：
      - 预平滑方向场：高斯预平滑后估计结构张量，避免沿噪声伪结构扩散
      - log(I+eps) 替代 log(1+I)：恢复完整动态范围（~19x）
      - 乘法均值校正替代加法：利用条纹图均值=0.5先验
      - Lee 滤波预前处理：MMSE 初始去噪

    参数接口：
        image:              输入含噪图像 [0,1]
        sigma_grad:         梯度预平滑尺度 σ
        rho:                结构张量积分尺度 ρ
        num_iter:           迭代步数
        dt:                 时间步长
        alpha:              Weickert 背景扩散率
        C:                  Weickert 相干性灵敏度
        use_homomorphic:    同态映射开关（None=自动）
        median_window:      中值滤波窗口
        force_noise_type:   强制噪声类型（None=自动）
        noise_threshold:    感知器阈值
        block_size:         感知器图块大小
        pre_smooth_sigma:   散斑路径预平滑 sigma（默认 2.0）
        use_lee_prefilter:  散斑路径是否启用 Lee 预滤波（默认 True）
        fringe_mean_prior:  条纹图干净均值先验（默认 0.5）

    返回：
        u: 去噪后的图像 [0,1]
    """
    image = image.astype(np.float64)

    # ============================================================
    # Step 0: 噪声类型物理感知
    # ============================================================
    if force_noise_type is not None:
        is_multiplicative = (force_noise_type == 'speckle')
        slope, correlation = 0.0, 0.0
        print(f"  [噪声感知] 强制模式: {force_noise_type}")
    else:
        is_multiplicative, slope, correlation = classify_noise_type(
            image, block_size=block_size, threshold=noise_threshold)

    M_in = np.mean(image)

    # ============================================================
    # Step 1: 双态演化分流
    # ============================================================
    if is_multiplicative:
        # --- 散斑路径 v5 ---
        if use_homomorphic is None:
            use_homomorphic = True
        rho_actual = max(rho, 8.0)
        median_window_actual = median_window
        num_iter_actual = max(num_iter, 80)

        # v5: Lee 滤波预前处理
        if use_lee_prefilter:
            image_pre = lee_filter(image)
            print(f"  [Lee 预滤波] 窗口=7 | 均值: {np.mean(image_pre):.4f}")
        else:
            image_pre = image.copy()

        # v5: 预平滑获取可靠方向场
        image_presmooth = gaussian_filter(image_pre, sigma=pre_smooth_sigma)

        # v5: 正确的同态映射 log(I+eps)
        eps = 1e-6
        u_input = np.log(np.clip(image_pre, eps, 1.0))

        # v5: 在预平滑图像的对数域计算固定扩散张量
        u_presmooth_log = np.log(np.clip(image_presmooth, eps, 1.0))
        J11, J12, J22 = compute_structure_tensor(
            u_presmooth_log, sigma_grad, rho_actual, median_window=median_window_actual)
        D11, D12, D22 = build_diffusion_tensor(J11, J12, J22, alpha=alpha, C=C)

        print(f"  [散斑路径 v5] ρ={rho_actual:.1f}, iter={num_iter_actual}, "
              f"pre_smooth_σ={pre_smooth_sigma}, Lee={'ON' if use_lee_prefilter else 'OFF'}")
        print(f"  [同态映射] log(I+eps) | 范围: [{u_input.min():.2f}, {u_input.max():.2f}]")

        # CED 迭代（固定扩散张量）
        u = u_input.copy()
        engine_name = "Numba JIT" if HAS_NUMBA else "NumPy"
        print(f"  [引擎: {engine_name}] CED 迭代 | σ={sigma_grad}, ρ={rho_actual}, α={alpha}, C={C}")
        t_start = time.time()

        for n in range(num_iter_actual):
            u = _pde_step(u, D11, D12, D22, dt)
            if (n + 1) % 20 == 0:
                elapsed = time.time() - t_start
                print(f"  迭代 {n+1}/{num_iter_actual} | 范围: [{u.min():.2f}, {u.max():.2f}] | 用时: {elapsed:.1f}s")

        total_time = time.time() - t_start
        print(f"  CED 完成 | 总用时: {total_time:.1f}s")

        # 逆映射
        u = np.exp(u)

        # v5: 乘法均值校正（条纹图先验 mean=0.5）
        M_out = np.mean(u)
        correction = fringe_mean_prior / M_out
        u = u * correction
        print(f"  [均值校正] 乘法 | M_out={M_out:.4f}, 校正因子={correction:.4f}")
        print(f"  [输出] 范围: [{u.min():.4f}, {u.max():.4f}], 均值: {np.mean(u):.4f}")

    else:
        # --- 高斯路径（保持 v4 最优参数）---
        if use_homomorphic is None:
            use_homomorphic = False
        rho_actual = min(rho, 2.0)
        median_window_actual = 1
        num_iter_actual = min(num_iter, 50)
        sigma_grad_actual = min(sigma_grad, 1.0)
        C_actual = min(C, 1e-8)
        dt_actual = max(dt, 0.25)
        print(f"  [高斯路径] ρ={rho_actual:.1f}, σ={sigma_grad_actual}, C={C_actual}, "
              f"dt={dt_actual}, 同态映射={'ON' if use_homomorphic else 'OFF'}")

        u_input = image.copy()

        u = u_input.copy()
        engine_name = "Numba JIT" if HAS_NUMBA else "NumPy"
        print(f"  [引擎: {engine_name}] CED 迭代 | σ={sigma_grad_actual}, ρ={rho_actual}, α={alpha}, C={C_actual}")
        t_start = time.time()

        for n in range(num_iter_actual):
            J11, J12, J22 = compute_structure_tensor(u, sigma_grad_actual, rho_actual, median_window=median_window_actual)
            D11, D12, D22 = build_diffusion_tensor(J11, J12, J22, alpha=alpha, C=C_actual)
            u = _pde_step(u, D11, D12, D22, dt_actual)
            if (n + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  迭代 {n+1}/{num_iter_actual} | 范围: [{u.min():.2f}, {u.max():.2f}] | 用时: {elapsed:.1f}s")

        total_time = time.time() - t_start
        print(f"  CED 完成 | 总用时: {total_time:.1f}s")

        if use_homomorphic:
            u = np.exp(u) - 1.0
            M_out = np.mean(u)
            u = u + (M_in - M_out)
            print(f"  [同态映射] exp(u)-1 + 加法均值平移 | M_in={M_in:.4f}, M_out={M_out:.4f}")

    return np.clip(u, 0, 1)


# ================================================================
# 主程序：批量处理 1Den 文件夹
# ================================================================
if __name__ == "__main__":
    import os
    import glob
    from matplotlib.image import imread

    np.random.seed(42)

    # ==================== 路径配置 ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "数据（待处理数据）", "1Den")
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== 参数配置 ====================
    sigma_grad = 1.0
    rho = 8.0
    num_iter = 100
    dt = 0.2
    alpha = 0.01
    C = 1e-6
    use_homomorphic = None
    median_window = 3
    force_noise_type = None
    noise_threshold = 0.3
    block_size = 32
    pre_smooth_sigma = 1.5
    use_lee_prefilter = True
    fringe_mean_prior = 0.5
    # =================================================

    # 收集输入文件（01.png ~ 10.png）
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not input_files:
        print(f"错误：在 {INPUT_DIR} 中未找到图片")
        exit(1)

    print("=" * 60)
    print("CED v5 — 批量去噪")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"待处理: {len(input_files)} 张图片")
    print(f"参数: σ={sigma_grad}, ρ={rho}, iter={num_iter}, α={alpha}, C={C}")
    print(f"PDE 求解器: {'Numba JIT' if HAS_NUMBA else 'NumPy'}")
    print()

    total_start = time.time()

    for idx, input_path in enumerate(input_files, 1):
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_name = f"1Den_{basename}.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # 读取并归一化
        img = imread(input_path)
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        img = img.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min() + 1e-10)

        print(f"[{idx}/{len(input_files)}] {os.path.basename(input_path)} → {output_name}")

        # CED 去噪
        denoised = ced_denoise(
            img,
            sigma_grad=sigma_grad,
            rho=rho,
            num_iter=num_iter,
            dt=dt,
            alpha=alpha,
            C=C,
            use_homomorphic=use_homomorphic,
            median_window=median_window,
            force_noise_type=force_noise_type,
            noise_threshold=noise_threshold,
            block_size=block_size,
            pre_smooth_sigma=pre_smooth_sigma,
            use_lee_prefilter=use_lee_prefilter,
            fringe_mean_prior=fringe_mean_prior,
        )
        denoised = np.clip(denoised, 0, 1)

        # 保存为 uint8 灰度 PNG
        plt.imsave(output_path, denoised, cmap='gray')
        print(f"  已保存: {output_path}")
        print()

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"全部完成 | 共 {len(input_files)} 张 | 总用时: {total_time:.1f}s")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)