import cv2
import numpy as np
import os

def calculate_mae(denoised_img, ideal_img):
    """
    计算两张图像之间的平均绝对误差 (MAE)。
    """
    # 检查图像尺寸是否一致，这是非常关键的物理边界条件
    if denoised_img.shape != ideal_img.shape:
        raise ValueError(f"图像尺寸不匹配! 预测图: {denoised_img.shape}, 真值图: {ideal_img.shape}")
    
    # 【致命细节提示】必须先将图像转换为浮点数 (float32)
    # 如果直接用 8 位无符号整数 (uint8) 相减，当出现负数时会发生截断或溢出（例如 10 - 20 = 246）
    f = denoised_img.astype(np.float32)
    g = ideal_img.astype(np.float32)
    
    # 计算绝对误差矩阵，并求整个矩阵的平均值
    mae = np.mean(np.abs(f - g))
    return mae

def evaluate_batch(denoised_dir, ideal_dir, num_images=10):
    """
    批量计算多张图像的 MAE 并求平均分。
    假设命名规则为: 
    预测图: 1Den_01.png 到 1Den_10.png
    真值图: 01_ideal.png 到 10_ideal.png (需根据你本地实际真值图名字修改)
    """
    total_mae = 0.0
    valid_count = 0
    
    print("开始计算评测指标 MAE...")
    print("-" * 30)
    
    for i in range(1, num_images + 1):
        # 构造文件名 (请根据实际情况调整命名规则)
        denoised_name = f"1Den_{i:02d}.png"
        ideal_name = f"{i:02d}_ideal.png" # 假设理想图像以此命名
        
        denoised_path = os.path.join(denoised_dir, denoised_name)
        ideal_path = os.path.join(ideal_dir, ideal_name)
        
        if not os.path.exists(denoised_path) or not os.path.exists(ideal_path):
            print(f"警告: 找不到文件 {denoised_name} 或其对应的真值图。跳过。")
            continue
            
        # 以灰度模式读取图像
        denoised_img = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)
        ideal_img = cv2.imread(ideal_path, cv2.IMREAD_GRAYSCALE)
        
        # 计算单张 MAE
        current_mae = calculate_mae(denoised_img, ideal_img)
        total_mae += current_mae
        valid_count += 1
        
        print(f"图像 {denoised_name} 的 MAE: {current_mae:.4f}")
        
    print("-" * 30)
    if valid_count > 0:
        final_score = total_mae / valid_count
        print(f"【最终得分】 {valid_count} 张图像的平均 MAE 为: {final_score:.4f}")
    else:
        print("未找到有效图像进行计算。")

# ================= 运行测试 =================
if __name__ == "__main__":
    # 单张图测试：结果图 vs 原图（使用相对路径避免编码问题）
    result_path = "image2_example_ced.png"
    ideal_path = "label_example.png"
    
    test_denoised = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
    test_ideal = cv2.imread(ideal_path, cv2.IMREAD_GRAYSCALE)
    
    if test_denoised is not None and test_ideal is not None:
        mae = calculate_mae(test_denoised, test_ideal)
        print(f"结果图: {result_path}")
        print(f"原  图: {ideal_path}")
        print(f"MAE: {mae:.4f} (越低越好)")
    else:
        print("错误: 无法读取图像文件，请检查路径是否正确。")
    
    # 批量测试接口预留（将文件夹路径替换为你本地的实际路径）
    # evaluate_batch(denoised_dir="./results", ideal_dir="./dataset/ideal")
    pass