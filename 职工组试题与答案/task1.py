# TODO: 导入Python库
# 要求：导入OpenCV、numpy、matplotlib等库用于图像处理
import os, cv2, random
from pathlib import Path
from typing import List, Tuple
import ...... as np  # 提示：numpy库，用于数值计算
import ...... as plt  # 提示：matplotlib.pyplot库，用于绘图

ROOT = Path(__file__).resolve().parent
REC_GT = ROOT / "rec_gt.txt"
OUT_DIR = ROOT / "processed"
IMG_SIZE = (96, 96)
TEST_RATIO = 0.2
TARGET_SUFFIX = "_crop_2.jpg"

def read_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            img_path, label = line.split("\t", 1)
            if img_path.endswith(TARGET_SUFFIX):
                pairs.append((img_path, label))
    return pairs

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_and_denoise(gray_img, save_prefix: Path):
    # TODO: 图像去噪 - 绘制原图直方图
    # 要求：使用matplotlib绘制直方图，保存原图与去噪结果
    plt.figure()
    plt.hist(......, ......, ......)
    plt.title("Histogram (Before Denoising)")
    plt.tight_layout()
    plt.savefig(......) # 提示：以"_hist_before.png"结尾
    plt.close()
    
    # TODO: 图像去噪 - 使用双边滤波进行保边去噪
    # 要求：使用cv2.bilateralFilter进行双边滤波，保持边缘清晰
    denoised = cv2.......
    
    plt.figure()
    plt.hist(denoised.ravel(), 256, (0, 256))
    plt.title("Histogram (After Bilateral)")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_hist_after.png")
    plt.close()
    return denoised

def required_filter(img_gray):
    # TODO: 图像滤波 - 使用均值滤波平滑图像
    # 要求：均值滤波，核大小为3x3
    img_gray_f = cv2.......
    return img_gray_f

def extra_ops(img_gray):
    return img_gray

def process_one(img_path: str, out_img_path: Path, is_test: bool):
    img = cv2.imread(img_path)
    if img is None: return False
    
    # TODO: 图像分辨率统一 - 批量修改为96×96并保存
    # 要求：将图像调整为96×96尺寸
    img = cv2.resize(......, ......)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = required_filter(gray)
    gray = extra_ops(gray)
    
    if is_test:
        # TODO: 图像增强 - 对test图片进行形态学开运算
        # 要求：对测试集图片进行形态学开运算
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.......
    
    save_prefix = out_img_path.with_suffix("")
    denoised = plot_and_denoise(gray, save_prefix)
    
    # TODO: 数据增强 - 旋转操作并保存
    # 要求：进行15度旋转
    h, w = denoised.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.......
    rotated = cv2.......
    # 提示：文件名以"_denoised.png"和"_rotated.png"结尾
    cv2.imwrite(str(out_img_path.with_name(out_img_path.stem + ......)), denoised)
    cv2.imwrite(str(out_img_path.with_name(out_img_path.stem + ......)), rotated)
    return True

def main():
    pairs = read_pairs(REC_GT)
    random.shuffle(pairs)
    n_test = int(len(pairs) * TEST_RATIO)
    test_pairs, train_pairs = pairs[:n_test], pairs[n_test:]
    
    for split, data in [("train", train_pairs), ("test", test_pairs)]:
        out_dir = OUT_DIR / split
        ensure_dir(out_dir)
        out_list = []
        
        for ip, label in data:
            out_img = out_dir / f"{Path(ip).stem}"
            if process_one(ip, out_img.with_suffix(".png"), split == "test"):
                out_list.append((out_img.with_name(out_img.stem + "_denoised.png"), label))
                out_list.append((out_img.with_name(out_img.stem + "_rotated.png"), label))
        
        with open(OUT_DIR / f"rec_gt_{split}.txt", "w", encoding="utf-8") as f:
            for pth, lab in out_list:
                f.write(f"{pth.as_posix()}\t{lab}\n")

if __name__ == "__main__":
    main()
