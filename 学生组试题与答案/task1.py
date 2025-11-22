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
IMG_SIZE = (80, 80)
TEST_RATIO = 0.2
TARGET_SUFFIX = "_crop_1.jpg"

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

def plot_and_sharpen(gray_img, save_prefix: Path):
    # TODO: 图像锐化 - 绘制原图直方图
    # 要求：使用matplotlib绘制直方图，保存原图与锐化结果，需有截图
    # 提示：hist函数需要展平的图像数据，使用ravel()方法
    plt.figure()
    plt.hist(......, ......, ......)
    plt.title("Histogram (Before Sharpening)")
    plt.tight_layout()
    plt.savefig(......)  # 提示：使用f-string格式化save_prefix变量，文件名以"_hist_before.png"结尾
    plt.close()
    
    # TODO: 图像锐化 - 使用拉普拉斯算子进行锐化
    # 要求：使用cv2.Laplacian进行拉普拉斯锐化，增强边缘
    # 提示：Laplacian函数两个参数(图像, 数据类型如cv2.CV_64F)
    laplacian = cv2.......
    sharpened = np.uint8(np.clip(gray_img - 0.3 * laplacian, 0, 255))
    
    plt.figure()
    plt.hist(sharpened.ravel(), 256, (0, 256))
    plt.title("Histogram (After Laplacian)")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_hist_after.png")
    plt.close()
    return sharpened

def required_filter(img_gray):
    # TODO: 图像滤波 - 使用高斯滤波平滑图像
    # 要求：使用cv2.GaussianBlur进行高斯滤波，核大小为5x5
    # 提示：GaussianBlur函数三个参数(图像, 核大小元组, sigmaX)
    img_gray_f = cv2.......
    return img_gray_f

def extra_ops(img_gray):
    return img_gray

def process_one(img_path: str, out_img_path: Path, is_test: bool):
    img = cv2.imread(img_path)
    if img is None: return False
    
    # TODO: 图像分辨率统一 - 批量修改为80×80并保存
    # 要求：使用cv2.resize将图像调整为80×80尺寸
    # 提示：resize函数两个参数(原图像, 目标尺寸)，目标尺寸IMG_SIZE已在第12行定义为(80, 80)
    img = cv2.resize(......, ......)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = required_filter(gray)
    gray = extra_ops(gray)
    
    if is_test:
        # TODO: 图像增强 - 对test图片进行形态学闭运算
        # 要求：使用cv2.morphologyEx对测试集图片进行形态学闭运算
        # 提示：morphologyEx需要三个参数：图像、操作类型常量(闭运算用cv2.MORPH_CLOSE)、核结构
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.......
    
    save_prefix = out_img_path.with_suffix("")
    sharpened = plot_and_sharpen(gray, save_prefix)
    
    # TODO: 数据增强 - 垂直镜像操作并保存
    # 要求：使用cv2.flip进行垂直镜像操作，并按照命名规范保存
    # 提示：flip函数两个参数(图像, flipCode)，垂直翻转flipCode=0，水平翻转flipCode=1
    flip_v = cv2.flip(......, ......)
    # 提示：查看main函数中第103-104行的命名规范，分别保存为"_sharpened.png"和"_flipv.png"
    cv2.imwrite(str(out_img_path.with_name(out_img_path.stem + ......)), sharpened)
    cv2.imwrite(str(out_img_path.with_name(out_img_path.stem + ......)), flip_v)
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
                out_list.append((out_img.with_name(out_img.stem + "_sharpened.png"), label))
                out_list.append((out_img.with_name(out_img.stem + "_flipv.png"), label))
        
        with open(OUT_DIR / f"rec_gt_{split}.txt", "w", encoding="utf-8") as f:
            for pth, lab in out_list:
                f.write(f"{pth.as_posix()}\t{lab}\n")

if __name__ == "__main__":
    main()

