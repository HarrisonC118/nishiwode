# ==================== 1. 导入库 ====================
import os
import cv2
import torch
import pandas as pd
from tkinter import Tk, Label, Button, Frame, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from task2 import BarcodeCNN, num_classes, decode_label, validate_number, IMG_H, IMG_W, SAVE_PATH

# ==================== 2. 全局变量 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
current_image_path = None
img_label = None
result_label = None
status_label = None

# ==================== 3. 模型加载 ====================
def load_model():
    """加载训练好的条码数量识别模型"""
    global model
    model = BarcodeCNN(nclass=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("Barcode quantity recognition model loaded successfully")

# ==================== 4. 图片预处理 ====================
def preprocess_image(image_path):
    """
    预处理图片用于模型识别
    Args:
        image_path: 图片路径
    Returns:
        处理后的tensor
    """
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 调整尺寸
    img = cv2.resize(img, (IMG_W, IMG_H))
    # 归一化
    img = (img / 255.0).astype("float32")
    # 转换为tensor并添加batch和channel维度
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor

# ==================== 5. 上传图片 ====================
def upload_image():
    """打开文件对话框选择图片并显示预览"""
    global current_image_path
    
    # TODO: 上传图片函数 - 打开文件对话框选择图片
    # 要求：使用filedialog.askopenfilename打开文件选择对话框，筛选jpg/jpeg/png格式
    path = filedialog.askopenfilename(
        title="Select EAN/UPC Barcode Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if not path:
        return
    
    current_image_path = path
    
    # TODO: 显示图片预览 - 限制图片尺寸并显示在界面
    # 要求：使用PIL读取图片，限制缩略图尺寸不超过400x300，使用ImageTk.PhotoImage转换并显示
    image = Image.open(path)
    image.thumbnail((400, 300))
    photo = ImageTk.PhotoImage(image)
    img_label.config(image=photo)
    img_label.image = photo  # 保持引用
    
    # 更新状态
    status_label.config(text=f"Selected: {Path(path).name}")
    result_label.config(text="Result: Waiting...")

# ==================== 6. 单张识别 ====================
def recognize_single():
    """识别当前上传的图片"""
    global current_image_path
    
    # TODO: 图片识别函数 - 检查是否已上传图片
    # 要求：判断current_image_path是否为空，若未上传则使用messagebox.showwarning弹出警告
    if not current_image_path:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return
    
    # TODO: 调用模型进行识别
    # 要求：调用preprocess_image预处理图片，使用模型推理并解码得到识别文本
    img_tensor = preprocess_image(current_image_path)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        barcode = decode_label(predicted.item())
    
    # TODO: 显示识别结果
    # 要求：使用config方法更新result_label显示识别结果，更新status_label显示完成状态
    is_valid = validate_number(barcode)
    result_text = f"Barcode: {barcode}\nValid: {'Yes' if is_valid else 'No'}"
    result_label.config(text=result_text)
    status_label.config(text="Recognition completed")

# ==================== 7. 批量识别 ====================
def batch_recognize():
    """批量识别文件夹中的所有图片"""
    # TODO: 批量识别函数 - 选择包含图片的文件夹
    # 要求：使用filedialog.askdirectory打开文件夹选择对话框
    folder = filedialog.askdirectory(title="Select folder containing EAN/UPC barcode images")
    
    if not folder:
        return
    
    results = []
    
    # TODO: 遍历文件夹并批量识别
    # 要求：使用os.listdir遍历文件夹，筛选jpg/jpeg/png图片，逐个识别并保存结果
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, filename)
            
            # 预处理
            img_tensor = preprocess_image(image_path)
            
            # 识别
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs.data, 1)
                barcode = decode_label(predicted.item())
            
            is_valid = validate_number(barcode)
            results.append((filename, barcode, is_valid))
    
    # TODO: 保存识别结果为CSV文件
    # 要求：使用pandas的DataFrame保存结果，列名为filename、barcode和valid，保存为batch_results.csv
    output_csv = Path(folder) / "batch_results.csv"
    df = pd.DataFrame(results, columns=["filename", "barcode", "valid"])
    df.to_csv(output_csv, index=False)
    
    # TODO: 弹窗提示完成
    # 要求：使用messagebox.showinfo显示完成提示，包含处理的图片数量
    messagebox.showinfo("Complete", f"Batch recognition completed!\nResults saved to: {output_csv}")
    status_label.config(text=f"Batch completed: {len(results)} images processed")

# ==================== 8. 创建GUI界面 ====================
def create_gui():
    """创建GUI界面"""
    global img_label, result_label, status_label
    
    # 创建主窗口
    root = Tk()
    root.title("EAN/UPC Barcode Recognition System - Task D")
    root.geometry("700x550")
    
    # 标题区域
    title_frame = Frame(root, bg="#d35400", height=60)
    title_frame.pack(fill="x")
    Label(title_frame, text="EAN/UPC Barcode Recognition System", 
          font=("Arial", 18, "bold"), fg="white", bg="#d35400").pack(pady=15)
    
    # 主内容区域
    main_frame = Frame(root, padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)
    
    # 左侧：图片预览区
    left_frame = Frame(main_frame, relief="ridge", bd=2)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    Label(left_frame, text="EAN/UPC Barcode Preview", font=("Arial", 12, "bold")).pack(pady=5)
    
    # TODO: 界面功能 - 创建图片显示区
    # 要求：创建Label组件用于显示图片，设置合适的尺寸和背景色
    img_label = Label(left_frame, text="Please upload an EAN/UPC barcode image", font=("Arial", 10), bg="#fdf2e9", width=50, height=15)
    img_label.pack(padx=10, pady=10, fill="both", expand=True)
    
    # 右侧：操作区
    right_frame = Frame(main_frame, relief="ridge", bd=2, width=200)
    right_frame.pack(side="right", fill="y")
    
    Label(right_frame, text="Control Panel", font=("Arial", 12, "bold")).pack(pady=10)
    
    # TODO: 界面功能 - 创建结果显示区
    # 要求：创建Label组件用于显示识别结果，设置字体和颜色
    result_label = Label(right_frame, text="Result: Waiting...", 
                        font=("Consolas", 10), fg="#d35400", wraplength=180)
    result_label.pack(pady=10)
    
    # TODO: 界面功能 - 创建上传按钮
    # 要求：创建Button组件，绑定upload_image函数，设置文本为"Upload Barcode Image"
    upload_btn = Button(right_frame, text="Upload Barcode Image", command=upload_image,
                        bg="#e67e22", fg="white", font=("Arial", 10, "bold"), width=18, height=2)
    upload_btn.pack(pady=5)
    
    # TODO: 界面功能 - 创建识别按钮
    # 要求：创建Button组件，绑定recognize_single函数，设置文本为"Recognize Barcode"
    recognize_btn = Button(right_frame, text="Recognize Barcode", command=recognize_single,
                           bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=18, height=2)
    recognize_btn.pack(pady=5)
    
    # TODO: 界面功能 - 创建批量识别按钮
    # 要求：创建Button组件，绑定batch_recognize函数，设置文本为"Batch Recognize"
    batch_btn = Button(right_frame, text="Batch Recognize", command=batch_recognize,
                       bg="#e67e22", fg="white", font=("Arial", 10, "bold"), width=18, height=2)
    batch_btn.pack(pady=5)
    
    # 底部状态栏
    status_frame = Frame(root, bg="#34495e", height=30)
    status_frame.pack(fill="x", side="bottom")
    status_label = Label(status_frame, text="Ready", font=("Arial", 9), fg="white", bg="#34495e", anchor="w")
    status_label.pack(fill="x", padx=10)
    
    return root

# ==================== 9. 主函数 ====================
def main():
    """主函数"""
    # 加载模型
    load_model()
    
    # 创建GUI
    root = create_gui()
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()





