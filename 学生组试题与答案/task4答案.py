from pathlib import Path

# TODO: 导入Python库
# 要求：导入pandas库用于数据处理，导入matplotlib.pyplot库用于绘图
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LOG_CSV = ROOT / "train_log.csv"

def main():
    if not LOG_CSV.exists():
        raise FileNotFoundError(f"未找到日志文件：{LOG_CSV}，请先完成任务二训练生成日志。")
    
    # TODO: 准确率计算 - 读取训练日志数据
    # 要求：使用pandas读取CSV文件，获取epoch、准确率(val_acc)、损失值(val_loss)等数据
    log = pd.read_csv(LOG_CSV)
    
    # 创建第一个图形窗口，绘制准确率柱状图
    plt.figure()
    
    # TODO: 柱状图绘制 - 使用bar函数绘制柱状图
    # 要求：绘制Accuracy per Epoch柱状图，X轴为Epoch，Y轴为Accuracy
    plt.bar(log['epoch'], log['val_acc'], color='orange', alpha=0.7)
    
    # TODO: 设置图表标题
    # 要求：设置标题为"EAN/UPC Barcode Recognition Accuracy"
    plt.title("EAN/UPC Barcode Recognition Accuracy")
    
    # TODO: 设置坐标轴标签
    # 要求：x轴标签为"Epoch"，y轴标签为"Accuracy"
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 创建第二个图形窗口，绘制损失曲线图
    plt.figure()
    plt.plot(log['epoch'], log['val_loss'], marker='s', linewidth=2, markersize=4, color='red')
    plt.title("EAN/UPC Barcode Recognition Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # TODO: 结果展示 - 显示所有图形窗口
    # 要求：使用matplotlib显示所有创建的图形窗口
    plt.show()
    print("EAN/UPC条码识别训练结果图表已显示")

if __name__ == "__main__":
    main()




