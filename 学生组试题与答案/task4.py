from pathlib import Path

# TODO: 导入Python库
# 要求：导入pandas库用于数据处理，导入matplotlib.pyplot库用于绘图
# 提示：pandas常用于CSV/Excel数据处理，matplotlib.pyplot常用别名plt
import ...... as pd  # 提示：pandas库，用于数据处理
import ...... as plt  # 提示：matplotlib.pyplot库，用于绘图

ROOT = Path(__file__).resolve().parent
LOG_CSV = ROOT / "train_log.csv"

def main():
    if not LOG_CSV.exists():
        raise FileNotFoundError(f"未找到日志文件：{LOG_CSV}，请先完成任务二训练生成日志。")
    
    # TODO: 准确率计算 - 读取训练日志数据
    # 要求：使用pandas读取CSV文件，获取epoch、准确率(val_acc)、损失值(val_loss)等数据
    # 提示：pandas中用于读取CSV文件的函数
    log = pd.......  # 提示：读取CSV文件的函数
    
    # 创建第一个图形窗口，绘制准确率柱状图
    plt.figure()
    
    # TODO: 柱状图绘制 - 使用bar函数绘制柱状图
    # 要求：绘制Accuracy per Epoch柱状图，X轴为Epoch，Y轴为Accuracy
    # 提示：matplotlib中绘制柱状图的函数，需要x轴和y轴数据
    plt.......  # 提示：绘制柱状图的函数，参数为(log['epoch'], log['val_acc'])
    
    # TODO: 设置图表标题
    # 要求：设置标题为"EAN/UPC Barcode Recognition Accuracy"
    # 提示：matplotlib中设置图表标题的函数
    plt.......  # 提示：设置标题的函数
    
    # TODO: 设置坐标轴标签
    # 要求：x轴标签为"Epoch"，y轴标签为"Accuracy"
    # 提示：matplotlib中设置坐标轴标签的函数
    plt.......  # 提示：设置x轴标签的函数
    plt.......  # 提示：设置y轴标签的函数
    
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
    # 提示：matplotlib中用于显示图形的函数，通常以show开头
    plt.......  # 提示：显示图形的函数
    print("EAN/UPC条码识别训练结果图表已显示")

if __name__ == "__main__":
    main()

