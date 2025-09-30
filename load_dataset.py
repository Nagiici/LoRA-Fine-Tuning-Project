"""
导入 Hugging Face 足球数据集
Dataset: ybelkada/football-dataset
"""

from datasets import load_dataset

def load_football_dataset():
    """加载足球数据集"""
    print("正在加载数据集...")

    # 加载数据集
    dataset = load_dataset("ybelkada/football-dataset", split="train")

    print(f"数据集加载完成！")
    print(f"数据集大小: {len(dataset)}")
    print(f"\n数据集特征: {dataset.features}")
    print(f"\n前3条数据示例:")
    for i in range(min(3, len(dataset))):
        print(f"\n样本 {i+1}:")
        print(dataset[i])

    return dataset

if __name__ == "__main__":
    dataset = load_football_dataset()

    # 保存为本地文件（可选）
    # dataset.save_to_disk("./football_dataset")

    # 转换为 pandas DataFrame（可选）
    # df = dataset.to_pandas()
    # df.to_csv("football_dataset.csv", index=False)