# BLIP-2 LoRA 微调项目

使用 LoRA 技术对 BLIP-2 模型进行图像字幕生成任务的微调。

## 项目结构

```
Lora/
├── blip2_lora_int8_fine_tune.py  # 模型微调脚本
├── blip2_lora_inference.py       # 数据集推理脚本
├── blip2_inference_custom.py     # 自定义图片推理脚本
├── load_dataset.py               # 数据集加载脚本
├── load_model.py                 # 模型加载脚本
├── model_loading_example.py      # 模型导入机制说明
├── run_example.py                # 完整运行示例
├── requirements.txt              # 项目依赖
├── output/                       # 微调模型输出目录
├── test_images/                  # 测试图片目录
└── README.md                     # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖列表：**
- `datasets` - HuggingFace 数据集库
- `transformers` - 模型加载和推理
- `torch` - PyTorch 深度学习框架
- `Pillow` - 图像处理
- `peft` - 参数高效微调（LoRA）
- `accelerate` - 模型加速
- `sentencepiece` - 分词器
- `matplotlib` - 可视化训练曲线

### 2. 运行完整示例（推荐）

```bash
python run_example.py
```

这个脚本会：
- ✓ 检查依赖是否完整
- ✓ 引导下载数据集和模型
- ✓ 显示微调和推理的完整命令

## 详细使用步骤

### 步骤 1: 准备数据集

**方式 1：自动下载 HuggingFace 数据集**
```bash
python load_dataset.py
```

**方式 2：使用自定义数据集**
- 数据格式要求：
  ```python
  {
      "image": PIL.Image,  # 图像对象
      "text": str          # 图像描述文本
  }
  ```

### 步骤 2: 模型微调

**基础用法（使用默认参数）：**
```bash
python blip2_lora_int8_fine_tune.py
```

**自定义参数：**
```bash
python blip2_lora_int8_fine_tune.py \
    --pretrain-model-path "Salesforce/blip2-opt-2.7b" \
    --train-dataset-path "ybelkada/football-dataset" \
    --output-path "./output/blip2_lora"
```

**参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrain-model-path` | 预训练模型路径或 HuggingFace ID | `Salesforce/blip2-opt-2.7b` |
| `--train-dataset-path` | 训练数据集路径或 HuggingFace ID | `ybelkada/football-dataset` |
| `--output-path` | 微调后模型的输出目录 | `./output/blip2_lora` |

**训练配置：**
- LoRA rank (r): 16
- LoRA alpha: 32
- Dropout: 0.05
- Batch size: 2
- Learning rate: 5e-5
- Epochs: 11
- 量化：Windows 自动使用 float16/float32（Linux/Mac 可用 8-bit）

**输出文件：**
- `./output/blip2_lora/` - LoRA 权重
- `pytorch_image2text_blip2_loss_curve.png` - 训练损失曲线

### 步骤 3: 模型推理

#### 方式 1: 使用 HuggingFace 数据集测试（原始方式）

```bash
python blip2_lora_inference.py
```

- 使用 `ybelkada/football-dataset` 数据集
- 测试第一张图片
- 对比原始标签和生成结果

#### 方式 2: 使用自定义图片（推荐）

```bash
python blip2_inference_custom.py
```

**支持三种推理模式：**

1. **单张图片推理**
   ```python
   # 修改脚本中的路径
   image_path = "./test_image.jpg"
   ```

2. **批量图片推理**
   ```bash
   # 创建图片文件夹
   mkdir test_images
   # 将图片放入 test_images/ 文件夹
   # 运行脚本会自动处理所有图片
   ```

3. **数据集对比测试**
   - 自动加载 HuggingFace 数据集
   - 显示原始标签 vs 生成描述

## 模型导入机制

项目支持从 HuggingFace 或本地路径导入模型：

```python
# HuggingFace 模型 ID（自动下载）
model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-opt-2.7b")

# 本地路径
model = AutoModelForVision2Seq.from_pretrained("./saved_model/blip2")

# 智能检测（推荐）
pretrain_model_path = "Salesforce/blip2-opt-2.7b"  # 可随时改为本地路径
model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path)
```

**工作原理：**
1. `from_pretrained()` 检查路径是否为本地目录
2. 如果是本地目录 → 直接加载
3. 如果不是 → 视为 HuggingFace ID，从 Hub 下载
4. 下载的模型自动缓存到 `~/.cache/huggingface/hub/`

详细说明请运行：
```bash
python model_loading_example.py
```

## 系统兼容性

### Windows 系统

✓ **已自动适配 Windows**
- 自动禁用 8-bit 量化（Windows 不支持）
- 使用 float16（GPU）或 float32（CPU）
- 无需手动配置

**符号链接警告（可忽略）：**
```bash
# 可选：设置环境变量禁用警告
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Linux/Mac 系统

✓ **支持 8-bit 量化**
- 自动检测并启用 bitsandbytes
- 显存占用更少（约 5-6GB）

## 系统要求

### 硬件要求

**GPU 训练（推荐）：**
- NVIDIA GPU：至少 8GB 显存（建议 12GB+）
- 内存：16GB+
- 存储：10GB+（模型 + 数据集）

**CPU 训练：**
- 内存：32GB+
- 训练时间：数小时（不推荐）

**推理：**
- GPU：4GB+ 显存
- CPU：8GB+ 内存

### 软件要求

- Python 3.8+
- CUDA 11.0+（GPU 训练）
- PyTorch 2.0+

## 常见问题

### Q: ImportError: To support decoding images, please install 'Pillow'
```bash
pip install Pillow
```

### Q: CUDA out of memory
**解决方案：**
1. 减小 batch size（修改脚本中的 `batch_size=2` 为 `batch_size=1`）
2. 使用更小的模型
3. 使用 CPU（训练会很慢）

### Q: bitsandbytes 错误（Windows）
**不用担心！** 脚本已自动适配 Windows，会使用 float16/float32 代替 8-bit 量化。

### Q: 模型下载失败或很慢
```bash
# 使用 HuggingFace 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
set HF_ENDPOINT=https://hf-mirror.com     # Windows
```

### Q: 如何查看训练进度？
- 训练过程会实时显示损失（Loss）
- 每 10 个 batch 显示一次生成示例
- 训练结束后生成损失曲线图

### Q: 如何使用微调后的模型？
```python
# 推理时会自动加载微调权重
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("./output/blip2_lora")
model = AutoModelForVision2Seq.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "./output/blip2_lora")
```

## 数据集说明

### 训练数据集

**默认使用：** `ybelkada/football-dataset`
- 6 张足球相关图片
- 每张图片配有描述文本
- 自动从 HuggingFace 下载

### 推理数据集

**三种选择：**
1. **HuggingFace 数据集** - 使用 `blip2_lora_inference.py`
2. **本地单张图片** - 修改 `blip2_inference_custom.py` 中的 `image_path`
3. **本地图片文件夹** - 将图片放入 `test_images/` 目录

### 自定义数据集

创建符合以下格式的数据集：
```python
from datasets import Dataset
from PIL import Image

# 准备数据
data = {
    "image": [Image.open("img1.jpg"), Image.open("img2.jpg")],
    "text": ["描述1", "描述2"]
}

# 创建数据集
dataset = Dataset.from_dict(data)
dataset.save_to_disk("./my_dataset")

# 使用自定义数据集
python blip2_lora_int8_fine_tune.py --train-dataset-path "./my_dataset"
```

## 项目技术栈

| 技术 | 说明 | 链接 |
|------|------|------|
| **BLIP-2** | Salesforce 视觉-语言预训练模型 | [论文](https://arxiv.org/abs/2301.12597) |
| **LoRA** | 低秩适应微调技术 | [论文](https://arxiv.org/abs/2106.09685) |
| **PEFT** | 参数高效微调库 | [GitHub](https://github.com/huggingface/peft) |
| **Transformers** | HuggingFace 模型库 | [文档](https://huggingface.co/docs/transformers) |
| **PyTorch** | 深度学习框架 | [官网](https://pytorch.org/) |

## 进阶使用

### 调整 LoRA 参数

编辑 [blip2_lora_int8_fine_tune.py:112-116](blip2_lora_int8_fine_tune.py#L112-116)：

```python
config = LoraConfig(
    r=16,              # LoRA rank（越大越强但显存占用越多）
    lora_alpha=32,     # LoRA alpha（通常是 r 的 2 倍）
    lora_dropout=0.05, # Dropout 率
    bias="none",       # 偏置训练策略
)
```

### 修改训练参数

编辑 [blip2_lora_int8_fine_tune.py:143-153](blip2_lora_int8_fine_tune.py#L143-153)：

```python
# 批次大小（显存不足时减小）
batch_size=2

# 学习率
lr=5e-5

# 训练轮数
for epoch in range(11):  # 修改为你想要的轮数
```

### 保存完整模型（而非 LoRA 权重）

```python
# 在训练结束后添加
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./output/full_model")
```

## 许可证

本项目基于 MIT 许可证。使用的模型和数据集请遵循其各自的许可证。

## 参考资源

- [BLIP-2 模型卡片](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [LoRA 技术详解](https://huggingface.co/blog/lora)
- [Transformers 教程](https://huggingface.co/docs/transformers/training)
