
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
from PIL import Image
import os

print("正在加载模型...")

# 修改为你的 LoRA 模型输出路径
peft_model_id = "./output/blip2_lora"

# 1. 加载 LoRA 配置
config = PeftConfig.from_pretrained(peft_model_id)
processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)

# 2. 加载基础模型（Windows 不使用 8-bit 量化）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

if device == "cuda":
    print("使用 float16 加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16
    )
else:
    print("使用 float32 加载模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        config.base_model_name_or_path
    )

model.to(device)

# 3. 加载 LoRA 权重
print("加载 LoRA 权重...")
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()

print("模型加载完成！\n")

# 使用 HuggingFace 数据集 ID
print("加载数据集...")
dataset = load_dataset("ybelkada/football-dataset", split="train")

item = dataset[2]
print(f"数据集加载完成，共 {len(dataset)} 条数据\n")


encoding = processor(images=item["image"], padding="max_length", return_tensors="pt")
# remove batch dimension
encoding = {k: v.squeeze() for k, v in encoding.items()}
encoding["text"] = item["text"]

print(encoding.keys())

processed_batch = {}
for key in encoding.keys():
    if key != "text":
        processed_batch[key] = torch.stack([example[key] for example in [encoding]])
    else:
        text_inputs = processor.tokenizer(
            [example["text"] for example in [encoding]], padding=True, return_tensors="pt"
        )
        processed_batch["input_ids"] = text_inputs["input_ids"]
        processed_batch["attention_mask"] = text_inputs["attention_mask"]


# 根据设备选择数据类型
dtype = torch.float16 if device == "cuda" else torch.float32
pixel_values = processed_batch.pop("pixel_values").to(device, dtype)

print("=" * 60)
print("开始生成图像描述...")
print("=" * 60)
generated_output = model.generate(pixel_values=pixel_values)
generated_text = processor.batch_decode(generated_output, skip_special_tokens=True)

print(f"\n原始标签: {item['text']}")
print(f"生成文本: {generated_text[0]}")
print("\n" + "=" * 60)



