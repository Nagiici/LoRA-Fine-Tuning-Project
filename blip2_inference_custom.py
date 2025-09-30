"""
使用自定义图片进行 BLIP-2 LoRA 模型推理
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, PeftConfig
from PIL import Image
import os

def load_model(peft_model_id):
    """加载微调后的模型"""
    print("正在加载模型...")

    config = PeftConfig.from_pretrained(peft_model_id)
    processor = AutoProcessor.from_pretrained(config.base_model_name_or_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    if device == "cuda":
        model = AutoModelForVision2Seq.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            config.base_model_name_or_path
        )

    model.to(device)
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.eval()

    print("模型加载完成！\n")
    return model, processor, device

def generate_caption(image_path, model, processor, device):
    """为图片生成描述"""
    # 加载图片
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 - {image_path}")
        return None

    image = Image.open(image_path).convert('RGB')
    print(f"已加载图片: {image_path}")
    print(f"图片尺寸: {image.size}")

    # 处理图片
    inputs = processor(images=image, return_tensors="pt")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pixel_values = inputs["pixel_values"].to(device, dtype)

    # 生成描述
    print("正在生成图像描述...")
    with torch.no_grad():
        generated_ids = model.generate(pixel_values=pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

def main():
    print("=" * 60)
    print("BLIP-2 LoRA 自定义图片推理")
    print("=" * 60 + "\n")

    # 配置
    peft_model_id = "./output/blip2_lora"

    # 加载模型
    model, processor, device = load_model(peft_model_id)

    # 方式 1: 使用单张图片
    # print("\n方式 1: 单张图片推理")
    # print("-" * 60)

    # image_path = "./test_image.jpg"  # 修改为你的图片路径

    # if os.path.exists(image_path):
    #     caption = generate_caption(image_path, model, processor, device)
    #     print(f"\n生成的描述: {caption}")
    # else:
    #     print(f"图片不存在: {image_path}")
    #     print("请将图片放在项目目录，或修改 image_path 变量")

    # 方式 2: 批量处理多张图片
    print("\n\n方式 2: 批量图片推理")
    print("-" * 60)

    image_folder = "./test_images"  # 图片文件夹

    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        print(f"找到 {len(image_files)} 张图片")

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            print(f"\n处理: {img_file}")
            caption = generate_caption(img_path, model, processor, device)
            print(f"描述: {caption}")
    else:
        print(f"文件夹不存在: {image_folder}")
        print("创建 test_images 文件夹并放入图片")

    # # 方式 3: 使用 HuggingFace 数据集（原始方式）
    # print("\n\n方式 3: 使用 HuggingFace 数据集")
    # print("-" * 60)

    # try:
    #     from datasets import load_dataset
    #     dataset = load_dataset("ybelkada/football-dataset", split="train")

    #     print(f"数据集加载完成，共 {len(dataset)} 条数据")

    #     # 测试前 3 张图片
    #     for i in range(min(3, len(dataset))):
    #         item = dataset[i]
    #         image = item["image"]
    #         original_text = item["text"]

    #         # 处理图片
    #         inputs = processor(images=image, return_tensors="pt")
    #         dtype = torch.float16 if device == "cuda" else torch.float32
    #         pixel_values = inputs["pixel_values"].to(device, dtype)

    #         # 生成描述
    #         with torch.no_grad():
    #             generated_ids = model.generate(pixel_values=pixel_values)
    #             generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #         print(f"\n图片 {i+1}:")
    #         print(f"  原始标签: {original_text}")
    #         print(f"  生成描述: {generated_text}")

    # except Exception as e:
    #     print(f"加载数据集失败: {e}")

    # print("\n" + "=" * 60)
    # print("推理完成！")
    # print("=" * 60)

if __name__ == "__main__":
    main()