"""
导入 Hugging Face BLIP-2 模型
Model: Salesforce/blip2-opt-2.7b
用于图像字幕生成和视觉问答
"""

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

def load_blip2_model():
    """加载 BLIP-2 模型"""
    print("正在加载 BLIP-2 模型...")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载处理器和模型
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.to(device)

    print("模型加载完成！")
    return processor, model, device

def generate_caption(image_path, processor, model, device):
    """为图像生成描述"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 处理图像
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

    # 生成描述
    generated_ids = model.generate(**inputs, max_length=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

def answer_question(image_path, question, processor, model, device):
    """对图像进行视觉问答"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 处理图像和问题
    inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

    # 生成答案
    generated_ids = model.generate(**inputs, max_length=50)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text

if __name__ == "__main__":
    # 加载模型
    processor, model, device = load_blip2_model()

    # 示例使用（需要提供图像路径）
    # caption = generate_caption("path/to/image.jpg", processor, model, device)
    # print(f"图像描述: {caption}")

    # answer = answer_question("path/to/image.jpg", "What is in this image?", processor, model, device)
    # print(f"答案: {answer}")