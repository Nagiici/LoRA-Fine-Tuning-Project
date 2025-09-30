
# coding=utf-8

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import argparse

from peft import LoraConfig, get_peft_model
import os
import matplotlib.pyplot as plt
from PIL import Image



class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = item["text"]
        return encoding


def plot(loss_list, output_path):
    plt.figure(figsize=(10,5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4007f', label="image2text train/loss curve")

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(output_path+'/pytorch_image2text_blip2_loss_curve.png')
    # plt.show()


def main():

    parser = argparse.ArgumentParser(description="BLIP-2 LoRA Fine-tuning")
    parser.add_argument("--pretrain-model-path", dest="pretrain_model_path", required=False, type=str, default="Salesforce/blip2-opt-2.7b", help="预训练模型路径")
    parser.add_argument("--train-dataset-path", type=str, default="ybelkada/football-dataset", help="训练集路径")
    parser.add_argument("--test-dataset-path", type=str, default="ybelkada/football-dataset", help="测试集路径")
    parser.add_argument("--output-path", type=str, default="./output/blip2_lora",help="模型输出路径")
    parser.add_argument("--batch-size", type=int, default=2, help="训练批次大小")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=11, help="训练轮数")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA秩")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha参数,通常为LoRA秩的两倍")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--print-freq", type=int, default=10, help="生成输出的频率(每n个batch)")

    args = parser.parse_args()
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    output_path = args.output_path
    pretrain_model_path = args.pretrain_model_path
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    print_freq = args.print_freq

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #pretrain_model_path="/workspace/model/blip2-opt-2.7b"
    #train_dataset_path = "/workspace/data/pytorch_data/multimodal/blip2/ybelkada___football-dataset/default-80f5618dafa96df9/0.0.0/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"
    peft_model_id = output_path


    # We load our model and processor using `transformers`
    # 检查是否支持 8-bit 量化（Windows 通常不支持）
    use_8bit = False
    if os.name != 'nt':  # 非 Windows 系统
        try:
            import bitsandbytes
            use_8bit = True
            print("使用 8-bit 量化加载模型...")
        except ImportError:
            print("bitsandbytes 不可用，使用 float32 加载模型...")
    else:
        print("Windows 系统检测到，使用 float32 加载模型（不使用 8-bit 量化）...")

    if use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForVision2Seq.from_pretrained(
            pretrain_model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        # 使用 float16 以减少显存占用
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        if device == "cuda":
            model = AutoModelForVision2Seq.from_pretrained(
                pretrain_model_path,
                torch_dtype=torch.float16
            )
            model.to(device)
        else:
            model = AutoModelForVision2Seq.from_pretrained(pretrain_model_path)
            model.to(device)

    processor = AutoProcessor.from_pretrained(pretrain_model_path)


    # Let's define the LoraConfig
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )
    # Get our peft model and print the number of trainable parameters
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Let's load the dataset here!
    # dataset = load_dataset("ybelkada/football-dataset", split="train")


    def collator(batch):
        # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    dataset = load_dataset(train_dataset_path, split="train")

    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 确定数据类型
    dtype = torch.float16 if device == "cuda" else torch.float32

    model.train()
    loss_list = []
    for epoch in range(epochs):
        print("Epoch:", epoch)
        sum_loss_list = []
        for idx, batch in enumerate(train_dataloader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device, dtype)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

                loss = outputs.loss

                print("Loss:", loss.item())

                sum_loss_list.append(float(loss.item()))
                
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                if idx % print_freq == 0:
                        generated_output = model.generate(pixel_values=pixel_values)
                        print(processor.batch_decode(generated_output, skip_special_tokens=True))
            
        avg_sum_loss = sum(sum_loss_list)/len(sum_loss_list)
        print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
        loss_list.append(float(avg_sum_loss))


    if not os.path.exists(peft_model_id):
        os.makedirs(peft_model_id)

    print("model_output:", peft_model_id)
    model.save_pretrained(peft_model_id)
    plot(loss_list, peft_model_id)



if __name__ == "__main__":
    main()
