from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import json

processor = LlavaNextProcessor.from_pretrained("/root/ly/model/llava-v1.6-mistral-7b-hf/")

model = LlavaNextForConditionalGeneration.from_pretrained("/root/ly/model/llava-v1.6-mistral-7b-hf/", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:6")

IMG_PATH = "./data/memes/"
SAVE_PATH = "data/meme-cap-main/result/llava/zeroshot0.json"
with open("./data/meme-cap-main/data/memes-testocr2com-ft.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    result = []
    result_file = open(SAVE_PATH, mode='w')
    for item in data:
        if "com" not in item:
            continue
        com = item["com"]
        title = item["title"]
        image_caption = " ".join(item["img_captions"])
        ocr_text = item["ocr_text"]
        metaphors = item["metaphors"]
        rationals = []
        for i in range(len(metaphors)):
            metaphor = metaphors[i]["metaphor"]
            meaning = metaphors[i]["meaning"]
            if "not related to the meme context" not in meaning:
                rationals.append(f"“{metaphor}” is a metaphor for “{meaning}”. ")
        rational = "".join(rationals)
        img_name = IMG_PATH + item["img_fname"]

        image = Image.open(img_name)

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": "What is the meme poster trying to convey? Answer:"},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:6")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=200)

        print(processor.decode(output[0], skip_special_tokens=True))
