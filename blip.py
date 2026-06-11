import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import json
from PIL import Image
from LAVIS.lavis.models import load_model_and_preprocess
# setup device to use
DEVICE = "cuda:4"
# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=device)
IMG_PATH = "./data/memes/"
SAVE_PATH = "data/meme-cap-main/result/blip/zeroshot0.json"
with open("./data/meme-cap-main/data/memes-testocr2.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    # print(data)
    result = []
    result_file = open(SAVE_PATH, mode='w')
    for item in data:
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
        query_image = Image.open(img_name)
        image = vis_processors["eval"](query_image).unsqueeze(0).to(device)
        # <image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. What is the meme poster trying to convey? Rationale: “{rational}”Answer:
        prompt = "What is the meme poster trying to convey? Answer:"
        print(model.generate({"image": image, "prompt": prompt}))