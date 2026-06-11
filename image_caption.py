import torch
import json
from PIL import Image
from lavis.models import load_model_and_preprocess
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
IMG_PATH = "./data/met-meme/Eimages/Eimages/Eimages/"
with open("./data/met-meme/memes-imgcap.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    result = []
    result_file = open("data/met-meme/memes-imgcap.json", mode='w')
    for item in data:
        img_name = IMG_PATH + item["img_fname"]
        query_image = Image.open(img_name).convert('RGB')
        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](query_image).unsqueeze(0).to(device)
        # generate caption
        generated_text = model.generate({"image": image})
        # ['a large fountain spewing water into the air']
        print("Generated text: ", generated_text)
        item["img_captions"] = generated_text
        result.append(item)

    json.dump(result, result_file, indent=4)