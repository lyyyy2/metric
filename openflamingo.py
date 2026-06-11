import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from torch.nn import DataParallel
from PIL import Image
import requests
import json
import re
import random
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
from transformers import AutoModelForCausalLM, AutoConfig

# # torch.cuda.empty_cache()
DEVICE = "cuda:4"
# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")
# os.environ['CUDA_VISIBLE_DEVICES']='1,2'
# device_ids = [0,1]

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-7b",
    tokenizer_path="anas-awadalla/mpt-7b",
    cross_attn_every_n_layers=4,
    # cache_dir="./cache"  # Defaults to ~/.cache
)
# model = DataParallel(model, device_ids=device_ids)
# model = model.cuda()

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
# checkpoint_path = "./openflamingo/checkpoint.pt"
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to(device)
#
# """
# Step 1: Load images
# """
# demo_image_one = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True
#     ).raw
# )
#
# demo_image_two = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028137.jpg",
#         stream=True
#     ).raw
# )
#
# query_image = Image.open(
#     requests.get(
#         "http://images.cocodataset.org/test-stuff2017/000000028352.jpg",
#         stream=True
#     ).raw
# )
#
#
# """
# Step 2: Preprocessing images
# Details: For OpenFlamingo, we expect the image to be a torch tensor of shape
#  batch_size x num_media x num_frames x channels x height x width.
#  In this case batch_size = 1, num_media = 3, num_frames = 1,
#  channels = 3, height = 224, width = 224.
# """
# vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
# vision_x = torch.cat(vision_x, dim=0)
# vision_x = vision_x.unsqueeze(1).unsqueeze(0)
#
# """
# Step 3: Preprocessing text
# Details: In the text we expect an <image> special token to indicate where an image is.
#  We also expect an <|endofchunk|> special token to indicate the end of the text
#  portion associated with an image.
# """
# tokenizer.padding_side = "left" # For generation padding tokens should be on the left
# lang_x = tokenizer(
#     ["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
#     return_tensors="pt",
# )
#
#
# """
# Step 4: Generate text
# """
# generated_text = model.generate(
#     vision_x=vision_x,
#     lang_x=lang_x["input_ids"],
#     attention_mask=lang_x["attention_mask"],
#     max_new_tokens=20,
#     num_beams=3,
#     pad_token_id=tokenizer.eos_token_id
# )
#
# print("Generated text: ", tokenizer.decode(generated_text[0]))
demo_image1 = Image.open("./data/memes/memes_bpet7l.png")
demo_image2 = Image.open("./data/memes/memes_ctxvmu.png")
# demo_image3 = Image.open("./data/memes/memes_cxvuqd.png")
# demo_image4 = Image.open("./data/memes/memes_cyop7a.png")

IMG_PATH = "./data/memes/"
SAVE_PATH = "data/meme-cap-main/result/flamingo/com-ft4.json"
with open("./data/meme-cap-main/data/memes-testocr2com-ft.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    # random.seed(42)
    # data = random.sample(data, 30)
    # print(data)
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
        query_image = Image.open(img_name)
        # zero shot
        # <image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. What is the meme poster trying to convey? Rationale: “{rational}”Answer:
        vision_x = image_processor(query_image).unsqueeze(0).unsqueeze(1).unsqueeze(0)
        lang_x = tokenizer(
            [f"<image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. Here is the metaphorical reasoning chain of the meme: {com}\nWhat is the meme poster trying to convey? Answer:"],
            return_tensors="pt",
        )
        # few shot
        # <image>This is a meme with the title “For real though”. The image description is “Person in Spider Man outfit gives a lecture on stage.”. The following text is written inside the meme: “Reddit should add an,undo button so if,you accidentally,scroll to the top,you can go back to,where you were”. What is the meme poster trying to convey? Rationale: “a lecture” is a metaphor for “complaint”. “spider man” is a metaphor for “Meme poster”. “crowd” is a metaphor for “meme readers”. Answer: Meme poster is frustrated about the format of the website and is making a suggestion for improvement.<|endofchunk|>
        # <image>This is a meme with the title “And that's a fact”. The image description is “Identical chubby animated dogs carry a white banner.”. The following text is written inside the meme: “Google is better than Reddit to search for something on Reddit”. What is the meme poster trying to convey? Rationale: “Two dogs” is a metaphor for “Meme poster”. Answer: Meme poster is saying that searching Google plus the term you want to search on reddit is better than searching reddit itself.<|endofchunk|>
        # vision_x = [image_processor(demo_image1).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
        # vision_x = torch.cat(vision_x, dim=0)
        # vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        # tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        # lang_x = tokenizer(
        #     [f"<image>This is a meme with the title “For real though”. The image description is “Person in Spider Man outfit gives a lecture on stage.”. The following text is written inside the meme: “Reddit should add an,undo button so if,you accidentally,scroll to the top,you can go back to,where you were”. What is the meme poster trying to convey? Rationale: “a lecture” is a metaphor for “complaint”. “spider man” is a metaphor for “Meme poster”. “crowd” is a metaphor for “meme readers”. Answer: Meme poster is frustrated about the format of the website and is making a suggestion for improvement.<|endofchunk|>"
        #      f"<image>This is a meme with the title “And that's a fact”. The image description is “Identical chubby animated dogs carry a white banner.”. The following text is written inside the meme: “Google is better than Reddit to search for something on Reddit”. What is the meme poster trying to convey? Rationale: “Two dogs” is a metaphor for “Meme poster”. Answer: Meme poster is saying that searching Google plus the term you want to search on reddit is better than searching reddit itself.<|endofchunk|>"
        #      # f"<image>The image description is “Quentin Tarantino is standing sad in different rooms.”. What is the meme poster trying to convey? Answer: Meme poster is feeling sad.<|endofchunk|>"
        #      # f"<image>The image description is “A black man looking at camera tearing up, second image of man pulling down glasses with one hand and laughing ”. What is the meme poster trying to convey? Answer: Meme poster is saying that Reddit is more fun than Instagram.<|endofchunk|>"
        #      f"<image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. Rationale: “{rational}”What is the meme poster trying to convey? Answer:"],
        #     return_tensors="pt",
        # )
        # print(title)

        try:
            generated_text = model.generate(
                vision_x=vision_x.to(device),
                lang_x=lang_x["input_ids"].to(device),
                attention_mask=lang_x["attention_mask"].to(device),
                max_new_tokens=120,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=4
            )
            text = re.findall('Answer: (.*?)<', tokenizer.decode(generated_text[0]))
            print("Generated text: ", text)
            item["generated_caption"] = text
            result.append(item)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                # json.dump(result, result_file, indent=4)
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

    json.dump(result, result_file, indent=4)
