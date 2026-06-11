import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import re
from peft import AutoPeftModelForCausalLM

torch.manual_seed(42)
DEVICE = "cuda:6"
# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

model = AutoPeftModelForCausalLM.from_pretrained(
    "/root/ly/qwen/model/modal_selection/", # path to the output directory
    device_map="auto",
    trust_remote_code=True
).eval()

# IMG_PATH = "./data/met-meme/Eimages/Eimages/Eimages/"
# FILE_PATH = "./data/met-meme/result/qwen-vl/zeroshot0.json"
IMG_PATH = "./data/memes/"
SAVE_PATH = "./data/meme-cap-main/result/qwen-vl-ft/zeroshot0.json"

with open("./data/meme-cap-main/data/memes-testocr2.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    # random.seed(42)
    # data = random.sample(data, 30)
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

        # zero shot
        # <image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. What is the meme poster trying to convey? Rationale: “{rational}”Answer:
        query = tokenizer.from_list_format([
            {'image': img_name}, # Either a local path or an url
            {'text': f'What is the meme poster trying to convey? Answer:'},
        ])
        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        print(response)
        # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
        if '<|endoftext|>' in response:
            # print(item["generated_caption"])
            if len(re.findall('Answer: (.*?)<', response)) == 0:
                text = ""
            else:
                text = re.findall('Answer: (.*?)<', response)[-1]
        else:
            # print(item["generated_caption"])
            if len(re.findall('(?<=Answer: ).*$', response)) == 0:
                text = ""
            else:
                text = re.findall('(?<=Answer: ).*$', response)[-1]
        item["generated_caption"] = [text]
        result.append(item)

    json.dump(result, result_file, indent=4)