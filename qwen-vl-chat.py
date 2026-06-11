import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import re

torch.manual_seed(42)
DEVICE = "cuda:7"
# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=DEVICE, trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# 默认gpu进行推理，需要约24GB显存
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# IMG_PATH = "./data/met-meme/Eimages/Eimages/Eimages/"
# FILE_PATH = "./data/met-meme/result/qwen-vl/zeroshot0.json"
IMG_PATH = "./data/memes/"
SAVE_PATH = "./data/meme-cap-main/result/qwen-vl-chat/com.json"

with open("./data/meme-cap-main/data/memes-testocr2com.json", 'r', encoding='utf-8') as f:
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

        # zero shot
        # <image>This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. What is the meme poster trying to convey? Rationale: “{rational}”Answer:
        query = tokenizer.from_list_format([
            {'image': img_name}, # Either a local path or an url
            {'text': f"Here is the metaphorical reasoning chain of the meme: {com}\nWhat is the meme poster trying to convey? ONLY output the answer of the question and the answer should be concise."},
            # {'text': 'Analyze the provided meme image and identify all metaphorical elements. For each metaphor found:'
            #          '1. Clearly describe the visual metaphor/symbol used (the "metaphor")'
            #          '2. Explain its underlying meaning or real-world reference (the "meaning")'
            #          '3. Format results in JSON with exactly this structure:'
            #          '{'
            #          '  "metaphors": ['
            #          '      {'
            #          '          "metaphor": "explicit visual element/text reference",'
            #          '          "meaning": "what it symbolically represents"'
            #          '      }'
            #          '  ]'
            #          '}'
            #          'Consider:'
            #          '- Cultural references'
            #          '- Visual symbolism'
            #          '- Textual puns/wordplay'
            #          '- Popular culture parodies'
            #          '- Historical/political analogies'
            #          'Requirements:'
            #          '1. Be thorough but concise'
            #          '2. Separate distinct metaphors'
            #          '3. Use simple English-only explanations'
            #          '4. Output ONLY valid JSON (no commentary)'
            #          '5. Include both visual and textual elements'},
        ])
        response, history = model.chat(tokenizer, query=query, history=None)
        print(response)
        # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
        # if '<|endoftext|>' in response:
        #     # print(item["generated_caption"])
        #     if len(re.findall('Answer: (.*?)<', response)) == 0:
        #         text = ""
        #     else:
        #         text = re.findall('Answer: (.*?)<', response)[-1]
        # else:
        #     # print(item["generated_caption"])
        #     if len(re.findall('(?<=Answer: ).*$', response)) == 0:
        #         text = ""
        #     else:
        #         text = re.findall('(?<=Answer: ).*$', response)[-1]
        item["generated_caption"] = response
        result.append(item)

    json.dump(result, result_file, indent=4)