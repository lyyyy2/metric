import os
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json
import re
# import easyocr
# import torch
# from paddleocr import PaddleOCR

# DEVICE = "cuda:2"
# # Set up the device.
# if torch.cuda.is_available():
#     device = torch.device(DEVICE)
#     print(f"Using {DEVICE}.")
# else:
#     device = torch.device("cpu")
#     print(f"CUDA not available, using CPU.")

IMG_PATH = "./data/memes/"
SAVE_PATH = "./data/meme-cap-main/result/flamingo/twoshot_5.json"
def process(save_path):
    with open("./data/meme-cap-main/result/flamingo/twoshot5.json", "r", encoding='utf-8') as f:
        data = json.load(f)
        result = []
        for item in data:
            if'<|endofchunk|>' in item["generated_caption"]:
                # print(item["generated_caption"])
                if len(re.findall('Answer: (.*?)<', item["generated_caption"])) == 0:
                    text = ""
                else:
                    text = re.findall('Answer: (.*?)<', item["generated_caption"])[-1]
            else:
                # print(item["generated_caption"])
                if len(re.findall('(?<=Answer: ).*$', item["generated_caption"])) == 0:
                    text = ""
                else:
                    text = re.findall('(?<=Answer: ).*$', item["generated_caption"])[-1]
            item["generated_caption"] = [text]
            result.append(item)

        result_file = open(save_path, mode='w')
        json.dump(result, result_file, indent=4)

# def ocr(save_path=None):
#     reader = easyocr.Reader(lang_list=['en'], gpu=True, download_enabled=False, model_storage_directory="./model")
#     # reader.device = device
#     with open("./data/meme-cap-main/data/memes-test.json", 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for item in data:
#             img_name = IMG_PATH + item["img_fname"]
#             ocr = reader.readtext(image=img_name, detail=0, paragraph=True)
#             item['ocr_text'] = '\n'.join(ocr)
#             print(item['ocr_text'])
#         result_file = open('./data/meme-cap-main/data/memes-testocr.json', mode='w')
#         json.dump(data, result_file, indent=4)

def paddle_ocr():
    with open("./data/meme-cap-main/data/memes-trainval.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        ocr = PaddleOCR(use_angle_cls=True, lang="en")
        for item in data:
            img_name = IMG_PATH + item["img_fname"]
            result = ocr.ocr(img_name, cls=True)
            text = []
            for idx in range(len(result)):
                res = result[idx]
                print(res)
                for line in res:
                    text.append(line[1][0])
            text = ','.join(text)
            print(text)
            item['ocr_text'] = text
        result_file = open('./data/meme-cap-main/data/memes-trainval2.json', mode='w')
        json.dump(data, result_file, indent=4)

def get_caption():
    with open("./data/meme-cap-main/data/memes-trainval.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        caption = {}
        annotation = []
        result_file = open('./data/meme-cap-main/data/filter_cap.json', mode='w')
        for item in data:
            result = {}
            result["image_id"] = "memes_" + item["post_id"]
            result["caption"] = " ".join(item["meme_captions"])
            annotation.append(result)
        caption["annotations"] = annotation
        json.dump(caption, result_file)

def count_metaphor(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        count = 0
        length = len(data)
        for item in data:
            # print(item["metaphors"])
            metaphors = item["metaphors"]
            vlength = 0
            tlength = 0
            if len(metaphors) == 0:
                length -= 1
                continue
            for m in metaphors:
                vlength += len(m["metaphor"].split())
                tlength += len(m["meaning"].split())
            count += vlength / len(metaphors)

    return count / length

def get_30meme(file, save_file):
    with open('./data/meme30.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
        names = []
        for line in data:
            names.append(line.replace('\n', ''))

    with open(file, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
        result = []
        result_file = open(save_file, mode='w')
        for item in data1:
            if item['post_id'] in names:
                result.append(item)
        json.dump(result, result_file, indent=4)

def data_format():
    text = pd.read_csv('./data/met-meme/E_text.csv', encoding='gbk')
    label = pd.read_csv('./data/met-meme/label_E.csv', encoding='gbk')
    merge_df = pd.merge(text, label, on='file_name')
    # metaphor_df = merge_df[(merge_df['metaphor occurrence'] == 1) & (merge_df['metaphor category'] == 'complementary')]
    metaphor_df = merge_df[merge_df['metaphor occurrence'] == 1]
    np.random.seed(42)

    result = []
    for _, row in metaphor_df.iterrows():
        metaphor = []
        meaning = []
        metaphor.extend(str(row['source domain']).split(';'))
        meaning.extend(str(row['target domain']).split(';'))
        metaphors = []
        if len(metaphor) != len(meaning):
            continue
        for i in range(len(metaphor)):
            metaphors.append({
                "metaphor": metaphor[i],
                "meaning": meaning[i]
            })
        result.append({
            "category": "memes",
            "metaphor category": str(row['metaphor category']),
            "img_captions": [],
            "meme_captions": [],
            "title": "",
            "url": "",
            "img_fname": row['file_name'],
            "metaphors": metaphors,
            "post_id": "",
            "ocr_text": str(row['text']) if str(row['text']) != 'nan' else ""
        })
    result_file = './data/met-meme/memes.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def merge_llm_metaphor(file_path, llm_file_path, output_file_path):
    result = []
    result_file = open(output_file_path, mode='w')
    with open(file_path, "r", encoding="utf-8") as f1:
        data = json.load(f1)
        with open(llm_file_path, "r", encoding="utf-8") as f2:
            llm_data = json.load(f2)
            for item in data:
                for meme in llm_data:
                    if item["post_id"] == meme["post_id"]:
                        item["llm_metaphors"] = meme["llm_metaphors"]
                        break
                result.append(item)
    json.dump(result, result_file, indent=4)


# ocr()
# paddle_ocr()
# process(SAVE_PATH)
# get_caption()
# print(count_metaphor("C:/Users/15013/Documents/幽默计算/qwen/data/metmeme-test.json"))
# get_30meme('./data/meme-cap-main/result/minigpt4/finetune0.json', './data/meme-cap-main/evaluation/minigpt4/finetune0-original.json')
# data_format()
merge_llm_metaphor('./data/meme-cap-main/evaluation/llama/zeroshot2.json', './data/meme-cap-main/evaluation/test-qwen.json', './data/meme-cap-main/evaluation/llama/zeroshot2-qwen.json')