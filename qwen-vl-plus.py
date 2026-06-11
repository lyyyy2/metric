from openai import OpenAI
import re
import base64
import json

IMG_PATH = "./data/memes/"
SAVE_PATH = "./data/meme-cap-main/qwen-vl-plus/zeroshot0.json"

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    api_key="sk-2f93eb50c9fd41c2a01134591127c159",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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
        print(item["img_fname"])

        base64_image = encode_image(img_name)

        # f'This is a meme with the title “{title}”. The image description is “{image_caption}”. The following text is written inside the meme: “{ocr_text}”. What is the meme poster trying to convey? Rationale: “{rational}”Answer:'
        try:
            completion = client.chat.completions.create(
                model="qwen-vl-plus-latest",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            },
                            {"type": "text", "text":
                                f'What is the meme poster trying to convey? ONLY output the answer of the question and the answer should be concise.'}
                        ]
                    }
                ]
            )
            answer = completion.choices[0].message.content
            print(answer)
            item["generated_caption"] = [answer]
            result.append(item)

        except Exception as e:
            print(f"Error processing {item['img_fname']}: {e}")
            continue

    json.dump(result, result_file, indent=4)