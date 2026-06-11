from openai import OpenAI
import re
import base64
import json

IMG_PATH = "./data/memes/"
# SAVE_PATH = "./data/meme-cap-main/evaluation/test-qwen.json"

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

client = OpenAI(
    api_key="sk-2f93eb50c9fd41c2a01134591127c159",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

with open("./data/meme-cap-main/evaluation/minigpt4/zeroshot4-v2.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    result = 0
    count = 0
    for item in data:
        if item["generated_caption"] == "":
            continue
        metaphors = item["metaphors"]
        rationals = []
        for i in range(len(metaphors)):
            metaphor = metaphors[i]["metaphor"]
            meaning = metaphors[i]["meaning"]
            if "not related to the meme context" not in meaning:
                rationals.append(f"“{metaphor}” is a metaphor for “{meaning}”. ")
        rational = "".join(rationals)
        text = [item["generated_caption"]]
        img_name = IMG_PATH + item["img_fname"]
        print(item["img_fname"])

        base64_image = encode_image(img_name)

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
                                f"Generated Text:{text}"
                                "Evaluate the generated text's metaphorical comprehension of the provided meme using human-like holistic analysis. Output ONLY a 0-10 integer score."
                                "Scoring Protocol:"
                                "1. Primary Metaphor Extraction (Mental Process):"
                                "- Identify core metaphorical elements from the meme"
                                "- Determine their contextual relationships"
                                "2. Interpretation Assessment:"
                                "a) Literal Alignment: Does the text reference key visual/textual elements?"
                                "b) Conceptual Depth: Does it reveal:"
                                "   - Social commentary"
                                "   - Cultural nuance"
                                "   - Satirical intent"
                                "c) Coherence: Logical mapping between elements and meanings"
                                "3. Grading Matrix:"
                                "10 = Captures all metaphors with contextual depth"
                                "8-9 = Misses 1 subtle metaphor"
                                "6-7 = Surface-level but accurate"
                                "4-5 = Partial understanding"
                                "2-3 = Literal description only"
                                "0-1 = Irrelevant response"
                                "Critical Constraints:"
                                "- NEVER output analysis steps"
                                "- Score MUST reflect meme's metaphorical complexity"
                                "- Prioritize cultural coherence over literal matching"}
                        ]
                    }
                ]
            )
            answer = completion.choices[0].message.content
            print(answer)
            result += int(answer)
            count += 1

        except Exception as e:
            print(f"Error processing {item['img_fname']}: {e}")
            continue

    print("Score: ", result / count)