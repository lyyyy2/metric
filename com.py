from openai import OpenAI
import re
import base64
import json

IMG_PATH = "./data/memes/"
SAVE_PATH = "./data/meme-cap-main/data/memes-testocr2com.json"

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
        # title = item["title"]
        # image_caption = " ".join(item["img_captions"])
        # ocr_text = item["ocr_text"]
        # metaphors = item["metaphors"]
        # rationals = []
        # for i in range(len(metaphors)):
        #     metaphor = metaphors[i]["metaphor"]
        #     meaning = metaphors[i]["meaning"]
        #     if "not related to the meme context" not in meaning:
        #         rationals.append(f"“{metaphor}” is a metaphor for “{meaning}”. ")
        # rational = "".join(rationals)
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
                                'Analyze the provided meme through its multimodal metaphorical structure using this exact chain of thought:'
                                '1. Metaphor Type Classification'
                                'Identify primary interaction pattern:[text-dominant/image-dominant/complementary]'
                                '2. Metaphor Deconstruction'
                                'For each identified metaphor:'
                                'a) Literal element description (text/visual)'
                                'b) Cultural/conceptual reference'
                                'c) Mapping logic to real-world concept'
                                '3. Coherence Analysis'
                                'Explain how elements interact to create humor/satire'
                                'Output Requirements:'
                                '- Use ONLY this structure:'
                                '   1. Metaphor Type: [classification]'
                                '      Rationale: [20-word explanation]'
                                '   2. Metaphor Breakdown:'
                                '   a) [Visual/Text Element]:'
                                '      - Vehicle: [literal description]'
                                '      - Mapped Meaning: [interpretation]'
                                '      - Mapping Mechanism: [connection logic]'
                                '   b) [Next Element]: [...]'
                                '   3. Interaction Synthesis:'
                                '      [Combined metaphorical effect explanation]  '
                                '- Maintain strict English technical terminology  '
                                '- Exclude markdown/casual language  '
                                '- Prioritize conceptual accuracy over length'
                                'Example Output:'
                                '1. Metaphor Type: Complementary'
                                '   Rationale: Text clarifies ambiguous visual symbolism through wordplay'
                                '2. Metaphor Breakdown:'
                                '   a) Visual Element: Melting clock'
                                '      - Vehicle: Surreal timepiece deformation'
                                '      - Mapped Meaning: Wasted potential'
                                '      - Mapping Mechanism: Dali-esque symbolism → productivity critique  '
                                '   b) Text Element: "Deadlinosaurus Rex"'
                                '      - Vehicle: Dinosaur pun'
                                '      - Mapped Meaning: Primordial work pressure'
                                '      - Mapping Mechanism: Paleontological metaphor → archaic systems  '
                                '3. Interaction Synthesis:'
                                '   Surreal imagery combines with evolutionary wordplay to satirize outdated workplace expectations through temporal distortion metaphors.'
                                'Critical Constraints:'
                                '1. NEVER include interpretation confidence scores'
                                '2. ALWAYS maintain vehicle/meaning separation'
                                '3. AVOID subjective emotional analysis'
                                '4. PRESERVE cause-effect relationships in mapping'}
                        ]
                    }
                ]
            )
            answer = completion.choices[0].message.content
            print(answer)
            item["com"] = answer
            result.append(item)

        except Exception as e:
            print(f"Error processing {item['img_fname']}: {e}")
            continue

    json.dump(result, result_file, indent=4)