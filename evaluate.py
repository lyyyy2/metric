import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,7,5'
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from bert_score import score
import json
import torch
import math
import re
import emoji
import time
from sentence_transformers import SentenceTransformer, util
from similarities import ClipSimilarity, BM25Similarity, BertSimilarity, SimHashSimilarity
from PIL import Image
from summa import keywords
from keybert import KeyBERT

DEVICE = "cuda:5"
# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")

# IMG_PATH = "./data/met-meme/Eimages/Eimages/Eimages/"
# FILE_PATH = "/root/ly/kymkb/result/llava1.6/metmeme_final.json"
# IMG_PATH = "./data/memes/"
# FILE_PATH = "/root/ly/kymkb/result/llava1.6/memecap_final.json"
IMG_PATH = ""
FILE_PATH = "/root/ly/kymkb/result/qwen2.5-vl-7b/memeinterpret_non.json"
references = []
candidates = []
metaphors = []
reference_sentence = []
candidate_sentence = []
image_caption = []
image_name = []

with open(FILE_PATH, 'r', encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        # if item["metaphor category"] != "complementary":
        #     continue
        # if "llm_metaphors" not in item:
        #     continue
        reference = []
        candidate = []
        sentence = []
        # item["generated_caption"] = [item["generated_caption"]]
        # item["generated_caption"][0] = re.sub(u"\u4e00-\u9fa5\u2018\u2019\u201c\u201d\u200b", '', item["generated_caption"][0])
        # item["generated_caption"][0] = item["generated_caption"][0].replace(u'\u200b', '')
        # item["generated_caption"][0] = re.sub('\d\. ', '', item["generated_caption"][0])
        # item["generated_caption"][0] = re.sub('\d\.\d ', '', item["generated_caption"][0])
        # item["generated_caption"][0] = re.sub('\d', '', item["generated_caption"][0])
        # item["generated_caption"][0] = emoji.demojize(item["generated_caption"][0])
        if len(item["generated_caption"]) == 0 or item["generated_caption"][0] == '':
            continue
        # item["generated_caption"][0] = item["generated_caption"][0][0:120]
        for text in item["meme_captions"]:
            reference.append(text.split(' '))
            sentence.append(text)
        candidate = item["generated_caption"][0].split(' ')
        if "metaphors" not in item:
            metaphor = []
        else:
            metaphor = item["metaphors"]
        # metaphor = item["llm_metaphors"]
        # print(reference)
        # print(candidate)
        # print(sentence)
        references.append(reference)
        candidates.append(candidate)
        metaphors.append(metaphor)
        image_caption.append(item["img_captions"])
        image_name.append(IMG_PATH + item["img_fname"])
        reference_sentence.append(sentence)
        candidate_sentence.append(item["generated_caption"][0])

def BLEUScore(references, candidates):
    start_time = time.time()
    score = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\BLEU total time：{run_time:.2f} s")
    return score

def ROUGE_L(references, candidates):
    # print(references)
    # print(candidates)
    score = 0
    rouge = Rouge()
    length = len(references)
    start_time = time.time()
    for i in range(len(references)):
        candidate = [candidates[i].replace(u'\u200b', '')] * len(references[i])
        if candidate[0] == '':
            length -= 1
            continue
        rouge_score = rouge.get_scores(hyps=candidate, refs=references[i], avg=True)
        score += rouge_score["rouge-l"]["p"]
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\ROUGE total time：{run_time:.2f} s")
    return score / length

def BERTScore(references, candidates):
    total = 0
    length = len(references)
    start_time = time.time()
    for i in range(len(references)):
        candidate = re.sub(u"\u4e00-\u9fa5\u2018\u2019\u201c\u201d\u200b", '', candidates[i])
        candidate = candidate.replace(u'\u200b', '')
        if candidate == '':
            length -= 1
            continue
        total += score([candidate], [references[i]], lang="en", device=device, model_type="microsoft/deberta-xlarge-mnli")[2]
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nBERTScore total time：{run_time:.2f} s")
    return total / length

def clip_similarity(image_names, candidates):
    m = ClipSimilarity(model_name_or_path="openai/clip-vit-base-patch32")
    imgs = [Image.open(i) for i in image_names]
    sim_scores = m.similarity(imgs, candidates)
    similarity = [sim_scores[j][j] for j in range(len(image_names))]
    for item in similarity:
        print("CLIPSimilarity:{:.4f}".format(item))
    return similarity

def cosine_similarity(model, meaning, n_gram):
    sentences = [meaning, n_gram]
    embeddings = model.encode(sentences)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return sim.tolist()[0][0]

def get_keywords(candidate):
    words = keywords.keywords(candidate, ratio=1).split("\n")
    return words

def keybert_words(candidate, n):
    kw_model = KeyBERT(model='./sentencetransformer/paraphrase-MiniLM-L6-v2')
    length = len(candidate.split(' '))
    # if length >= 1 and length <= 10:
    #     keywords = candidate.split(' ')
    # else:
    #     keywords = kw_model.extract_keywords(candidate, keyphrase_ngram_range=(1, n), stop_words='english',
    #                                          use_maxsum=True, nr_candidates=6, top_n=3)
    #     keywords = [word[0] for word in keywords]
    if length > 24 and n > 2:
        keywords = kw_model.extract_keywords(candidate, keyphrase_ngram_range=(1, n), stop_words='english',
                                         use_maxsum=True, nr_candidates=16, top_n=8)
        keywords = [word[0] for word in keywords]
    elif length >= 1 and length <= 10:
        keywords = candidate.split(' ')
    elif length > 10 and length <= 15:
        keywords = kw_model.extract_keywords(candidate, keyphrase_ngram_range=(1, n), stop_words='english',
                                         use_maxsum=True, nr_candidates=6, top_n=3)
        keywords = [word[0] for word in keywords]
    else:
        if n == 1 or n == 2:
            keywords = kw_model.extract_keywords(candidate, keyphrase_ngram_range=(1, n+1), stop_words='english',
                                                 use_maxsum=True, nr_candidates=4, top_n=3)
        else:
            keywords = kw_model.extract_keywords(candidate, keyphrase_ngram_range=(1, n), stop_words='english',
                                         use_maxsum=True, nr_candidates=10, top_n=5)
        keywords = [word[0] for word in keywords]
    if len(keywords) == 0:
        keywords = candidate.split(' ')
    print(keywords)
    return keywords

def compute_metaphor(candidate, metaphor):
    count = 0
    candidate = candidate.strip().split(" ")
    for item in metaphor:
        n = len(item.strip().split(" "))
        # print(item)
        i = 0
        while i + n < len(candidate):
            n_gram = " ".join(candidate[i:i+n]).lower()
            # print(n_gram)
            if n_gram == item:
                count += 1
            i += 1
    # print(count)
    return count / len(metaphor)

def compute_metaphor_v2(candidate, metaphor, alpha):
    count = 0
    cover = 0
    candidate_sentence = candidate.strip().split(" ")
    for item in metaphor:
        if item.lower() in candidate.lower():
            cover += 1
        n = len(item.strip().split(" "))
        i = 0
        while i + n < len(candidate_sentence):
            n_gram = " ".join(candidate_sentence[i:i+n]).lower()
            if n_gram == item:
                count += 1
            i += 1
    cover_score = cover / len(metaphor)
    count_score = 1 / (1 + count * cover_score)
    # metaphor_score = alpha * count_score + (1 - alpha) * cover_score
    metaphor_score = count_score
    return metaphor_score

def compute_meaning(model, candidate, meanings, threshold):
    score = 0
    length = len(meanings)
    for meaning in meanings:
        if meaning == "not related to the meme context":
            length -= 1
            if length == 0:
                return 0
            continue
        # if meaning in candidate:
        #     score += 1
        # else:
        words = meaning.strip().split(' ')
        grams = candidate.strip().split(' ')
        n = len(words)
        i = 0
        f_n = 1
        # sim = 0
        while n > 0 and i < len(words):
            count = 0
            # temp_s = 0
            m = " ".join(words[i:i+n])
            j = 0
            while j + n < len(grams):
                sim = cosine_similarity(model, m, " ".join(grams[j:j+n]))
                # sim = model.similarity(m, " ".join(grams[j:j+n]))
                # print(sim)
                # if sim[0] > threshold:
                if sim > threshold:
                    # temp_s += cos_sim
                    count += 1
                j += 1
            score += f_n * count
            i += 1
            if i + n >= len(words):
                # score += sim / (len(words) + 1 - n)
                i = 0
                n -= 1
                # sim = 0
                f_n = 1 - math.exp(-n)
                # f_n = n / len(words)
    return score / length

def meaning_density(metaphors):
    amount = 0
    for metaphor in metaphors:
        length = len(metaphor)
        for item in metaphor:
            if item["meaning"] == "not related to the meme context":
                length -= 1
        amount += length
    return amount / len(metaphors)

def compute_meaning_v2(model, candidate, meanings, keyword):
    score = 0
    length = len(meanings)
    for meaning in meanings:
        sim = []
        if meaning == "not related to the meme context":
            length -= 1
            if length == 0:
                return 0
            continue
        if meaning.lower() in candidate.lower():
            score += 1
        else:
            for word in keyword:
                sim.append(float(model.similarity(meaning, word)[0]))
            score += max(sim)
    return score / length

def metaphor_eval(candidates, metaphors, image_captions):
    model = SentenceTransformer('./sentencetransformer/paraphrase-MiniLM-L6-v2')
    model.to(device)
    model2 = BertSimilarity(model_name_or_path="bert-base-uncased", device=device)
    # model2 = SimHashSimilarity()
    density = meaning_density(metaphors)
    metaphor_scores = 0
    meaning_scores = 0
    reduction_scores = 0
    length = len(candidates)
    start_time = time.time()
    for i in range(len(candidates)):
        metaphor = []
        meaning = []
        candidate = candidates[i]
        metaphor_sim = 0
        if isinstance(candidate, str):
            candidate = re.sub(u"\u4e00-\u9fa5\u2018\u2019\u201c\u201d\u200b", '', candidate)
            candidate = candidate.replace(u'\u200b', '')
        if candidate == "":
            length -= 1
            continue
        if len(metaphors[i]) == 0:
            length -= 1
            continue
        for item in metaphors[i]:
            metaphor.append(item["metaphor"].lower())
            meaning.append(item["meaning"])
        #     metaphor_sim += cosine_similarity(model, item["metaphor"], item["meaning"])
        # metaphor_sim /= len(metaphors[i])
        print(metaphor)
        caption_sim = 0
        metaphor_count = 0
        meaning_score = 0
        if isinstance(candidate, list):
            n = max([len(m.split(' ')) for m in meaning])
            candidate = [candidate[1]]
            for c in candidate:
                c = re.sub(u"\u4e00-\u9fa5"u"\u2018\u2019\u201c\u201d\u200b", '', c)
                c = c.replace(u'\u200b', '')
                for img_caption in image_captions[i]:
                    caption_sim += cosine_similarity(model, c, img_caption)
                # metaphor_count += compute_metaphor(c, metaphor)
                metaphor_count += compute_metaphor_v2(c, metaphor, 0.5)
                # meaning_score += compute_meaning(model, c, meaning, 0.8)
                # keyword = get_keywords(c)
                keyword = keybert_words(c, n)
                meaning_score += compute_meaning_v2(model2, c, meaning, keyword)
            caption_sim /= (len(image_captions[i]) * len(candidate))
            if caption_sim < 0:
                caption_sim = 0
            metaphor_count /= len(candidate)
            # metaphor_score = caption_sim * metaphor_count
            metaphor_score = (1 - caption_sim) * metaphor_count
            meaning_score /= len(candidate)
            print("metaphor_score:{0:.4f}".format(metaphor_score))
            print("meaning_score:{0:.4f}".format(meaning_score))
            metaphor_scores += metaphor_score
            meaning_scores += meaning_score
        else:
            for img_caption in image_captions[i]:
                caption_sim += cosine_similarity(model, candidate, img_caption)
            caption_sim /= len(image_captions[i])
            if caption_sim < 0:
                caption_sim = 0
            # metaphor_count = compute_metaphor(candidate, metaphor)
            # metaphor_score = caption_sim * metaphor_count
            metaphor_score = (1 - caption_sim) * compute_metaphor_v2(candidate, metaphor, 0.5)
            # meaning_score = compute_meaning(model, candidate, meaning, 0.8)
            # keyword = get_keywords(candidate)
            n = max([len(m.split(' ')) for m in meaning])
            keyword = keybert_words(candidate, n)
            meaning_score = compute_meaning_v2(model2, candidate, meaning, keyword)
            print("metaphor_score:{0:.4f}".format(metaphor_score))
            print("meaning_score:{0:.4f}".format(meaning_score))
            reduction_scores += meaning_score - metaphor_score
            metaphor_scores += metaphor_score
            meaning_scores += meaning_score
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\TS time：{run_time:.2f} s")
    return metaphor_scores/length, meaning_scores/length, reduction_scores/length

# metaphor_score, meaning_score, reduction_score = metaphor_eval(candidate_sentence, metaphors, image_caption)
# print("metaphor_score:{0:.4f}".format(metaphor_score))
# print("meaning_score:{0:.4f}".format(meaning_score))
# # print("reduction_score:{0:.4f}".format(reduction_score))
print("BLEU:{0:.4f}".format(BLEUScore(references, candidates)))
print("ROUGE_L:{0:.4f}".format(ROUGE_L(reference_sentence, candidate_sentence)))
print(f"BERTScore:{BERTScore(reference_sentence, candidate_sentence)}")
# clip_similarity(image_name, [caption[0] for caption in image_caption])
# print(get_keywords(candidate_sentence))
# keybert_words("The meme poster is trying to say that he is watching a tv show and he is very excited about it.", 7)
