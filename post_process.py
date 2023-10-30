import json
import os

def parse_answer(str):
    return str.replace(" ", "").replace("\n", "")

def add_answer(path):
    result = []
    with open("result/ret.json", 'r', encoding='utf-8') as f1, open(path, 'r', encoding="utf-8") as f2:
        samples = json.load(f1)
        new_samples = json.load(f2)
        for sample, new_sample in zip(samples, new_samples):
            # sample["answer_1"] = parse_answer(sample["answer_1"])
            # sample["answer_2"] = parse_answer(sample["answer_2"])
            sample["answer_1"] = parse_answer(new_sample["answer"])
            # sample["answer_3"] = ""
            result.append(sample)

    with open("data/ret.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def classify_answer():
    with open("result/0.3_top_p0.7_answer.json", 'r', encoding='utf-8') as f:
        samples = json.load(f)

    empty = []
    good_answer = []
    for sample in samples:
        if len(sample["keyword"]) == 0:
            empty.append(sample)
            continue
        if sample["keyword"][0][1] > -90:
            good_answer.append(sample)
            continue

    with open("result/not_found.json", "w", encoding='utf-8') as f:
        json.dump(empty, f, ensure_ascii=False, indent=4)

    with open("result/good_answer.json", "w", encoding='utf-8') as f:
        json.dump(good_answer, f, ensure_ascii=False, indent=4)

def add_answer_from_scratch():
    result = []
    with open("data/测试问题.json", 'r', encoding='utf-8') as f1, open("result/0.3_top_p0.7_answer.json", 'r', encoding="utf-8") as f2:
        samples = json.load(f1)
        new_samples = json.load(f2)
        for sample, new_sample in zip(samples, new_samples):
            # sample["answer_1"] = new_sample["answer"]
            sample["answer_2"] = parse_answer(new_sample["answer"])
            sample["answer_3"] = ""
            result.append(sample)

    with open("result/ret.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    temperature = 0.5
    top_p = 0.6
    add_answer(f"result/{temperature}_top_p{top_p}_answer.json")
    # add_answer_from_scratch()