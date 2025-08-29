import json
import os
import random
import io
import fitz
import tqdm

output_root = "clean"


def load_jsonl(file_path, debug=False):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                if debug:
                    print(f"Error decoding line: {line}\nError: {e}")
                continue
    return data


def pdf_to_text(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text


# ============================== Writing ==============================
print("=======Generating Writing Dataset=======")
writing_root = "raw/writing/"
max_samples_split = [25, 25]

# ------------------------ writing_bench ------------------------
data_path = os.path.join(writing_root, "writing_bench", "benchmark_all.jsonl")
data = load_jsonl(data_path)
data = [sample for sample in data if sample["lang"] == "en"]

# randomly shuffle data
random.shuffle(data)

data = data[: max_samples_split[0]]

print(f"Saving {len(data)} writing_bench samples...")

output_data = []
for sample in data:
    meta_data = {
        "domains": [sample["domain1"], sample["domain2"]],
        "source": "writing_bench",
    }

    output_data.append(
        {
            "prompt": sample["query"],
            "answer": sample["checklist"],
            "meta_data": meta_data,
        }
    )

# ------------------------ litbench ------------------------
data_path = os.path.join(writing_root, "litbench", "litbench_train.jsonl")
data = load_jsonl(data_path)

# randomly shuffle data
random.shuffle(data)

template_prompt = "Write a creative short story based on this:\n{writing_prompt}"

print(f"Saving {max_samples_split[1]} litbench samples...")
for sample in data[: max_samples_split[1]]:
    meta_data = {"source": "litbench"}

    answer = f"Story A: {sample['story_a']}\nStory B: {sample['story_b']}\nChosen Story: {sample['chosen_story']}\nRationale: {sample['rationale']}"

    output_data.append(
        {
            "prompt": template_prompt.format(writing_prompt=sample["prompt"]),
            "answer": answer,
            "meta_data": meta_data,
        }
    )

# write to file to output_root
os.makedirs(output_root, exist_ok=True)
output_path = os.path.join(output_root, "writing.json")

print(f"[DONE] Saving {len(output_data)} writing samples")

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)

print("========================================\n")

# ============================== Roleplay =============================
print("=======Generating Roleplay Dataset=======")
roleplay_root = "raw/roleplay/"
max_samples_split = [50]

# ------------------------ character_100------------------------
train_data_path = os.path.join(roleplay_root, "character_100", "train.json")
dev_data_path = os.path.join(roleplay_root, "character_100", "dev.json")

train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
dev_data = json.load(open(dev_data_path, "r", encoding="utf-8"))

data = train_data + dev_data

random.shuffle(data)

output_data = []

data = data[: max_samples_split[0]]

for sample in data:
    meta_data = {"source": "character_100"}

    output_data.append(
        {"prompt": sample["input"], "answer": sample["output"], "meta_data": meta_data}
    )

print(f"Saving {len(output_data)} character_100 samples...")

print(f"[DONE] Saving {len(output_data)} roleplay samples")

with open(os.path.join(output_root, "roleplay.json"), "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)

print("========================================\n")

# ============================== Reasoning =============================
print("=======Generating Reasoning Dataset=======")
reasoning_root = "raw/reasoning/"
max_samples_split = [25, 25]

# ------------------------ logiqa2_logic ------------------------
logic_train_path = os.path.join(reasoning_root, "logiqa2", "logic", "train.txt")
logic_test_path = os.path.join(reasoning_root, "logiqa2", "logic", "test.txt")
logic_dev_path = os.path.join(reasoning_root, "logiqa2", "logic", "dev.txt")


logic_train_data = load_jsonl(logic_train_path)
logic_test_data = load_jsonl(logic_test_path)
logic_dev_data = load_jsonl(logic_dev_path)

data = logic_train_data + logic_test_data + logic_dev_data

random.shuffle(data)

print(f"Saving {max_samples_split[0]} logiqa2_logic samples...")

output_data = []

for sample in data[: max_samples_split[0]]:
    meta_data = {"category": sample["type"], "source": "logiqa2_logic"}

    # build the prompt
    prompt = f"{sample['text']}\nQuestion: {sample['question']}\n"
    for idx, option in enumerate(sample["options"]):
        prompt += f"{chr(65+idx)}. {option}\n"

    output_data.append(
        {
            "prompt": prompt,
            "answer": chr(65 + int(sample["answer"])),
            "meta_data": meta_data,
        }
    )

# =------------------------ logiqa2_nli ------------------------
nli_train_path = os.path.join(reasoning_root, "logiqa2", "nli", "train_new.txt")
nli_test_path = os.path.join(reasoning_root, "logiqa2", "nli", "test_new.txt")
nli_dev_path = os.path.join(reasoning_root, "logiqa2", "nli", "test_new.txt")

nli_train_data = load_jsonl(nli_train_path)
nli_test_data = load_jsonl(nli_test_path)
nli_dev_data = load_jsonl(nli_dev_path)

data = nli_train_data + nli_test_data + nli_dev_data

random.shuffle(data)

print(f"Saving {max_samples_split[1]} logiqa2_nli samples...")
for sample in data[: max_samples_split[1]]:
    meta_data = {"category": sample["type"], "source": "logiqa2_nli"}

    # build the prompt
    # prompt = f"You are given a premise and a hypothesis. Decide whether the hypothesis logically follows from the premise. Answer only with 'entailment' or 'not-entailment' with no other text or explanations. Premise: {sample['premise']} Hypothesis: {sample['hypothesis']}"

    prompt = f"You are given a premise and a hypothesis. Decide whether the hypothesis logically follows from the premise. Premise: {sample['premise']} Hypothesis: {sample['hypothesis']}"

    output_data.append(
        {"prompt": prompt, "answer": sample["label"], "meta_data": meta_data}
    )

print(f"[DONE] Saving {len(output_data)} reasoning samples")
with open(os.path.join(output_root, "reasoning.json"), "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
print("========================================\n")

# ============================== Math =============================
print("=======Generating Math Dataset=======")
math_root = "raw/math/"
max_samples_split = [10, 10, 10, 10, 10]

# ------------------------ mathbench ------------------------
data_path = os.path.join(math_root, "mathbench")

arithmetic_path = os.path.join(data_path, "arithmetic", "cloze_en.jsonl")

primary_path = os.path.join(data_path, "primary", "cloze_en.jsonl")
primary_know_path = os.path.join(
    data_path, "primary_knowledge", "single_choice_en.jsonl"
)

middle_path = os.path.join(data_path, "middle", "single_choice_en.jsonl")
middle_know_path = os.path.join(data_path, "middle_knowledge", "single_choice_en.jsonl")

high_path = os.path.join(data_path, "high", "single_choice_en.jsonl")
high_knowledge_path = os.path.join(
    data_path, "high_knowledge", "single_choice_en.jsonl"
)

college_path = os.path.join(data_path, "college", "single_choice_en.jsonl")
college_knowledge_path = os.path.join(
    data_path, "college_knowledge", "single_choice_en.jsonl"
)

arithmetic_data = load_jsonl(arithmetic_path)

primary_data = load_jsonl(primary_path) + load_jsonl(primary_know_path)
middle_data = load_jsonl(middle_path) + load_jsonl(middle_know_path)
high_data = load_jsonl(high_path) + load_jsonl(high_knowledge_path)
college_data = load_jsonl(college_path) + load_jsonl(college_knowledge_path)

random.shuffle(arithmetic_data)
random.shuffle(primary_data)
random.shuffle(middle_data)
random.shuffle(high_data)
random.shuffle(college_data)

output_data = []

for sample in arithmetic_data[: max_samples_split[0]]:
    meta_data = {
        "category": "arithmetic",
        "topic": sample["topic"],
        "source": "mathbench",
    }

    output_data.append(
        {
            "prompt": sample["question"],
            "answer": sample["answer"],
            "meta_data": meta_data,
        }
    )

for sample in primary_data[: max_samples_split[1]]:
    meta_data = {"category": "primary", "topic": sample["topic"], "source": "mathbench"}

    output_data.append(
        {
            "prompt": sample["question"],
            "answer": sample["answer"],
            "meta_data": meta_data,
        }
    )

for sample in middle_data[: max_samples_split[2]]:
    meta_data = {"category": "middle", "topic": sample["topic"], "source": "mathbench"}

    output_data.append(
        {
            "prompt": sample["question"],
            "answer": sample["answer"],
            "meta_data": meta_data,
        }
    )

for sample in high_data[: max_samples_split[3]]:
    meta_data = {"category": "high", "topic": sample["topic"], "source": "mathbench"}

    output_data.append(
        {
            "prompt": sample["question"],
            "answer": sample["answer"],
            "meta_data": meta_data,
        }
    )

for sample in college_data[: max_samples_split[4]]:
    meta_data = {"category": "college", "topic": sample["topic"], "source": "mathbench"}

    output_data.append(
        {
            "prompt": sample["question"],
            "answer": sample["answer"],
            "meta_data": meta_data,
        }
    )

print(f"Saving {len(output_data)} mathbench samples...")

print(f"[DONE] Saving {len(output_data)} math samples")

with open(os.path.join(output_root, "math.json"), 'w', encoding='utf-8') as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
print("========================================\n")

# ============================== Code =============================
print("=======Generating Code Dataset=======")
code_root = "raw/code/"
max_samples_split = [50]

# ------------------------ LeetCode ------------------------
data_path = os.path.join(code_root, "leetcode", "LeetCodeDataset.jsonl")
data = load_jsonl(data_path)

random.shuffle(data)

output_data = []

target_per = max_samples_split[0] // 3
counts = {"Easy": 0, "Medium": 0, "Hard": 0}
output_data = []

for sample in data:
    d = sample.get("difficulty")
    if d in counts and counts[d] < target_per:
        meta_data = {
            "difficulty": d,
            "tags": sample["tags"],
            "source": "leetcode_dataset",
        }
        output_data.append(
            {
                "prompt": sample["query"],
                "answer": sample["response"],
                "meta_data": meta_data,
            }
        )
        counts[d] += 1
        if sum(counts.values()) == target_per * 3:
            break

print(f"Saving {len(output_data)} leetcode_dataset samples...")

print(f"[DONE] Saving {len(output_data)} code samples")

with open(os.path.join(output_root, "code.json"), "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
print("========================================\n")


# ============================== Extraction ==============================
print("=======Generating Extraction Dataset=======")
extraction_root = "raw/extraction/"
max_samples_split = [50]

# ------------------------ docbench ------------------------
output_data = []

data_path = os.path.join(extraction_root, "docbench", "data")

for sample in tqdm.tqdm(os.listdir(data_path)):
    sample_files = os.listdir(os.path.join(data_path, sample))

    # get jsonl file
    data_file = [f for f in sample_files if f.endswith(".jsonl")][0]
    # get pdf file
    document_file = [f for f in sample_files if f.endswith(".pdf")][0]

    data = load_jsonl(os.path.join(data_path, sample, data_file))

    pdf_content = pdf_to_text(os.path.join(data_path, sample, document_file))

    for question in data:
        if question["type"] != "text-only":
            continue

        meta_data = {"evidence": question["evidence"], "source": "docbench"}

        full_query = f"Context: {pdf_content}\n\nQuestion: {question['question']}\n"

        output_data.append({
            "prompt": full_query,
            "answer": question["answer"],
            "meta_data": meta_data
        })

output_data = output_data[:max_samples_split[0]]

print(f"Saving {len(output_data)} docbench samples...")

print(f"[DONE] Saving {len(output_data)} extraction samples")

with open(os.path.join(output_root, "extraction.json"), 'w', encoding='utf-8') as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
print("========================================\n")

# ============================== stem ==============================
print("=======Generating STEM Dataset=======")
stem_root = "raw/stem/"
max_samples_split = [25, 25]

# ------------------------ arc ------------------------
output_data = []

challenge_data_path = os.path.join(stem_root, "arc", "arc_challenge.json")
challenge_data = load_jsonl(challenge_data_path)

random.shuffle(challenge_data)

for sample in challenge_data[: max_samples_split[0]]:
    meta_data = {"difficulty": "Challenge", "source": "arc_challenge"}

    # prompt = (
    #     sample["question"]
    #     + " Answer with only the letter choice with no other text or explanations."
    # )

    prompt = sample["question"]

    # iterate over text and label together
    for idx, (text, label) in enumerate(
        zip(sample["choices"]["text"], sample["choices"]["label"])
    ):
        prompt += f"\n{label}. {text}"

    output_data.append(
        {"prompt": prompt, "answer": sample["answerKey"], "meta_data": meta_data}
    )

print(f"Saving {max_samples_split[0]} arc_challenge samples...")

easy_data_path = os.path.join(stem_root, "arc", "arc_easy.json")
easy_data = load_jsonl(easy_data_path)

random.shuffle(easy_data)

for sample in easy_data[: max_samples_split[1]]:
    meta_data = {"difficulty": "Easy", "source": "arc_easy"}

    # prompt = (
    #     sample["question"]
    #     + " Answer with only the letter choice with no other text or explanations."
    # )

    prompt = sample["question"]

    # iterate over text and label together
    for idx, (text, label) in enumerate(
        zip(sample["choices"]["text"], sample["choices"]["label"])
    ):
        prompt += f"\n{label}. {text}"

    output_data.append(
        {"prompt": prompt, "answer": sample["answerKey"], "meta_data": meta_data}
    )


print(f"Saving {max_samples_split[1]} arc_easy samples...")

print(f"[DONE] Saving {len(output_data)} stem samples")
with open(os.path.join(output_root, "stem.json"), "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)
print("========================================\n")

# ============================== humanities ==============================
print("=======Generating Humanities Dataset=======")
humanities_root = "raw/humanities/"
max_samples_split = [50]

# ------------------------ mmlu ------------------------
output_data = []

data_path = os.path.join(stem_root, "humanities", "humanities_data.json")
data = load_jsonl(challenge_data_path)

random.shuffle(data)

for sample in data[:max_samples_split[0]]:
    meta_data = {"source": "mmlu_human"}

    # prompt = sample['question'] + " Answer with only the letter choice with no other text or explanations."
    prompt = sample["question"]

    for idx, (text, label) in enumerate(zip(sample["choices"]["text"], sample["choices"]["label"])):
        prompt += f"\n{label}. {text}"

    output_data.append({
        "prompt": prompt,
        "answer": sample["answerKey"],
        "meta_data": meta_data
    })

print(f"Saving {max_samples_split[0]} mmlu_human samples...")

print(f"[DONE] Saving {len(output_data)} humanities samples")

with open(os.path.join(output_root, "humanities.json"), 'w', encoding='utf-8') as file:
    json.dump(output_data, file, indent=2, ensure_ascii=False)

print("========================================\n")

