import json
from openai import OpenAI
import tqdm

client = OpenAI()

data_path = "question.jsonl"

def call_openai(prompt, model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.8
    )
    return response.choices[0].message.content.strip()

# read in jsonl
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    return [json.loads(line) for line in lines]

data = read_jsonl(data_path)

distribution = {}
for sample in data:
    if sample["category"] not in distribution:
        distribution[sample["category"]] = 0

    distribution[sample["category"]] += 1

print(distribution)
exit()
# construct dataset with more turns
new_data = []
for sample in tqdm.tqdm(data):
    curr_sample = sample
    
    prompt = (
        f"Given these questions: {curr_sample['turns']}, generate 10 similar follow-up questions. "
        "Return only a valid JSON array of strings, with no extra text, explanations, or markdown formatting."
    )

    response = call_openai(prompt)
    follow_up_turns = json.loads(response)

    curr_sample["turns"].extend(follow_up_turns)

    new_data.append(curr_sample)

# write new_data to jsonl
with open("new_question.jsonl", 'w', encoding='utf-8') as file:
    for sample in new_data:
        file.write(json.dumps(sample, ensure_ascii=False) + '\n')
