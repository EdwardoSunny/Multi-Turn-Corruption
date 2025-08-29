import json
from openai import OpenAI
import tqdm
import

client = OpenAI()

root_data = "./clean"

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

def generate_followups(filename):
    data_path = os.path.join(root_data, filename)

    data = read_jsonl(data_path)

    # construct dataset with more turns
    new_data = []
    for sample in tqdm.tqdm(data):
        curr_sample = sample

        prompt = (
            f"Given this questions: {curr_sample['turns']}, generate 10 follow-up questions. "
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
