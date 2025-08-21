import json
from openai import OpenAI
import re
import tqdm
from collections import defaultdict
import numpy as np
import datetime
import random
import os
import concurrent.futures
import threading
from functools import partial

client = OpenAI()
data_path = "new_question.jsonl"

# Configuration
NUM_EXPERIMENTS = 1  # Number of times to repeat the entire experiment
MAX_WORKERS = 2  # Adjust based on your OpenAI rate limits

# Thread-safe logging
log_lock = threading.Lock()

# Create logs directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"experiment_logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# Judge prompts:
writing_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these WRITING dimensions (use them to guide your comparison):
1) Clarity & Coherence
2) Structure & Organization
3) Creativity & Originality
4) Adherence to Prompt

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "writing",
  "answer_1": {
    "clarity_coherence": 1-5,
    "structure_organization": 1-5,
    "creativity_originality": 1-5,
    "adherence_to_prompt": 1-5
  },
  "answer_2": {
    "clarity_coherence": 1-5,
    "structure_organization": 1-5,
    "creativity_originality": 1-5,
    "adherence_to_prompt": 1-5
  }
}
"""

roleplay_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these ROLEPLAY dimensions (use them to guide your comparison):
1) Character Consistency
2) Immersion & Believability
3) Empathy & Responsiveness
4) Tone Matching

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "roleplay",
  "answer_1": {
    "character_consistency": 1-5,
    "immersion_believability": 1-5,
    "empathy_responsiveness": 1-5,
    "tone_matching": 1-5
  },
  "answer_2": {
    "character_consistency": 1-5,
    "immersion_believability": 1-5,
    "empathy_responsiveness": 1-5,
    "tone_matching": 1-5
  }
}
"""

reasoning_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these REASONING dimensions (use them to guide your comparison):
1) Logical Coherence
2) Depth of Analysis
3) Transparency
4) Relevance

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "reasoning",
  "answer_1": {
    "logical_coherence": 1-5,
    "depth_of_analysis": 1-5,
    "transparency": 1-5,
    "relevance": 1-5
  },
  "answer_2": {
    "logical_coherence": 1-5,
    "depth_of_analysis": 1-5,
    "transparency": 1-5,
    "relevance": 1-5
  }
}
"""

math_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these MATH dimensions (use them to guide your comparison):
1) Correctness
2) Step-by-Step Clarity
3) Problem Understanding
4) Completeness

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "math",
  "answer_1": {
    "correctness": 1-5,
    "step_by_step_clarity": 1-5,
    "problem_understanding": 1-5,
    "completeness": 1-5
  },
  "answer_2": {
    "correctness": 1-5,
    "step_by_step_clarity": 1-5,
    "problem_understanding": 1-5,
    "completeness": 1-5
  }
}
"""

coding_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these CODING dimensions (use them to guide your comparison):
1) Correctness
2) Readability
3) Efficiency
4) Error Handling & Robustness

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "coding",
  "answer_1": {
    "correctness": 1-5,
    "readability": 1-5,
    "efficiency": 1-5,
    "error_handling_robustness": 1-5
  },
  "answer_2": {
    "correctness": 1-5,
    "readability": 1-5,
    "efficiency": 1-5,
    "error_handling_robustness": 1-5
  }
}
"""

extraction_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these EXTRACTION dimensions (use them to guide your comparison):
1) Accuracy
2) Completeness
3) Relevance
4) Consistency

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "extraction",
  "answer_1": {
    "accuracy": 1-5,
    "completeness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5
  },
  "answer_2": {
    "accuracy": 1-5,
    "completeness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5
  }
}
"""

stem_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these STEM dimensions (use them to guide your comparison):
1) Scientific Accuracy
2) Conceptual Depth
3) Explanatory Clarity
4) Application

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "stem",
  "answer_1": {
    "scientific_accuracy": 1-5,
    "conceptual_depth": 1-5,
    "explanatory_clarity": 1-5,
    "application": 1-5
  },
  "answer_2": {
    "scientific_accuracy": 1-5,
    "conceptual_depth": 1-5,
    "explanatory_clarity": 1-5,
    "application": 1-5
  }
}
"""

humanities_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given two answers numbered 1 and 2 and you must evaluate them using the following method:

Begin your evaluation by comparing the two answers. Identify and correct any mistakes. Be as objective as possible.

Evaluate ONLY on these HUMANITIES dimensions (use them to guide your comparison):
1) Interpretive Depth
2) Contextual Awareness
3) Clarity of Argument
4) Use of Evidence

Score these dimensions by comparing the answers against each other and seeing which one is better or worse at certain categories.

Scoring for each dimension (per answer): 1=Poor, 2=Weak, 3=Adequate, 4=Good, 5=Excellent.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation comparing answer 1 and answer 2 (1–5 short sentences).
2) Then output ONLY this JSON object with per-dimension scores for BOTH answers and nothing else:
{
  "domain": "humanities",
  "answer_1": {
    "interpretive_depth": 1-5,
    "contextual_awareness": 1-5,
    "clarity_of_argument": 1-5,
    "use_of_evidence": 1-5
  },
  "answer_2": {
    "interpretive_depth": 1-5,
    "contextual_awareness": 1-5,
    "clarity_of_argument": 1-5,
    "use_of_evidence": 1-5
  }
}
"""

category_to_judge_prompt_map = {
    "writing": writing_prompt,
    "roleplay": roleplay_prompt,
    "reasoning": reasoning_prompt,
    "math": math_prompt,
    "coding": coding_prompt,
    "extraction": extraction_prompt,
    "stem": stem_prompt,
    "humanities": humanities_prompt
}

print(f"📁 Logging to directory: {log_dir}")

def call_openai(messages, model="gpt-4.1", temperature=0.8):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def safe_log_write(filename, log_entry):
    """Thread-safe logging function"""
    with log_lock:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def calculate_preference_stats(scores):
    """Calculate preference statistics from 3-way scores (0=tie, 1=reference, 2=context)"""
    if not scores:
        return {
            "context_preference_rate": 0.0,
            "reference_preference_rate": 0.0,
            "tie_rate": 0.0,
            "total_comparisons": 0,
            "context_wins": 0,
            "reference_wins": 0,
            "ties": 0
        }

    context_wins = scores.count(2)
    reference_wins = scores.count(1)
    ties = scores.count(0)
    total = len(scores)

    return {
        "context_preference_rate": context_wins / total if total > 0 else 0,
        "reference_preference_rate": reference_wins / total if total > 0 else 0,
        "tie_rate": ties / total if total > 0 else 0,
        "total_comparisons": total,
        "context_wins": context_wins,
        "reference_wins": reference_wins,
        "ties": ties
    }

def judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id=None, exp_id=None, turn_type=None):
    """
    Judges two answers, parses detailed scores, determines a winner, and logs everything.
    - ref_answer is mapped to answer_1
    - assistant_answer (with context) is mapped to answer_2
    """
    system_prompt = category_to_judge_prompt_map[target_category]
    prompt = f"1. {ref_answer}\n\n2. {assistant_answer}\n\nPlease provide your evaluation as specified."
    response = call_openai([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature=0.0)

    pattern = re.compile(r'\{.*\}', re.DOTALL)
    match = pattern.search(response)

    if not match:
        # Retry if JSON parsing fails
        return judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id, exp_id, turn_type)

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        ref_scores = data.get('answer_1', {})
        context_scores = data.get('answer_2', {})
        
        # Ensure scores are dicts before summing
        ref_total = sum(ref_scores.values()) if isinstance(ref_scores, dict) else 0
        context_total = sum(context_scores.values()) if isinstance(context_scores, dict) else 0

    except json.JSONDecodeError:
        # Retry if JSON is invalid
        return judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id, exp_id, turn_type)

    # Determine winner: 0=tie, 1=reference preferred, 2=context preferred
    if context_total > ref_total:
        choice = 2
    elif ref_total > context_total:
        choice = 1
    else:
        choice = 0

    choice_interpretation = {0: "tie", 1: "reference_preferred", 2: "context_preferred"}

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "sample_id": sample_id,
        "turn_type": turn_type,
        "question": latest_question,
        "reference_answer": ref_answer,
        "context_answer": assistant_answer,
        "judge_response_raw": response,
        "judge_parsed_scores": data,
        "reference_total_score": ref_total,
        "context_total_score": context_total,
        "judge_choice": choice,
        "choice_interpretation": choice_interpretation[choice],
    }

    log_filename = f"{log_dir}/judge_logs_exp{exp_id}.jsonl"
    safe_log_write(log_filename, log_entry)

    return {"choice": choice, "scores": data}

def generate_mt_response(turn_questions, curr_question, sample_id=None, exp_id=None, turn_type=None):
    """
    Generates a response from the model, either with or without context turns.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    conversation_log = []

    # Add context turns
    for prompt in turn_questions:
        messages.append({"role": "user", "content": prompt})
        response = call_openai(messages)
        messages.append({"role": "assistant", "content": response})
        conversation_log.append({"user": prompt, "assistant": response})

    # Add current question and get the final response
    messages.append({"role": "user", "content": curr_question})
    final_response = call_openai(messages)
    conversation_log.append({"user": curr_question, "assistant": final_response})

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "sample_id": sample_id,
        "turn_type": turn_type,
        "num_context_turns": len(turn_questions),
        "conversation": conversation_log,
        "final_response": final_response
    }

    log_filename = f"{log_dir}/conversation_logs_exp{exp_id}.jsonl"
    safe_log_write(log_filename, log_entry)

    return final_response

def generate_and_judge_single_response(context_turns, target_question, reference_response, target_category, sample_id, exp_id, turn_type):
    """Generates a single response and judges it - designed for parallel execution."""
    # Generate response with context
    contextual_response = generate_mt_response(context_turns, target_question, sample_id, exp_id, turn_type)
    
    # Judge the response against the reference
    judgement = judge_response(target_question, reference_response, contextual_response, target_category, sample_id, exp_id, turn_type)
    
    return turn_type, judgement

def process_single_sample(context_sample, curr_sample, exp_id, sample_id, context_type):
    """
    Processes a single sample by generating a reference response and comparing it
    against contextual responses with varying numbers of turns.
    """
    target_question = curr_sample["turns"][0]
    target_category = curr_sample["category"]

    # Generate reference response (no context)
    reference_response = generate_mt_response([], target_question, sample_id, exp_id, "reference")

    context_configs = [
        (context_sample["turns"][:1], f"{context_type}_1_turn"),
        (context_sample["turns"][:3], f"{context_type}_3_turns"),
        (context_sample["turns"][:6], f"{context_type}_6_turns"),
        (context_sample["turns"][:9], f"{context_type}_9_turns"),
        (context_sample["turns"][:12], f"{context_type}_12_turns"),
    ]

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_turn_type = {
            executor.submit(
                generate_and_judge_single_response,
                context_turns, target_question, reference_response, target_category,
                sample_id, exp_id, turn_type
            ): turn_type
            for context_turns, turn_type in context_configs
        }

        for future in concurrent.futures.as_completed(future_to_turn_type):
            turn_type_full = future_to_turn_type[future]
            turn_key = turn_type_full.replace(f"{context_type}_", "")
            try:
                _, judgement = future.result()
                results[turn_key] = judgement
            except Exception as exc:
                print(f'Turn {turn_type_full} generated an exception: {exc}')
                results[turn_key] = {"choice": -1, "scores": {}} # Error code

    summary_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "sample_id": sample_id,
        "context_type": context_type,
        "target_category": curr_sample["category"],
        "context_category": context_sample["category"],
        "target_question": target_question,
        "results": results # Contains both choice and detailed scores
    }

    log_filename = f"{log_dir}/sample_summary_exp{exp_id}.jsonl"
    safe_log_write(log_filename, summary_log)

    return {
        'experiment_id': exp_id,
        'sample_id': sample_id,
        'context_type': context_type,
        'target_category': curr_sample["category"],
        'context_category': context_sample["category"],
        'results': results
    }

def process_sample_with_both_contexts(sample_idx, sample, data, exp_id):
    """Processes a single sample with both same and different category contexts."""
    current_category = sample['category']
    
    same_category_samples = [s for s in data if s['category'] == current_category and s != sample]
    diff_category_samples = [s for s in data if s['category'] != current_category]
    
    if not same_category_samples or not diff_category_samples:
        return None, None
    
    same_category_context = random.choice(same_category_samples)
    diff_category_context = random.choice(diff_category_samples)
    
    same_category_result = process_single_sample(same_category_context, sample, exp_id, sample_idx, "same_category")
    diff_category_result = process_single_sample(diff_category_context, sample, exp_id, sample_idx, "different_category")
    
    return same_category_result, diff_category_result

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file.readlines()]

def run_single_experiment(data, exp_id):
    """Runs a single experiment, collecting detailed results."""
    print(f"\n🔬 Running Experiment {exp_id + 1}/{NUM_EXPERIMENTS}")
    
    # Store all detailed results for aggregation
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        process_func = partial(process_sample_with_both_contexts, data=data, exp_id=exp_id)
        
        future_to_sample = {
            executor.submit(process_func, i, sample): i
            for i, sample in enumerate(data)
        }
        
        with tqdm.tqdm(total=len(data), desc=f"Experiment {exp_id + 1}") as pbar:
            for future in concurrent.futures.as_completed(future_to_sample):
                try:
                    same_cat_res, diff_cat_res = future.result()
                    if same_cat_res:
                        all_results.append(same_cat_res)
                    if diff_cat_res:
                        all_results.append(diff_cat_res)
                except Exception as exc:
                    sample_idx = future_to_sample[future]
                    print(f'Sample {sample_idx} generated an exception: {exc}')
                pbar.update(1)

    # Log experiment summary
    exp_summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "total_samples_processed": len(data),
        "raw_results": all_results
    }
    
    log_filename = f"{log_dir}/experiment_summary.jsonl"
    safe_log_write(log_filename, exp_summary)
    
    return all_results

# --- Main Execution ---
data = read_jsonl(data_path)
distribution = {cat: sum(1 for s in data if s['category'] == cat) for cat in set(s['category'] for s in data)}

print("Distribution:", distribution)
print(f"Running {NUM_EXPERIMENTS} experiments with {len(data)} samples each")
print(f"Using {MAX_WORKERS} parallel workers")
# 1 ref gen + (5 context gen + 5 judge calls) per context type = 1 + 10 + 10 = 21
print(f"Total API calls will be approximately: {len(data) * 21 * NUM_EXPERIMENTS}")

# Run all experiments and collect raw results
all_exp_raw_results = []
for exp_id in range(NUM_EXPERIMENTS):
    exp_results = run_single_experiment(data, exp_id)
    all_exp_raw_results.extend(exp_results)

# --- Final Aggregation and Reporting ---
print("\n📊 Calculating final statistics across all experiments...")

final_results = {}
turn_keys = ["1_turn", "3_turns", "6_turns", "9_turns", "12_turns"]

# Initialize structure
for context_type in ["same_category", "different_category"]:
    final_results[context_type] = defaultdict(lambda: {
        "preference_scores": {turn: [] for turn in turn_keys},
        "dimensional_scores": {
            "reference": {turn: defaultdict(list) for turn in turn_keys},
            "context": {turn: defaultdict(list) for turn in turn_keys}
        }
    })

# Populate with data from all experiments
for res in all_exp_raw_results:
    context_type = res['context_type']
    category = res['target_category']
    
    for turn_key, judgement in res['results'].items():
        if turn_key in turn_keys:
            # Aggregate preference scores
            final_results[context_type][category]["preference_scores"][turn_key].append(judgement['choice'])
            
            # Aggregate dimensional scores
            if isinstance(judgement.get('scores'), dict):
                ref_scores = judgement['scores'].get('answer_1', {})
                context_scores = judgement['scores'].get('answer_2', {})
                if isinstance(ref_scores, dict):
                    for dim, score in ref_scores.items():
                        final_results[context_type][category]["dimensional_scores"]["reference"][turn_key][dim].append(score)
                if isinstance(context_scores, dict):
                    for dim, score in context_scores.items():
                        final_results[context_type][category]["dimensional_scores"]["context"][turn_key][dim].append(score)


# Calculate final statistics
final_stats = {}
for context_type, categories in final_results.items():
    final_stats[context_type] = {}
    for category, data in categories.items():
        final_stats[context_type][category] = {
            "preference_stats": {},
            "dimensional_stats": {
                "reference": {turn: {} for turn in turn_keys},
                "context": {turn: {} for turn in turn_keys}
            }
        }
        # Calculate preference stats
        for turn_key, scores in data['preference_scores'].items():
            final_stats[context_type][category]["preference_stats"][turn_key] = calculate_preference_stats(scores)
        
        # Calculate dimensional stats (averages)
        for model_type in ["reference", "context"]:
            for turn_key, dims in data['dimensional_scores'][model_type].items():
                for dim, scores in dims.items():
                    avg_score = np.mean(scores) if scores else 0
                    final_stats[context_type][category]["dimensional_stats"][model_type][turn_key][dim] = round(avg_score, 3)

# --- Print Final Report ---
for context_type, stats_by_cat in final_stats.items():
    print(f"\n\n{'='*35} {context_type.replace('_', ' ').upper()} CONTEXT RESULTS {'='*35}")
    for category, cat_stats in sorted(stats_by_cat.items()):
        
        # --- Preference Report ---
        print(f"\n📂 Category: {category} - Preference Report")
        print(f"{'Turn':<12} {'Context %':<10} {'Ref %':<10} {'Tie %':<10} {'C/R/T':<12} {'Total':<8}")
        print("-" * 65)
        for turn_key in turn_keys:
            s = cat_stats["preference_stats"].get(turn_key, {})
            if s:
                counts_str = f"{s.get('context_wins',0)}/{s.get('reference_wins',0)}/{s.get('ties',0)}"
                print(f"{turn_key:<12} {s.get('context_preference_rate',0):<10.3f} {s.get('reference_preference_rate',0):<10.3f} {s.get('tie_rate',0):<10.3f} {counts_str:<12} {s.get('total_comparisons',0):<8}")

        # --- Dimensional Score Report ---
        print(f"\n📂 Category: {category} - Dimensional Score Report (Avg Score 1-5)")
        dim_stats = cat_stats.get("dimensional_stats", {})
        
        # Get all unique dimension names for header
        all_dims = set()
        if dim_stats.get("context", {}).get("1_turn", {}):
            all_dims.update(dim_stats["context"]["1_turn"].keys())
        
        if not all_dims: continue
        
        header = f"{'Turn':<12} {'Model':<12}" + "".join([f"{dim[:12]:<15}" for dim in sorted(list(all_dims))])
        print(header)
        print("-" * len(header))
        
        for turn_key in turn_keys:
            for model_type in ["Reference", "Context"]:
                row = f"{turn_key:<12} {model_type:<12}"
                model_data = dim_stats.get(model_type.lower(), {}).get(turn_key, {})
                for dim in sorted(list(all_dims)):
                    avg_score = model_data.get(dim, 0.0)
                    row += f"{avg_score:<15.3f}"
                print(row)
            if turn_key != turn_keys[-1]: print("." * len(header)) # Separator line


# Save final combined results to a JSON file
output_data = {
    "experiment_config": {
        "num_experiments": NUM_EXPERIMENTS,
        "num_samples_per_experiment": len(data),
    },
    "final_statistics": final_stats,
    "dataset_distribution": distribution
}

with open("combined_results.json", 'w', encoding='utf-8') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=2)

print(f"\n\n💾 Final aggregated results saved to: combined_results.json")
print(f"📁 Raw logs for debugging are in the directory: {log_dir}/")
print("\n🎯 Experiment completed!")
