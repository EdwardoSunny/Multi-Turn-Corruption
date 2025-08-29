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
writing_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four WRITING dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

WRITING DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Clarity & Coherence
- 5 (Excellent): Meaning is always clear on first read; zero contradictions; precise wording; logical flow with no confusing jumps; ≤1 minor grammar/usage error per 200 words.
- 4 (Good): Mostly clear with rare minor ambiguities that do not impede understanding; flow is solid; ≤3 minor errors per 200 words; no material contradictions.
- 3 (Adequate): Understandable overall but with noticeable awkward phrases, vague references, or a few choppy transitions; may have one minor lapse in logic; ≤6 minor errors per 200 words.
- 2 (Weak): Frequent ambiguity or muddled passages; multiple confusing jumps or unclear references; may include a contradiction; >6 minor errors or ≥1 major error per 200 words.
- 1 (Poor): Hard to follow; persistent ambiguity or incoherence; major logical breakdowns or contradictions; pervasive grammar issues.

2) Structure & Organization
- 5 (Excellent): Clear beginning, middle, and end; strong topic sentences; purposeful paragraphing; smooth transitions; information order optimizes reader comprehension.
- 4 (Good): Recognizable structure with effective paragraphing and mostly smooth transitions; minor ordering issues that do not harm comprehension.
- 3 (Adequate): Basic structure present but uneven (e.g., weak intro or conclusion, some mixed topics in paragraphs); transitions are inconsistent.
- 2 (Weak): Disorganized sections, missing key structural elements, or paragraphs that mix unrelated points; transitions largely absent.
- 1 (Poor): No discernible structure; ideas appear in a haphazard order; paragraphs (if any) do not group related ideas.

3) Creativity & Originality
- 5 (Excellent): Presents clearly original phrasing or framing; includes ≥2 specific, non-cliché examples/analogies or insightful observations that enhance understanding; avoids templates.
- 4 (Good): Some fresh phrasing or at least 1 specific, apt example/analogy; limited reliance on clichés or boilerplate.
- 3 (Adequate): Mostly conventional phrasing; ideas are correct but familiar; may use 1 mild cliché; few, if any, specific examples.
- 2 (Weak): Heavily generic or templated; clichés common; no meaningful examples or novel angles.
- 1 (Poor): Derivative or rote; reads as copy-paste boilerplate; no original elements.

4) Adherence to Prompt
Assess ONLY compliance with these instructions: (a) acts as an impartial judge, (b) focuses strictly on the four writing dimensions above, (c) produces the required two-part OUTPUT FORMAT exactly (concise explanation + JSON), (d) no extra content before/after the JSON, (e) correct JSON keys and integer scores 1–5.
- 5 (Excellent): Fully compliant with (a)–(e) with zero deviations.
- 4 (Good): One minor deviation that does not change meaning or format (e.g., slightly longer explanation).
- 3 (Adequate): Up to two minor deviations OR one moderate deviation (e.g., mild overreach beyond writing dimensions, but still outputs correct JSON).
- 2 (Weak): Multiple deviations or one major deviation (e.g., adds sections outside required outputs, or partially wrong JSON keys).
- 1 (Poor): Fails core instructions (e.g., ignores dimensions, or JSON malformed/missing).

SCORING RULES (for consistency):
- Count a “minor error” as a small grammar/usage/punctuation issue that does not alter meaning; a “major error” obscures meaning or changes it.
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer text must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "writing",
    "clarity_coherence": 1-5,
    "structure_organization": 1-5,
    "creativity_originality": 1-5,
    "adherence_to_prompt": 1-5
  },
  "scores_B": {
    "domain": "writing",
    "clarity_coherence": 1-5,
    "structure_organization": 1-5,
    "creativity_originality": 1-5,
    "adherence_to_prompt": 1-5
  }
}
"""

roleplay_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four ROLEPLAY dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

ROLEPLAY DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Character Consistency
- 5 (Excellent): The assistant stays fully in character throughout with no breaks; actions, wording, and style match the established persona perfectly.
- 4 (Good): Stays in character with only 1 minor slip or slightly out-of-character phrasing; overall consistent persona.
- 3 (Adequate): Mostly consistent but includes 2–3 noticeable slips, contradictions, or moments where the persona weakens.
- 2 (Weak): Frequent lapses in character; tone often drifts; roleplay persona only loosely maintained.
- 1 (Poor): Little or no character consistency; assistant sounds generic or out of character most of the time.

2) Immersion & Believability
- 5 (Excellent): Creates a vivid, immersive experience with rich detail, natural dialogue, and believable reactions; user can easily “forget” they are talking to an AI.
- 4 (Good): Generally immersive and believable; some details enhance realism; minor weak spots in believability.
- 3 (Adequate): Immersion present but thin; descriptions or reactions feel generic; noticeable gaps in believability.
- 2 (Weak): Minimal immersion; responses feel flat or artificial; difficult for user to stay engaged.
- 1 (Poor): No immersion; response breaks suspension of disbelief immediately.

3) Empathy & Responsiveness
- 5 (Excellent): Demonstrates deep emotional attunement; acknowledges user’s cues; responses feel supportive, adaptive, and context-aware.
- 4 (Good): Shows empathy and adjusts to user reasonably well; some responses could be more nuanced.
- 3 (Adequate): Some recognition of user’s emotions or context but inconsistent or superficial.
- 2 (Weak): Rare or minimal empathy; responses feel mechanical or dismissive; little adaptation to user.
- 1 (Poor): No empathy; completely ignores user’s emotions or context.

4) Tone Matching
- 5 (Excellent): Tone perfectly aligns with the roleplay scenario and user’s expectations; consistent style enhances immersion.
- 4 (Good): Tone mostly fits the roleplay; one or two mismatched phrases but overall appropriate.
- 3 (Adequate): Tone somewhat fits but is uneven; multiple mismatches or generic phrasing.
- 2 (Weak): Tone often mismatched; breaks roleplay mood repeatedly.
- 1 (Poor): Tone entirely inappropriate or inconsistent with the roleplay.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer text must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "roleplay",
    "character_consistency": 1-5,
    "immersion_believability": 1-5,
    "empathy_responsiveness": 1-5,
    "tone_matching": 1-5
  },
  "scores_B": {
    "domain": "roleplay",
    "character_consistency": 1-5,
    "immersion_believability": 1-5,
    "empathy_responsiveness": 1-5,
    "tone_matching": 1-5
  }
}
"""

reasoning_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four REASONING dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

REASONING DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Logical Coherence
- 5 (Excellent): Argument is fully consistent; no contradictions; every step follows logically; clear cause-effect or reasoning chain with no gaps.
- 4 (Good): Mostly coherent; one minor lapse or small unexplained jump; reasoning is still understandable overall.
- 3 (Adequate): Reasoning is generally followable but contains 2–3 unclear or weakly connected steps; may include minor inconsistency.
- 2 (Weak): Frequent lapses in logic; multiple unsupported jumps; reasoning often hard to follow.
- 1 (Poor): Largely incoherent; contradictory or circular reasoning; impossible to follow the logic.

2) Depth of Analysis
- 5 (Excellent): Thorough exploration of the question; considers multiple angles, nuances, or counterarguments; demonstrates strong insight.
- 4 (Good): Provides some depth with at least one nuanced observation; analysis goes beyond surface level but misses some angles.
- 3 (Adequate): Basic analysis; mostly surface-level; limited exploration of complexity; may miss key aspects.
- 2 (Weak): Shallow discussion; minimal elaboration; ignores important considerations.
- 1 (Poor): No meaningful analysis; oversimplified or purely restates the question.

3) Transparency
- 5 (Excellent): Clearly explains reasoning process step by step; assumptions stated explicitly; user can easily trace how conclusions were reached.
- 4 (Good): Mostly transparent; explains key reasoning steps but omits a few minor assumptions or intermediate links.
- 3 (Adequate): Some explanation of reasoning but inconsistent; at least half the steps or assumptions are unstated.
- 2 (Weak): Reasoning process is mostly hidden; only scattered hints at underlying logic.
- 1 (Poor): No explanation of reasoning; conclusions presented with no justification.

4) Relevance
- 5 (Excellent): Entire response is tightly focused on the user’s question; no digressions; every part contributes to answering it.
- 4 (Good): Mostly relevant; one minor digression or redundant point but overall on target.
- 3 (Adequate): Some off-topic material or filler; at least 70% directly relevant to the question.
- 2 (Weak): Frequent digressions; less than half the response directly addresses the question.
- 1 (Poor): Mostly irrelevant; response does not meaningfully address the question.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer text must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "reasoning",
    "logical_coherence": 1-5,
    "depth_of_analysis": 1-5,
    "transparency": 1-5,
    "relevance": 1-5
  },
  "scores_B": {
    "domain": "reasoning",
    "logical_coherence": 1-5,
    "depth_of_analysis": 1-5,
    "transparency": 1-5,
    "relevance": 1-5
  }
}
"""

math_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four MATH dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

MATH DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Correctness
- 5 (Excellent): Final answer is fully correct; all intermediate steps accurate; no mathematical errors.
- 4 (Good): Final answer correct with one minor arithmetic/notation slip OR intermediate step error that does not affect final result.
- 3 (Adequate): Final answer correct but with multiple minor errors OR final answer incorrect due to a small arithmetic/notation mistake while reasoning is mostly correct.
- 2 (Weak): Major conceptual or procedural mistake leads to incorrect final answer; some parts correct but reasoning flawed.
- 1 (Poor): Entirely incorrect reasoning and final answer; no meaningful correctness.

2) Step-by-Step Clarity
- 5 (Excellent): Each step is shown clearly and logically; no gaps; easy to follow throughout.
- 4 (Good): Most steps are clear; 1–2 small gaps or slightly rushed explanations, but overall understandable.
- 3 (Adequate): Some steps shown but with noticeable jumps; reader must infer key parts of the reasoning.
- 2 (Weak): Steps mostly unclear or missing; reasoning hard to follow.
- 1 (Poor): No coherent step-by-step explanation; reasoning opaque.

3) Problem Understanding
- 5 (Excellent): Demonstrates full understanding of the problem; correctly interprets all elements, constraints, and what is being asked.
- 4 (Good): Understands the problem well; one minor misinterpretation or overlooked detail.
- 3 (Adequate): General grasp of problem but at least one significant misunderstanding or misinterpretation.
- 2 (Weak): Partial understanding; overlooks key aspects or misinterprets core of the question.
- 1 (Poor): Fundamental misunderstanding of the problem; does not address what was asked.

4) Completeness
- 5 (Excellent): Fully solves the problem; includes final answer and all necessary justifications; nothing essential missing.
- 4 (Good): Mostly complete; final answer present but explanation slightly abbreviated OR misses a minor supporting detail.
- 3 (Adequate): Partial solution; provides some reasoning but skips important parts OR does not fully justify answer.
- 2 (Weak): Incomplete solution; only fragments of reasoning provided; final answer may be missing.
- 1 (Poor): No meaningful attempt to solve; response incomplete or irrelevant.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer solutions must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "math",
    "correctness": 1-5,
    "step_by_step_clarity": 1-5,
    "problem_understanding": 1-5,
    "completeness": 1-5
  },
  "scores_B": {
    "domain": "math",
    "correctness": 1-5,
    "step_by_step_clarity": 1-5,
    "problem_understanding": 1-5,
    "completeness": 1-5
  }
}
"""

coding_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four CODING dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

CODING DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Correctness
- 5 (Excellent): Code fully correct; produces intended results across typical inputs; no syntax or logic errors.
- 4 (Good): Code correct overall; minor syntax/logic slip easily fixed without changing structure; still works for intended purpose.
- 3 (Adequate): Code mostly works but contains 1–2 significant mistakes (logic, syntax, or edge cases) that prevent it from being fully correct without debugging.
- 2 (Weak): Code has major flaws; incorrect outputs for many cases; not runnable without substantial fixes.
- 1 (Poor): Entirely incorrect; code does not compile/run or bears little relation to the problem.

2) Readability
- 5 (Excellent): Code is very clear; good naming, formatting, and structure; easy for another programmer to read and understand.
- 4 (Good): Mostly clear; minor naming or formatting issues but overall understandable.
- 3 (Adequate): Readable with some effort; inconsistent naming/formatting; structure could confuse readers.
- 2 (Weak): Hard to read; poor formatting and naming; structure unclear.
- 1 (Poor): Unreadable; no meaningful formatting, poor style, or disorganized code.

3) Efficiency
- 5 (Excellent): Code uses optimal or near-optimal approach for the problem; efficient time and space complexity.
- 4 (Good): Reasonably efficient; not optimal but still acceptable for normal input sizes.
- 3 (Adequate): Works but inefficient; noticeable redundancy or unnecessary complexity; may not scale well.
- 2 (Weak): Very inefficient; poor algorithmic choices; performance issues likely even at moderate input sizes.
- 1 (Poor): Extremely inefficient; approach unusable in practice.

4) Error Handling & Robustness
- 5 (Excellent): Code anticipates and handles edge cases and errors gracefully; robust under varied inputs.
- 4 (Good): Handles most errors/edge cases; minor gaps but generally reliable.
- 3 (Adequate): Some basic error handling (e.g., minimal checks); several important cases not covered.
- 2 (Weak): Rare or poor error handling; fails on common edge cases.
- 1 (Poor): No error handling; fragile and breaks easily with unexpected inputs.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer solutions must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "coding",
    "correctness": 1-5,
    "readability": 1-5,
    "efficiency": 1-5,
    "error_handling_robustness": 1-5
  },
  "scores_B": {
    "domain": "coding",
    "correctness": 1-5,
    "readability": 1-5,
    "efficiency": 1-5,
    "error_handling_robustness": 1-5
  }
}
"""

extraction_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four EXTRACTION dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

EXTRACTION DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Accuracy
- 5 (Excellent): Extracted information exactly matches the source or intended content with no errors.
- 4 (Good): Extracted information is correct with only one minor slip (e.g., small wording difference that does not change meaning).
- 3 (Adequate): Mostly accurate but includes 2–3 minor errors OR one moderate misrepresentation.
- 2 (Weak): Several inaccuracies; core meaning partially lost.
- 1 (Poor): Mostly inaccurate; extracted content does not reflect the source meaning.

2) Completeness
- 5 (Excellent): Fully extracts all required/expected information with nothing essential missing.
- 4 (Good): Nearly complete; one minor piece of relevant information missing.
- 3 (Adequate): Partially complete; some important details missing but the main point present.
- 2 (Weak): Large gaps; misses multiple key elements.
- 1 (Poor): Very incomplete; extracts only fragments or irrelevant parts.

3) Relevance
- 5 (Excellent): Every extracted element is directly relevant to the question/task; no extraneous material.
- 4 (Good): Mostly relevant; one minor piece of extra or tangential information included.
- 3 (Adequate): Some irrelevant or off-topic content; at least 70% relevant.
- 2 (Weak): Many irrelevant details; less than half of content directly relevant.
- 1 (Poor): Largely irrelevant; extraction does not match the question/task.

4) Consistency
- 5 (Excellent): Extracted content is internally consistent; no contradictions; terminology and details uniform.
- 4 (Good): One minor inconsistency or slight terminology mismatch; overall coherent.
- 3 (Adequate): A few inconsistencies that do not fully undermine understanding.
- 2 (Weak): Multiple inconsistencies; makes interpretation confusing.
- 1 (Poor): Highly inconsistent; contradictory or unstable extraction.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer extractions must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "extraction",
    "accuracy": 1-5,
    "completeness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5
  },
  "scores_B": {
    "domain": "extraction",
    "accuracy": 1-5,
    "completeness": 1-5,
    "relevance": 1-5,
    "consistency": 1-5
  }
}
"""

stem_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four STEM dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

STEM DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Scientific Accuracy
- 5 (Excellent): Fully accurate; all facts, data, and explanations correct; no scientific errors.
- 4 (Good): Mostly accurate; one minor factual slip or slightly imprecise wording that does not alter meaning.
- 3 (Adequate): Generally accurate but contains 2–3 minor errors OR one moderate error affecting part of the explanation.
- 2 (Weak): Multiple inaccuracies or one major error undermining correctness.
- 1 (Poor): Largely inaccurate; fundamental misunderstandings or pervasive scientific errors.

2) Conceptual Depth
- 5 (Excellent): Thorough explanation showing strong grasp of underlying concepts; explores nuances and relationships; goes beyond surface detail.
- 4 (Good): Solid conceptual coverage with some nuance; one or two areas could be developed further.
- 3 (Adequate): Basic conceptual treatment; some understanding shown but leaves gaps or oversimplifies.
- 2 (Weak): Shallow understanding; significant gaps; important concepts missing or confused.
- 1 (Poor): Very little or no conceptual understanding demonstrated.

3) Explanatory Clarity
- 5 (Excellent): Explanation crystal clear; well-structured, precise wording; easy to understand even for non-expert; no ambiguity.
- 4 (Good): Generally clear and structured; minor awkward phrasing or small gaps but still understandable.
- 3 (Adequate): Somewhat clear; noticeable vagueness or awkwardness; requires effort to follow.
- 2 (Weak): Often unclear; poorly structured; confusing wording.
- 1 (Poor): Very unclear; explanation incoherent or unreadable.

4) Application
- 5 (Excellent): Effectively connects concepts to practical examples, real-world contexts, or problem-solving scenarios; enhances understanding.
- 4 (Good): Includes at least one relevant example or application; helpful but not deeply developed.
- 3 (Adequate): Mentions application vaguely or superficially; limited examples.
- 2 (Weak): Minimal or poorly chosen applications; little added value.
- 1 (Poor): No attempt to connect concepts to examples, applications, or practical use.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer answers must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "stem",
    "scientific_accuracy": 1-5,
    "conceptual_depth": 1-5,
    "explanatory_clarity": 1-5,
    "application": 1-5
  },
  "scores_B": {
    "domain": "stem",
    "scientific_accuracy": 1-5,
    "conceptual_depth": 1-5,
    "explanatory_clarity": 1-5,
    "application": 1-5
  }
}
"""

humanities_prompt = """Please act as an impartial judge and evaluate the quality of two AI assistant responses, "Answer A" and "Answer B", provided for the user's question. Your goal is to determine which response is better based on the criteria below.

You must evaluate both responses on the four HUMANITIES dimensions below. After scoring, you will declare a winner.

**IMPORTANT RULE: You must choose a winner. Ties are not allowed.** If the quality is very close, you must still decide which one is even slightly better and justify your choice.

HUMANITIES DIMENSIONS & SCORE DEFINITIONS (apply exactly as written):

1) Interpretive Depth
- 5 (Excellent): Offers a rich, nuanced interpretation; considers multiple perspectives; demonstrates deep critical thinking about meaning and significance.
- 4 (Good): Strong interpretation with some nuance; at least one insightful observation; minor gaps in depth.
- 3 (Adequate): Basic interpretation; shows some insight but limited complexity; may miss important layers of meaning.
- 2 (Weak): Shallow interpretation; little analysis; overlooks key interpretive elements.
- 1 (Poor): No meaningful interpretation; purely descriptive or superficial.

2) Contextual Awareness
- 5 (Excellent): Demonstrates full awareness of relevant historical, cultural, or intellectual context; integrates it seamlessly into the argument.
- 4 (Good): Shows solid contextual knowledge; at least one relevant contextual link made; minor gaps.
- 3 (Adequate): Some context included but underdeveloped; misses important background connections.
- 2 (Weak): Minimal context; shows only partial awareness of relevant background.
- 1 (Poor): No contextual awareness; response is isolated from relevant historical/cultural factors.

3) Clarity of Argument
- 5 (Excellent): Argument is precise, logical, and easy to follow; clear thesis and strong supporting structure.
- 4 (Good): Argument mostly clear and coherent; one or two minor ambiguities or lapses in flow.
- 3 (Adequate): Argument present but uneven; thesis or reasoning somewhat vague; requires effort to follow.
- 2 (Weak): Argument unclear or poorly structured; reasoning hard to follow.
- 1 (Poor): No discernible argument; incoherent or contradictory.

4) Use of Evidence
- 5 (Excellent): Evidence (quotes, references, or examples) consistently accurate, specific, and well-integrated to support claims.
- 4 (Good): Evidence generally solid; at least one specific example; minor gaps in integration or precision.
- 3 (Adequate): Some evidence used but limited, generic, or weakly tied to claims.
- 2 (Weak): Minimal or poorly chosen evidence; weakly supports argument.
- 1 (Poor): No evidence provided or evidence entirely irrelevant.

SCORING RULES (for consistency):
- When uncertain between two levels, choose the lower score.
- Do not reward length; longer answers must still meet the same thresholds.
- Evaluate each dimension independently; do not let one dimension influence another.

OUTPUT FORMAT (strict):
1) First, write a concise natural-language explanation for your decision, explaining why you chose the winner.
2) Then output ONLY this JSON object and nothing else:
{
  "winner": "A" or "B",
  "scores_A": {
    "domain": "humanities",
    "interpretive_depth": 1-5,
    "contextual_awareness": 1-5,
    "clarity_of_argument": 1-5,
    "use_of_evidence": 1-5
  },
  "scores_B": {
    "domain": "humanities",
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

def call_openai(messages, model="gpt-5", temperature=0.8):
    if model in ["gpt-5", "o4-mini", "o3-mini", "gpt-5-mini", "gpt-5-nano"]:
        response = client.responses.create(
            model="gpt-5",
            input=messages,
            reasoning={"effort": "minimal"}
        )

        output_texts = []
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        output_texts.append(c.text)

        response = "\n".join(output_texts)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        response = response.choices[0].message.content.strip() 

    return response 

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
    Judges two answers by having a model compare them directly and choose a winner.
    """
    system_prompt = category_to_judge_prompt_map[target_category]
    
    # Answer A is the reference, Answer B is the contextual/assistant answer
    user_prompt = f"""
[USER QUESTION]
{latest_question}

[ANSWER A]
{ref_answer}

[ANSWER B]
{assistant_answer}
"""

    judge_response_raw = call_openai(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0
    )

    pattern = re.compile(r'\{.*\}', re.DOTALL)
    match = pattern.search(judge_response_raw)

    if not match:
        # Retry if JSON parsing fails
        return judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id, exp_id, turn_type)

    try:
        parsed_json = json.loads(match.group(0))
        winner = parsed_json['winner']
        ref_scores = parsed_json['scores_A']
        context_scores = parsed_json['scores_B']

        # Sum only numeric fields; skip meta keys like "domain"
        ref_total = sum(v for k, v in ref_scores.items() if isinstance(v, (int, float)))
        context_total = sum(v for k, v in context_scores.items() if isinstance(v, (int, float)))

    except (json.JSONDecodeError, KeyError):
        # Retry if JSON is invalid or missing required keys
        return judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id, exp_id, turn_type)

    # Determine winner based on judge's explicit choice: 1=reference (A), 2=context (B)
    if winner.upper() == 'B':
        choice = 2
    elif winner.upper() == 'A':
        choice = 1
    else:
        # This case should not be reached if the prompt is followed. Retry as a fallback.
        return judge_response(latest_question, ref_answer, assistant_answer, target_category, sample_id, exp_id, turn_type)
    
    # Tie is now impossible based on the prompt's instruction.
    choice_interpretation = {1: "reference_preferred", 2: "context_preferred"}

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "sample_id": sample_id,
        "turn_type": turn_type,
        "question": latest_question,
        "reference_answer": ref_answer,
        "context_answer": assistant_answer,
        "judge_response_raw": judge_response_raw,
        "judge_parsed_scores": {
            "reference_scores": ref_scores,
            "context_scores": context_scores
        },
        "reference_total_score": ref_total,
        "context_total_score": context_total,
        "judge_choice": choice,
        "choice_interpretation": choice_interpretation[choice],
    }

    log_filename = f"{log_dir}/judge_logs_exp{exp_id}.jsonl"
    safe_log_write(log_filename, log_entry)

    return {
        "choice": choice,
        "reference_scores": ref_scores,
        "context_scores": context_scores,
        "reference_total": ref_total,
        "context_total": context_total
    }


def generate_mt_response(context_seed_question, turn_count, curr_question, sample_id=None, exp_id=None, turn_type=None):
    """
    Generates a response from the model, either with or without context turns.
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    conversation_log = []

    # Add context turns generated on the fly
    # generate seed response first
    if turn_count != 0 and context_seed_question is not None:
        messages.append({"role": "user", "content": context_seed_question})
        response = call_openai(messages)
        messages.append({"role": "assistant", "content": response})

        conversation_log.append({"user": context_seed_question, "assistant": response})
        last_interaction = {"user": context_seed_question, "assistant": response}

        for turn in range(turn_count):
            # get follow up question
            follow_up_question = call_openai([{"role": "user", "content": f"Generate a natural follow-up question to the assistant’s last response that I can ask to keep the conversation going, and output only the question with no extra text or formatting. Be creative with your follow up and ask good and insightful quesitons that a human/person would actually ask. Here is the latest interaction: {last_interaction}"}], model="gpt-4.1-nano", temperature=0.9)

            messages.append({"role": "user", "content": follow_up_question})
            response = call_openai(messages)
            messages.append({"role": "assistant", "content": response})

            conversation_log.append({"user": follow_up_question, "assistant": response})

            last_interaction = {"user": follow_up_question, "assistant": response}

    # Add current question and get the final response
    messages.append({"role": "user", "content": curr_question})
    final_response = call_openai(messages)
    conversation_log.append({"user": curr_question, "assistant": final_response})

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_id": exp_id,
        "sample_id": sample_id,
        "turn_type": turn_type,
        "num_context_turns": turn_count,
        "conversation": conversation_log,
        "final_response": final_response
    }

    log_filename = f"{log_dir}/conversation_logs_exp{exp_id}.jsonl"
    safe_log_write(log_filename, log_entry)

    return final_response

def generate_and_judge_single_response(context_turns, target_question, reference_response, target_category, sample_id, exp_id, turn_type, turn_count):
    """Generates a single response and judges it - designed for parallel execution."""
    # Generate response with context
    context_seed_question = context_turns[0]

    contextual_response = generate_mt_response(context_seed_question, turn_count, target_question, sample_id, exp_id, turn_type)
    
    # Judge the response against the reference
    judgement = judge_response(target_question, reference_response, contextual_response, target_category, sample_id, exp_id, turn_type)
    
    return turn_type, judgement

def process_single_sample(context_sample, curr_sample, exp_id, sample_id, context_type):
    """
    Processes a single sample by generating a reference response and comparing it
    against contextual responses with varying numbers of turns.
    """
    target_question = curr_sample["turns"][0] # seed question
    target_category = curr_sample["category"]

    # Generate reference response (no context)
    reference_response = generate_mt_response(None, 0, target_question, sample_id, exp_id, "reference")

    context_configs = [
        (context_sample["turns"][:1], f"{context_type}_1_turn", 1),
        (context_sample["turns"][:3], f"{context_type}_3_turns", 3),
        (context_sample["turns"][:6], f"{context_type}_6_turns", 6),
        (context_sample["turns"][:9], f"{context_type}_9_turns", 9),
        (context_sample["turns"][:12], f"{context_type}_12_turns", 12),
    ]


    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_turn_type = {
            executor.submit(
                generate_and_judge_single_response,
                context_turns, target_question, reference_response, target_category,
                sample_id, exp_id, turn_type, turn_count
            ): turn_type
            for context_turns, turn_type, turn_count in context_configs
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
    print(exp_results)
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
            # Aggregate preference scores (0/1/2)
            final_results[context_type][category]["preference_scores"][turn_key].append(judgement.get('choice', 0))
            
            # Aggregate dimensional scores from independent evals
            ref_scores = judgement.get('reference_scores', {}) or {}
            context_scores = judgement.get('context_scores', {}) or {}

            for dim, score in ref_scores.items():
                if dim != "domain" and isinstance(score, (int, float)):
                    final_results[context_type][category]["dimensional_scores"]["reference"][turn_key][dim].append(score)

            for dim, score in context_scores.items():
                if dim != "domain" and isinstance(score, (int, float)):
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
