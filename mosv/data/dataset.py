import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from datasets import load_dataset


LLAMA3_SYSTEM = "Answer the following question truthfully and concisely."

LLAMA3_PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n"
    "{system}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "{question}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

LETTERS = ["A", "B", "C", "D"]


@dataclass
class MCPair:
    prompt: str
    correct_answer: str
    incorrect_answer: str
    question: str
    category: str = ""
    source_dataset: str = "truthfulqa"


@dataclass
class GenerationItem:
    prompt: str
    question: str
    best_answer: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    category: str = ""
    source_dataset: str = "truthfulqa"


@dataclass
class MMLUEvalItem:
    prompt: str
    question: str
    choices: List[str]
    correct_letter: str
    correct_answer: str
    subject: str


def format_prompt(question: str) -> str:
    return LLAMA3_PROMPT_TEMPLATE.format(
        system=LLAMA3_SYSTEM,
        question=question,
    )


def format_mc_prompt(question: str, choices: List[str]) -> str:
    """Format question with A/B/C/D choices for generative MC eval."""
    choices_str = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    mc_question = f"{question}\n{choices_str}\nAnswer with A, B, C, or D."
    return LLAMA3_PROMPT_TEMPLATE.format(system=LLAMA3_SYSTEM, question=mc_question)


def load_truthfulqa_pairs(
    train_ratio: float = 0.7,
    seed: int = 42,
    max_pairs_per_question: int = None,
) -> Tuple[List[MCPair], List[MCPair]]:
    """
    Load TruthfulQA generation split as (correct, incorrect) MCPairs.
    Each question has multiple correct and incorrect answers — use all combos.
    Prompt is the question; answers are the raw answer strings (full text).
    """
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    items = list(dataset)
    random.seed(seed)
    random.shuffle(items)

    pairs = []
    for item in items:
        question = item["question"]
        corrects = item["correct_answers"]
        incorrects = item["incorrect_answers"]

        if not corrects or not incorrects:
            continue

        combos = [(c, inc) for c in corrects for inc in incorrects]
        if max_pairs_per_question:
            random.shuffle(combos)
            combos = combos[:max_pairs_per_question]

        for correct, incorrect in combos:
            pairs.append(MCPair(
                prompt=format_prompt(question),
                correct_answer=correct,
                incorrect_answer=incorrect,
                question=question,
                source_dataset="truthfulqa",
            ))

    split = int(len(pairs) * train_ratio)
    return pairs[:split], pairs[split:]


def load_mmlu_pairs(
    train_ratio: float = 0.7,
    seed: int = 42,
    max_per_subject: int = 100,
) -> Tuple[List[MCPair], List[MCPair]]:
    """
    Load MMLU test split as (correct, incorrect) MCPairs.
    Each question yields 3 pairs: (correct_text, wrong_text_k) for each wrong choice.
    Prompt is just the question — answers are full answer text, no A/B/C/D labels.
    """
    dataset = load_dataset("cais/mmlu", "all", split="test")
    items = list(dataset)
    random.seed(seed)
    random.shuffle(items)

    subject_counts: dict = {}
    pairs = []
    for item in items:
        subject = item.get("subject", "unknown")
        if subject_counts.get(subject, 0) >= max_per_subject:
            continue

        question = item["question"]
        choices = item["choices"]
        correct_idx = item["answer"]

        if not isinstance(correct_idx, int) or correct_idx >= len(choices):
            continue

        correct_text = choices[correct_idx]
        wrong_choices = [c for i, c in enumerate(choices) if i != correct_idx]

        for wrong_text in wrong_choices:
            pairs.append(MCPair(
                prompt=format_prompt(question),
                correct_answer=correct_text,
                incorrect_answer=wrong_text,
                question=question,
                category=subject,
                source_dataset="mmlu",
            ))

        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    split = int(len(pairs) * train_ratio)
    return pairs[:split], pairs[split:]


def load_combined_pairs(
    train_ratio: float = 0.7,
    seed: int = 42,
    max_tqa_pairs_per_question: int = None,
    mmlu_max_per_subject: int = 100,
) -> Tuple[List[MCPair], List[MCPair]]:
    """
    Combine TruthfulQA + MMLU into one training/test split.
    TruthfulQA: all correct x incorrect combos (~11k total pairs).
    MMLU test split: 3 pairs per question, capped per subject (~17k total pairs).
    """
    tqa_train, tqa_test = load_truthfulqa_pairs(
        train_ratio=train_ratio,
        seed=seed,
        max_pairs_per_question=max_tqa_pairs_per_question,
    )
    mmlu_train, mmlu_test = load_mmlu_pairs(
        train_ratio=train_ratio,
        seed=seed,
        max_per_subject=mmlu_max_per_subject,
    )

    combined_train = tqa_train + mmlu_train
    combined_test = tqa_test + mmlu_test

    random.seed(seed)
    random.shuffle(combined_train)
    random.shuffle(combined_test)

    print(f"  TruthfulQA: {len(tqa_train)} train, {len(tqa_test)} test pairs")
    print(f"  MMLU:       {len(mmlu_train)} train, {len(mmlu_test)} test pairs")
    print(f"  Combined:   {len(combined_train)} train, {len(combined_test)} test pairs")

    return combined_train, combined_test


def load_truthfulqa_generation(
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[List[GenerationItem], List[GenerationItem]]:
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    items = list(dataset)
    random.seed(seed)
    random.shuffle(items)

    gen_items = []
    for item in items:
        if not item["correct_answers"] or not item["incorrect_answers"]:
            continue
        gen_items.append(GenerationItem(
            prompt=format_prompt(item["question"]),
            question=item["question"],
            best_answer=item["best_answer"],
            correct_answers=item["correct_answers"],
            incorrect_answers=item["incorrect_answers"],
            category=item.get("category", ""),
        ))

    split = int(len(gen_items) * train_ratio)
    return gen_items[:split], gen_items[split:]


def load_mmlu_eval_items(seed: int = 42) -> List[MMLUEvalItem]:
    """
    Load MMLU validation split formatted for generative MC eval.
    Prompt includes A/B/C/D choices; model generates a letter; check against correct_letter.
    """
    dataset = load_dataset("cais/mmlu", "all", split="validation")
    items = list(dataset)
    random.seed(seed)
    random.shuffle(items)

    eval_items = []
    for item in items:
        choices = item["choices"]
        correct_idx = item["answer"]

        if not isinstance(correct_idx, int) or correct_idx >= len(choices):
            continue

        eval_items.append(MMLUEvalItem(
            prompt=format_mc_prompt(item["question"], choices),
            question=item["question"],
            choices=choices,
            correct_letter=LETTERS[correct_idx],
            correct_answer=choices[correct_idx],
            subject=item.get("subject", ""),
        ))

    return eval_items


def save_pairs(pairs: List[MCPair], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__) + "\n")


def load_pairs(path: str) -> List[MCPair]:
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append(MCPair(**d))
    return pairs


def save_gen_items(items: List[GenerationItem], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.__dict__) + "\n")


def load_gen_items(path: str) -> List[GenerationItem]:
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items.append(GenerationItem(**d))
    return items


def save_mmlu_eval(items: List[MMLUEvalItem], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.__dict__) + "\n")


def load_mmlu_eval(path: str) -> List[MMLUEvalItem]:
    items = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items.append(MMLUEvalItem(**d))
    return items
