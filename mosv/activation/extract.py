import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class ActivationExtractor:
    """Extracts residual stream activations at the last token position for all layers."""

    def __init__(self, model: PreTrainedModel, layers: List[int]):
        self.model = model
        self.layers = layers
        self.n_layers = len(layers)
        self.hidden_size = model.config.hidden_size
        self._hooks: List = []
        self._buffer: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, input: tuple, output: tuple) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self._buffer[layer_idx] = hidden[:, -1, :].detach().float().cpu()
        return hook

    def _register_hooks(self) -> None:
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor) -> np.ndarray:
        self._buffer.clear()
        self._register_hooks()
        try:
            self.model(input_ids)
        finally:
            self._remove_hooks()

        activations = np.stack(
            [self._buffer[l].squeeze(0).numpy() for l in self.layers],
            axis=0,
        )
        return activations

    @torch.no_grad()
    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        self._buffer.clear()
        self._register_hooks()
        try:
            self.model(input_ids, attention_mask=attention_mask)
        finally:
            self._remove_hooks()

        activations = np.stack(
            [self._buffer[l].numpy() for l in self.layers],
            axis=0,
        )
        return activations


def tokenize_pair(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    text = prompt + answer
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return tokens.input_ids.to(device)


CHECKPOINT_INTERVAL = 500


def collect_contrastive_activations(
    extractor: ActivationExtractor,
    tokenizer: PreTrainedTokenizer,
    pairs: list,
    device: torch.device,
    out_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract diff and prompt vectors for all pairs. Resumes from checkpoint if out_dir is set."""
    checkpoint_path = os.path.join(out_dir, "checkpoint.npz") if out_dir else None
    start_idx = 0
    diff_list = []
    prompt_list = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path)
        diff_list = list(ckpt["diff_vectors"])
        prompt_list = list(ckpt["prompt_vectors"])
        start_idx = len(diff_list)
        print(f"Resuming from checkpoint at pair {start_idx}/{len(pairs)}")

    for i, pair in enumerate(tqdm(pairs, initial=start_idx, total=len(pairs), desc="Activations")):
        if i < start_idx:
            continue

        ids_pos = tokenize_pair(tokenizer, pair.prompt, pair.correct_answer, device)
        ids_neg = tokenize_pair(tokenizer, pair.prompt, pair.incorrect_answer, device)
        ids_prompt = tokenizer(
            pair.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).input_ids.to(device)

        act_pos = extractor.extract(ids_pos)
        act_neg = extractor.extract(ids_neg)
        act_prompt = extractor.extract(ids_prompt)

        diff_list.append(act_pos - act_neg)
        prompt_list.append(act_prompt)

        if checkpoint_path and len(diff_list) % CHECKPOINT_INTERVAL == 0:
            np.savez(
                checkpoint_path,
                diff_vectors=np.stack(diff_list, axis=0),
                prompt_vectors=np.stack(prompt_list, axis=0),
            )

    diff_vectors = np.stack(diff_list, axis=0)
    prompt_vectors = np.stack(prompt_list, axis=0)
    labels = np.zeros(len(diff_list), dtype=np.int64)

    return diff_vectors, prompt_vectors, labels


def save_activations(
    diff_vectors: np.ndarray,
    prompt_vectors: np.ndarray,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "diff_vectors.npy"), diff_vectors)
    np.save(os.path.join(out_dir, "prompt_vectors.npy"), prompt_vectors)
    print(f"Saved activations to {out_dir}: diff {diff_vectors.shape}, prompt {prompt_vectors.shape}")


def load_activations(out_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    diff_vectors = np.load(os.path.join(out_dir, "diff_vectors.npy"))
    prompt_vectors = np.load(os.path.join(out_dir, "prompt_vectors.npy"))
    return diff_vectors, prompt_vectors
