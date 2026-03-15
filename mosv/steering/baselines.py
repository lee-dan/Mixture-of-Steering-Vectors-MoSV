from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class VanillaBaseline:
    """Unsteered LLaMA-3-8B Instruct. No hooks, no modifications."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 80,
        do_sample: bool = False,
    ) -> List[str]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self._device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = enc.input_ids.shape[1]
        return [
            self.tokenizer.decode(out[i, input_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]


class SingleVecBaseline:
    """CAA baseline: mean diff vector applied uniformly to every prompt."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        diff_vectors: np.ndarray,
        steer_layer: int,
        layer_pos: int,
        alpha: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steer_layer = steer_layer
        self.alpha = alpha
        self._device = next(model.parameters()).device

        mean_diff = diff_vectors[:, layer_pos, :].mean(axis=0)
        self.steering_vector = torch.from_numpy(mean_diff.astype(np.float32)).to(self._device)

    def _make_hook(self):
        v = self.steering_vector
        alpha = self.alpha

        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + alpha * v.to(hidden.dtype)
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        return hook

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        handle = self.model.model.layers[self.steer_layer].register_forward_hook(
            self._make_hook()
        )
        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            handle.remove()
        generated = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 80,
        do_sample: bool = False,
    ) -> List[str]:
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self._device)
        handle = self.model.model.layers[self.steer_layer].register_forward_hook(
            self._make_hook()
        )
        try:
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            handle.remove()
        input_len = enc.input_ids.shape[1]
        return [
            self.tokenizer.decode(out[i, input_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]
