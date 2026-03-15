from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from mosv.routing.model import MoSVRouter


class MoSV:
    """Mixture-of-Steering-Vectors inference wrapper. Injects a sparse weighted combination of steering vectors into the residual stream."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        router: MoSVRouter,
        steering_vectors: np.ndarray,
        steer_layer: int,
        alpha: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = router
        self.steer_layer = steer_layer
        self.alpha = alpha

        self.steering_vectors = torch.from_numpy(steering_vectors).to(
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )

    def _get_prompt_repr(self, input_ids: torch.Tensor) -> torch.Tensor:
        activations = {}

        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            activations["h"] = hidden[:, -1, :].detach().float()

        handle = self.model.model.layers[self.steer_layer].register_forward_hook(hook)
        with torch.no_grad():
            self.model(input_ids)
        handle.remove()
        return activations["h"]

    def _compute_composite_vector(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self._get_prompt_repr(input_ids)
        weights = self.router(h)
        composite = (weights.unsqueeze(-1) * self.steering_vectors.unsqueeze(0)).sum(1)
        return composite.squeeze(0)

    def _make_steer_hook(self, composite_v: torch.Tensor):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + self.alpha * composite_v.to(hidden.dtype)
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        composite_v = self._compute_composite_vector(inputs.input_ids)

        hook_handle = self.model.model.layers[self.steer_layer].register_forward_hook(
            self._make_steer_hook(composite_v)
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
            hook_handle.remove()

        generated = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 80,
        do_sample: bool = False,
    ) -> List[str]:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        # Single forward pass to get prompt representations for all items
        activations = {}

        def repr_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            activations["h"] = hidden[:, -1, :].detach().float()

        handle = self.model.model.layers[self.steer_layer].register_forward_hook(repr_hook)
        self.model(**enc)
        handle.remove()

        # Compute per-item composite steering vectors: (B, d_model)
        h = activations["h"]
        weights = self.router(h)  # (B, K)
        composite_vs = (weights.unsqueeze(-1) * self.steering_vectors.unsqueeze(0)).sum(1)

        alpha = self.alpha

        def steer_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + alpha * composite_vs.unsqueeze(1).to(hidden.dtype)
            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        handle = self.model.model.layers[self.steer_layer].register_forward_hook(steer_hook)
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

    def get_cluster(self, prompt: str) -> int:
        """Return the cluster index the router assigns to this prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        h = self._get_prompt_repr(inputs.input_ids)
        with torch.no_grad():
            return self.router(h).argmax(-1).item()

    def get_routing_weights(self, prompt: str) -> np.ndarray:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        h = self._get_prompt_repr(inputs.input_ids)
        with torch.no_grad():
            weights = self.router(h)
        return weights.squeeze(0).cpu().numpy()
