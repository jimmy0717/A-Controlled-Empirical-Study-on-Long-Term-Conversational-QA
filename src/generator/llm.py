"""LLM backend wrapper.

Three backends are supported transparently:

* ``hf``     - ``transformers.AutoModelForCausalLM`` (default, single-process,
               works on the AI-Studio A800 box without any extra setup).
* ``vllm``   - ``vllm.LLM`` for high-throughput batched generation.
* ``openai`` - any OpenAI-compatible HTTP endpoint (set ``OPENAI_API_KEY`` /
               ``OPENAI_BASE_URL`` in the environment).

The same :class:`LLM` class hides the differences and offers a small
``chat`` method plus a token counter, so the experiment runner is
backend-agnostic.

A ``diskcache`` is used to cache identical (model, prompt, params) calls so
that re-running the experiment is cheap and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import diskcache as dc

CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".llm_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_cache = dc.Cache(str(CACHE_DIR))


def _cache_key(model: str, messages: List[dict], **gen_kwargs) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "gen": sorted(gen_kwargs.items())},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class GenStats:
    n_calls: int = 0
    n_tokens_in: int = 0
    n_tokens_out: int = 0
    wallclock: float = 0.0


class LLM:
    """Thin three-backend wrapper for chat-style generation."""

    def __init__(
        self,
        model: str,
        backend: str = "hf",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
    ):
        self.model_name = model
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stats = GenStats()

        if backend == "hf":
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
                dtype, torch.bfloat16
            )
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            # transformers >=4.56 renamed the `torch_dtype` kwarg to `dtype` and
            # drops the old alias in 5.x. Pick the right one so the same code
            # loads on both 4.x and 5.x without a deprecation error.
            _tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
            _dtype_kw = ({"dtype": torch_dtype} if _tv >= (4, 56)
                         else {"torch_dtype": torch_dtype})
            # SDPA (PyTorch fused attention) is memory-frugal and faster on the
            # A800 prefill than the eager path, and needs no extra install
            # (unlike flash_attention_2). Fall back gracefully if unsupported.
            try:
                self.mdl = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=device,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                    **_dtype_kw,
                )
            except (ValueError, ImportError):
                self.mdl = AutoModelForCausalLM.from_pretrained(
                    model,
                    device_map=device,
                    trust_remote_code=True,
                    **_dtype_kw,
                )
            self.mdl.eval()
            self.device = device
        elif backend == "vllm":
            from vllm import LLM as _VLLM  # type: ignore

            self.vllm = _VLLM(model=model, dtype=dtype, gpu_memory_utilization=0.85)
        elif backend == "openai":
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "backend='openai' but OPENAI_API_KEY is not set. Either export "
                    "OPENAI_API_KEY (and OPENAI_BASE_URL for DashScope) or change the "
                    "judge/generator 'backend' to 'hf'/'vllm' in your YAML config."
                )
            self.client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL"),
                api_key=api_key,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")

    # ------------------------------------------------------------------
    def chat(self, messages: List[dict], **kwargs) -> str:
        """Run a single chat-completion call. Cached on disk."""
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        key = _cache_key(self.model_name, messages, **gen_kwargs)
        if key in _cache:
            return _cache[key]

        t0 = time.perf_counter()
        if self.backend == "hf":
            text = self._chat_hf(messages, **gen_kwargs)
        elif self.backend == "vllm":
            text = self._chat_vllm(messages, **gen_kwargs)
        else:
            text = self._chat_openai(messages, **gen_kwargs)
        self.stats.wallclock += time.perf_counter() - t0
        self.stats.n_calls += 1

        _cache[key] = text
        return text

    def chat_batch(self, batch_messages: List[List[dict]], **kwargs) -> List[str]:
        """Run many chat-completions efficiently, returning one string per input
        in order.

        Cache-aware: prompts already on disk are served from cache; the rest are
        executed as a *true* batch (HF left-padded ``generate`` / vLLM multi-prompt)
        or looped (OpenAI). For greedy decoding (``temperature=0``) the result is
        identical to calling :meth:`chat` on each item, because left-padding is
        masked out by the attention mask.
        """
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        results: List[Any] = [None] * len(batch_messages)
        pending: List[tuple] = []  # (idx, messages, cache_key)
        for i, messages in enumerate(batch_messages):
            key = _cache_key(self.model_name, messages, **gen_kwargs)
            if key in _cache:
                results[i] = _cache[key]
            else:
                pending.append((i, messages, key))

        if pending:
            t0 = time.perf_counter()
            msgs_only = [m for _, m, _ in pending]
            if self.backend == "hf":
                texts = self._chat_hf_batch(msgs_only, **gen_kwargs)
            elif self.backend == "vllm":
                texts = self._chat_vllm_batch(msgs_only, **gen_kwargs)
            else:
                texts = [self._chat_openai(m, **gen_kwargs) for m in msgs_only]
            self.stats.wallclock += time.perf_counter() - t0
            for (i, _m, key), text in zip(pending, texts):
                results[i] = text
                _cache[key] = text
                self.stats.n_calls += 1
        return results

    # ------------------ backend-specific implementations ----------------
    def _chat_hf(self, messages, **kw) -> str:
        import torch

        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        self.stats.n_tokens_in += int(enc["input_ids"].shape[1])
        do_sample = kw["temperature"] > 0
        with torch.no_grad():
            out = self.mdl.generate(
                **enc,
                max_new_tokens=kw["max_new_tokens"],
                do_sample=do_sample,
                temperature=max(kw["temperature"], 1e-5),
                top_p=kw["top_p"],
                pad_token_id=self.tok.eos_token_id,
            )
        gen_ids = out[0, enc["input_ids"].shape[1] :]
        self.stats.n_tokens_out += int(gen_ids.shape[0])
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()

    def _chat_hf_batch(self, batch_messages, _chunk: int = 0, **kw) -> List[str]:
        """Left-padded batched generation with **length-sorted, token-budgeted**
        dynamic batching.

        Why not a fixed chunk size? Decoder-only batching left-pads every prompt
        in a mini-batch to the longest one, so mixing a 6k-token session with a
        handful of 800-token ones wastes ~85% of the compute on padding. Instead
        we:

        1. sort prompts by token length (groups similar lengths together so
           padding waste is near-zero);
        2. greedily pack each batch until ``len(batch) * padded_len`` would exceed
           ``LLM_BATCH_TOKEN_BUDGET`` (default 16384) -- this auto-shrinks batches
           for long sessions (OOM-safe) and grows them for short ones (max
           throughput on the A800);
        3. restore the original order before returning.

        Greedy decoding (``temperature=0``) makes the per-item output identical to
        single-prompt :meth:`chat`. ``LLM_BATCH_MAX_SIZE`` caps the batch size and
        ``_chunk`` (if >0) forces a fixed size, overriding the budget logic.
        """
        import torch

        prompts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        do_sample = kw["temperature"] > 0

        # token length per prompt (no padding here -> cheap, just for planning)
        lengths = [len(self.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lengths[i])

        budget = int(os.environ.get("LLM_BATCH_TOKEN_BUDGET", "16384"))
        max_bs = int(os.environ.get("LLM_BATCH_MAX_SIZE", "64"))

        # ---- plan dynamic batches over the length-sorted indices
        batches: List[List[int]] = []
        i = 0
        while i < len(order):
            cur = [order[i]]
            maxlen = lengths[order[i]]
            j = i + 1
            while j < len(order):
                cand_max = max(maxlen, lengths[order[j]])
                fixed_ok = _chunk > 0 and len(cur) < _chunk
                budget_ok = _chunk <= 0 and (len(cur) + 1) * cand_max <= budget
                if (fixed_ok or budget_ok) and len(cur) < max_bs:
                    cur.append(order[j])
                    maxlen = cand_max
                    j += 1
                else:
                    break
            batches.append(cur)
            i = j

        prev_side = self.tok.padding_side
        self.tok.padding_side = "left"  # decoder-only models must left-pad
        results: List[Any] = [None] * len(prompts)
        try:
            for idxs in batches:
                chunk = [prompts[k] for k in idxs]
                enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
                self.stats.n_tokens_in += int(enc["attention_mask"].sum())
                with torch.no_grad():
                    out = self.mdl.generate(
                        **enc,
                        max_new_tokens=kw["max_new_tokens"],
                        do_sample=do_sample,
                        temperature=max(kw["temperature"], 1e-5),
                        top_p=kw["top_p"],
                        pad_token_id=self.tok.eos_token_id,
                    )
                gen = out[:, enc["input_ids"].shape[1] :]
                for k, row in zip(idxs, gen):
                    self.stats.n_tokens_out += int((row != self.tok.eos_token_id).sum())
                    results[k] = self.tok.decode(row, skip_special_tokens=True).strip()
        finally:
            self.tok.padding_side = prev_side
        return results  # original order restored

    def _chat_vllm(self, messages, **kw) -> str:
        from vllm import SamplingParams  # type: ignore

        prompt = self.vllm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        params = SamplingParams(
            max_tokens=kw["max_new_tokens"],
            temperature=kw["temperature"],
            top_p=kw["top_p"],
        )
        out = self.vllm.generate([prompt], params)[0]
        self.stats.n_tokens_in += len(out.prompt_token_ids)
        self.stats.n_tokens_out += len(out.outputs[0].token_ids)
        return out.outputs[0].text.strip()

    def _chat_vllm_batch(self, batch_messages, **kw) -> List[str]:
        from vllm import SamplingParams  # type: ignore

        tok = self.vllm.get_tokenizer()
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        params = SamplingParams(
            max_tokens=kw["max_new_tokens"],
            temperature=kw["temperature"],
            top_p=kw["top_p"],
        )
        outs = self.vllm.generate(prompts, params)  # order preserved
        texts: List[str] = []
        for o in outs:
            self.stats.n_tokens_in += len(o.prompt_token_ids)
            self.stats.n_tokens_out += len(o.outputs[0].token_ids)
            texts.append(o.outputs[0].text.strip())
        return texts

    def _chat_openai(self, messages, **kw) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kw["max_new_tokens"],
            temperature=kw["temperature"],
            top_p=kw["top_p"],
        )
        if resp.usage is not None:
            self.stats.n_tokens_in += resp.usage.prompt_tokens
            self.stats.n_tokens_out += resp.usage.completion_tokens
        return resp.choices[0].message.content.strip()
