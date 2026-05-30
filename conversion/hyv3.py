from __future__ import annotations

import json

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("HYV3ForCausalLM")
class HYV3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.HYV3
    eos_token = "<eos:6124c78e>"

    def set_vocab(self):
        self._set_vocab_gpt2()
        eos_id = self._get_eos_token_id()
        if eos_id is not None:
            logger.info(f"gguf: HYV3 EOS token {self.eos_token!r} id = {eos_id}")
            self.gguf_writer.add_eos_token_id(eos_id)
        else:
            logger.warning(
                f"gguf: HYV3 EOS token {self.eos_token!r} not found; "
                "generation may print it instead of stopping"
            )

    def _load_json(self, name: str):
        path = self.dir_model / name
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _token_content(value) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, str):
                return content
        return None

    def _find_token_id(self, token: str) -> int | None:
        tokenizer_config = self._load_json("tokenizer_config.json")
        if tokenizer_config is not None:
            added_tokens = tokenizer_config.get("added_tokens_decoder", {})
            if isinstance(added_tokens, dict):
                for token_id, data in added_tokens.items():
                    if self._token_content(data) == token:
                        return int(token_id)

        tokenizer = self._load_json("tokenizer.json")
        if tokenizer is not None:
            added_tokens = tokenizer.get("added_tokens", [])
            if isinstance(added_tokens, list):
                for data in added_tokens:
                    if self._token_content(data) == token:
                        token_id = data.get("id")
                        if isinstance(token_id, int):
                            return token_id

            vocab = tokenizer.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                token_id = vocab.get(token)
                if isinstance(token_id, int):
                    return token_id

        vocab = self._load_json("vocab.json")
        if vocab is not None:
            token_id = vocab.get(token)
            if isinstance(token_id, int):
                return token_id

        return None

    @staticmethod
    def _first_token_id(value) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return next((token_id for token_id in value if isinstance(token_id, int)), None)
        return None

    def _get_eos_token_id(self) -> int | None:
        token_id = self._find_token_id(self.eos_token)
        if token_id is not None:
            return token_id

        tokenizer_config = self._load_json("tokenizer_config.json")
        if tokenizer_config is not None:
            eos_token = self._token_content(tokenizer_config.get("eos_token"))
            if eos_token is not None:
                token_id = self._find_token_id(eos_token)
                if token_id is not None:
                    return token_id

        generation_config = self._load_json("generation_config.json")
        if generation_config is not None:
            token_id = self._first_token_id(generation_config.get("eos_token_id"))
            if token_id is not None:
                return token_id

        return self._first_token_id(self.hparams.get("eos_token_id"))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(
            self.hparams["moe_intermediate_size"] * self.hparams.get("num_shared_experts", 1)
        )
        self.gguf_writer.add_expert_weights_norm(self.hparams.get("route_norm", True))
        self.gguf_writer.add_expert_weights_scale(float(self.hparams.get("router_scaling_factor", 1.0)))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        logger.info("gguf: HYV3 sigmoid router with correction bias")

    _experts: list[dict[str, Tensor]] | None = None

    def _is_sparse_layer(self, bid: int | None) -> bool:
        if bid is None:
            return False
        layer_types = self.hparams.get("mlp_layer_types")
        if isinstance(layer_types, list) and bid < len(layer_types):
            return layer_types[bid] == "sparse"
        return bid >= int(self.hparams.get("first_k_dense_replace", 0))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model.layers.") and bid is not None and bid >= self.block_count:
            return

        if name.startswith("model.layers.") and ".mlp.experts." in name:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    merged = torch.stack(datas, dim=0)
                    yield from super().modify_tensors(merged, f"model.layers.{bid}.mlp.experts.{w_name}.weight", bid)
                return
            return

        if name.endswith(".mlp.router.gate.weight"):
            yield from super().modify_tensors(data_torch, name.replace(".mlp.router.gate.weight", ".mlp.gate.weight"), bid)
            return

        if name.endswith(".mlp.expert_bias"):
            mapped = name.replace(".mlp.expert_bias", ".mlp.gate.expert_bias.bias")
            yield from super().modify_tensors(data_torch, mapped, bid)
            return

        if ".mlp.shared_mlp." in name:
            yield from super().modify_tensors(data_torch, name.replace(".mlp.shared_mlp.", ".mlp.shared_experts."), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")
