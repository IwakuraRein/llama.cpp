from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("HYV3ForCausalLM")
class HYV3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.HYV3

    def set_vocab(self):
        self._set_vocab_gpt2()

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
