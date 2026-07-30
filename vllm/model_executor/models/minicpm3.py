# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2024 The ModelBest team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM3 model compatible with HuggingFace weights."""
from typing import Any, Dict, Optional, Union, List, Tuple
import math

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.minicpm import (MiniCPMDecoderLayer,
                                                MiniCPMForCausalLM,
                                                MiniCPMModel)
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_pp_group

from .utils import make_layers


class MiniCPM3Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                         self.q_lora_rank,
                                         bias=False,
                                         quant_config=quant_config)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                             self.num_heads * self.qk_head_dim,
                                             bias=False,
                                             quant_config=quant_config)

        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size,
                                                   self.kv_lora_rank +
                                                   self.qk_rope_head_dim,
                                                   bias=False,
                                                   quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config)
        # O projection.
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)

        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)
        self.merge_q_kv_a = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        long_prompt_offset: torch.Tensor,
        long_short_cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        import ixformer.inference.functions as ixf
        if hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16:
            if not self.merge_q_kv_a:
                self.qkv_weight = torch.cat([self.q_a_proj.weight, self.kv_a_proj_with_mqa.weight], dim=0)
                del self.q_a_proj
                del self.kv_a_proj_with_mqa
                self.merge_q_kv_a = True
            q_latent_cache = ixf.linear(hidden_states, self.qkv_weight)
            q, latent_cache = q_latent_cache.split([self.q_lora_rank, 
                                                    self.kv_lora_rank + self.qk_rope_head_dim], 
                                                    dim=-1)
        else:
            q, _ = self.q_a_proj(hidden_states)
            latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)

        q = self.q_a_layernorm(q)
        q, _ = self.q_b_proj(q)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                          dim=-1)
        
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv, _ = self.kv_b_proj(kv_a)
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
       
        q_pe, k_pe = ixf.minicpm3_fused_rope(
            positions,
            long_prompt_offset,
            long_short_cos_sin_cache,
            q_pe, latent_cache[:, :, self.kv_lora_rank:],
            out_query = q[..., self.qk_nope_head_dim:]
        )

        q = q.view(-1, self.num_local_heads * self.qk_head_dim)
        
        k, v = ixf.minicpm3_fused_copy_kv(k_nope, k_pe, v)
       
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)
        new_attn_output = attn_output.new_empty([attn_output.shape[0], attn_output.shape[1], self.v_head_dim])
        new_attn_output[:, :, :] = attn_output[:, :, :self.v_head_dim]
        attn_output = new_attn_output.view(-1, self.num_local_heads * self.v_head_dim)

        output, _ = self.o_proj(attn_output)
        return output


class MiniCPM3DecoderLayer(MiniCPMDecoderLayer):
    def __init__(self, config: PretrainedConfig, 
                 cache_config: CacheConfig | None = None, 
                 quant_config: QuantizationConfig | None = None) -> None:
        super().__init__(config, cache_config, quant_config)
        self.hidden_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def _init_attn_block(self):
        self.input_layernorm = RMSNorm(self.config.hidden_size,
                                       eps=self.config.rms_norm_eps)
        self.self_attn = MiniCPM3Attention(
            config=self.config,
            hidden_size=self.hidden_size,
            num_heads=self.config.num_attention_heads,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            v_head_dim=self.config.v_head_dim,
            q_lora_rank=self.config.q_lora_rank,
            kv_lora_rank=self.config.kv_lora_rank,
            rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            max_position_embeddings=self.max_position_embeddings,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        long_prompt_offset: Optional[torch.Tensor],
        long_short_cos_sin_cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(residual, hidden_states, self.hidden_scale)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            long_prompt_offset=long_prompt_offset,
            long_short_cos_sin_cache=long_short_cos_sin_cache,
        )

        hidden_states, residual = self.post_attention_layernorm(residual, hidden_states, self.hidden_scale)
        
        hidden_states = self.mlp(hidden_states)
       
        return hidden_states, residual


class MiniCPM3Model(MiniCPMModel):

    def _init_layers(
        self,
        prefix: str,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig],
        quant_config: Optional[QuantizationConfig],
    ):
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPM3DecoderLayer(config, cache_config,
                                                quant_config),
            prefix=f"{prefix}.layers")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        k = self.layers[self.start_layer].self_attn.rotary_emb.original_max_position_embeddings
        long_prompt_offset = (torch.any(positions > k).float() *
                              torch.full_like(positions, k)).long()
        long_short_cos_sin_cache = (
            self.layers[self.start_layer].self_attn.rotary_emb.long_short_cos_sin_cache.to(input_ids.device))


        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
                long_prompt_offset=long_prompt_offset,
                long_short_cos_sin_cache=long_short_cos_sin_cache,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        hidden_states, residual = self.norm(residual, hidden_states, self.layers[self.start_layer].hidden_scale)
       
        return hidden_states


class MiniCPM3ForCausalLM(MiniCPMForCausalLM):
    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "kv_a_proj_with_mqa",
        "q_a_proj",
        "q_b_proj",
        "kv_b_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ]

    # `embedding_modules` and `embedding_padding_modules`
    # are inherited from MiniCPMForCausalLM

    def _init_model(self):
        self.model = MiniCPM3Model(config=self.config,
                                   cache_config=self.cache_config,
                                   quant_config=self.quant_config,
                                   lora_config=self.lora_config)
