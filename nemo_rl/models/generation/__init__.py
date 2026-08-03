# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, cast

from transformers import PreTrainedTokenizerBase

from nemo_rl.models.generation.interfaces import GenerationConfig, GenerationInterface
from nemo_rl.models.generation.trtllm import TrtllmConfig
from nemo_rl.models.generation.vllm import VllmConfig
from nemo_rl.models.generation.vllm.config import VLLM_SPARSE_REFIT_TRANSPORTS
from nemo_rl.distributed.worker_group_utils import recursive_merge_options

if TYPE_CHECKING:
    from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
    from nemo_rl.models.policy import PolicyConfig

TokenizerType = PreTrainedTokenizerBase


class GenerationServerGroups:
    """Expose independent generation servers through one rollout interface."""

    def __init__(self, generations: list[GenerationInterface]) -> None:
        if not generations:
            raise ValueError("At least one generation server group is required")
        self.generations = generations
        self.cfg = generations[0].cfg
        self.dp_openai_server_base_urls = [
            url
            for generation in generations
            for url in getattr(generation, "dp_openai_server_base_urls", [])
        ]
        self._next_generation_index = 0

    def openai_server_base_urls(self) -> list[str | None]:
        urls: list[str | None] = []
        for generation in self.generations:
            method = getattr(generation, "openai_server_base_urls", None)
            urls.extend(
                method()
                if method is not None
                else getattr(generation, "dp_openai_server_base_urls", [])
            )
        return urls

    def prepare_for_generation(self, *args: object, **kwargs: object) -> None:
        for generation in self.generations:
            generation.prepare_for_generation(*args, **kwargs)

    def _next_generation(self) -> GenerationInterface:
        generation = self.generations[self._next_generation_index]
        self._next_generation_index = (self._next_generation_index + 1) % len(
            self.generations
        )
        return generation

    def generate(self, *args: object, **kwargs: object):
        return self._next_generation().generate(*args, **kwargs)

    async def generate_async(self, *args: object, **kwargs: object):
        generate_async = getattr(self._next_generation(), "generate_async", None)
        if generate_async is None:
            raise NotImplementedError(
                "The configured generation backend does not support async generation."
            )
        async for result in generate_async(*args, **kwargs):
            yield result

    def clear_logger_metrics(self) -> None:
        for generation in self.generations:
            generation.clear_logger_metrics()

    def finish_generation(self, *args: object, **kwargs: object) -> None:
        for generation in self.generations:
            generation.finish_generation(*args, **kwargs)

    def shutdown(self) -> None:
        for generation in self.generations:
            generation.shutdown()


def setup_generation_only(
    policy_config: "PolicyConfig",
    cluster_config: "ClusterConfig",
    *,
    enable_server_groups: bool = False,
) -> tuple["GenerationInterface | GenerationServerGroups", "RayVirtualCluster"]:
    """Start the configured generation backend without policy workers."""
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster

    generation_config = policy_config["generation"]
    assert generation_config is not None, "A generation config is required"
    backend = generation_config["backend"]
    if backend not in ("vllm", "sglang"):
        raise NotImplementedError(
            f"--debug-rollout-only supports vLLM and SGLang; got {backend!r}."
        )

    colocated = generation_config["colocated"]
    if colocated["enabled"]:
        num_nodes = cluster_config["num_nodes"]
        gpus_per_node = cluster_config["gpus_per_node"]
    else:
        resources = colocated["resources"]
        num_nodes = resources["num_nodes"]
        gpus_per_node = resources["gpus_per_node"]
        if cluster_config["num_nodes"] == 1:
            if num_nodes not in (None, 1):
                raise ValueError(
                    "Single-node generation requires num_nodes to be 1 or null."
                )
            num_nodes = 1
    if num_nodes is None or gpus_per_node is None or gpus_per_node <= 0:
        raise ValueError("--debug-rollout-only requires explicit generation resources.")

    server_groups = generation_config.get("server_groups")
    if server_groups:
        if not enable_server_groups:
            raise ValueError(
                "policy.generation.server_groups requires the NeMo-Gym rollout path."
            )
        if backend != "vllm":
            raise NotImplementedError("Server groups currently require vLLM.")
        if not isinstance(server_groups, list):
            raise TypeError("policy.generation.server_groups must be a list")
        if not colocated["enabled"]:
            raise ValueError("Server groups require colocated generation.")
        if sum(int(group.get("count", 0)) for group in server_groups) != num_nodes:
            raise ValueError(
                f"The server-group counts must equal cluster.num_nodes ({num_nodes})."
            )

        from nemo_rl.models.generation.vllm import VllmGeneration
        from nemo_rl.models.generation.vllm.config import normalize_vllm_refit_config
        from nemo_rl.models.megatron.router_replay import (
            configure_vllm_for_router_replay,
        )

        generations: list[GenerationInterface] = []
        first_cluster: RayVirtualCluster | None = None
        for group_index, group in enumerate(server_groups):
            count = int(group.get("count", 0))
            if count <= 0:
                raise ValueError("Each server group count must be positive.")
            overrides = group.get("overrides", {})
            if not isinstance(overrides, dict):
                raise TypeError("Each server-group overrides value must be a mapping.")
            group_config = recursive_merge_options(
                deepcopy(generation_config), overrides
            )
            group_config.pop("server_groups", None)
            group_policy = deepcopy(policy_config)
            group_policy["generation"] = group_config
            group_config["model_name"] = policy_config["model_name"]
            configure_vllm_for_router_replay(group_policy)
            vllm_config = cast(VllmConfig, group_config)
            normalize_vllm_refit_config(vllm_config)
            vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = (
                policy_config.get("hf_config_overrides", {})
            )
            group_cluster = RayVirtualCluster(
                name=f"debug_rollout_group_{group_index}",
                bundle_ct_per_node_list=[gpus_per_node] * count,
                use_gpus=True,
                num_gpus_per_node=gpus_per_node,
                max_colocated_worker_groups=1,
                port_range_low=cluster_config.get("master_port_range_low"),
                port_range_high=cluster_config.get("master_port_range_high"),
                segment_size=cluster_config.get("segment_size"),
            )
            generations.append(
                VllmGeneration(
                    group_cluster,
                    vllm_config,
                    name_prefix=f"vllm_policy_group_{group_index}",
                )
            )
            first_cluster = first_cluster or group_cluster
        assert first_cluster is not None
        grouped_generation = GenerationServerGroups(generations)
        grouped_generation.finish_generation()
        return grouped_generation, first_cluster

    cluster = RayVirtualCluster(
        name="debug_rollout_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=cluster_config.get("segment_size"),
    )
    generation_config["model_name"] = policy_config["model_name"]

    if backend == "vllm":
        from nemo_rl.models.generation.vllm import VllmGeneration
        from nemo_rl.models.generation.vllm.config import normalize_vllm_refit_config
        from nemo_rl.models.megatron.router_replay import (
            configure_vllm_for_router_replay,
        )

        vllm_config = cast(VllmConfig, generation_config)
        normalize_vllm_refit_config(vllm_config)
        configure_vllm_for_router_replay(policy_config)
        vllm_config.setdefault("vllm_kwargs", {})["hf_overrides"] = policy_config.get(
            "hf_config_overrides", {}
        )
        generation: GenerationInterface = VllmGeneration(cluster, vllm_config)
    else:
        from nemo_rl.models.generation.sglang.config import SGLangConfig
        from nemo_rl.models.generation.sglang.sglang_generation import (
            SGLangGeneration,
        )

        sglang_config = cast(SGLangConfig, generation_config)
        sglang_config["sglang_cfg"].setdefault(
            "model_path", policy_config["model_name"]
        )
        generation = SGLangGeneration(cluster, sglang_config)

    generation.finish_generation()
    return generation, cluster


def configure_generation_config(
    config: GenerationConfig,
    tokenizer: TokenizerType,
    is_eval: bool = False,
    has_refit_draft_weights: bool = False,
    trains_mtp: bool = False,
) -> GenerationConfig:
    """Apply specific configurations to generation config."""
    # tokenizer setting
    if "_pad_token_id" in config:
        warnings.warn(
            "'_pad_token_id' found in generation config and will be overridden with tokenizer.pad_token_id. "
            "Note: '_pad_token_id' is intended for internal use and has no effect when set in user-provided configs.",
            UserWarning,
        )
    config["_pad_token_id"] = tokenizer.pad_token_id
    if config["stop_token_ids"] is None:
        config["stop_token_ids"] = [tokenizer.eos_token_id]

    # vllm setting
    if config["backend"] == "vllm":
        config = cast(VllmConfig, config)
        if config.get("real_quant"):
            export_cpu_offload = config.get("real_quant_export_cpu_offload")
            if not isinstance(export_cpu_offload, bool):
                raise ValueError(
                    "generation.real_quant_export_cpu_offload must be a boolean"
                )
            colocated = config.get("colocated")
            if not export_cpu_offload and (
                colocated is None
                or not colocated["enabled"]
                or config.get("refit_transport") is not None
            ):
                raise ValueError(
                    "generation.real_quant_export_cpu_offload=false requires "
                    "colocated CUDA-IPC refit with no explicit refit_transport"
                )

        # set load_format
        config["vllm_cfg"]["load_format"] = (
            "auto"
            if is_eval or config.get("refit_transport") in VLLM_SPARSE_REFIT_TRANSPORTS
            else "dummy"
        )
        speculative_config = config.get("vllm_kwargs", {}).get("speculative_config")
        if speculative_config and not is_eval and not has_refit_draft_weights:
            # Speculative decoding needs real draft weights at startup, since the
            # draft is not covered by the initial refit.
            if speculative_config.get("method") not in ("deepseek_mtp", "mtp"):
                # Non-MTP methods (e.g. Eagle) must read the drafter's real
                # weights from the checkpoint, so load everything.
                warnings.warn(
                    "Speculative decoding is enabled without draft refit sync. "
                    "Setting vllm_cfg['load_format'] to 'auto' so the drafter does "
                    "not start from dummy weights."
                )
                config["vllm_cfg"]["load_format"] = "auto"

        # MTP draft weights arrive via refit if the trainer trains the MTP layer.
        # If the trainer does not train the MTP layer, the weights need to be
        # loaded from the checkpoint.
        config["_mtp_weights_from_refit"] = trains_mtp

        # Respect the skip_tokenizer_init setting from the config. VLMs for example, require this to be False.
        if "skip_tokenizer_init" not in config["vllm_cfg"]:
            # set skip_tokenizer_init
            if (
                is_eval
                or config["stop_strings"] is not None
                or config["vllm_cfg"].get("expose_http_server", None)
            ):
                config["vllm_cfg"]["skip_tokenizer_init"] = False
            else:
                config["vllm_cfg"]["skip_tokenizer_init"] = True

    elif config["backend"] == "trtllm":
        config = cast(TrtllmConfig, config)

    return config
