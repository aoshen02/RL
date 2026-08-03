# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Lifecycle orchestration for ``--debug-rollout-only``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from nemo_rl.environments.interfaces import EnvironmentInterface
    from nemo_rl.models.generation.interfaces import GenerationConfig


def should_use_async_rollouts(generation_config: "GenerationConfig | None") -> bool:
    """Return whether a generation configuration uses asynchronous rollouts."""
    if generation_config is None:
        return False

    backend = generation_config.get("backend", "")
    if backend == "sglang":
        return bool(generation_config.get("use_async_rollouts", False))
    if backend == "vllm":
        return bool(generation_config.get("vllm_cfg", {}).get("async_engine", False))
    if backend == "trtllm":
        assert generation_config.get("trtllm_cfg", {}).get("async_engine", False), (
            "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
            "synchronous engine path (async_engine=false) is no longer supported."
        )
        return True
    if backend == "megatron":
        return bool(
            generation_config.get("mcore_generation_config", {}).get(
                "async_engine", False
            )
        )
    return False


def _algorithm_values(algorithm_config: dict[str, Any] | BaseModel) -> dict[str, Any]:
    return (
        algorithm_config.model_dump()
        if isinstance(algorithm_config, BaseModel)
        else algorithm_config
    )


def run_rollout_only(
    *,
    policy_config: Any,
    cluster_config: Any,
    data_config: Any,
    logger_config: Any,
    log_hyperparams: dict[str, Any],
    dataset: Any,
    tokenizer: "PreTrainedTokenizerBase",
    task_to_env: dict[str, "EnvironmentInterface"] | None,
    batch_size: int,
    num_generations_per_prompt: int,
    max_num_steps: int,
    max_rollout_turns: int,
    use_async_rollouts: bool,
    env_config: dict[str, Any],
    use_nemo_gym: bool = False,
    effort_config: Any = None,
    reward_penalty_config: dict[str, Any] | BaseModel | None = None,
) -> None:
    """Run normal rollouts without constructing training-side workers.

    This function owns only the generation, environment, dataloader, and logger
    lifecycle. The actual rollout primitives remain in ``rollouts.py``.
    """
    import ray
    from torchdata.stateful_dataloader import StatefulDataLoader
    from wandb import Table

    from nemo_rl.data.collate_fn import rl_collate_fn
    from nemo_rl.experience.rollouts import (
        get_nemo_gym_thinking_tags,
        run_async_multi_turn_rollout,
        run_multi_turn_rollout,
        run_nemo_gym_rollout_sync,
        should_mask_flagged_samples,
    )
    from nemo_rl.models.generation import setup_generation_only
    from nemo_rl.utils.logger import Logger

    if data_config.get("use_multiple_dataloader") or isinstance(dataset, dict):
        raise NotImplementedError(
            "--debug-rollout-only requires one training dataloader."
        )

    generation_config = policy_config["generation"]
    assert generation_config is not None, "A generation config is required"
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    logger = Logger(logger_config)
    logger.log_hyperparams(log_hyperparams)
    generation = None
    nemo_gym_actor = None

    try:
        generation, _ = setup_generation_only(policy_config, cluster_config)
        if use_nemo_gym:
            if generation_config["backend"] != "vllm":
                raise NotImplementedError("NeMo-Gym rollout-only requires vLLM.")

            from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
            from nemo_rl.models.generation.interfaces import (
                resolve_routed_experts_dtype_name_for_model,
            )
            from nemo_rl.models.generation.vllm import VllmGeneration
            from nemo_rl.models.megatron.router_replay import router_replay_enabled

            vllm_generation = cast(VllmGeneration, generation)
            router_replay = router_replay_enabled(policy_config)
            nemo_gym_actor = spinup_nemo_gym_actor(
                env_configs=env_config,
                base_urls=vllm_generation.dp_openai_server_base_urls,
                model_name=policy_config["model_name"],
                enable_router_replay=router_replay,
                routed_experts_dtype=(
                    resolve_routed_experts_dtype_name_for_model(
                        policy_config["model_name"]
                    )
                    if router_replay
                    else "int16"
                ),
                use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
            )
            task_to_env = {"nemo_gym": nemo_gym_actor}
        elif task_to_env is None:
            raise ValueError("Native rollout-only requires task_to_env.")

        for step, batch in enumerate(dataloader):
            if step >= max_num_steps:
                break
            batch = batch.repeat_interleave(num_generations_per_prompt)
            generation.prepare_for_generation()
            generation.clear_logger_metrics()
            try:
                if use_nemo_gym:
                    result = run_nemo_gym_rollout_sync(
                        policy_generation=generation,
                        input_batch=batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=policy_config["max_total_sequence_length"],
                        generation_config={
                            **generation_config,
                            "stop_token_ids": None,
                            "stop_strings": None,
                        },
                        log_full_result_tables=True,
                        max_rollout_turns=None,
                        greedy=False,
                        effort_config=effort_config,
                        reward_penalty_config=reward_penalty_config,
                        thinking_tags=get_nemo_gym_thinking_tags(env_config),
                        mask_env_flagged_samples=should_mask_flagged_samples(
                            env_config
                        ),
                    )
                    metrics = result.rollout_metrics
                elif use_async_rollouts:
                    _, metrics = run_async_multi_turn_rollout(
                        policy_generation=generation,
                        input_batch=batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=policy_config["max_total_sequence_length"],
                        max_rollout_turns=max_rollout_turns,
                        greedy=False,
                    )
                else:
                    _, metrics = run_multi_turn_rollout(
                        policy_generation=generation,
                        input_batch=batch,
                        tokenizer=tokenizer,
                        task_to_env=task_to_env,
                        max_seq_len=policy_config["max_total_sequence_length"],
                        max_rollout_turns=max_rollout_turns,
                        greedy=False,
                    )
            finally:
                generation.finish_generation()

            rows = [
                row[0]
                for key, value in metrics.items()
                if "full_result" in key and isinstance(value, Table)
                for row in value.data
            ]
            if rows:
                logger.log_string_list_as_jsonl(rows, "debug_rollout_only.jsonl")
            logger.log_metrics(
                {key: value for key, value in metrics.items() if not isinstance(value, Table)},
                step=step,
                prefix="debug_rollout_only",
            )
    finally:
        if nemo_gym_actor is not None:
            try:
                ray.get(nemo_gym_actor.shutdown.remote())
            finally:
                ray.kill(nemo_gym_actor)
        if generation is not None:
            generation.shutdown()
        logger.finish()


def maybe_run_rollout_only(
    args: Any,
    config: Any,
    dataset: Any,
    tokenizer: "PreTrainedTokenizerBase",
    task_to_env: dict[str, "EnvironmentInterface"] | None,
    algorithm_config: dict[str, Any] | BaseModel,
    *,
    use_nemo_gym: bool = False,
    effort_config: Any = None,
    reward_penalty_config: dict[str, Any] | BaseModel | None = None,
) -> bool:
    """Run rollout-only mode when requested and report whether it was selected."""
    if not getattr(args, "debug_rollout_only", False):
        return False

    values = _algorithm_values(algorithm_config)
    required = (
        "num_prompts_per_step",
        "num_generations_per_prompt",
        "max_num_steps",
        "max_rollout_turns",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            "--debug-rollout-only requires algorithm settings: " + ", ".join(missing)
        )

    run_rollout_only(
        policy_config=config.policy,
        cluster_config=config.cluster,
        data_config=config.data,
        logger_config=config.logger,
        log_hyperparams=config.model_dump(),
        dataset=dataset,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        batch_size=values["num_prompts_per_step"],
        num_generations_per_prompt=values["num_generations_per_prompt"],
        max_num_steps=values["max_num_steps"],
        max_rollout_turns=values["max_rollout_turns"],
        use_async_rollouts=should_use_async_rollouts(config.policy["generation"]),
        env_config=config.env,
        use_nemo_gym=use_nemo_gym,
        effort_config=effort_config,
        reward_penalty_config=reward_penalty_config,
    )
    return True
