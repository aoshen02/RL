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

"""Async GRPO launcher driven by the SingleController actor.

Builds the full SC actor args driver-side via setup_single_controller and hands them
to SingleControllerActor. Mirrors run_grpo.py for config loading so the same YAML
files apply. data_plane.enabled=true is mandatory.
"""

import argparse
import os
import pprint
import sys

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    setup_single_controller,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.experience.rollout_only import maybe_run_rollout_only
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    add_debug_rollout_only_argument,
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

# Drop examples/ from sys.path so examples/nemo_gym/ (no __init__.py) doesn't
# shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run async GRPO training via SingleController"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    add_debug_rollout_only_argument(parser)
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    rollout_only = getattr(args, "debug_rollout_only", False)

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_1B_megatron_single_controller.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    dp_cfg = config.data_plane
    if not dp_cfg.get("enabled", False):
        raise ValueError(
            "run_grpo_single_controller requires data_plane.enabled=true. "
            "Use examples/run_grpo.py for the legacy / sync paths."
        )

    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for SC-driven async GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        is_eval=rollout_only,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    use_nemo_gym = bool(config.env.get("should_use_nemo_gym"))
    if use_nemo_gym:
        setup_nemo_gym_config(config, tokenizer)

    # Keep this branch before setup_single_controller: that factory creates
    # the trainer, optimizer, data-plane client, weight synchronizer, and all
    # other training-side workers. Rollout-only must not construct any of them.
    if rollout_only:
        if use_nemo_gym:
            dataset, _val_dataset = setup_response_data(
                tokenizer, config.data, env_configs=None
            )
            task_to_env = None
        else:
            dataset, _val_dataset, task_to_env, _val_task_to_env = (
                setup_response_data(
                    tokenizer, config.data, env_configs=config.env
                )
            )
        if maybe_run_rollout_only(
            args,
            config,
            dataset,
            tokenizer,
            task_to_env,
            config.grpo,
            use_nemo_gym=use_nemo_gym,
        ):
            return

    actor_args = setup_single_controller(config, tokenizer)

    print("🚀 Launching SingleControllerActor")
    sc = SingleControllerActor.remote(master_config=config, actor_args=actor_args)
    try:
        result = ray.get(sc.run.remote())
        print(f"SC run complete: {result}")
    finally:
        # Drain env actors before generation to avoid in-flight requests during shutdown.
        for env_name, handle in actor_args.env_handles.items():
            try:
                ray.get(handle.shutdown.remote())
            except Exception as e:
                print(f"Env {env_name!r} shutdown failed: {e}")

        for resource_name, resource in (
            ("Generation", actor_args.gen_handle),
            ("Trainer", actor_args.trainer_handle),
        ):
            try:
                resource.shutdown()
            except Exception as e:
                print(f"{resource_name} shutdown failed: {e}")


if __name__ == "__main__":
    main()
