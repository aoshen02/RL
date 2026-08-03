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

from argparse import Namespace

import pytest

from nemo_rl.experience.rollout_only import (
    maybe_run_rollout_only,
    should_use_async_rollouts,
)


def test_disabled_rollout_only_does_not_read_config() -> None:
    assert not maybe_run_rollout_only(
        Namespace(debug_rollout_only=False),
        object(),
        object(),
        object(),
        object(),
        {},
    )


def test_rollout_only_reports_missing_algorithm_setting() -> None:
    with pytest.raises(ValueError, match="max_num_steps"):
        maybe_run_rollout_only(
            Namespace(debug_rollout_only=True),
            object(),
            object(),
            object(),
            object(),
            {
                "num_prompts_per_step": 1,
                "num_generations_per_prompt": 1,
                "max_rollout_turns": 1,
            },
        )


@pytest.mark.parametrize(
    ("generation_config", "expected"),
    [
        (None, False),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": True}}, True),
        ({"backend": "vllm", "vllm_cfg": {"async_engine": False}}, False),
        ({"backend": "sglang", "use_async_rollouts": True}, True),
        ({"backend": "megatron", "mcore_generation_config": {}}, False),
    ],
)
def test_should_use_async_rollouts(
    generation_config: dict | None,
    expected: bool,
) -> None:
    assert should_use_async_rollouts(generation_config) is expected
