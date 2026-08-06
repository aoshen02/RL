from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.generation.vllm.multi_group import (
    MultiVllmGeneration,
    _merge,
    build_group_configs,
)


def _base_config(**extra):
    cfg = {
        "backend": "vllm",
        "model_name": "test-model",
        "vllm_cfg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_model_len": 4096,
        },
        "vllm_kwargs": {"hf_overrides": {"rope_scaling": {"factor": 2.0}}},
    }
    cfg.update(extra)
    return cfg


def test_merge_is_deep_and_non_mutating():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    merged = _merge(base, {"a": {"c": 9}, "e": 4})
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}
    assert base == {"a": {"b": 1, "c": 2}, "d": 3}


def test_build_group_configs_absent_returns_none():
    assert build_group_configs(_base_config()) is None
    assert build_group_configs(_base_config(server_groups=None)) is None
    assert build_group_configs(_base_config(server_groups=[])) is None


def test_build_group_configs_merges_and_inherits_setup_mutations():
    cfg = _base_config(
        server_groups=[
            {"name": "low_latency", "gpus": 2},
            {
                "name": "high_throughput",
                "gpus": 4,
                "overrides": {
                    "vllm_cfg": {"tensor_parallel_size": 4},
                    "vllm_kwargs": {"max_num_seqs": 512},
                },
            },
        ]
    )
    configs = build_group_configs(cfg)
    assert set(configs) == {"low_latency", "high_throughput"}
    for group_cfg in configs.values():
        assert "server_groups" not in group_cfg
        assert group_cfg["model_name"] == "test-model"
        assert group_cfg["vllm_kwargs"]["hf_overrides"] == {
            "rope_scaling": {"factor": 2.0}
        }
    ht = configs["high_throughput"]
    assert ht["vllm_cfg"]["tensor_parallel_size"] == 4
    assert ht["vllm_cfg"]["max_model_len"] == 4096
    assert ht["vllm_kwargs"]["max_num_seqs"] == 512
    ht["vllm_cfg"]["tensor_parallel_size"] = 8
    assert configs["low_latency"]["vllm_cfg"]["tensor_parallel_size"] == 1
    assert cfg["vllm_cfg"]["tensor_parallel_size"] == 1


@pytest.mark.parametrize(
    "groups",
    [
        [{"name": "a", "gpus": 2}, {"name": "a", "gpus": 2}],
        [{"name": "a", "gpus": 2, "overrides": {"colocated": {"enabled": True}}}],
        [{"name": "a", "gpus": 2, "overrides": {"vllm_cfg": {"async_engine": False}}}],
        [
            {
                "name": "a",
                "gpus": 2,
                "overrides": {"vllm_kwargs": {"max_model_len": 128}},
            }
        ],
    ],
)
def test_build_group_configs_rejects_invalid(groups):
    with pytest.raises(AssertionError):
        build_group_configs(_base_config(server_groups=groups))


def _mock_group(num_workers, urls):
    g = MagicMock()
    g.worker_group.workers = [object()] * num_workers
    g.dp_openai_server_base_urls = urls
    return g


def _make_wrapper(mock_groups):
    with patch(
        "nemo_rl.models.generation.vllm.multi_group.VllmGeneration",
        side_effect=list(mock_groups.values()),
    ) as ctor:
        wrapper = MultiVllmGeneration(
            clusters={n: MagicMock() for n in mock_groups},
            configs={n: _base_config() for n in mock_groups},
            defer_model_load=True,
        )
    assert [c.kwargs["name_prefix"] for c in ctor.call_args_list] == [
        f"vllm_{n}" for n in mock_groups
    ]
    return wrapper


def test_wrapper_url_concat_and_group_urls():
    wrapper = _make_wrapper(
        {
            "low_latency": _mock_group(2, ["http://a:1/v1", "http://a:2/v1"]),
            "high_throughput": _mock_group(4, ["http://b:1/v1"]),
        }
    )
    assert wrapper.dp_openai_server_base_urls == [
        "http://a:1/v1",
        "http://a:2/v1",
        "http://b:1/v1",
    ]
    assert wrapper.server_group_urls == {
        "low_latency": ["http://a:1/v1", "http://a:2/v1"],
        "high_throughput": ["http://b:1/v1"],
    }


def test_wrapper_init_collective_offsets_ranks_into_one_world():
    groups = {
        "low_latency": _mock_group(2, []),
        "high_throughput": _mock_group(4, []),
    }
    groups["low_latency"].init_collective.return_value = ["f0"]
    groups["high_throughput"].init_collective.return_value = ["f1", "f2"]
    wrapper = _make_wrapper(groups)
    futures = wrapper.init_collective("ip", 1234, 14, train_world_size=8)
    assert futures == ["f0", "f1", "f2"]
    assert groups["low_latency"].init_collective.call_args.kwargs["rank_offset"] == 0
    assert (
        groups["high_throughput"].init_collective.call_args.kwargs["rank_offset"] == 2
    )


def test_wrapper_fans_out_lifecycle_and_concatenates_futures():
    groups = {"a": _mock_group(1, []), "b": _mock_group(1, [])}
    groups["a"].update_weights_from_collective.return_value = ["fa"]
    groups["b"].update_weights_from_collective.return_value = ["fb"]
    groups["a"].get_logger_metrics.return_value = {"m": 1}
    groups["b"].get_logger_metrics.return_value = {"m": 2}
    wrapper = _make_wrapper(groups)
    assert wrapper.update_weights_from_collective() == ["fa", "fb"]
    wrapper.prepare_refit_info({"w": None})
    wrapper.clear_logger_metrics()
    for g in groups.values():
        g.prepare_refit_info.assert_called_once_with({"w": None})
        g.clear_logger_metrics.assert_called_once()
    assert wrapper.get_logger_metrics() == {"a/m": 1, "b/m": 2}
    with pytest.raises(NotImplementedError):
        wrapper.generate(None)
