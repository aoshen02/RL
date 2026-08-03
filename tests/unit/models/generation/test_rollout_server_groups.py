from nemo_rl.models.generation import setup_generation_only


def test_setup_generation_only_builds_independent_server_groups(monkeypatch):
    clusters = []

    class Cluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            clusters.append(self)

    class Generation:
        def __init__(self, cluster, config, name_prefix):
            self.cfg = config
            self.dp_openai_server_base_urls = [f"http://{name_prefix}/v1"]

        def finish_generation(self):
            return None

        def shutdown(self):
            return None

    monkeypatch.setattr(
        "nemo_rl.distributed.virtual_cluster.RayVirtualCluster", Cluster
    )
    monkeypatch.setattr("nemo_rl.models.generation.vllm.VllmGeneration", Generation)
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.config.normalize_vllm_refit_config",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.router_replay.configure_vllm_for_router_replay",
        lambda _config: None,
    )

    policy_config = {
        "model_name": "model",
        "generation": {
            "backend": "vllm",
            "colocated": {"enabled": True},
            "vllm_cfg": {"tensor_parallel_size": 1},
            "server_groups": [
                {"count": 1, "overrides": {"vllm_cfg": {"tensor_parallel_size": 2}}},
                {"count": 2, "overrides": {"vllm_cfg": {"tensor_parallel_size": 4}}},
            ],
        },
    }
    generation, first_cluster = setup_generation_only(
        policy_config,
        {"num_nodes": 3, "gpus_per_node": 8},
        enable_server_groups=True,
    )

    assert len(clusters) == 2
    assert first_cluster is clusters[0]
    assert [cluster.kwargs["bundle_ct_per_node_list"] for cluster in clusters] == [
        [8],
        [8, 8],
    ]
    assert generation.dp_openai_server_base_urls == [
        "http://vllm_policy_group_0/v1",
        "http://vllm_policy_group_1/v1",
    ]
    assert [
        group.cfg["vllm_cfg"]["tensor_parallel_size"]
        for group in generation.generations
    ] == [2, 4]
