from __future__ import annotations

import numpy as np

from plugrl_server.server.ray_inference import RayInferenceWorkerGroup


def test_ray_inference_worker_group_partitions_requests_round_robin():
    group = object.__new__(RayInferenceWorkerGroup)
    group._workers = [object(), object(), object()]
    group._next_worker_index = 0
    group._route_key_to_worker_index = {}
    group._worker_index_to_route_keys = {0: set(), 1: set(), 2: set()}
    group._route_key_to_last_env_ids = {}

    requests = [
        {"id": "r0", "env_ids": np.asarray([0])},
        {"id": "r1", "env_ids": np.asarray([1, 2])},
        {"id": "r2", "env_ids": np.asarray([3])},
        {"id": "r3", "env_ids": np.asarray([4, 5, 6])},
    ]

    assignments = group._partition_requests(requests)

    assert [worker_index for worker_index, _ in assignments] == [0, 1, 2]
    assert [request["id"] for request in assignments[0][1]] == ["r0", "r3"]
    assert [request["id"] for request in assignments[1][1]] == ["r1"]
    assert [request["id"] for request in assignments[2][1]] == ["r2"]
    assert group._next_worker_index == 4


def test_ray_inference_worker_group_as_metrics_exposes_per_worker_stats():
    group = object.__new__(RayInferenceWorkerGroup)
    group._last_known_worker_metrics = {
        1: {"requests_served": 2.0, "model_step": 7.0},
        0: {"requests_served": 5.0, "model_step": 7.0},
    }

    metrics = group.as_metrics()

    assert sorted(metrics["inference_workers"].keys()) == ["worker_00", "worker_01"]
    assert metrics["inference_workers"]["worker_00"]["requests_served"] == 5.0
    assert metrics["inference_workers"]["worker_01"]["model_step"] == 7.0


def test_ray_inference_worker_group_partitions_requests_sticky_by_route_key():
    group = object.__new__(RayInferenceWorkerGroup)
    group._workers = [object(), object(), object(), object()]
    group._next_worker_index = 0
    group._route_key_to_worker_index = {}
    group._worker_index_to_route_keys = {0: set(), 1: set(), 2: set(), 3: set()}
    group._route_key_to_last_env_ids = {}

    requests = [
        {"id": "r0", "env_ids": np.asarray([0]), "route_key": "proc-0"},
        {"id": "r1", "env_ids": np.asarray([1]), "route_key": "proc-1"},
        {"id": "r2", "env_ids": np.asarray([2]), "route_key": "proc-2"},
        {"id": "r3", "env_ids": np.asarray([3]), "route_key": "proc-3"},
        {"id": "r4", "env_ids": np.asarray([4]), "route_key": "proc-0"},
        {"id": "r5", "env_ids": np.asarray([5]), "route_key": "proc-2"},
    ]

    assignments = group._partition_requests(requests)

    assert [worker_index for worker_index, _ in assignments] == [0, 1, 2, 3]
    assert [request["id"] for request in assignments[0][1]] == ["r0", "r4"]
    assert [request["id"] for request in assignments[1][1]] == ["r1"]
    assert [request["id"] for request in assignments[2][1]] == ["r2", "r5"]
    assert [request["id"] for request in assignments[3][1]] == ["r3"]
    assert group._route_key_to_worker_index == {
        "proc-0": 0,
        "proc-1": 1,
        "proc-2": 2,
        "proc-3": 3,
    }


def test_ray_inference_worker_group_balances_32_isolated_routes_across_4_workers():
    group = object.__new__(RayInferenceWorkerGroup)
    group._workers = [object(), object(), object(), object()]
    group._next_worker_index = 0
    group._route_key_to_worker_index = {}
    group._worker_index_to_route_keys = {0: set(), 1: set(), 2: set(), 3: set()}
    group._route_key_to_last_env_ids = {}

    requests = [
        {
            "id": f"r{i}",
            "env_ids": np.asarray([i]),
            "route_key": f"proc-{i}",
        }
        for i in range(32)
    ]

    assignments = group._partition_requests(requests)

    assert [worker_index for worker_index, _ in assignments] == [0, 1, 2, 3]
    assert [len(shard) for _, shard in assignments] == [8, 8, 8, 8]
    assert group._next_worker_index == 32
    assert group._route_key_to_worker_index == {
        f"proc-{i}": i % 4 for i in range(32)
    }
    assignments_by_worker = group.describe_route_assignments()
    for worker_index in range(4):
        assert {
            (entry["route_key"], tuple(entry["last_env_ids"]))
            for entry in assignments_by_worker[worker_index]
        } == {
            (f"proc-{i}", (i,)) for i in range(worker_index, 32, 4)
        }
