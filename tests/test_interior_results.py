import os
import numpy as np
import pytest
from types import SimpleNamespace

from ptsnet.parallel.worker import Worker
from ptsnet.simulation.sim import PTSNETSimulation

from ptsnet.simulation.interior import (
    PipeOutputRequest,
    assemble_pipe_output_results,
    resolve_pipe_output_requests,
    select_owned_pipe_points,
)


def test_fractional_pipe_output_resolves_adjacent_mesh_points():
    requests = [
        PipeOutputRequest(
            name="sensor",
            pipe_name="P2",
            position=0.5,
            position_type="fraction",
            variables=("head", "flowrate"),
        )
    ]

    resolved = resolve_pipe_output_requests(
        pipe_labels=["P1", "P2"],
        segments=[4, 3],
        requests=requests,
    )

    output = resolved["sensor"]
    assert output.point_indices == (6, 7)
    assert output.interpolation_weight == pytest.approx(0.5)


def test_mesh_index_is_zero_based_among_interior_points():
    requests = [
        PipeOutputRequest(
            name="sensor",
            pipe_name="P1",
            position=1,
            position_type="index",
            variables=("head",),
        )
    ]

    resolved = resolve_pipe_output_requests(
        pipe_labels=["P1"],
        segments=[4],
        requests=requests,
    )

    output = resolved["sensor"]
    assert output.point_indices == (2, 2)
    assert output.interpolation_weight == 0.0


@pytest.mark.parametrize("position", [-0.1, 1.1])
def test_fractional_position_must_be_within_pipe(position):
    with pytest.raises(ValueError, match="fraction"):
        request = PipeOutputRequest(
            name="sensor",
            pipe_name="P1",
            position=position,
            position_type="fraction",
        )
        resolve_pipe_output_requests(["P1"], [4], [request])


def test_mesh_index_cannot_select_pipe_endpoints():
    request = PipeOutputRequest(
        name="sensor",
        pipe_name="P1",
        position=3,
        position_type="index",
    )

    with pytest.raises(ValueError, match="interior"):
        resolve_pipe_output_requests(["P1"], [4], [request])


def test_unknown_pipe_output_is_rejected():
    request = PipeOutputRequest(
        name="sensor",
        pipe_name="missing",
        position=0.5,
    )
    with pytest.raises(ValueError, match="unknown pipe"):
        resolve_pipe_output_requests(["P1"], [4], [request])


def test_pipe_output_requests_reject_duplicate_names_and_persisted_models():
    simulation = PTSNETSimulation.__new__(PTSNETSimulation)
    simulation.settings = SimpleNamespace(
        is_initialized=False,
        active_persistance=False,
    )
    simulation.worker = None
    simulation.pipe_output_requests = []
    simulation.add_pipe_output("sensor", "P1", 0.5)

    with pytest.raises(ValueError, match="duplicate"):
        simulation.add_pipe_output("sensor", "P1", 0.25)

    simulation.settings.active_persistance = True
    with pytest.raises(NotImplementedError, match="persisted workspace"):
        simulation.add_pipe_output("other", "P1", 0.25)


def test_requested_locations_interpolate_only_selected_history():
    request = PipeOutputRequest(
        name="sensor",
        pipe_name="P1",
        position=0.375,
        position_type="fraction",
        variables=("head",),
    )
    resolved = resolve_pipe_output_requests(["P1"], [4], [request])
    histories = {
        1: {"head": np.array([10.0, 12.0, 14.0])},
        2: {"head": np.array([20.0, 24.0, 28.0])},
        3: {"head": np.array([100.0, 100.0, 100.0])},
    }

    outputs = assemble_pipe_output_results(resolved, histories)

    np.testing.assert_allclose(outputs["sensor"]["head"], [15.0, 18.0, 21.0])
    assert set(outputs["sensor"]) == {"head"}


def test_exact_point_output_copies_requested_history():
    request = PipeOutputRequest(
        name="sensor",
        pipe_name="P1",
        position=1,
        position_type="index",
        variables=("head", "flowrate"),
    )
    resolved = resolve_pipe_output_requests(["P1"], [4], [request])
    histories = {
        2: {
            "head": np.array([10.0, 11.0]),
            "flowrate": np.array([1.0, 2.0]),
        }
    }

    outputs = assemble_pipe_output_results(resolved, histories)

    np.testing.assert_array_equal(outputs["sensor"]["head"], [10.0, 11.0])
    np.testing.assert_array_equal(outputs["sensor"]["flowrate"], [1.0, 2.0])


def test_worker_records_only_requested_points_for_each_time_step():
    worker = Worker.__new__(Worker)
    worker.time_steps = 3
    worker.partition = {"points": {"local_idx": {1: 0, 3: 1}}}
    worker.interior_point_histories = {
        1: {"head": np.zeros(3)},
        3: {"flowrate": np.zeros(3)},
    }
    worker.mem_pool_points = SimpleNamespace(
        head=np.array([[100.0, 10.0], [200.0, 20.0]]),
        flowrate=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    worker._record_interior_point_values(time_index=1, memory_column=1)

    np.testing.assert_array_equal(
        worker.interior_point_histories[1]["head"],
        [0.0, 10.0, 0.0],
    )
    np.testing.assert_array_equal(
        worker.interior_point_histories[3]["flowrate"],
        [0.0, 4.0, 0.0],
    )


def test_worker_capture_ownership_selects_only_points_owned_by_rank():
    requested = {
        1: ("head",),
        2: ("flowrate",),
    }
    processors = np.array([0, 0, 1, 1])

    assert select_owned_pipe_points(requested, processors, {1: 0}, rank=0) == {
        1: ("head",)
    }
    assert select_owned_pipe_points(requested, processors, {2: 0}, rank=1) == {
        2: ("flowrate",)
    }


def test_worker_capture_ownership_rejects_missing_owned_point():
    with pytest.raises(RuntimeError, match="owner worker"):
        select_owned_pipe_points({1: ("head",)}, [0, 0], {}, rank=0)


def test_simulation_registers_pipe_output_before_initialization():
    simulation = PTSNETSimulation.__new__(PTSNETSimulation)
    simulation.settings = SimpleNamespace(
        is_initialized=False,
        active_persistance=False,
    )
    simulation.worker = None
    simulation.pipe_output_requests = []

    request = simulation.add_pipe_output(
        "midpoint",
        "P1",
        0.5,
        position_type="fraction",
        variables=("head",),
    )

    assert simulation.pipe_output_requests == [request]
    simulation.settings.is_initialized = True
    with pytest.raises(RuntimeError, match="before initialization"):
        simulation.add_pipe_output("late", "P1", 0.25)


def test_simulation_collects_selected_histories_on_root():
    request = PipeOutputRequest(
        name="midpoint",
        pipe_name="P1",
        position=0.375,
        position_type="fraction",
        variables=("head",),
    )
    simulation = PTSNETSimulation.__new__(PTSNETSimulation)
    simulation._interior_results_collected = False
    simulation.resolved_pipe_outputs = resolve_pipe_output_requests(
        ["P1"], [4], [request]
    )
    simulation.worker = SimpleNamespace(
        interior_point_histories={
            1: {"head": np.array([10.0, 12.0, 14.0])},
        }
    )

    class FakeCommunicator:
        rank = 0

        @staticmethod
        def gather(payload, root):
            assert root == 0
            return [
                payload,
                {2: {"head": np.array([20.0, 24.0, 28.0])}},
            ]

    simulation.router = {"main": FakeCommunicator()}
    simulation.results = {}
    simulation.interior_results = {}

    simulation._collect_interior_results()

    np.testing.assert_allclose(
        simulation.interior_results["midpoint"]["head"],
        [15.0, 18.0, 21.0],
    )
    assert simulation.results["pipe.interior"] is simulation.interior_results
    assert simulation._interior_results_collected


def test_non_root_keeps_no_global_interior_result_table():
    simulation = PTSNETSimulation.__new__(PTSNETSimulation)
    simulation._interior_results_collected = False
    simulation.resolved_pipe_outputs = {
        "sensor": object(),
    }
    simulation.worker = SimpleNamespace(interior_point_histories={})
    simulation.router = {
        "main": SimpleNamespace(rank=1, gather=lambda payload, root: None)
    }
    simulation.results = {}
    simulation.interior_results = {"stale": {}}

    simulation._collect_interior_results()

    assert simulation.interior_results == {}
    assert "pipe.interior" not in simulation.results
    assert simulation._interior_results_collected


def test_live_simulation_captures_fraction_and_mesh_index(tmp_path, monkeypatch):
    input_file = os.path.join(
        os.path.dirname(__file__), "..", "ptsnet", "examples", "TNET3.inp"
    )
    monkeypatch.chdir(tmp_path)
    try:
        simulation = PTSNETSimulation(
            workspace_name="interior-output",
            inpfile=input_file,
            settings={
                "time_step": 0.01,
                "duration": 0.05,
                "default_wave_speed": 1000,
                "warnings_on": False,
                "show_progress": False,
                "save_results": False,
            },
        )
    except OSError as error:
        if "libepanet" in str(error):
            pytest.skip("WNTR EPANET shared library is unavailable")
        raise

    pipe_name = simulation.wn.pipe_name_list[0]
    simulation.add_pipe_output(
        "midpoint",
        pipe_name,
        0.5,
        position_type="fraction",
        variables=("head", "flowrate"),
    )
    simulation.add_pipe_output(
        "first-interior-point",
        pipe_name,
        0,
        position_type="index",
        variables=("head",),
    )
    simulation.run()

    midpoint = simulation.interior_results["midpoint"]
    mesh_point = simulation.interior_results["first-interior-point"]
    assert midpoint["head"].shape == (simulation.settings.time_steps,)
    assert midpoint["flowrate"].shape == (simulation.settings.time_steps,)
    assert mesh_point["head"].shape == (simulation.settings.time_steps,)
    assert np.isfinite(midpoint["head"]).all()
    assert np.isfinite(midpoint["flowrate"]).all()


@pytest.mark.parametrize("variables", [5, 1.5])
def test_output_variables_must_be_iterable(variables):
    with pytest.raises(ValueError, match="variables"):
        PipeOutputRequest(
            name="sensor",
            pipe_name="P1",
            position=0.5,
            variables=variables,
        )


def test_reinitialization_clears_interior_result_state():
    simulation = PTSNETSimulation.__new__(PTSNETSimulation)
    simulation.settings = SimpleNamespace(
        defined_wave_speeds=True,
        is_initialized=True,
        updated_settings=True,
        warnings_on=False,
    )
    simulation.element_settings = {
        setting_type: SimpleNamespace(_sort=lambda: None)
        for setting_type in simulation.SETTING_TYPES
    }
    simulation.initializer = SimpleNamespace(
        ss={"valve": SimpleNamespace(curve_index=np.array([], dtype=int))},
        create_secondary_elements=lambda: None,
        create_secondary_selectors=lambda: None,
    )
    simulation.pipe_output_requests = []
    simulation.resolved_pipe_outputs = {"old": object()}
    simulation.interior_point_variables = {1: ("head",)}
    simulation.interior_results = {"old": {"head": np.array([1.0])}}
    simulation._interior_results_collected = True
    simulation._distribute_work = lambda: None
    simulation.save_init_data = lambda: None

    simulation.initialize()

    assert simulation.interior_results == {}
    assert simulation._interior_results_collected is False
