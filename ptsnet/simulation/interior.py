"""Selected pipe-point output requests for PTSNET simulations."""

import math
from numbers import Integral, Real

import numpy as np

_SUPPORTED_VARIABLES = ("head", "flowrate")
_SUPPORTED_LOCATION_TYPES = ("fraction", "index")


class PipeOutputRequest(object):
    """Describe one requested output at a location in a pipe."""

    def __init__(
        self,
        name,
        pipe_name,
        position,
        position_type="fraction",
        variables=None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("output name must be a non-empty string")
        if not isinstance(pipe_name, str) or not pipe_name:
            raise ValueError("pipe_name must be a non-empty string")
        if position_type not in _SUPPORTED_LOCATION_TYPES:
            raise ValueError("unsupported position_type: %s" % position_type)

        if position_type == "fraction":
            if isinstance(position, bool) or not isinstance(position, Real):
                raise TypeError("fractional position must be a real number")
            if not 0.0 <= float(position) <= 1.0:
                raise ValueError("fractional position must be between 0 and 1")
            position = float(position)
        else:
            if isinstance(position, bool) or not isinstance(position, Integral):
                raise TypeError("mesh position must be an integer index")
            if position < 0:
                raise ValueError("mesh index must be non-negative")
            position = int(position)

        if variables is None:
            variables = _SUPPORTED_VARIABLES
        elif isinstance(variables, str):
            variables = (variables,)
        else:
            try:
                variables = tuple(variables)
            except TypeError as error:
                raise ValueError(
                    "variables must be a sequence of supported variable names"
                ) from error
        if not variables:
            raise ValueError("at least one output variable must be requested")
        unsupported = set(variables).difference(_SUPPORTED_VARIABLES)
        if unsupported:
            raise ValueError("unsupported output variables: %s" % sorted(unsupported))
        if len(set(variables)) != len(variables):
            raise ValueError("output variables must be unique")

        self.name = name
        self.pipe_name = pipe_name
        self.position = position
        self.position_type = position_type
        self.variables = tuple(variables)


class ResolvedPipeOutputRequest(object):
    """A request resolved to one or two global mesh points."""

    def __init__(self, request, point_indices, interpolation_weight):
        self.name = request.name
        self.pipe_name = request.pipe_name
        self.position = request.position
        self.position_type = request.position_type
        self.variables = request.variables
        self.point_indices = tuple(point_indices)
        self.interpolation_weight = float(interpolation_weight)


def resolve_pipe_output_requests(pipe_labels, segments, requests):
    """Resolve pipe-relative requests to global computational point indices."""
    pipe_labels = list(pipe_labels)
    segments = np.asarray(segments)
    if len(pipe_labels) != len(segments):
        raise ValueError("pipe labels and segment counts must have equal lengths")
    if len(set(pipe_labels)) != len(pipe_labels):
        raise ValueError("pipe labels must be unique")

    pipe_indexes = {name: index for index, name in enumerate(pipe_labels)}
    starts = np.concatenate(([0], np.cumsum(segments[:-1].astype(int) + 1)))
    resolved = {}

    for request in requests:
        if not isinstance(request, PipeOutputRequest):
            raise TypeError("requests must contain PipeOutputRequest instances")
        if request.name in resolved:
            raise ValueError("duplicate output name: %s" % request.name)
        if request.pipe_name not in pipe_indexes:
            raise ValueError("unknown pipe: %s" % request.pipe_name)

        pipe_index = pipe_indexes[request.pipe_name]
        segment_count = int(segments[pipe_index])
        if segment_count < 2:
            raise ValueError("pipe must have at least two segments")
        start = int(starts[pipe_index])

        if request.position_type == "fraction":
            coordinate = request.position * segment_count
            nearest = round(coordinate)
            if math.isclose(coordinate, nearest, rel_tol=1e-12, abs_tol=1e-12):
                point = start + int(nearest)
                point_indices = (point, point)
                weight = 0.0
            else:
                lower = int(math.floor(coordinate))
                upper = lower + 1
                point_indices = (start + lower, start + upper)
                weight = coordinate - lower
        else:
            interior_count = segment_count - 1
            if request.position >= interior_count:
                raise ValueError(
                    "mesh index %d is outside the %d interior points on pipe %s"
                    % (request.position, interior_count, request.pipe_name)
                )
            point = start + 1 + request.position
            point_indices = (point, point)
            weight = 0.0

        resolved[request.name] = ResolvedPipeOutputRequest(
            request, point_indices, weight
        )

    return resolved


def requested_pipe_point_variables(resolved_requests):
    """Return unique global point indices and variables needed by the requests."""
    point_variables = {}
    for request in resolved_requests.values():
        for point_index in set(request.point_indices):
            point_variables.setdefault(point_index, set()).update(request.variables)
    return {
        point_index: tuple(sorted(variables))
        for point_index, variables in point_variables.items()
    }


def select_owned_pipe_points(point_variables, processors, local_to_global, rank):
    """Select requested global points owned by one worker rank."""
    processors = np.asarray(processors)
    owned_points = {}
    for point_index, variables in point_variables.items():
        if isinstance(point_index, bool) or not isinstance(point_index, Integral):
            raise ValueError("global point indices must be integers")
        point_index = int(point_index)
        if point_index < 0 or point_index >= len(processors):
            raise ValueError("global point index is outside the mesh")
        if int(processors[point_index]) != rank:
            continue
        if point_index not in local_to_global:
            raise RuntimeError(
                "selected interior point %d is missing from its owner worker"
                % point_index
            )
        owned_points[point_index] = tuple(variables)
    return owned_points


def assemble_pipe_output_results(resolved_requests, point_histories):
    """Interpolate selected point histories into named pipe-output series."""
    outputs = {}
    for name, request in resolved_requests.items():
        output = {}
        lower_index, upper_index = request.point_indices
        for variable in request.variables:
            try:
                lower = np.asarray(point_histories[lower_index][variable], dtype=float)
                upper = np.asarray(point_histories[upper_index][variable], dtype=float)
            except KeyError as error:
                raise ValueError(
                    "missing captured %s history for output %s" % (variable, name)
                ) from error
            if lower.shape != upper.shape:
                raise ValueError(
                    "captured point histories have different shapes for output %s"
                    % name
                )
            weight = request.interpolation_weight
            output[variable] = (
                lower.copy()
                if weight == 0.0
                else (lower * (1.0 - weight) + upper * weight)
            )
        outputs[name] = output
    return outputs
