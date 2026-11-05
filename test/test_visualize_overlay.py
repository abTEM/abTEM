"""Overlaying and exploding the ensemble axes of a MeasurementsEnsemble in ``show()``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from abtem.core.axes import OrdinalAxis, ParameterAxis, ThicknessAxis
from abtem.measurements import MeasurementsEnsemble


def _lines(visualization):
    return [
        [(line.get_label(), line.get_ydata()) for line in ax.lines]
        for ax in visualization.axes.fig.axes
    ]


def test_overlay_labels_leave_out_the_x_axis():
    # the last axis is drawn along x, so it contributes no labels: two lines are
    # drawn and two labels are needed
    array = np.random.default_rng(0).random((3, 2, 5))
    measurement = MeasurementsEnsemble(
        array,
        ensemble_axes_metadata=[
            ParameterAxis(values=(1.0, 2.0, 3.0)),
            OrdinalAxis(values=("single", "double")),
            ThicknessAxis(values=(1.0, 2.0, 3.0, 4.0, 5.0)),
        ],
    )

    visualization = measurement.show(explode=(0,), overlay=(1, 2))
    lines = _lines(visualization)
    plt.close("all")

    assert len(lines) == 3
    for i, panel in enumerate(lines):
        assert [label for label, _ in panel] == ["single", "double"]
        np.testing.assert_array_equal(
            np.array([y for _, y in panel]), array[i]
        )


def test_overlay_of_an_axis_before_an_indexed_axis():
    # the axes of the overlay are not indexed, and must not shift the default index
    # of the axis that follows them
    array = np.random.default_rng(1).random((2, 3, 4, 5))
    measurement = MeasurementsEnsemble(
        array,
        ensemble_axes_metadata=[
            ParameterAxis(values=(1.0, 2.0)),
            ParameterAxis(values=(1.0, 2.0, 3.0)),
            ParameterAxis(values=(1.0, 2.0, 3.0, 4.0)),
            ParameterAxis(values=(1.0, 2.0, 3.0, 4.0, 5.0)),
        ],
    )

    visualization = measurement.show(explode=(0,), overlay=(1, 3))
    lines = _lines(visualization)
    plt.close("all")

    assert len(lines) == 2
    for i, panel in enumerate(lines):
        np.testing.assert_array_equal(
            np.array([y for _, y in panel]), array[i, :, 0, :]
        )
