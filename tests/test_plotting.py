from __future__ import annotations

import matplotlib.pyplot as plt

from seqcourse import plot_distribution, plot_frequency, plot_index, plot_representatives, representative_sequences


def test_plot_functions_return_figure_and_axes(toy_sequences) -> None:
    functions = [
        lambda: plot_distribution(toy_sequences),
        lambda: plot_frequency(toy_sequences),
        lambda: plot_index(toy_sequences),
        lambda: plot_representatives(representative_sequences(toy_sequences, criterion="freq", diss=None, method="LCS")),
    ]
    for function in functions:
        figure, axes = function()
        assert figure is not None
        assert axes is not None
        plt.close(figure)
