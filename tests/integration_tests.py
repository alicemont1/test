import pytest
import numpy as np
from earthkit.data import from_source
from earthkit import regrid
import earthkit.plots
import tempfile
from pathlib import Path
import shutil
from matplotlib.testing.compare import compare_images




def generate_grid_cells_plot(data, filename=None, title=None, legend=False, **kwargs):
    chart = earthkit.plots.Map(**kwargs)
    chart.grid_cells(
        data
    )

    chart.title(title)
    if legend:
        chart.legend()
    chart.gridlines()
    chart.coastlines()
    chart.save(filename)

def generate_figure(data, filename=None, title=None, **kwargs):
    figure = earthkit.plots.Figure(columns=3, size=(10, 5), domain="Europe")

    for i, _ in enumerate(range(len(data.ls()))):
        figure.add_map(1, i)
    figure.contourf(data)
    figure.land()
    figure.coastlines()
    figure.title(title)

    figure.save(filename)

import matplotlib.pyplot as plt

def generate_figure_matplotlib(data, filename=None, title=None, **kwargs):
    # Create EarthKit figure
    figure = earthkit.plots.Figure(**kwargs)

    # Add map layers to the figure
    for i, _ in enumerate(range(len(data.ls()))):
        figure.add_map(1, i)

    # Add the data contour and other features to the figure
    figure.contourf(data)
    figure.land()
    figure.coastlines()
    figure.title(title)

    # Now, we need to extract the matplotlib figure object that EarthKit uses under the hood.
    # EarthKit internally uses matplotlib, so we can directly access the figure.
    fig = plt.gcf()  # Get the current matplotlib figure object

    # Save the figure using matplotlib's savefig
    fig.savefig(filename)  # Saving the figure with Matplotlib's savefig method

    # Optionally close the figure to avoid memory issues
    plt.close(fig)


def compare_plot_images(expected_path, actual_path, tol=1):
    diff = compare_images(str(expected_path), str(actual_path), tol=tol)
    assert diff is None, f"Plot image mismatch: {diff}"




def test_example_domain():
    data = from_source("file", "./climate-dt/data/climate-dt-earthkit-example-domain.grib")
    assert data is not None

    out_grid = {"grid": [0.1, 0.1]}
    data_interpolated = regrid.interpolate(data, out_grid=out_grid, method="linear")
    xr = data_interpolated.to_xarray()

    #Check if 
    expected_coords = {"latitude", "longitude"}
    expected_vars = {"10u", "10v", "sp"}

    assert set(xr.coords) == expected_coords
    assert set(xr.data_vars) == expected_vars

    # Check the shape and contents are as expected
    assert xr.sp.shape == data_interpolated[0].shape == (1801, 3600) # surface pressure
   

    # Temporary directory for plots
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        file1 = "tests/images/climate-dt-earthkit-example-domain1.png"
        file2 = "tests//images/climate-dt-earthkit-example-domain2.png"
        file3 = "tests//images/climate-dt-earthkit-example-domain3.png"
        tmp_file1 = tmpdir / "plot_extent.png"
        tmp_file2 = tmpdir / "plot_domain.png"
        tmp_file3= tmpdir / "figure_domains.png"

        # generate_grid_cells_plot(
        #     data[0],
        #     filename=tmp_file1,
        #     title="Surface Pressure",
        #     legend=False,
        #     extent=[-180, 180, -90, 90]
        # )
        # generate_grid_cells_plot(
        #     data_interpolated[0],
        #     filename=tmp_file2,
        #     legend=True,
        #     domain="Europe"
        # )

        # TODO save figure not working
        generate_figure_matplotlib(
            data,
            filename=tmp_file3,
            title="Surface pressure, 10m U wind component at 10 meter V wind component",
            columns=3, size=(10, 5), domain="Europe"
        )

        # Compare plots visually
        # compare_plot_images(file1, tmp_file1)
        # compare_plot_images(file2, tmp_file2)
        compare_plot_images(file3, tmp_file3)