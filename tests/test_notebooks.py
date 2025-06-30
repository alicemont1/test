import os
import pytest
import nbformat
import base64
from io import BytesIO
from PIL import Image
import imagehash

from pytest_notebook.nb_regression import NBRegressionFixture
from pytest_notebook.diffing import filter_diff, diff_to_string



NOTEBOOK_PATHS = [
    # "climate-dt/hell.ipynb"
    # 'climate-dt/climate-dt-earthkit-aoi-example.ipynb', #
    # 'climate-dt/climate-dt-earthkit-area-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-example-domain.ipynb',
    'climate-dt/climate-dt-earthkit-example.ipynb',
    # # 'climate-dt/climate-dt-earthkit-fe-boundingbox.ipynb',
    # # 'climate-dt/climate-dt-earthkit-fe-polygon.ipynb',
    # # 'climate-dt/climate-dt-earthkit-fe-story-nudging.ipynb', #
    # # 'climate-dt/climate-dt-earthkit-fe-timeseries.ipynb', #
    # # 'climate-dt/climate-dt-earthkit-fe-trajectory.ipynb',
    # # 'climate-dt/climate-dt-earthkit-fe-verticalprofile.ipynb',
    # # 'climate-dt/climate-dt-earthkit-grid-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-healpix-interpolate.ipynb',
    # # 'climate-dt/climate-dt-healpix-data.ipynb',
    # # 'climate-dt/climate-dt-healpix-ocean-example.ipynb',

    # # 'extremes-dt/extremes-dt-earthkit-example-domain.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-boundingbox.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-country.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-polygon.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-timeseries.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-trajectory.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-trajectory4d.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-verticalprofile.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-fe-wave.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example-regrid.ipynb',
    # # 'extremes-dt/extremes-dt-earthkit-example.ipynb',


]
# Static paths we always ignore
BASE_IGNORES = (
    '/metadata/language_info/',
    '/cells/*/execution_count',
    '/cells/*/outputs/*/execution_count',
    "/cells/10/outputs/1/data/text/html"
)

# Map tags to ignore paths
TAG_IGNORES = {
    "skip-text-html": "/cells/{idx}/outputs/1/data/text/html",
    "skip-text-plain": "/cells/{idx}/outputs/*/data/text/plain",
    "skip-outputs": "/cells/{idx}/outputs",
    "skip-image": "/cells/{idx}/outputs/*/data/image/png",
}

# Tags that require image comparison instead of ignoring outright
TAG_IMAGE_CHECKS = {"check-image"}


def perceptual_hash(b64_string: str):
    """Convert base64 PNG to perceptual hash."""
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return imagehash.phash(image)


def analyze_tags(nb):
    """Return paths to ignore and image check targets based on tags."""
    ignore_paths = []
    image_checks = []

    for idx, cell in enumerate(nb.cells):
        tags = set(cell.metadata.get("tags", []))

        # Add ignore paths for each matching tag
        for tag, path_template in TAG_IGNORES.items():
            if tag in tags:
                ignore_paths.append(path_template.format(idx=idx))

        # If any "check-image" tag is present, add all output indexes
        if TAG_IMAGE_CHECKS.intersection(tags):
            for output_idx, _ in enumerate(cell.get("outputs", [])):
                if cell.outputs[output_idx].get("data", {}).get("image/png"):
                    image_checks.append((idx, output_idx))

    return ignore_paths, image_checks


def compare_images(result, image_checks_initial, image_checks_final):
    """Compare image hashes and remove diffs for perceptually identical images."""
    remove_paths = []

    for (cell_idx, output_idx_final), (_, output_idx_initial) in zip(image_checks_final, image_checks_initial):


        png1 = result.nb_initial.cells[cell_idx].outputs[output_idx_initial].data["image/png"] #necessary so that other tags are ignored
        png2 = result.nb_final.cells[cell_idx].outputs[output_idx_final].data["image/png"]

        
        # Handle case where base64 is split in a list
        png1 = "".join(png1) if isinstance(png1, list) else png1
        png2 = "".join(png2) if isinstance(png2, list) else png2

        if perceptual_hash(png1) == perceptual_hash(png2):
            remove_paths.append(f"/cells/{cell_idx}/outputs/{output_idx_final}/data/image/png")
            remove_paths.append(f"/cells/{cell_idx}/outputs/{output_idx_initial}/data/image/png")

    return filter_diff(result.diff_filtered, remove_paths=remove_paths)

# @pytest.mark.parametrize("nb_file", [os.getenv("PYTEST_NB_FILE")])
# NOTEBOOK_PATHS = os.environ.get("NOTEBOOKS", "").split()

@pytest.mark.parametrize("nb_file", NOTEBOOK_PATHS)
def test_changed_notebook(nb_file, nb_regression: NBRegressionFixture):
    nb = nbformat.read(nb_file, as_version=4)

    ignore_paths, image_checks = analyze_tags(nb)
    # Set working directory to the notebook's parent directory
    nb_regression.exec_cwd = os.path.dirname(nb_file)
    nb_regression.diff_ignore = BASE_IGNORES + tuple(ignore_paths)

    result = nb_regression.check(nb_file, raise_errors=False)

    _, image_checks_final = analyze_tags(result.nb_final)

    _, image_checks_initial = analyze_tags(result.nb_initial)

    if result.diff_filtered:
        if image_checks:
            filtered_diff = compare_images(result, image_checks_initial, image_checks_final)
            if filtered_diff:
                diff_str = diff_to_string(result.nb_final, filtered_diff, use_git=False, use_diff=True)
                pytest.fail(diff_str)
        else:
            pytest.fail(result.diff_string)
