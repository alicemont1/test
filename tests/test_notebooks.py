import base64
import tempfile
from io import BytesIO
from pathlib import Path

import nbformat
import pytest
from PIL import Image
import imagehash

from pytest_notebook.nb_regression import NBRegressionFixture
from pytest_notebook.diffing import filter_diff, diff_to_string

NOTEBOOK_PATHS = [
    # 'climate-dt/climate-dt-earthkit-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-aoi-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-area-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-example-domain.ipynb',
    # 'climate-dt/climate-dt-earthkit-fe-boundingbox.ipynb',
    # 'climate-dt/climate-dt-earthkit-fe-polygon.ipynb',
    'climate-dt/climate-dt-earthkit-fe-story-nudging.ipynb', #f
    'climate-dt/climate-dt-earthkit-fe-timeseries.ipynb', #f
    'climate-dt/climate-dt-earthkit-fe-trajectory.ipynb', #f
    'climate-dt/climate-dt-earthkit-fe-verticalprofile.ipynb', #f
    # 'climate-dt/climate-dt-earthkit-grid-example.ipynb',
    # 'climate-dt/climate-dt-earthkit-healpix-interpolate.ipynb',
    # 'climate-dt/climate-dt-healpix-data.ipynb',
    # 'climate-dt/climate-dt-healpix-ocean-example.ipynb',

    'extremes-dt/extremes-dt-earthkit-example-domain.ipynb', #f
    # 'extremes-dt/extremes-dt-earthkit-example-fe-boundingbox.ipynb',
    # 'extremes-dt/extremes-dt-earthkit-example-fe-country.ipynb',
    # 'extremes-dt/extremes-dt-earthkit-example-fe-polygon.ipynb',
    'extremes-dt/extremes-dt-earthkit-example-fe-timeseries.ipynb', #f
    # 'extremes-dt/extremes-dt-earthkit-example-fe-trajectory.ipynb',
    # 'extremes-dt/extremes-dt-earthkit-example-fe-trajectory4d.ipynb',
    'extremes-dt/extremes-dt-earthkit-example-fe-verticalprofile.ipynb', #f
    'extremes-dt/extremes-dt-earthkit-example-fe-wave.ipynb', #f
    'extremes-dt/extremes-dt-earthkit-example-regrid.ipynb', #f
    # 'extremes-dt/extremes-dt-earthkit-example.ipynb',

    'on-demand-extremes-dt/on-demand-extremes-dt-example.ipynb',
]

# Paths in notebooks to always ignore during diffing
BASE_IGNORES = (
    '/metadata/language_info/',
    '/cells/*/execution_count',
    '/cells/*/outputs/*/execution_count',
    '/cells/*/outputs/*/data/text/html',
)

# Tags that instruct diff-ignore behavior
TAG_IGNORES = {
    "skip-text-plain": "/cells/{idx}/outputs/*/data/text/plain",
    "skip-outputs": "/cells/{idx}/outputs",
    "skip-image": "/cells/{idx}/outputs/*/data/image/png",
}

# Tags that require perceptual image comparison instead of ignoring
TAG_IMAGE_CHECKS = {"check-image"}


def perceptual_hash(b64_png: str):
    """Convert base64-encoded PNG to perceptual hash."""
    data = base64.b64decode(b64_png)
    img = Image.open(BytesIO(data)).convert("RGB")
    return imagehash.phash(img)


def analyze_tags(nb):
    """Analyze notebook for tag-based ignore paths and image checks."""
    ignore_paths = []
    image_checks = []

    for idx, cell in enumerate(nb.cells):
        tags = set(cell.metadata.get("tags", []))

        # Collect ignore paths
        for tag, template in TAG_IGNORES.items():
            if tag in tags:
                ignore_paths.append(template.format(idx=idx))

        # Collect image check indices
        if TAG_IMAGE_CHECKS & tags:
            for output_idx, output in enumerate(cell.get("outputs", [])):
                if "image/png" in output.get("data", {}):
                    image_checks.append((idx, output_idx))

    return ignore_paths, image_checks


def compare_images(result, checks_initial, checks_final, threshold=4):
    """Filter out perceptually identical image diffs."""
    paths_to_remove = []

    for (cell_idx, out_idx_f), (_, out_idx_i) in zip(checks_final, checks_initial):
        png1 = result.nb_initial.cells[cell_idx].outputs[out_idx_i].data["image/png"]
        png2 = result.nb_final.cells[cell_idx].outputs[out_idx_f].data["image/png"]

        # Convert list-of-strings to single base64 string if needed
        png1 = "".join(png1) if isinstance(png1, list) else png1
        png2 = "".join(png2) if isinstance(png2, list) else png2

        if perceptual_hash(png1) - perceptual_hash(png2) <= threshold:
            paths_to_remove.append(f"/cells/{cell_idx}/outputs/{out_idx_f}/data/image/png")

    return filter_diff(result.diff_filtered, remove_paths=paths_to_remove)


def inject_silence_stderr_cell(nb):
    """Insert a code cell to suppress stderr output."""
    patch_code = """
    LIVE_REQUEST = False
    import sys
    class DevNull:
        def write(self, msg): pass
        def flush(self): pass

    sys.stderr = DevNull()
    """
    silence_cell = nbformat.v4.new_code_cell(source=patch_code)
    nb.cells.insert(0, silence_cell)


@pytest.mark.parametrize("nb_file", NOTEBOOK_PATHS)
def test_changed_notebook(nb_file, nb_regression: NBRegressionFixture):
    # Load and patch notebook
    nb = nbformat.read(nb_file, as_version=4)
    inject_silence_stderr_cell(nb)

    # Analyze for tag-based ignore paths and image checks
    ignore_paths, image_checks = analyze_tags(nb)

    # Save modified notebook to temporary file
    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False, mode="w") as tmp:
        nbformat.write(nb, tmp)
        patched_path = tmp.name

    # Setup regression execution
    nb_regression.exec_notebook = True
    nb_regression.exec_cwd = str(Path(nb_file).parent)
    nb_regression.diff_ignore = BASE_IGNORES + tuple(ignore_paths)

    result = nb_regression.check(patched_path, raise_errors=False)

    # Post-process diff if perceptual image comparisons are needed
    if result.diff_filtered:
        _, final_checks = analyze_tags(result.nb_final)
        _, initial_checks = analyze_tags(result.nb_initial)

        if image_checks:
            filtered = compare_images(result, initial_checks, final_checks, threshold=5)
            if filtered:
                diff_str = diff_to_string(result.nb_final, filtered, use_git=True, use_diff=True, use_color=True)
                pytest.fail(diff_str)
        else:
            pytest.fail(result.diff_string)