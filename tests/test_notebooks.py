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
    'climate-dt/climate-dt-earthkit-healpix-interpolate.ipynb',
]

BASE_IGNORES = (
    '/metadata/language_info/',
    '/cells/*/execution_count',
    '/cells/*/outputs/*/execution_count'
)

TAG_IGNORES = {
    "skip-text-html": "/cells/{idx}/outputs/*/data/text/html",
    "skip-text-plain": "/cells/{idx}/outputs/*/data/text/plain",
    "skip-outputs": "/cells/{idx}/outputs",
    "skip-image": "/cells/{idx}/outputs/*/data/image/png",
}

TAG_IMAGE_CHECKS = {"check-image"}


def perceptual_hash(b64_string: str):
    """Convert base64 PNG to perceptual hash."""
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return imagehash.phash(image)


def analyze_tags(nb):
    """Return ignore paths and image check targets based on tags."""
    ignore_paths = []
    image_checks = []

    for idx, cell in enumerate(nb.cells):
        tags = set(cell.metadata.get("tags", []))

        for tag, path_template in TAG_IGNORES.items():
            if tag in tags:
                ignore_paths.append(path_template.format(idx=idx))

        if TAG_IMAGE_CHECKS.intersection(tags):
            for output_idx, output in enumerate(cell.get("outputs", [])):
                if output.get("data", {}).get("image/png"):
                    image_checks.append((idx, output_idx))

    return ignore_paths, image_checks


def get_image_outputs_by_cell(nb):
    """Return dict: cell_idx -> list of output indexes with 'image/png'."""
    image_outputs = {}
    for cell_idx, cell in enumerate(nb.cells):
        outputs = cell.get("outputs", [])
        png_outputs = []
        for output_idx, output in enumerate(outputs):
            if output.get("data", {}).get("image/png"):
                png_outputs.append(output_idx)
        if png_outputs:
            image_outputs[cell_idx] = png_outputs
    return image_outputs


def compare_images(result):
    """
    Compare images from nb_initial and nb_final by perceptual hash.
    Remove diffs for perceptually identical images.
    """
    nb_initial_images = get_image_outputs_by_cell(result.nb_initial)
    nb_final_images = get_image_outputs_by_cell(result.nb_final)

    remove_paths = []

    for cell_idx in nb_initial_images.keys() & nb_final_images.keys():
        for out_idx_initial in nb_initial_images[cell_idx]:
            png1 = result.nb_initial.cells[cell_idx].outputs[out_idx_initial].data["image/png"]
            if isinstance(png1, list):
                png1 = "".join(png1)
            hash1 = perceptual_hash(png1)

            for out_idx_final in nb_final_images[cell_idx]:
                png2 = result.nb_final.cells[cell_idx].outputs[out_idx_final].data["image/png"]
                if isinstance(png2, list):
                    png2 = "".join(png2)
                hash2 = perceptual_hash(png2)

                if hash1 == hash2:
                    remove_paths.append(f"/cells/{cell_idx}/outputs/{out_idx_final}/data/image/png")

    return filter_diff(result.diff_filtered, remove_paths=remove_paths)


@pytest.mark.parametrize("nb_file", NOTEBOOK_PATHS)
def test_changed_notebook(nb_file, nb_regression: NBRegressionFixture):
    nb = nbformat.read(nb_file, as_version=4)

    ignore_paths, image_checks = analyze_tags(nb)
    nb_regression.exec_cwd = os.path.dirname(nb_file)
    nb_regression.diff_ignore = BASE_IGNORES + tuple(ignore_paths)

    result = nb_regression.check(nb_file, raise_errors=False)

    # Remove or properly handle warnings check if needed
    # For example, if you want to allow any warnings but not fail, just skip this line:
    # Or if expecting a specific warning, do:
    # with pytest.warns(SomeWarning):
    #     pass

    if result.diff_filtered:
        if image_checks:
            filtered_diff = compare_images(result)
            if filtered_diff:
                diff_str = diff_to_string(result.nb_final, filtered_diff, use_git=False, use_diff=True)
                pytest.fail(diff_str)
        else:
            pytest.fail(result.diff_string)
