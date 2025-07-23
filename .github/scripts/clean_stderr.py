import os
import nbformat

def remove_stderr(nb, target_folder):
    for cell in nb.cells:
        if "outputs" in cell:
            cell.outputs = [
                output for output in cell.outputs
                if output.get("name") != "stderr"
            ]
    tmp_file = os.path.join(target_folder, "tmp.ipynb")
    with open(tmp_file, "w") as f:
        nbformat.write(nb, f)
    return tmp_file

def clean_all_notebooks(folder="."):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".ipynb"):
                path = os.path.join(root, file)
                nb = nbformat.read(path, as_version=4)
                for cell in nb.cells:
                    if "outputs" in cell:
                        cell.outputs = [
                            o for o in cell.outputs if o.get("name") != "stderr"
                        ]
                nbformat.write(nb, path)

if __name__ == "__main__":
    clean_all_notebooks()
