"""Execute the standalone notebook without depending on nbclient/nbformat."""

from __future__ import annotations

import json
import base64
import io
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "source",
    nargs="?",
    type=Path,
    default=ROOT / "README.ipynb",
    help="input notebook (default: README.ipynb beside this script)",
)
parser.add_argument(
    "target",
    nargs="?",
    type=Path,
    default=ROOT / "README.executed.ipynb",
    help="executed notebook to write",
)
args = parser.parse_args()
source = args.source.resolve()
target = args.target.resolve()
notebook = json.loads(source.read_text())
shell = InteractiveShell.instance()


for number, cell in enumerate(notebook["cells"], 1):
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    cell["outputs"] = []
    with capture_output(stdout=True, stderr=True, display=True) as captured:
        result = shell.run_cell(text, store_history=False, silent=False)
    if captured.stdout:
        cell["outputs"].append(
            {"output_type": "stream", "name": "stdout", "text": captured.stdout}
        )
    if captured.stderr:
        cell["outputs"].append(
            {"output_type": "stream", "name": "stderr", "text": captured.stderr}
        )
    for output in captured.outputs:
        cell["outputs"].append(
            {
                "output_type": "display_data",
                "metadata": output.metadata or {},
                "data": output.data,
            }
        )
    # A plain InteractiveShell has no notebook display hook.  Preserve every
    # matplotlib figure explicitly, then close it so a later cell cannot
    # redisplay stale figures.
    for figure_number in plt.get_fignums():
        figure = plt.figure(figure_number)
        memory = io.BytesIO()
        figure.savefig(memory, format="png", dpi=150, bbox_inches="tight")
        cell["outputs"].append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(memory.getvalue()).decode()},
            }
        )
        plt.close(figure)
    if result.error_before_exec or result.error_in_exec:
        error = result.error_before_exec or result.error_in_exec
        raise RuntimeError(f"cell {number} ({cell.get('id','')}) failed") from error
    cell["execution_count"] = number
    print(f"PASS cell {number:02d} {cell.get('id','')}", flush=True)

target.write_text(json.dumps(notebook, indent=1))
print(target)
