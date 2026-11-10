"""Execute the notebook using this Python environment, from any directory."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parent
path = root / (sys.argv[1] if len(sys.argv) > 1 else 'Endterm_Denis_Shin_Abylay_Latiyev_Nikita_Vassilenko.ipynb')
with TemporaryDirectory(prefix='notebook-kernel-') as temporary:
    kernel = Path(temporary) / 'notebook'
    kernel.mkdir()
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Notebook', 'language': 'python',
        'env': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    }))
    manager = KernelManager(kernel_name='notebook',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(notebook, km=manager, timeout=600,
                       resources={'metadata': {'path': str(root)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
print(f'{path.name} executed; outputs and artifacts saved.')
