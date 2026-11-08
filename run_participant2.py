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
path = root / 'Participant2_Preprocessing_Baseline.ipynb'
with TemporaryDirectory(prefix='participant2-kernel-') as temporary:
    kernel = Path(temporary) / 'participant2'
    kernel.mkdir()
    (kernel / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Participant 2', 'language': 'python',
        'env': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    }))
    manager = KernelManager(kernel_name='participant2',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(notebook, km=manager, timeout=600,
                       resources={'metadata': {'path': str(root)}}).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    nbformat.write(notebook, path)
print('Participant 2 notebook executed; outputs and artifacts saved.')
