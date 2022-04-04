import os
import subprocess
from pathlib import Path

import astropy.io.fits as fits


def get_template(script):
    return f"""
.. automodule:: examples.{script}
  :members:

.. literalinclude:: ../../examples/{script}.py
  :language: python
  :linenos:
  :start-after: "__main__":
  :end-at: sim.run()
  :tab-width: 0
  :dedent: 4

.. raw:: html
   :file: _static/plots/example_results/{script}.html
   
   """


os.chdir(Path(__file__).parent.parent.parent)
doc_content = """
Examples - python scripts
=========================
"""

for script in sorted(list(Path('examples').glob('*.py'))):
    if "__init__" not in str(script):
        subprocess.call(f"python {script}", shell=True)
        fits_path = Path(f"{Path(f'{script}').stem}.fits")
        fits.getdata(fits_path)

        Path(f"{Path(f'{script}').stem}.fits").unlink(True)
        doc_content += get_template(script.stem)

with open("docs/source/example_direct.rst", "w") as f:
    f.write(doc_content)
