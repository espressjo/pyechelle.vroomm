# This script creates the kernels for the different slits and writes them to a file.
# TODO: 'no photon noise' is not implemented yet, right now the code created is identical for both cases.
import inspect
from pathlib import Path

from jinja2 import FileSystemLoader, Environment
from numba.core.registry import CPUDispatcher
from numba.cuda.dispatcher import CUDADispatcher

import pyechelle.slit


def extract_slit_code(slit_function: CUDADispatcher) -> str:
    source_code = inspect.getsource(slit_function)
    # remove first two lines
    source_code = source_code.split("\n")[2:]
    # remove last line
    source_code = source_code[:-2]
    # remove indentation
    source_code = inspect.cleandoc("\n".join(source_code))
    return source_code


# all available cuda slits
available_slits = available_cuda_slits = [
    f
    for n, f in vars(pyechelle.slit).items()
    if isinstance(f, CUDADispatcher) or isinstance(f, CPUDispatcher)
]

# write kernels to file
path = Path(__file__).parent.parent.joinpath("pyechelle/_kernels.py")

file_loader = FileSystemLoader("pyechelle/kernel_templates")
env = Environment(loader=file_loader)

with open(path, "w") as f:
    import_template = env.get_template("import_template.jinja")
    f.write(import_template.render())
    f.write("\n\n")
    for slit in available_slits:
        slitname = slit.func_code.co_name
        for sourcetype in ["ListLike", "Continuous"]:
            for photonnoise in [True, False]:
                if "singlemode" in slitname:
                    transformation_code = env.get_template(
                        "singlemode_transform.jinja"
                    ).render(source_type=sourcetype, slit_name=slitname)
                else:
                    slit_code = extract_slit_code(slit)
                    transformation_code = env.get_template(
                        "transformation_template.jinja"
                    ).render(slit_code=slit_code, slit_name=slitname)
                kernel_code = env.get_template("kernel_template.jinja").render(
                    slit_name=slitname,
                    source_type=sourcetype,
                    photonnoise=photonnoise,
                    transformation_code=transformation_code,
                )

                f.write(kernel_code)
                f.write("\n\n")
