import argparse
import itertools
import re
import sys
import textwrap
from pathlib import Path

import pyechelle
from pyechelle import spectrograph
from pyechelle.efficiency import GratingEfficiency

dir_path = Path(__file__).resolve().parent.parent.joinpath("models")
models = [x.stem for x in dir_path.glob('*.hdf')]


def parseNumList(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


def rel_to_full_path(filename):
    script_dir = Path(__file__).resolve().parent.parent.joinpath("models")
    return script_dir.joinpath(f"{filename}.hdf")


_HELP = "help"
_DESCRIPTION = "description"
_FORMAT_CLASS = "formatter_class"

_KEYWORDS_ARGS = ("Args:",)
_KEYWORDS_OTHERS = ("Returns:", "Raises:", "Yields:", "Usage:")
_KEYWORDS = _KEYWORDS_ARGS + _KEYWORDS_OTHERS


def _checker(keywords):
    """Generate a checker which tests a given value not starts with keywords."""
    def _(v):
        """Check a given value matches to keywords."""
        for k in keywords:
            if k in v:
                return False
        return True
    return _


def _parse_doc(doc):
    """Parse a docstring.
    Parse a docstring and extract three components; headline, description,
    and map of arguments to help texts.
    Args:
      doc: docstring.
    Returns:
      a dictionary.
    """
    lines = doc.split("\n")
    descriptions = list(itertools.takewhile(_checker(_KEYWORDS), lines))

    if len(descriptions) < 3:
        description = lines[0]
    else:
        description = "{0}\n\n{1}".format(
            lines[0], textwrap.dedent("\n".join(descriptions[2:])))

    args = list(itertools.takewhile(
        _checker(_KEYWORDS_OTHERS),
        itertools.dropwhile(_checker(_KEYWORDS_ARGS), lines)))
    argmap = {}
    if len(args) > 1:
        for pair in args[1:]:
            kv = [v.strip() for v in pair.split(":")]
            if len(kv) >= 2:
                argmap[kv[0]] = ":".join(kv[1:])

    return dict(headline=descriptions[0], description=description, args=argmap)


def main(args):
    # generate flat list for all fields to simulate
    if any(isinstance(el, list) for el in args.fiber):
        fibers = [item for sublist in args.fiber for item in sublist]
    else:
        fibers = args.fiber

    # generate flat list of all sources to simulate
    sources = args.sources
    if len(sources) == 1:
        sources = [sources[0]] * len(fibers)  # generate list of same length than 'fields' if only one source given

    assert len(fibers) == len(sources), 'Number of sources needs to match number of fields (or be 1).'
    for fiber, source_name in zip(fibers, sources):
        spec = spectrograph.ZEMAX(args.s, fiber, args.n_lookup)
        # ccd = read_ccd_from_hdf(args.s)
        # telescope = Telescope(args.d_primary, args.d_secondary)
        source = getattr(pyechelle.sources, source_name)
        if not args.no_blaze_efficiency:
            efficiency = GratingEfficiency(spec.blaze, spec.blaze, spec.gpmm)

        if args.orders is None:
            orders = spec.order_keys
        else:
            orders = [item for sublist in args.orders for item in sublist]
            # TODO: Check that order exists


if __name__ == "__main__":
    # clize.run(main)
    # sp = clize.run(spectrograph.ZEMAX, alt=Phoenix)
    # phoenix = clize.run(Phoenix)
    # print(sp)
    # print(sp.parameters())
    d = _parse_doc(spectrograph.ZEMAX.__init__.__doc__)
    print(d['args'])

    # clize.run(main, alt=spectrograph.Spectrograph)
    parser = argparse.ArgumentParser(description='PyEchelle Simulator')
    parser.add_argument('-s', nargs='?', type=rel_to_full_path, default=sys.stdin, required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. Options "
                             f"are {','.join(models)}")
    parser.add_argument('--fiber', type=parseNumList, default=1, required=True)
    parser.add_argument('--n_lookup', type=int, default=10000, required=False)

    parser.add_argument('--d_primary', type=float, required=False, default=1.0)
    parser.add_argument('--d_secondary', type=float, required=False, default=0)

    parser.add_argument('--orders', type=parseNumList, nargs='+', required=False,
                        help='Echelle order numbers to simulate... '
                             'if not specified, all orders of the spectrograph are simulated')
    parser.add_argument('--sources', nargs='+', choices=['Phoenix', 'Dark', 'Flat', 'Etalon'], required=True)
    parser.add_argument('--no_blaze_efficiency', default=True, action='store_false')
    # parser.add_argument('model', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
    #                     help="Filename of spectrograph model. Model file needs to be located in models/ folder.")

    args = parser.parse_args()
    main(args)
    # spectrograph.ZEMAX(args.model)
    #
    # # print(args.accumulate(args.integers))
    # print(args)
