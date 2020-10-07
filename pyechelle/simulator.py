import argparse
import itertools
import sys
import textwrap
from pathlib import Path

from pyechelle import spectrograph
from pyechelle.telescope import Telescope

dir_path = Path(__file__).resolve().parent.parent.joinpath("models")
models = [x.stem for x in dir_path.glob('*.hdf')]


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
    spec = spectrograph.ZEMAX(args.model, args.fiber, args.n_lookup)

    telescope = Telescope(args.d_primary, args.d_secondary)


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
    parser.add_argument('model', nargs='?', type=rel_to_full_path, default=sys.stdin,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. Options "
                             f"are {','.join(models)}")
    parser.add_argument('n_lookup', type=int)

    # parser.add_argument('model', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
    #                     help="Filename of spectrograph model. Model file needs to be located in models/ folder.")

    args = parser.parse_args()
    # spectrograph.ZEMAX(args.model)
    #
    # # print(args.accumulate(args.integers))
    # print(args)
