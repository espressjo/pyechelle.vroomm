import argparse
import sys
from pyechelle import spectrograph
from pyechelle.sources import Phoenix
import docopt
from pathlib import Path

dir_path = Path(__file__).resolve().parent.parent.joinpath("models")
models = [x.stem for x in dir_path.glob('*.hdf')]

def rel_to_full_path(filename):
    script_dir = Path(__file__).resolve().parent.parent.joinpath("models")
    return script_dir.joinpath(f"{filename}.hdf")

def main():
    pass

if __name__ == "__main__":
    # clize.run(main)
    # sp = clize.run(spectrograph.ZEMAX, alt=Phoenix)
    # phoenix = clize.run(Phoenix)
    # print(sp)
    # print(sp.parameters())
    # clize.run(main, alt=spectrograph.Spectrograph)
    parser = argparse.ArgumentParser(description='PyEchelle Simulator')
    parser.add_argument('model', nargs='?', type=rel_to_full_path, default=sys.stdin,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. Options "
                             f"are {','.join(models)}")
    parser.add_argument('n')

    # parser.add_argument('model', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
    #                     help="Filename of spectrograph model. Model file needs to be located in models/ folder.")

    args = parser.parse_args()
    spectrograph.ZEMAX(args.model)
    #
    # # print(args.accumulate(args.integers))
    # print(args)
