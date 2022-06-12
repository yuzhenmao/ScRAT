import argparse
import argcomplete
import logging


def _loglevel(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    return numeric_level


def _str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


def parser():
    parser = argparse.ArgumentParser("cloudpred", description="Classification for point clouds")
    parser.add_argument("dir", type=str,
                        help="root directory of data")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="seed for RNG")
    parser.add_argument("-t", "--transform", default="log",
                        choices=["none", "log"],
                        help="preprocessing on data")
    parser.add_argument("--pc", type=_str2bool, default=True,
                        help="project onto principal components")
    parser.add_argument("-d", "--dims", type=int, default=10,
                        help="dimension of principal components")
    parser.add_argument("-l", "--loglevel", type=_loglevel, default=logging.DEBUG)
    parser.add_argument("--logfile", type=str,
                        default=None,
                        help="file to store logs")
    parser.add_argument("--cloudpred", action='store_true',
                        help="train with cloudpred classifier")
    parser.add_argument("--linear", action='store_true',
                        help="train with linear classifier")
    parser.add_argument("--generative", action='store_true',
                        help="train with generative classifier")
    parser.add_argument("--genpat", action='store_true',
                        help="train with generative classifier by patient")
    parser.add_argument("--deepset", action='store_true',
                        help="train with deepset classifier")
    parser.add_argument("--calibrate", action='store_true',
                        help="calibrate size before training density")
    parser.add_argument("-c", "--centers", type=int, default=[2], nargs="+",
                        help="number of centers")
    parser.add_argument("-f", "--figroot", type=str, default=None,
                        help="root for optional figures")
    parser.add_argument("--valid", type=int_or_float, default=0.25,
                        help="root for optional figures")
    parser.add_argument("--test", type=int_or_float, default=0.25,
                        help="root for optional figures")
    parser.add_argument("--regression", action='store_true',
                        help="train as a regression task instead of classification")
    parser.add_argument("--train_patients", type=int, default=None,
                        help="limit number of training patients")
    parser.add_argument("--cells", type=int, default=None,
                        help="limit number of cells")

    # TODO: specify lr

    return parser
