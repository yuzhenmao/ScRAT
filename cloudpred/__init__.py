# WARNING: parser must be called very early on for argcomplete.
# argcomplete evaluates the package until the parser is constructed before it
# can generate completions. Because of this, the parser must be constructed
# before the full package is imported to behave in a usable way. Note that
# running
# > python -m cloudpred
# will actually import the entire package (along with dependencies like
# pytorch, numpy, and pandas), before running __main__.py, which takes
# about 0.5-1 seconds.
# See Performance section of https://argcomplete.readthedocs.io/en/latest/

from .parser import parser
parser()

import scipy.sparse as _

from cloudpred.__version__ import __version__
from cloudpred.main import main

import cloudpred.utils as utils
import cloudpred.generative as generative
import cloudpred.genpat as genpat
import cloudpred.cloudpred as cloudpred
import cloudpred.deepset as deepset
