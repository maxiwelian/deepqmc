from collections import namedtuple

from .base import WaveFunction
from .paulinet import PauliNet
from .ferminet import FermiNet, Pretrainer

__all__ = ['WaveFunction', 'PauliNet', 'FermiNet', 'Pretrainer']

AnsatzSpec = namedtuple('AnsatzSpec', 'name entry defaults uses_workdir')

ANSATZES = [AnsatzSpec('paulinet', PauliNet.from_hf, PauliNet.DEFAULTS(), True),
            AnsatzSpec('ferminet', FermiNet, {}, True)]
ANSATZES = {spec.name: spec for spec in ANSATZES}
