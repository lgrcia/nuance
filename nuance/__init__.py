import jax
from jax.config import config

DEVICES_COUNT = jax.device_count()
config.update("jax_enable_x64", True)

from nuance.combined import CombinedNuance
from nuance.nuance import Nuance
from nuance.search_data import SearchData
