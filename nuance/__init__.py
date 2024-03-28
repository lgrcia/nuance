import jax

DEVICES_COUNT = jax.device_count()

config = jax.config
config.update("jax_enable_x64", True)

from nuance.combined import CombinedNuance
from nuance.nuance import Nuance
from nuance.search_data import SearchData
from nuance.star import Star
