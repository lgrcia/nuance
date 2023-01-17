from jax.config import config
config.update("jax_enable_x64", True)

from .nuance import Nuance
from .search_data import SearchData