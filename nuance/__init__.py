import jax

DEVICES_COUNT = jax.device_count()

config = jax.config
config.update("jax_enable_x64", True)

from nuance.linear_search import linear_search
from nuance.periodic_search import periodic_search
from nuance.star import Star
