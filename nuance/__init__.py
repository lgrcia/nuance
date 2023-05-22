import os
from multiprocessing import cpu_count

CPU_counts = cpu_count()
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CPU_counts}"
from jax.config import config

config.update("jax_enable_x64", True)

from nuance.nuance import Nuance
from nuance.search_data import SearchData
