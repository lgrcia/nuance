import pytest


@pytest.mark.parametrize("cpu_count", [None, 1, 2, 4])
def test_cpu(cpu_count):
    pytest.skip("Figure out later")
    import os

    import jax

    if cpu_count is None:
        cpu_count = os.cpu_count()

    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count}"

    import nuance

    assert jax.device_count() == cpu_count
