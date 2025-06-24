import jax.numpy as jnp


def phase_coverage(time, gap=0.1):
    """Returns the coverage of given phases, in number of times observed

    Parameters
    ----------
    times : list of arrays
        an array of observed times
    gap : float, optional
        minimum gap between observations to be considered an independent
        segment, by default 0.1

    Returns
    -------
    function
        a function that computes the phases coverage for a given period,
        i.e. with signature :code:`fun(float) -> array`
    """
    # we pre-compute segments_times: the pairs of (time_min, time_max) of each segment
    diff_time = jnp.diff(time)
    segment_idxs = jnp.flatnonzero(diff_time > gap)
    segment_idxs = jnp.sort(
        jnp.hstack([0, *segment_idxs, *segment_idxs + 1, len(time) - 1])
    )
    cuts_time = time[segment_idxs]
    segments_times = cuts_time.reshape(-1, 2)

    def fun(period, phases):
        """Returns the coverage of given phases, in number of times observed

        Parameters:
        ----------
        period: float or array
            the period  to compute the coverage for

        Returns
        -------
        array
            phases coverage for the given period
        """

        sampled = phases.copy()
        complete = jnp.ones_like(phases)

        raw_segments_phases = ((segments_times + 0.5 * period) % period) / period

        # segments_phases is unordered, some of them are
        # (0.6, 0.5) which corresponds to a segment that wraps around
        # the phase 1.0. We need to fix this and split it in (0.0, 0.5) and (0.6, 1.0)
        # we allocate segments_phases_2 for the extra split, as JAX requires fixed size arrays
        #
        # cases:
        # - |   0-----1 |
        #
        # - |        0--+
        #   +-----1     |
        #
        # - |--0
        #    +----------+
        #           1---|
        #
        # - |        0--+
        #   +-----------+
        #   +---1       |
        #
        # | : bounds of full phase segment
        # 0, 1 : start, end of the actual segment
        # + : wrap around the phase 1.0
        #
        # is_positive : 0 < 1
        # is_full : +-----------+

        n = raw_segments_phases.shape[0]

        full = jnp.floor((segments_times[:, 1] - segments_times[:, 0]) / period)
        is_positive = jnp.array(raw_segments_phases[:, 1] >= raw_segments_phases[:, 0])
        is_full = full > 0

        condition = jnp.logical_and(is_positive, jnp.logical_not(is_full))

        segments_phases_1 = jnp.where(
            condition,
            raw_segments_phases.T,
            jnp.vstack([jnp.zeros(n), raw_segments_phases[:, 1]]),
        ).T

        segments_phases_2 = jnp.where(
            condition,
            jnp.zeros_like(raw_segments_phases).T,
            jnp.vstack([raw_segments_phases[:, 0], jnp.ones(n)]),
        ).T

        # we now have clean segments from which to compute the overlap
        # on the grid of sampled phases
        clean_segments_phases = jnp.vstack([segments_phases_1, segments_phases_2])

        overlap = jnp.array(
            (sampled[:, None] >= clean_segments_phases[:, 0])
            & (sampled[:, None] <= clean_segments_phases[:, 1])
        ).astype(float)

        return jnp.sum(overlap, 1) + complete * jnp.sum(full)

    return fun
