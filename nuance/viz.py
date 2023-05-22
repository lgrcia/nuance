import matplotlib.pyplot as plt


def plot_result(nu, search):
    plt.subplot(2, 2, (1, 3))
    plt.plot(search.periods, search.Q_snr)

    mean, astro, noise = nu.models(*search.best)

    plt.subplot(2, 2, 2)
    plt.plot(nu.time, nu.flux, ".", c="0.8")
    plt.plot(nu.time, astro + 1.0, c="k", label="found")
    _ = plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(nu.time, nu.flux - noise - mean, ".", c="0.8")
    plt.plot(nu.time, astro, c="k", label="found")
    _ = plt.legend()
