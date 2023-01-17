import matplotlib.pyplot as plt

def show(, t0s, Ds)

plt.figure(None, (6, 6))
plt.subplot(211)
plt.plot(time, diff_flux, ".", c="0.8")
plt.plot(time, mean + astro + noise, c="k")
plt.xlim(time.min(), time.max())
plt.text(0.02, 0.05, f"depth: {nu.depth(t0, D)[0]:2.0e}", transform=plt.gca().transAxes)
plt.subplot(212)
plt.plot(i, j, "x", c="k")
plt.imshow(ls.T, aspect='auto', origin="lower", extent=(t0s.min(), t0s.max(), Ds.min(), Ds.max()))