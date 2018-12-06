import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

# Fixing random state for reproducibility
# np.random.seed(19680801)

dt = 0.01
# t = np.arange(0, 10, dt)

t = np.arange(1,6)


r1 = [0.9512, 0.9588, 0.9661, 0.9726, 0.9783]
r2 = [0.9547, 0.9563, 0.9515, 0.9507, 0.9476]

s1 = [0.9513, 0.9593, 0.9671, 0.9737, 0.9789]
s2 = [0.9546, 0.9552, 0.9529, 0.9504, 0.9485]
# nse = np.random.randn(len(t))
# r = np.exp(-t / 0.05)

# cnse = np.convolve(nse, r) * dt
# cnse = cnse[:len(t)]
# s = 0.1 * np.sin(2 * np.pi * t) + cnse

# fig, (ax0, ax1) = plt.subplots(nrows=2, constrained_layout=True)

# # plt.subplot(211)
# # plt.plot(t, r1, t, r2)
# # plt.set_title("")
# ax0.psd(xn, NFFT=301, Fs=fs, window=mlab.window_none, pad_to=1024,
#         scale_by_freq=True)
# ax0.set_title('LSTM attention')
# ax0.set_yticks(r2)
# ax0.set_yticks(r1)
# ax0.set_xticks(t)
# ax0.grid(True)
# ax0.set_ylim(yrange)



# plt.subplot(212)
# plt.plot(t, s1, t, s2)


# plt.show()


# plt.rcParams['savefig.facecolor'] = "0.8"

def example_plot(ax,  data1, data2, t,title , fontsize=12):
	ax.plot(t, data1, t, data2)
	# ax.locator_params(nbins=3)
	ax.set_xlabel( "Epoch", fontsize=fontsize)
	ax.set_ylabel( "Accuracy", fontsize=fontsize)
	ax.legend(['train acc', 'test acc'])
	ax.set_title(title, fontsize=fontsize)

plt.close('all')
fig, (ax1, ax2) = plt.subplots(nrows=2)
example_plot(ax1, r1, r2, t, "LSTM", fontsize=12)
example_plot(ax2, s1, s2, t, "LSTM self attentive structure" ,fontsize=12)
plt.tight_layout()

plt.show()
