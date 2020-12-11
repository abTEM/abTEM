from abtem.measure import Measurement
import matplotlib.pyplot as plt

measurement = Measurement.read('STEM_MoS2.hdf5')

measurement.tile((5, 3)).interpolate(.01).show()
plt.show()