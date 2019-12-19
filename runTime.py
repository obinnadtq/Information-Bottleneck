import numpy as np
import matplotlib.pyplot as plt

exec_times = [35.888031005859375, 35.66608691215515, 35.69620895385742, 36.54158878326416, 35.61477184295654, 35.812420129776,
              35.53462815284729, 35.586766958236694, 35.57995104789734, 35.53414726257324]
plt.hist(exec_times)
plt.title('Histogram for Run times for Iterative Information Bottleneck Algorithm')
plt.xlabel('Run times')
plt.ylabel('Count')
plt.show()
print(np.mean(exec_times))
print(np.median(exec_times))
