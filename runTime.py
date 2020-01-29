import numpy as np
import matplotlib.pyplot as plt

exec_times = [0.013311982154846191, 0.02320575714111328, 0.10530996322631836, 0.47330546379089355]
num_of_clusters = [8, 16, 32, 64]
plt.plot(num_of_clusters, exec_times, 'r', linewidth=2)
plt.title('Plot of Medians of Execution Times vs Number of output clusters')
plt.grid()
plt.xlabel('Number of Output Clusters')
plt.ylabel('Median of Execution Time')
plt.show()


