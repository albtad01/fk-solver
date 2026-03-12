import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

processes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
times = np.array([2422.912, 1336.381, 751.122, 436.718, 237.379, 122.979, 86.459, 340.4])
memory = np.array([1386.785, 2250.618, 3594.989, 4458.447, 6304.562, 10738.0, 20697.175, 41313.585])

plt.figure(figsize=(8, 6))
plt.loglog(processes, times, marker='o', label='Numerical Time', linewidth=2)
plt.xlabel('Number of Processes')
plt.ylabel('Time (s)')
plt.title('Scalability Plot')

# Plot a first-order reference line
time_ref = times[0]
reference_line = time_ref * (processes / processes[0])**(-1)  # Negative slope for 1st order
plt.loglog(processes, reference_line, '--', label='First-Order Reference', linewidth=1.5)

plt.gca().xaxis.set_major_formatter(NullFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.gca().set_xticks(processes)
plt.gca().set_xticklabels([str(p) for p in processes])
plt.gca().set_yticks(np.array([1600, 800, 400, 200, 100, 50]))
plt.gca().set_yticklabels([str(t) for t in np.array([1600, 800, 400, 200, 100, 50])])

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save the plot as a file
plt.savefig('../media/scalability_plot.png', dpi=300)  # Save as a high-resolution PNG file

p = np.zeros(6)
# Calculate the rate between the points
for i in range(len(p)):
    p[i] = np.log(times[i+1] / times[i]) / np.log(processes[i] / processes[i+1])
print(f'Estimated order of scalability: {p}')

# Memory Usage Plot
plt.figure(figsize=(8, 6))
plt.loglog(processes, memory, marker='o', label='Memory Usage', linewidth=2)
plt.xlabel('Number of Processes')
plt.ylabel('Memory (MB)')
plt.title('Memory Usage Plot')

plt.gca().xaxis.set_major_formatter(NullFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.gca().set_xticks(processes)
plt.gca().set_xticklabels([str(p) for p in processes])
plt.gca().set_yticks(np.array([1000, 2000, 4000, 8000, 16000, 32000]))
plt.gca().set_yticklabels([str(t) for t in np.array([1000, 2000, 4000, 8000, 16000, 32000])])
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('../media/memory_plot.png', dpi=300)
