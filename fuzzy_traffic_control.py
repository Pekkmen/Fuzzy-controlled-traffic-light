import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Generate ranges for the membership functions
# Waiting cars
sum_queue_range = np.arange(0, 21, 1)
# Waiting time
waiting_time_range = np.arange(0, 151, 1)
# Urgency
urgency_range = np.arange(0, 11, 1)
# Inner lane queue
inner_lane_queue_range = np.arange(0, 11, 1)
# Outer lane queue
outer_lane_queue_range = np.arange(0, 10, 1)

# Generate fuzzy membership functions
sum_queue_mf = {
    "zero": fuzz.trapmf(sum_queue_range, [0, 0, 0, 1]),
    "low": fuzz.trimf(sum_queue_range, [0, 6, 12]),
    "medium": fuzz.trimf(sum_queue_range, [6, 12, 18]),
    "high": fuzz.trapmf(sum_queue_range, [12, 18, 20, 20]),
}

waiting_time_mf = {
    "zero": fuzz.trapmf(waiting_time_range, [0, 0, 30, 60]),
    "short": fuzz.trimf(waiting_time_range, [30, 60, 90]),
    "medium": fuzz.trimf(waiting_time_range, [60, 90, 120]),
    "long": fuzz.trapmf(waiting_time_range, [90, 120, 150, 150]),
}

urgency_mf = {
    "zero": fuzz.trapmf(urgency_range, [0, 0, 2, 4]),
    "low": fuzz.trimf(urgency_range, [2, 4, 6]),
    "medium": fuzz.trimf(urgency_range, [4, 6, 8]),
    "high": fuzz.trapmf(urgency_range, [6, 8, 10, 10]),
}

inner_lane_queue_mf = {
    "zero": fuzz.trapmf(inner_lane_queue_range, [0, 0, 2, 4]),
    "low": fuzz.trimf(inner_lane_queue_range, [2, 4, 6]),
    "medium": fuzz.trimf(inner_lane_queue_range, [4, 6, 8]),
    "high": fuzz.trapmf(inner_lane_queue_range, [6, 8, 10, 10]),
}

# Creating the plots
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)

ax0.set_xticks([0, 1, 6, 12, 18, 20])
for key, value in sum_queue_mf.items():
    ax0.plot(sum_queue_range, value, linewidth=2, label=f'{key}')
ax0.set_title('Várakozó autók összmennyiség')
ax0.legend()

ax1.set_xticks(np.arange(0, 151, 30))
for key, value in waiting_time_mf.items():
    ax1.plot(waiting_time_range, value, linewidth=2, label=f'{key}')
ax1.set_title('Várakozási idő')
ax1.legend()

ax2.set_xticks(np.arange(0, 11, 2))
for key, value in urgency_mf.items():
    ax2.plot(urgency_range, value, linewidth=2, label=f'{key}')
ax2.set_title('Várakozási időből adódó prioritásszint')
ax2.legend()

ax3.set_xticks(np.arange(0, 11, 2))
for key, value in inner_lane_queue_mf.items():
    ax3.plot(inner_lane_queue_range, value, linewidth=2, label=f'{key}')
ax3.set_title('Várakozó autók száma a belső sávban')
ax3.legend()

plt.tight_layout()
plt.show()
