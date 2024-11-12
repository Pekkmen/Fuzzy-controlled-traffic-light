import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Generate ranges for the membership functions
# Sum of cars waiting in a given direction 
sum_queue_range = np.arange(0, 21, 1)
# Time elapsed since the last green phase in a given direction
waiting_time_range = np.arange(0, 151, 1)
# Output for the inputs (sum_queue, waiting_time): the more cars wait in one
# direction, the more urgent they become.
urgency_range = np.arange(0, 11, 1)
# Number of cars waiting in the inner lane (the lane that turns left) in a
# given direction OR in the outer lane (the lane that goes straight or turns
# right) in the same given direction
lane_queue_range = np.arange(0, 11, 1)
# Output for the inputs (inner_lane_queue, outer_lane_queue): 
extension_time_range = np.arange(0, 41, 1)

# Generate fuzzy membership functions
sum_queue_mf = {
    "negligible": fuzz.trapmf(sum_queue_range, [0, 0, 0, 1]),
    "few": fuzz.trimf(sum_queue_range, [0, 6, 12]),
    "medium": fuzz.trimf(sum_queue_range, [6, 12, 18]),
    "many": fuzz.trapmf(sum_queue_range, [12, 18, 20, 20]),
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
    "negligible": fuzz.trapmf(lane_queue_range, [0, 0, 2, 4]),
    "few": fuzz.trimf(lane_queue_range, [2, 4, 6]),
    "medium": fuzz.trimf(lane_queue_range, [4, 6, 8]),
    "many": fuzz.trapmf(lane_queue_range, [6, 8, 10, 10]),
}

# Might need to modify one single lane later, so keep the boilerplate code
outer_lane_queue_mf = {
    "negligible": fuzz.trapmf(lane_queue_range, [0, 0, 2, 4]),
    "few": fuzz.trimf(lane_queue_range, [2, 4, 6]),
    "medium": fuzz.trimf(lane_queue_range, [4, 6, 8]),
    "many": fuzz.trapmf(lane_queue_range, [6, 8, 10, 10]),
}

extension_time_mf = {
    "zero": fuzz.trapmf(extension_time_range, [0, 0, 0, 1]),
    "short": fuzz.trimf(extension_time_range, [0, 10, 20]),
    "medium": fuzz.trimf(extension_time_range, [10, 20, 30]),
    "long": fuzz.trapmf(extension_time_range, [20, 30, 40, 40]),
}

# Create plots for the membership functions
fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(10, 8))
# Orders of the colors for the membership functions
colors = ('c', 'g', 'orange', 'r')

ax0.set_xticks([0, 1, 6, 12, 18, 20])
for (key, value), color in zip(sum_queue_mf.items(), colors):
    ax0.plot(sum_queue_range, value, color, linewidth=2, label=f'{key}')
ax0.set_title('INPUT: Várakozó autók összmennyiség')

ax1.set_xticks(np.arange(0, 151, 30))
for (key, value), color in zip(waiting_time_mf.items(), colors):
    ax1.plot(waiting_time_range, value, color, linewidth=2, label=f'{key}')
ax1.set_title('INPUT: Várakozási idő')

ax2.set_xticks(np.arange(0, 11, 2))
for (key, value), color in zip(urgency_mf.items(), colors):
    ax2.plot(urgency_range, value, color, linewidth=2, label=f'{key}')
ax2.set_title('OUTPUT: Várakozási időből adódó prioritásszint')

ax3.set_xticks(np.arange(0, 11, 2))
for (key, value), color in zip(inner_lane_queue_mf.items(), colors):
    ax3.plot(lane_queue_range, value, color, linewidth=2, label=f'{key}')
ax3.set_title('INPUT: Várakozó autók száma a belső sávban')

ax4.set_xticks(np.arange(0, 11, 2))                                             
for (key, value), color in zip(outer_lane_queue_mf.items(), colors):            
    ax4.plot(lane_queue_range, value, color, linewidth=2, label=f'{key}')       
ax4.set_title('INPUT: Várakozó autók száma a külső sávban') 

ax5.set_xticks([0, 1, 10, 20, 30, 40])
for (key, value), color in zip(extension_time_mf.items(), colors):            
    ax5.plot(extension_time_range, value, color, linewidth=2, label=f'{key}')       
ax5.set_title('OUTPUT: Zöld lámpa idejéhez adott idő')

# Place the legends to the right of the plots
for ax in (ax0, ax1, ax2, ax3, ax4):
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
