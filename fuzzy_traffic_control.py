import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def interpret_memberships(universe, dict_mf, value):
    calc_mfs = {}
    for label, membership_func in dict_mf.items():
        calc_mfs[label] = fuzz.interp_membership(
            universe, membership_func, value
        )
    return calc_mfs  

def urgency_rule_activation(first_antec, second_antec, consec_mf):
    active_consec = {}
    rules = [-1]*17

    # active_consec['zero'] = np.fmin(np.fmin(first_antec['zero'], second_antec['negligible']), consec_mf['zero'])
    rules[1] = np.fmin(np.fmin(first_antec['zero'], second_antec['negligible']), consec_mf['zero'])
    rules[2] = np.fmin(np.fmin(first_antec['zero'], second_antec['short']), consec_mf['low'])
    rules[3] = np.fmin(np.fmin(first_antec['zero'], second_antec['medium']), consec_mf['medium'])
    rules[4] = np.fmin(np.fmin(first_antec['zero'], second_antec['long']), consec_mf['high'])
    rules[5] = np.fmin(np.fmin(first_antec['few'], second_antec['negligible']), consec_mf['zero'])
    rules[6] = np.fmin(np.fmin(first_antec['few'], second_antec['short']), consec_mf['low'])
    rules[7] = np.fmin(np.fmin(first_antec['few'], second_antec['medium']), consec_mf['medium'])
    rules[8] = np.fmin(np.fmin(first_antec['few'], second_antec['long']), consec_mf['high'])
    rules[9] = np.fmin(np.fmin(first_antec['medium'], second_antec['negligible']), consec_mf['low'])
    rules[10] = np.fmin(np.fmin(first_antec['medium'], second_antec['short']), consec_mf['medium'])
    rules[11] = np.fmin(np.fmin(first_antec['medium'], second_antec['medium']), consec_mf['medium'])
    rules[12] = np.fmin(np.fmin(first_antec['medium'], second_antec['long']), consec_mf['high'])
    rules[13] = np.fmin(np.fmin(first_antec['many'], second_antec['negligible']), consec_mf['medium'])
    rules[14] = np.fmin(np.fmin(first_antec['many'], second_antec['short']), consec_mf['high'])
    rules[15] = np.fmin(np.fmin(first_antec['many'], second_antec['medium']), consec_mf['high'])
    rules[16] = np.fmin(np.fmin(first_antec['many'], second_antec['long']), consec_mf['high'])
    
    active_consec['zero'] = np.fmax(rules[1], rules[5])
    active_consec['low'] = np.fmax(np.fmax(rules[2], rules[6]), rules[9])
    active_consec['medium'] = np.fmax(np.fmax(np.fmax(np.fmax(rules[3], rules[7]), rules[10]), rules[11]), rules[13])
    active_consec['high'] = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(rules[4], rules[8]), rules[12]), rules[14]), rules[15]), rules[16])

    # active_consec['low'] = np.fmin(np.fmin(first_antec['medium'], second_antec['negligible']), consec_mf['low'])
    
    return active_consec

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
    "zero": fuzz.trapmf(sum_queue_range, [0, 0, 0, 1]),
    "few": fuzz.trimf(sum_queue_range, [0, 6, 12]),
    "medium": fuzz.trimf(sum_queue_range, [6, 12, 18]),
    "many": fuzz.trapmf(sum_queue_range, [12, 18, 20, 20]),
}

waiting_time_mf = {
    "negligible": fuzz.trapmf(waiting_time_range, [0, 0, 30, 60]),
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
for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# TEST VARS
q_cars_n, q_cars_e, q_cars_s, q_cars_w = 9, 3, 11, 12
w_n, w_e, w_s, w_w = 30, 0, 90, 60

# Fuzzy membership function activation for the given values
n_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_n)
e_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_e)
s_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_s)
w_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_w)
print(f"n_sum_queue = {n_sum_queue}\ne_sum_queue = {e_sum_queue}\ns_sum_queue = {s_sum_queue}\nw_sum_queue = {w_sum_queue}\n\n")
n_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_n)
e_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_e)
s_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_s)
w_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_w)
print(f"n_wait_t={n_wait_t}\ne_wait_t={e_wait_t}\ns_wait_t={s_wait_t}\nw_wait_t={w_wait_t}\n\n")

# FUZZY RULE SETS
# Traffic urgency decision
north_urgency = urgency_rule_activation(w_sum_queue, w_wait_t, urgency_mf)

for key, value in north_urgency.items():
    print(f"{key} = {value}")

urgency0 = np.zeros_like(urgency_range)
fig, ax0 = plt.subplots()
# Visualization before aggregation
for value_urgency in north_urgency.values():
    for key, value in urgency_mf.items():
        ax0.fill_between(urgency_range, urgency0, value_urgency, alpha=0.7)
        ax0.plot(urgency_range, value, linewidth=0.5, linestyle='--', )
ax0.set_title('Output membership activity')

plt.show()