import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def interpret_memberships(universe, dict_mf, fuzzy_element):
    """
    Calculates the fuzzy membership values of fuzzy_element for the 
    memberships defined in dict_mf
    """
    calc_m = {}
    for label, membership_func in dict_mf.items():
        calc_m[label] = fuzz.interp_membership(
            universe, membership_func, fuzzy_element
        )
    return calc_m 

def urgency_rule_activation(first_antec, second_antec, conseq_mf):
    """
    Assembles the rule base and calculates the fuzzy membership values of the 
    antecedents for each urgency fuzzy set
    """
    active_conseq = {}
    rules = [-1]*17

    # First variable of the antecedents (sum of waiting cars)
    first_antec_values = ['zero', 'zero', 'zero', 'zero', 
                          'few', 'few', 'few', 'few', 
                          'medium', 'medium', 'medium', 'medium', 
                          'many', 'many', 'many', 'many']
    # Second variable of the antecedents (waiting time since the last green 
    # phase)
    second_antec_values = ['negligible', 'short', 'medium', 'long', 
                           'negligible', 'short', 'medium', 'long', 
                           'negligible', 'short', 'medium', 'long', 
                           'negligible', 'short', 'medium', 'long']
    # Variable for the consequents
    conseq_mf_values = ['zero', 'low', 'medium', 'high', 
                        'zero', 'low', 'medium', 'high', 
                        'low', 'medium', 'medium', 'high', 
                        'medium', 'high', 'high', 'high']
    # e.g.:
    # antecedent = (IF sum_of_waiting_cars = zero AND waiting_time = negligible)
    # consequent = (THEN urgency = zero)

    # Assembling the fuzzy rule base and calculating the fuzzy membership values 
    # for the given rules
    print("\nDEBUG\n")
    for i, (value1, value2, value3) in enumerate(
            zip(first_antec_values, second_antec_values, conseq_mf_values),
            start=1):
        rules[i] = np.fmin(np.fmin(
            first_antec[value1], second_antec[value2]), conseq_mf[value3])
        print(f"    {rules[i]}")

    # Calculating the fuzzy membership values for the output urgency fuzzy sets 
    # (the level of urgency)
    # The urgency is 'zero' if rule 1 OR rule 5 is activated
    active_conseq['zero'] = np.fmax(rules[1], rules[5])
    # The urgency is 'low' if rule 2 OR rule 6 OR rule 9 is activated, etc...
    active_conseq['low'] = np.fmax(np.fmax(rules[2], rules[6]), rules[9])
    active_conseq['medium'] = np.fmax(np.fmax(np.fmax(np.fmax(
        rules[3],rules[7]), rules[10]), rules[11]), rules[13])
    active_conseq['high'] = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
        rules[4], rules[8]), rules[12]), rules[14]), rules[15]), rules[16])
    
    return active_conseq

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
fig_m, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(10, 8))

# Colors for the membership functions
colors = {
    'zero': 'c',
    'negligible': 'c',
    'few': 'g',
    'short': 'g',
    'low': 'g',
    'medium': 'orange',
    'many': 'r',
    'long': 'r',
    'high': 'r',
}
axes = [ax0, ax1, ax2, ax3, ax4, ax5]
ranges = [sum_queue_range, waiting_time_range, urgency_range, lane_queue_range,
          lane_queue_range, extension_time_range]
membership_functions = [sum_queue_mf, waiting_time_mf, urgency_mf,
                        inner_lane_queue_mf, outer_lane_queue_mf,
                        extension_time_mf]
titles = ['INPUT: Várakozó autók összmennyiség',
          'INPUT: Várakozási idő',
          'OUTPUT: Várakozási időből adódó prioritásszint',
          'INPUT: Várakozó autók száma a belső sávban',
          'INPUT: Várakozó autók száma a külső sávban',
          'OUTPUT: Zöld lámpa idejéhez adott idő']
xticks = [[0, 1, 6, 12, 18, 20],
          np.arange(0, 151, 30),
          np.arange(0, 11, 2),
          np.arange(0, 11, 2),
          np.arange(0, 11, 2),
          [0, 1, 10, 20, 30, 40]]

# Visualize the membership functions
for mf, u_range, ax, title, xtick in zip(
        membership_functions, ranges, axes,titles, xticks):
    for key, value in mf.items():
        ax.plot(u_range, value, colors[key], linewidth=2, label=f'{key}')
    ax.set_xticks(xtick)
    ax.set_title(title)

# Place the legends to the right of the plots
for ax in axes:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# TEST VARS
q_cars_n, q_cars_e, q_cars_s, q_cars_w = 7, 19, 5, 11
w_n, w_e, w_s, w_w = 60, 0, 120, 30

# Calculate the fuzzy memberships of queue and waiting time for each direction
n_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_n)
e_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_e)
s_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_s)
w_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_w)
print(f"\nDEBUG\nn_sum_queue = {n_sum_queue}\ne_sum_queue = {e_sum_queue}\n"
      f"s_sum_queue = {s_sum_queue}\nw_sum_queue = {w_sum_queue}\n\n")
n_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_n)
e_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_e)
s_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_s)
w_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_w)
print(f"\nDEBUG\nn_wait_t={n_wait_t}\ne_wait_t={e_wait_t}\n"
      f"s_wait_t={s_wait_t}\nw_wait_t={w_wait_t}\n\n")

# Traffic urgency decision
north_urgency = urgency_rule_activation(n_sum_queue, n_wait_t, urgency_mf)
east_urgency = urgency_rule_activation(e_sum_queue, e_wait_t, urgency_mf)
south_urgency = urgency_rule_activation(s_sum_queue, s_wait_t, urgency_mf)
west_urgency = urgency_rule_activation(w_sum_queue, w_wait_t, urgency_mf)

print("\nDEBUG\n")
for key, value in north_urgency.items():
    print(f"{key} = {value}")

urgency0 = np.zeros_like(urgency_range)
f_u, (ax_n, ax_e, ax_s, ax_w) = plt.subplots(nrows=4, figsize=(10, 8))

urgencies = [north_urgency, east_urgency, south_urgency, west_urgency]
axes = [ax_n, ax_e, ax_s, ax_w]
titles = ["Észak - sürgősség", "Kelet - sürgősség", "Dél - sürgősség",
          "Nyugat - sürgősség"]

# Visualization before aggregation
for (urgency, ax, title) in zip(urgencies, axes, titles):
    for key, value_urgency in urgency.items():
        ax.fill_between(urgency_range, urgency0, value_urgency,
                        color=colors[key], alpha=0.7)
    # Draw the outlines of the membership functions
    for key, value in urgency_mf.items():    
        ax.plot(urgency_range, value, linewidth=1.5, color=colors[key],
                label=f'{key}')
    ax.set_title(title)
plt.tight_layout()

# Place the legends to the right of the plots
for ax in axes:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

print("\nDEBUG\n")
# Aggregate all four urgency output membership functions together
aggregated_urgencies = {
    'north': -1,
    'east': -1,
    'south': -1,
    'west': -1,
}
defuzz_results, defuzz_plt = dict(aggregated_urgencies), dict(aggregated_urgencies)

for urgency, key in zip(urgencies, aggregated_urgencies.keys()):
    u_zero, u_low, u_medium, u_high = urgency.values()
    # DEBUG
    # for asd in [u_zero, u_low, u_medium, u_high]:
    #     print(f"    urgency {asd}")
    # print('\n')
    aggregated_urgencies[key] = np.fmax(u_zero, np.fmax(
        u_low, np.fmax(u_medium, u_high)))
    
print(f"\nDEBUG\n    agg: {aggregated_urgencies}\n    defuzz:{defuzz_results}")

# Defuzzify aggregated outputs
for key, value in aggregated_urgencies.items():
    defuzz_results[key] = fuzz.defuzz(urgency_range, aggregated_urgencies[key], "centroid")
    # This is only necessary for the plot
    defuzz_plt[key] = fuzz.interp_membership(urgency_range, aggregated_urgencies[key], defuzz_results[key])
print(f"\n\n{defuzz_results}\n\n")
fig_d, (ax_n_d, ax_e_d, ax_s_d, ax_w_d) = plt.subplots(nrows=4, figsize=(10, 8))

axes_d = [ax_n_d, ax_e_d, ax_s_d, ax_w_d]

# Visualization of the defuzzified results
for (key, value), urgency, ax, title in zip(aggregated_urgencies.items(), urgencies, axes_d, titles):
    ax.fill_between(urgency_range, urgency0, value,
                    facecolor='orange', alpha=0.7)
    ax.plot([defuzz_results[key], defuzz_results[key]], [0, defuzz_plt[key]], 'k', linewidth=1.5, alpha=0.9)
    # Draw the outlines of the membership functions
    for key, value in urgency_mf.items():    
        ax.plot(urgency_range, value, linewidth=1.5, color=colors[key],
                label=f'{key}')
    ax.set_title(title)
plt.tight_layout()

plt.show()