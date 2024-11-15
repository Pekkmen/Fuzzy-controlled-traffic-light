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

def rule_activation(first_antec, second_antec, conseq_mf, case):
    """
    Assembles the rule base and calculates the fuzzy membership values of the 
    antecedents for each urgency fuzzy set
    """
    active_conseq = {}
    rules = [-1]*17

    if case == 'urgency':
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
        # antecedent := (IF sum_of_waiting_cars = zero AND waiting_time = 
        # negligible)
        # consequent := (THEN urgency = zero)

        # Assembling the fuzzy rule base and calculating the fuzzy membership 
        # values for the given rules

        for i, (value1, value2, value3) in enumerate(
                zip(first_antec_values, second_antec_values, conseq_mf_values),
                start=1):
            rules[i] = np.fmin(np.fmin(
                first_antec[value1], second_antec[value2]), conseq_mf[value3])

        # Calculating the fuzzy membership values for the output urgency fuzzy 
        # sets (the level of urgency)
        # The urgency is 'zero' if rule 1 OR rule 5 is activated
        active_conseq['zero'] = np.fmax(rules[1], rules[5])
        # The urgency is 'low' if rule 2 OR rule 6 OR rule 9 is activated, etc.
        active_conseq['low'] = np.fmax(np.fmax(rules[2], rules[6]), rules[9])
        active_conseq['medium'] = np.fmax(np.fmax(np.fmax(np.fmax(
            rules[3],rules[7]), rules[10]), rules[11]), rules[13])
        active_conseq['high'] = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
            rules[4], rules[8]), rules[12]), rules[14]), rules[15]), rules[16])
    
    elif case == 'extension':
        # First variable of the antecedents (sum of waiting cars)
        first_antec_values = ['negligible', 'negligible', 'negligible', 
                              'negligible', 'few', 'few', 'few', 'few', 
                            'medium', 'medium', 'medium', 'medium', 
                            'many', 'many', 'many', 'many']
        # Second variable of the antecedents (waiting time since the last green 
        # phase)
        second_antec_values = ['negligible', 'few', 'medium', 'many', 
                            'negligible', 'few', 'medium', 'many', 
                            'negligible', 'few', 'medium', 'many', 
                            'negligible', 'few', 'medium', 'many']
        # Variable for the consequents
        conseq_mf_values = ['zero', 'short', 'medium', 'long', 
                            'short', 'short', 'medium', 'long', 
                            'medium', 'medium', 'medium', 'long', 
                            'long', 'long', 'long', 'long']

        # e.g.:
        # antecedent = (IF sum_of_waiting_cars = zero AND 
        # waiting_time = negligible) consequent = (THEN urgency = zero)

        # Assembling the fuzzy rule base and calculating the fuzzy membership 
        # values for the given rules
        for i, (value1, value2, value3) in enumerate(
                zip(first_antec_values, second_antec_values, conseq_mf_values),
                start=1):
            rules[i] = np.fmin(np.fmin(
                first_antec[value1], second_antec[value2]), conseq_mf[value3])

        # Calculating the fuzzy membership values for the output extension time 
        # fuzzy sets (the length of green phase extension)
        # The extension is 'zero' if rule 1 is activated
        active_conseq['zero'] = rules[1]
        # The urgency is 'short' if rule 2 OR rule 5 OR rule 6 is activated, etc
        active_conseq['short'] = np.fmax(np.fmax(rules[2], rules[5]), rules[6])
        active_conseq['medium'] = np.fmax(np.fmax(np.fmax(np.fmax(
            rules[3],rules[7]), rules[9]), rules[10]), rules[11])
        active_conseq['long'] = np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(np.fmax(
            rules[4], rules[8]), rules[12]), rules[13]), rules[14]), rules[15]),
            rules[16])

    else:
        print(f"ERROR: BAD CASE GIVEN!")
        exit(1)
    
    return active_conseq

def extension_rule_activation(inner_queue, outer_queue, extension_time_mf):
    pass

# Generate ranges for the membership functions
# Sum of cars waiting in a given direction 
sum_queue_range = np.arange(0, 21, 0.01)
# Time elapsed since the last green phase in a given direction
waiting_time_range = np.arange(0, 151, 0.01)
# Output for the inputs (sum_queue, waiting_time): the more cars wait in one
# direction, the more urgent they become.
urgency_range = np.arange(0, 11, 0.01)
# Number of cars waiting in the inner lane (the lane that turns left) in a
# given direction OR in the outer lane (the lane that goes straight or turns
# right) in the same given direction
lane_queue_range = np.arange(0, 11, 0.01)
# Output for the inputs (inner_lane_queue, outer_lane_queue): 
extension_time_range = np.arange(0, 41, 0.01)

# Generate fuzzy membership functions
sum_queue_mf = {
    "zero": fuzz.trapmf(sum_queue_range, [0, 0, 0, 1]),
    "few": fuzz.trimf(sum_queue_range, [0, 6, 12]),
    "medium": fuzz.trimf(sum_queue_range, [6, 12, 18]),
    "many": fuzz.trapmf(sum_queue_range, [12, 18, 20, 21]),
}

waiting_time_mf = {
    "negligible": fuzz.trapmf(waiting_time_range, [0, 0, 30, 60]),
    "short": fuzz.trimf(waiting_time_range, [30, 60, 90]),
    "medium": fuzz.trimf(waiting_time_range, [60, 90, 120]),
    "long": fuzz.trapmf(waiting_time_range, [90, 120, 150, 151]),
}

urgency_mf = {
    "zero": fuzz.trapmf(urgency_range, [0, 0, 2, 4]),
    "low": fuzz.trimf(urgency_range, [2, 4, 6]),
    "medium": fuzz.trimf(urgency_range, [4, 6, 8]),
    "high": fuzz.trapmf(urgency_range, [6, 8, 10, 11]),
}

inner_lane_queue_mf = {
    "negligible": fuzz.trapmf(lane_queue_range, [0, 0, 2, 4]),
    "few": fuzz.trimf(lane_queue_range, [2, 4, 6]),
    "medium": fuzz.trimf(lane_queue_range, [4, 6, 8]),
    "many": fuzz.trapmf(lane_queue_range, [6, 8, 10, 11]),
}

# Might need to modify one single lane later, so keep the boilerplate code
outer_lane_queue_mf = {
    "negligible": fuzz.trapmf(lane_queue_range, [0, 0, 2, 4]),
    "few": fuzz.trimf(lane_queue_range, [2, 4, 6]),
    "medium": fuzz.trimf(lane_queue_range, [4, 6, 8]),
    "many": fuzz.trapmf(lane_queue_range, [6, 8, 10, 11]),
}

extension_time_mf = {
    "zero": fuzz.trapmf(extension_time_range, [0, 0, 0, 1]),
    "short": fuzz.trimf(extension_time_range, [0, 10, 20]),
    "medium": fuzz.trimf(extension_time_range, [10, 20, 30]),
    "long": fuzz.trapmf(extension_time_range, [20, 30, 40, 41]),
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
titles = ['INPUT: Várakozó autók összmennyisége',
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
    # Cut off the redundant end of the plot
    ax.set_xlim([0, u_range[-1]-0.99])

# Place the legends to the right of the plots
for ax in axes:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# TEST VARS
# Waiting cars in specficic lanes in each directions
n_lane_queues = {
    'inner': 0,
    'outer': 9,
}
e_lane_queues = {
    'inner': 9,
    'outer': 4,
}
s_lane_queues = {
    'inner': 3,
    'outer': 2,
}
w_lane_queues = {
    'inner': 7,
    'outer': 2,
}
# q_outer_n, q_outer_e, q_outer_s, q_outer_w = 5, 10, 0, 2
# q_inner_n, q_inner_e, q_inner_s, q_inner_w = 2, 8, 5, 9
lane_queues = {
    'north': n_lane_queues,
    'east': e_lane_queues,
    'south': s_lane_queues,
    'west': w_lane_queues,
}

# Sum of waiting cars and their waiting time in each direction
q_cars_n = sum(n_lane_queues.values())
q_cars_e = sum(e_lane_queues.values())
q_cars_s = sum(s_lane_queues.values())
q_cars_w = sum(w_lane_queues.values())
w_n, w_e, w_s, w_w = 30, 60, 0, 120

# THIS PART IS FOR TESTS ONLY
# Calculate fuzzy memberships of a queue for each lane in a given direction
# n_outer_queue = interpret_memberships(lane_queue_range, outer_lane_queue_mf,
#                                       q_outer_n)
# e_outer_queue = interpret_memberships(lane_queue_range, outer_lane_queue_mf,
#                                       q_outer_e)
# s_outer_queue = interpret_memberships(lane_queue_range, outer_lane_queue_mf,
#                                       q_outer_s)
# w_outer_queue = interpret_memberships(lane_queue_range, outer_lane_queue_mf,
#                                       q_outer_w)
# print(f"\nLANE DEBUG\n")
# print(f"n_outer_queue = {n_outer_queue}\ne_outer_queue = {e_outer_queue}\n"
#       f"s_outer_queue = {s_outer_queue}\nw_outer_queue = {w_outer_queue}\n")
# n_inner_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
#                                       q_inner_n)
# e_inner_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
#                                       q_inner_e)
# s_inner_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
#                                       q_inner_s)
# w_inner_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
#                                       q_inner_w)
# print(f"n_inner_queue = {n_inner_queue}\ne_inner_queue = {e_inner_queue}\n"
#       f"s_inner_queue = {s_inner_queue}\nw_inner_queue = {w_inner_queue}\n")

# Calculate the fuzzy memberships of queue and waiting time for each direction
n_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_n)
e_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_e)
s_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_s)
w_sum_queue = interpret_memberships(sum_queue_range, sum_queue_mf, q_cars_w)

n_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_n)
e_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_e)
s_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_s)
w_wait_t = interpret_memberships(waiting_time_range, waiting_time_mf, w_w)

# Traffic urgency fuzzy membership values
north_urgency = rule_activation(n_sum_queue, n_wait_t, urgency_mf, 'urgency')
east_urgency = rule_activation(e_sum_queue, e_wait_t, urgency_mf, 'urgency')
south_urgency = rule_activation(s_sum_queue, s_wait_t, urgency_mf, 'urgency')
west_urgency = rule_activation(w_sum_queue, w_wait_t, urgency_mf, 'urgency')

# print("\nDEBUG\n")
# for key, value in north_urgency.items():
#     print(f"{key} = {value}")

# Lower boundary
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
        # Cut off the redundant end of the plot
        ax.set_xlim([0, urgency_range[-1]-0.99])
    ax.set_title(title)

# Place the legends to the right of the plots
for ax in axes:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Aggregate all four urgency output membership functions together
directions = ['north', 'east', 'south', 'west']
aggregated_urgencies, defuzz_results, defuzz_plt = {}, {}, {}

for urgency, key in zip(urgencies, directions):
    u_zero, u_low, u_medium, u_high = urgency.values()
    # The aggregated result will be the union of all output membership functions
    # union = max
    aggregated_urgencies[key] = np.fmax(u_zero, np.fmax(
        u_low, np.fmax(u_medium, u_high)))

# Defuzzify aggregated outputs
for key, value in aggregated_urgencies.items():
    defuzz_results[key] = fuzz.defuzz(urgency_range, aggregated_urgencies[key],
                                      "centroid")
    # This is only necessary for the plot
    defuzz_plt[key] = fuzz.interp_membership(
        urgency_range,aggregated_urgencies[key], defuzz_results[key])
# print(f"\n\n    defuzz: {defuzz_results}\n\n")
fig_d, (ax_n_d, ax_e_d, ax_s_d, ax_w_d) = plt.subplots(nrows=4, figsize=(10, 8))

axes_d = [ax_n_d, ax_e_d, ax_s_d, ax_w_d]
titles = ["Észak - sürgősség (defuzzfikált)",
          "Kelet - sürgősség (defuzzfikált)",
          "Dél - sürgősség (defuzzfikált)",
          "Nyugat - sürgősség (defuzzfikált)"]

# Visualization of the defuzzified results
for (key, value), urgency, ax, title in zip(aggregated_urgencies.items(), 
                                            urgencies, axes_d, titles):
    # Draw the filled aggregated output
    ax.fill_between(urgency_range, urgency0, value,
                    facecolor='peachpuff', alpha=0.7)
    ax.plot(urgency_range, value, linewidth=1.5, color='k')
    # Draw a vertical line to mark the crisp output
    ax.plot([defuzz_results[key], defuzz_results[key]], [0, defuzz_plt[key]],
            'k', linewidth=1.5, alpha=0.9)
            
    # Draw the outlines of the membership functions
    for key, value in urgency_mf.items():    
        ax.plot(urgency_range, value, linewidth=1.5, color=colors[key],
                label=f'{key}', linestyle='dashed', alpha=0.5)
        # Cut off the redundant end of the plot
        ax.set_xlim([0, urgency_range[-1]-0.99])
    ax.set_title(title)
    
# Place the legends to the right of the plots
for ax in axes:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Need to also receive an index to print directions in Hungarian
max_key, max_index = max(
    enumerate(defuzz_results.items()), key=lambda x: x[1][1])[1][0], max(
        enumerate(defuzz_results.items()), key=lambda x: x[1][1])[0]

# Print urgency calculation results
sums_print = [n_sum_queue, e_sum_queue, s_sum_queue, w_sum_queue]
ws_print = [n_wait_t, e_wait_t, s_wait_t, w_wait_t]
directions_print = ["Észak", "Kelet", "Dél", "Nyugat"]
print("\nAz egyes irányok fuzzy tagsági értékei:")
for (direction_print, sum_print, w_print, urgency_print, dir) in zip(
        directions_print, sums_print, ws_print, urgencies, directions):
    print("-----------------------------------------------")
    print(f"    {direction_print}:")
    print("    várakozó autók összmennyisége alapján")
    for key, value in sum_print.items():
        print(f"        {key} = {value}")
    print("    várakozási idő alapján")
    for key, value in w_print.items():
        print(f"        {key} = {value}")
    print(f"\n    ez alapján a prioritás fuzzy tagsági értéi:")
    for key, value in urgency_print.items():
        print(f"        {key} = {max(value)}")
    print(f"\n    az összesített következtetés defuzzifikált eredménye:")
    print(f"        {defuzz_results[dir]}")
print(f"\n\nA következő irány kapja a zöld lámpát: {directions_print[max_index]}")

# Calculate fuzzy memberships of lane queues for the most urgent direction
inner_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
                                    lane_queues[max_key]['inner'])
outer_queue = interpret_memberships(lane_queue_range, inner_lane_queue_mf,
                                    lane_queues[max_key]['outer'])

# Green phase extension fuzzy membership values
extension = rule_activation(inner_queue, outer_queue, extension_time_mf,
                            'extension')

# Lower boundary
extension0 = np.zeros_like(extension_time_range)
f_ext, ax_ext = plt.subplots(figsize=(8, 4))
title = 'Zöld lámpa fázis meghosszabbításának ideje'

# Visualization before aggregation
for key, value in extension.items():
    ax_ext.fill_between(extension_time_range, extension0, value,
                    color=colors[key], alpha=0.7)
# Draw the outlines of the membership functions
for key, value in extension_time_mf.items():    
    ax_ext.plot(extension_time_range, value, linewidth=1.5, color=colors[key],
            label=f'{key}')
    # Cut off the redundant end of the plot
    ax_ext.set_xlim([0, extension_time_range[-1]-0.99])
ax_ext.set_title(title)
# Place the legends to the right of the plots
ax_ext.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Aggregate all four extension time output membership functions together
aggregated_extensions, defuzz_ext_results, defuzz_ext_plt = {}, {}, {}

ext_zero, ext_short, ext_medium, ext_long = extension.values()
# The aggregated result will be the union of all output membership functions
# union = max
aggregated_extensions[key] = np.fmax(ext_zero, np.fmax(
    ext_short, np.fmax(ext_medium, ext_long)))
    

# Defuzzify aggregated outputs
for key, value in aggregated_extensions.items():
    defuzz_ext_results[key] = fuzz.defuzz(
        extension_time_range, aggregated_extensions[key], "centroid")

    # This is only necessary for the plot
    defuzz_ext_plt[key] = fuzz.interp_membership(
        extension_time_range, aggregated_extensions[key],
        defuzz_ext_results[key])

fig_d, ax_ext_def = plt.subplots(figsize=(8, 4))
title = 'Zöld lámpa fázis meghosszabbításának ideje (defuzzifikált)'

# Visualization of the defuzzified result
for key, value in aggregated_extensions.items():
    # Draw the filled aggregated output
    ax_ext_def.fill_between(extension_time_range, extension0, value,
                    facecolor='peachpuff', alpha=0.7)
    ax_ext_def.plot(extension_time_range, value, linewidth=1.5, color='k')
    # Draw a vertical line to mark the crisp output
    ax_ext_def.plot([defuzz_ext_results[key], defuzz_ext_results[key]], 
                    [0, defuzz_ext_plt[key]], 'k', linewidth=1.5, alpha=0.9)
            
    # Draw the outlines of the membership functions
    for key, value in extension_time_mf.items():    
        ax_ext_def.plot(extension_time_range, value, linewidth=1.5,
                        color=colors[key], label=f'{key}', linestyle='dashed',
                        alpha=0.5)
        # Cut off the redundant end of the plot
        ax_ext_def.set_xlim([0, extension_time_range[-1]-0.99])
    ax_ext_def.set_title(title)
    
# Place the legends to the right of the plots
ax_ext_def.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

lanes_title_print = ["Belső sáv", "Külső sáv"]
lanes_print = [inner_queue, outer_queue]
print("\nA kiválasztott irány sávjainak fuzzy tagsági értékei:")
for (lane_title, lane) in zip(lanes_title_print, lanes_print):
    print("-----------------------------------------------")
    print(f"    {lane_title}:")
    print("    várakozó autók mennyisége alapján")
    for lane_mf in lane.items():
        print(f"        {lane_mf[0]} = {lane_mf[1]}")
print(f"\nA kiválasztott irány zöld lámpa időtartamát "
      f"{max(defuzz_ext_results.values())} másodperccel kell meghosszabbítani.")
plt.show()