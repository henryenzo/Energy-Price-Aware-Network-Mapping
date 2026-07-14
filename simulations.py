"""
    This program will consitue the main simulation engine for the project. Now that we have a clean model class and test models, we can run simulations and compare the results for differnent hyperparameters and different models
"""

from model_class import NetworkMapping, json_parser
import matplotlib.pyplot as plt
import numpy as np
import time



def run_simulation(model_name, W, N, time_stride=1):
    """
        This function will run the simulation for a given model and hyperparameters. It will return the overall cost for the whole time horizon.
    """
    model = json_parser(model_name)
    network_mapping = NetworkMapping(model, W=W, S=1, N=N, time_stride=time_stride)
    network_mapping.run()
    return network_mapping.overall_cost

def trace_cost_curve_for_different_W(model_name, W_list, N, time_stride=1):
    """
        This function will run the simulation for a given model and trace the cost per k for different values of W. It will return a list of costs for each value of W.
    """
    cost_curve = []
    overall_costs = []
    optim_duration = []
    delay_curve = []
    for W in W_list:
        model = json_parser(model_name)
        network_mapping = NetworkMapping(model, W=W, S=1, N=N, time_stride=time_stride)
        start_time = time.time()
        network_mapping.run()
        end_time = time.time()
        print(f"Simulation duration for W={W}: {end_time - start_time} seconds")
        cost_curve.append([network_mapping.cost[k]["effective_cost"] for k in range(N)]) # only keep the effective cost for the cost curve
        overall_costs.append(network_mapping.overall_cost)
        optim_duration.append((end_time - start_time)/N) # average optimization duration per time slot
        delay_curve.append([cost["link_delay"] for cost in network_mapping.cost]) # only keep the delay cost for the delay curve
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    axes = axes.flatten()
    # Cost per time slot for each W
    ax = axes[0]
    for i, W in enumerate(W_list):
        ax.plot(range(len(cost_curve[i])), cost_curve[i], label=f"W={W}")
    ax.set_xlabel("Time slot k")
    ax.set_ylabel("Cost at time slot k")
    ax.set_title(f"Cost per time slot k for different values of W (N={N})")
    ax.legend()
    ax.grid()

    # overall cost as a function of W
    ax2 = axes[1]
    ax2.plot(W_list, overall_costs, marker='o')
    ax2.set_xlabel("W")
    ax2.set_ylabel("Overall cost")
    ax2.set_title(f"Overall cost vs W (N={N})")
    ax2.grid()

    # # Optimization duration as a function of W
    # ax3 = axes[2]
    # ax3.plot(W_list, optim_duration, marker='o')
    # ax3.set_xlabel("W")
    # ax3.set_ylabel("Optimization duration (s)")
    # ax3.set_title(f"Optimization duration vs W (N={N})")
    # ax3.grid()

    # individual cost components for the last W
    ax3 = axes[2]
    last_W_costs = [cost_curve[-1][k] for k in range(N)]
    for cost_component in ["energy_cost", "usage_cost", "disposal_cost", "link_delay_cost", "migration_cost"]:
        try:
            component_costs = [network_mapping.cost[k][cost_component] for k in range(N)]
            ax3.plot(range(len(component_costs)), component_costs, label=cost_component)
        except:
            pass
    energy_costs = [network_mapping.cost[k][cost_component] for k in range(N)]
    ax3.plot(range(len(energy_costs)), np.array(energy_costs) - np.mean(energy_costs), label="energy_cost_diff")
    ax3.set_xlabel("Time slot k")
    ax3.set_ylabel("Cost at time slot k")
    ax3.set_title(f"Cost components per time slot k for W={W_list[-1]} (N={N})")
    ax3.legend()
    ax3.grid()

    # Total delay
    ax4 = axes[3]
    for i, W in enumerate(W_list):
        ax4.plot(range(len(delay_curve[i])), 1000*np.array(delay_curve[i]), label=f"W={W}")   # *1000 to convert to ms
    ax4.set_xlabel("Time slot k")
    ax4.set_ylabel("Delay (ms)")
    ax4.set_title(f"Delay vs Time slot k for different values of W (N={N})")
    ax4.legend()
    ax4.grid()

    plt.tight_layout()
    plt.savefig(f"cost_curve_W_{model_name}_N{N}.png")
    plt.show()
    

if __name__ == "__main__":
    model_name = "nobel-eu"
    W_list = [1, 2, 3, 4, 5]
    N = 10
    time_stride = 2
    trace_cost_curve_for_different_W(model_name, W_list, N, time_stride)

    #run_simulation(model_name, W=3, N=N, time_stride=time_stride)
