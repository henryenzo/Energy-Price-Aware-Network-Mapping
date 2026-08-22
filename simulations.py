"""
    This program will consitue the main simulation engine for the project. Now that we have a clean model class and test models, we can run simulations and compare the results for differnent hyperparameters and different models
"""

#from model_class import NetworkMapping, json_parser
from optim_relaxed import json_parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import pickle
import time
import subprocess, os

plt.rcParams["text.usetex"] = True # to use LaTeX in the plots

MODEL_FILE = "load_models.json"     # where the generated rho models live, same file as the batch
TIME_LIMIT = 1200 # s
FRAMEWORKS = ["model_class", "optim_on_whole_window", "optim_relaxed"]

def make_network_mapping(model, W, framework=None, **kwargs):
    """
        Picks the exact (fully binary) model for W in [1, 4] and the relaxed one for W=0 (static) or W>=5. \n
        The exact model gives a much tighter Gurobi bound on this range (it lets Gurobi apply RLT cuts on the binary products
        it would otherwise lose by relaxing), but its branch-and-bound blows up past W=4 -- that's when the relaxed model
        (built for large W, see optim_relaxed.py) takes over. \n
        Passing `framework` forces one single module for the whole W range instead, so the three of them can be compared on the same scenario.
    """
    if framework is None:
        framework = "optim_relaxed" if (W <= 0 or W >= 5) else "optim_on_whole_window"
    NetworkMapping = importlib.import_module(framework).NetworkMapping
    return NetworkMapping(model, W=W, time_limit=TIME_LIMIT, **kwargs)

def W_label(W):
    """ legend and tick label of one W : W=0 is the static optimized over the whole horizon, W=-1 the one optimized on the first time slot only """
    return {0: r"Static (oracle)", -1: r"Static (myopic)"}.get(W, rf"$W={W}$")


def get_time_index(prices_csv, N, time_stride, offset):
    """ reads the datetime index of the price CSV so the per-time-slot plots can show the clock time instead of the time slot $k$ """
    df = pd.read_csv(f"data/{prices_csv}", sep=",", decimal=".", encoding="utf-8-sig", index_col="datetime", parse_dates=True)
    return df.index[offset : offset + N * time_stride : time_stride]


def format_time_ticks(time_index):
    """ "%Hh%M" tick labels, with the date prepended if the simulation spans more than one calendar day """
    fmt = "%d/%m %Hh%M" if time_index[0].date() != time_index[-1].date() else "%Hh%M"
    return [t.strftime(fmt) for t in time_index]


def hyperparams_caption(W_list, time_stride, start_time):
    """ mini caption recalling the hyperparameters of the figure : the W's compared, the duration of a time slot, the S used by the underlying model (see `make_network_mapping`), and the simulation's start date """
    step_min = 15 * time_stride
    return (f"W = {list(W_list)}  --  step = {step_min} min (stride={time_stride})  --  "
            f"S=1 for W in [1,4], S=W otherwise  --  start = {start_time.strftime('%Y-%m-%d %H:%M %Z')}")


def add_hyperparams_caption(fig, W_list, time_stride, start_time):
    """ adds the mini caption at the bottom-right of the figure, reserving a thin margin below the subplots so it doesn't overlap their tick labels """
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 1))
    fig.text(0.995, 0.005, hyperparams_caption(W_list, time_stride, start_time),
              fontsize=6.5, family='monospace', ha='right', va='bottom',
              bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6', alpha=0.85, pad=0.3))


def plot_as_emf(figure, **kwargs):
    filepath = kwargs.get('filename', None)

    if filepath is not None:
        path, filename = os.path.split(filepath)
        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')
        emf_filepath = os.path.join(path, filename+'.emf')

        figure.savefig(svg_filepath, format='svg')

        subprocess.call(["inkscape", svg_filepath, '--export-type=emf']) #emf_filepath
        os.remove(svg_filepath)


def run_simulation(model_name, W, N, time_stride=1, offset=0, prices_csv="energy_prices.csv", framework=None):
    """
        This function will run the simulation for a given model and hyperparameters. It will return the overall cost for the whole time horizon.
    """
    model = json_parser(model_name, MODEL_FILE)
    network_mapping = make_network_mapping(model, W, framework=framework, S=1, N=N, time_stride=time_stride, offset=offset, prices_csv=prices_csv)
    network_mapping.run()
    return network_mapping.overall_cost

def trace_cost_curve_for_different_W(model_name, W_list, N, time_stride=1, prices_csv="energy_prices.csv", offset=0, framework=None):
    """
        This function will run the simulation for a given model and trace the cost per k for different values of W. It will return a list of costs for each value of W.
    """
    time_index = get_time_index(prices_csv, N, time_stride, offset)
    time_ticks = format_time_ticks(time_index)

    cost_curve = []
    overall_costs = []
    optim_duration = []
    delay_curve = []
    for W in W_list:
        model = json_parser(model_name, MODEL_FILE)
        network_mapping = make_network_mapping(model, W, framework=framework, S=1, N=N, time_stride=time_stride, offset=offset, prices_csv=prices_csv)
        start_time = time.time()
        network_mapping.run()
        end_time = time.time()
        print(f"Simulation duration for W={W}: {end_time - start_time} seconds")
        print(f"Worst MIP gap for W={W}: {100*max(network_mapping.mip_gaps, default=0):.2f}% ({len(network_mapping.statuses)} solves, statuses {sorted(set(network_mapping.statuses))})")
        cost_curve.append([network_mapping.cost[k]["effective_cost"] for k in range(N)]) # only keep the effective cost for the cost curve
        overall_costs.append(network_mapping.overall_cost)
        optim_duration.append((end_time - start_time)/N) # average optimization duration per time slot
        delay_curve.append([cost["link_delay"] for cost in network_mapping.cost]) # only keep the delay cost for the delay curve
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    # Cost per time slot for each W
    ax = axes[0]
    for i, W in enumerate(W_list):
        ax.plot(range(len(cost_curve[i])), cost_curve[i], label=W_label(W))
    ax.set_xticks(range(N))
    ax.set_xticklabels(time_ticks, rotation=45, ha='right')
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Cost at time slot $k$")
    ax.set_title(rf"Cost per time slot $k$ for different values of $W$ ($N={N}$)")
    ax.legend()
    ax.grid()

    # overall cost as a function of W
    ax2 = axes[1]
    ax2.bar(W_list, overall_costs, width=0.6, color='tab:blue')
    ax2.set_xticks(list(W_list))
    ax2.set_xticklabels([W_label(W) for W in W_list], rotation=45, ha='right')
    ax2.set_xlabel(r"$W$")
    ax2.set_ylabel(r"Overall cost")
    ax2.set_title(rf"Overall cost vs $W$ ($N={N}$)")
    ax2.grid()

    # secondary y-axis: normalized relative to W=0 
    ax2b = ax2.twinx()
    base = overall_costs[list(W_list).index(0)] if 0 in W_list and overall_costs[list(W_list).index(0)] != 0 else 1
    normalized = [oc / base for oc in overall_costs]
    ax2b.plot(W_list, normalized, marker='s', linestyle='--', color='tab:orange')
    ax2b.set_ylabel(r"Normalized")
    ax2b.set_ylim(min(normalized) * 0.95, max(normalized) * 1.05)

    # individual cost components for the last W
    ax3 = axes[2]
    ax3_secondary = ax3.twinx()
    last_W_costs = [cost_curve[-1][k] for k in range(N)]
    cost_components = ["energy_cost", "migration_cost"] #"usage_cost", "disposal_cost",
    n_components = len(cost_components)
    bar_width = 0.8 / n_components
    for j, cost_component in enumerate(cost_components):
        try:
            component_costs = [network_mapping.cost[k][cost_component] for k in range(N)]
            x_positions = [k + (j - (n_components - 1) / 2) * bar_width for k in range(len(component_costs))]
            ax3.bar(x_positions, component_costs, width=bar_width, label=cost_component)
        except:
            pass
    ax3_secondary.plot(range(len(network_mapping.migrations)), network_mapping.migrations, label="Number of migrations", marker='o', linestyle='None', color='tab:purple')
    energy_costs = [network_mapping.cost[k][cost_component] for k in range(N)]
    ax3.set_xticks(range(N))
    ax3.set_xticklabels(time_ticks, rotation=45, ha='right')
    ax3.set_xlabel(r"Time")
    ax3.set_ylabel(r"Cost at time slot $k$")
    ax3_secondary.set_ylabel(r"Number of migrations")
    ax3.set_title(rf"Cost components per time slot $k$ and number of migrations for $W={W_list[-1]}$ ($N={N}$)")
    ax3.legend()
    ax3.grid()

    # Total delay
    ax4 = axes[3]
    # for i, W in enumerate(W_list):
    #     ax4.plot(range(len(delay_curve[i])), 1000*np.array(delay_curve[i]), label=rf"$W={W}$")   # *1000 to convert to ms
    # ax4.set_xticks(range(N))
    # ax4.set_xticklabels(time_ticks, rotation=45, ha='right')
    # ax4.set_xlabel(r"Time")
    # ax4.set_ylabel(r"Latency (ms)")
    # ax4.set_title(rf"Latency vs Time slot $k$ for different values of $W$ ($N={N}$)")
    # ax4.legend()
    # ax4.grid()

    # Relative gain of the dynamic model over the static one, same definition as in the batch
    positive_W = [W for W in W_list if W > 0]
    for W_static in [W for W in (0, -1) if W in W_list]:
        static_cost = overall_costs[list(W_list).index(W_static)]
        gains = [100 * (static_cost - overall_costs[list(W_list).index(W)]) / static_cost for W in positive_W]
        ax4.plot(positive_W, gains, marker='s', label=rf"vs {W_label(W_static)}")
    ax4.axhline(0, color='k', linewidth=0.8)
    ax4.set_xticks(positive_W)
    ax4.set_xlabel(r"$W$")
    ax4.set_ylabel(r"Gain over the static model (\%)")
    ax4.set_title(rf"Relative saving vs $W$ ($N={N}$)")
    ax4.legend()
    ax4.grid()

    # Histogram of average price for each country, and the aggregate time spent by VNFs in each country for the last W
    ax5 = axes[4]
    ax5_secondary = ax5.twinx()
    country_prices = network_mapping.price_per_country
    country_time_spent = {country: 0 for country in country_prices.keys()}
    for k in range(N):
        for v in network_mapping.virtual_nodes:
            country = model["node_country"][network_mapping.placement[k][v]]
            country_time_spent[country] += 1
    
    width = 0.35
    countries = list(country_prices.keys())
    x = np.arange(len(countries))
    avg_prices = [np.mean(country_prices[country]) for country in countries]
    avg_time_spent = [country_time_spent[country] for country in countries]
    bars1 = ax5.bar(x - width/2, avg_prices, width, label='Avg. Price', color='tab:blue')
    bars2 = ax5_secondary.bar(x + width/2, avg_time_spent, width, label='Avg. Time Spent', color='tab:orange')
    ax5.set_xlabel(r"Country")
    ax5.set_ylabel(r"Average energy price")
    ax5_secondary.set_ylabel(r"Aggregate time spent")
    ax5.set_title(rf"Avg. Price and Time Spent per Country for $W={W_list[-1]}$ ($N={N}$)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(countries, rotation=45)
    ax5.legend([bars1, bars2], ['Avg. Price', 'Avg. Time Spent'])

    # Optimization duration as a function of W
    ax6 = axes[5]
    ax6.plot(W_list, optim_duration, marker='o')
    ax6.set_xlabel(r"$W$")
    ax6.set_ylabel(r"Optimization duration (s)")
    ax6.set_title(rf"Optimization duration vs $W$ ($N={N}$)")
    ax6.grid()

    add_hyperparams_caption(fig, W_list, time_stride, time_index[0])

    figname = f"cost_curve_W_{model_name}_N{N}" + (f"_{framework}" if framework is not None else "")

    plt.tight_layout()
    plt.savefig(f"plots/simulations/{figname}.svg")              # saves as a mere svg file

    save_pkl(fig, figname)                # saves as an adjustable python pickle file
    fig = plt.gcf()
    # plot_as_emf(fig, filename=f"cost_curve_W_{model_name}_N{N}")  # saves as an emf file, editable on poweproint or Inkscape
    plt.show()

def save_pkl(fig, figname="figure"):
    with open(f"plots/simulations/{figname}.pkl", "wb") as f:
        pickle.dump(fig, f)

def show_pkl(figname="figure", filename=None):
    if filename is None:
        filename = f"plots/simulations/{figname}.pkl"

    with open(filename, "rb") as f:
        fig = pickle.load(f)
    plt.show()
    

if __name__ == "__main__":
    model_name = "nobel-eu-8SFC"
    W_list = [-1, 0, 1, 2, 3, 4, 5]
    N = 24
    time_stride = 2
    offset = 896    # 2026-07-17 06:00 UTC in energy_prices_batch.csv
    prices_csv = "energy_prices_batch.csv"
    #prices_csv = "energy_prices_3days.csv"

    for framework in FRAMEWORKS:
        trace_cost_curve_for_different_W(model_name, W_list, N, time_stride, prices_csv, offset=offset, framework=framework)

    # show_pkl(figname=f"cost_curve_W_{model_name}_N{N}")


    #run_simulation(model_name, W=3, N=N, time_stride=time_stride, prices_csv=prices_csv, offset=offset)
