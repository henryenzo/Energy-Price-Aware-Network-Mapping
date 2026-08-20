""" ABOUT THIS FILE.
    This file runs a batch of simulations whose only purpose is to quantify the gain of the dynamic (foresighted) model over the static one, on many different scenarios instead of a single one.
    A scenario is a quadruplet (duration, date, starting hour, load factor) and is explored one axis at a time around a baseline case, so that each figure isolates the effect of one parameter :
    - duration : how long the lifecycle of the SFC lasts (number of time slots N),
    - date : which day of the price CSV the lifecycle starts on,
    - starting hour : at what UTC hour of the day it starts (everything here is reasoned in UTC : the price CSV is indexed in UTC, and a "same time slot" only means something if it's the same instant for every country),
    - load factor (rho) : how full the infrastructure is. total CPU demand over total CPU capacity. Contrary to the three other axes, changing rho means changing the model and not only the time window, so each rho gets its own generated entry in load_models.json (that i generated in cost_vs_load.py). The baseline rho reproduces the fill of nobel-eu-1SFC (about 11%).
    Each scenario is run for W=0 (static, no migration) and for every W of W_LIST, and the gain is the relative saving of the dynamic model with respect to the static one : \n
    $gain(W) = (C_{static} - C_W) / C_{static} * 100$ \n
    Precisions about the results : the static model is not so naive a baseline, run_nomig optimizes the best fixed placement knowing the prices of the whole horizon. A dynamic model with a short window can therefore be beaten by it, which is exactly what a negative gain means here.

    Created on July 30th, 2026 by Enzo Henry
"""

from optim_relaxed import json_parser
from stochastic_engine import fetch_energy_prices
from cost_vs_load import generate_nobel_eu_entry, append_entry_to_json, compute_rho
import matplotlib
matplotlib.use("macosx")    # native backend on my machine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import time
from pathlib import Path

plt.rcParams["text.usetex"] = True # to use LaTeX in the plots
plt.rcParams["font.family"] = "serif"

# All prints should go to out_console.txt as well so I can read all of them easily after 
OUT_CONSOLE = Path(__file__).resolve().parent / "simulations_bat" / "out_console.txt" 
OUT_CONSOLE.write_text("")
def print_out(x):
    print(x)
    with OUT_CONSOLE.open("a", encoding="utf-8") as f:
        f.write(f"{x}\n")


# configuration of the batch
SOURCE_MODEL = "nobel-eu"  # topology, prices and distances the generated models with varying loads are built upon
MODEL_FILE = "load_models.json"  # where the generated rho models live, created by ensure_load_models if it doesn't exist yet (so it works for you too)
BATCH_NAME = "nobel-eu-rho"  # only used to name the results file and the figures, the model of each scenario is carried by the case itself
PRICES_CSV = "energy_prices_batch.csv"  # named differently from my other CSVs, fetched once and reused by every scenario
FETCH_START = pd.Timestamp("2026-07-08", tz="UTC")  # only used if the CSV doesn't exist yet
FETCH_END = pd.Timestamp("2026-07-29", tz="UTC")

W_LIST = [1, 2, 3, 4, 5]  # dynamic models to compare to the static one (W=0 is always run as the baseline)
TIME_STRIDE = 2  # 2 * 15 minutes = 1 time slot per half hour

BASELINE_N = 24 # baseline duration, in time slots
BASELINE_DATE = None  # if None then it'll be the middle date of the CSV, so the batch doesn't depend on the span I fetched
BASELINE_HOUR = 0  # midnight UTC
BASELINE_RHO = 0.11  # baseline load factor : same fill as nobel-eu-1SFC (16 cores demanded / 140 available = 11.4%)

DURATIONS = [6, 12, 24, 48]  # duration axis, in time slots
DATE_STEP_DAYS = 3  # date axis : one scenario every DATE_STEP_DAYS days of the CSV
START_HOURS = [0, 6, 12, 18]  # starting hour axis, Zulu time
RHO_LIST = [0.11, 0.25, 0.50, 0.75]  # load factor axis. Setting it to [BASELINE_RHO] makes it run only the baseline rho in case that's a bad idea to use it

# size of the generated models, see Section 5 of my draft paper
SFC_LEN = 8
VNF_CPU, VNF_MEM, VNF_BW = 4, 2, 2
NODE_CPU, NODE_MEM = 64, 256  # Dell PowerEdge HS5610 (2 x Xeon Gold 6448Y), the SPECpower reference server that I chose in Section 5

RESUME = True  # skip the scenarios already present in the results file

PRICES_DF = None  # dataframe of the prices, filled by `ensure_prices_csv` : it is needed to convert the dates and hours into offsets, and to compute the price spread of each scenario

OUTPUT_DIR = Path(__file__).resolve().parent / "simulations_bat"
PLOTS_DIR = OUTPUT_DIR / "plots"
RESULTS_DIR = OUTPUT_DIR / "results"
GRAPHS_DIR_NAME = "simulations_bat/graphs"  # relative to the model file, as expected by NetworkMapping

TIME_LIMIT = 60 # s

def add_hyperparams_caption(fig, W_list, entries, stride = TIME_STRIDE):
    """ 
        bottom-right caption with the figure's hyperparameters : W list, step duration, S rule, and the scenarios' start date/hour ("varies (x-axis)" when that's the axis being studied) \n
        Function generated by Claude AI
    """
    step_min = 15 * stride
    dates = {entry["date"] for entry in entries}
    hours = {entry["hour"] for entry in entries}
    date_str = dates.pop() if len(dates) == 1 else "varies (x-axis)"
    hour_str = f"{hours.pop():02d}h" if len(hours) == 1 else "varies (x-axis)"
    caption = (f"W = {list(W_list)}  --  step = {step_min} min (stride={stride})  --  S=1 for W in [1,4], S=W otherwise  --  start = {date_str} {hour_str}")
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 1))
    fig.text(0.995, 0.005, caption, fontsize=6.5, family='monospace', ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.6', alpha=0.85, pad=0.3))


def latex_safe(text):
    """
        escapes underscores and percent signs so it don't break LaTeX rendering once and for all
    """
    return str(text).replace("_", r"\_").replace("%", r"\%")


def n_sfc_for_rho(rho, cpu_req = None, cpu_avail = None, sfc_len = None, n_nodes = None):
    """
        Number of SFC instances whose CPU demand gets as close as possible to rho from the bottom \n
        Only the real VNFs of each SFC are counted cause the two fictitious ones carry no load.
    """
    # baseline if not sth else
    if cpu_req is None:
        cpu_req = VNF_CPU
    if cpu_avail is None:
        cpu_avail = NODE_CPU
    if sfc_len is None:
        sfc_len = SFC_LEN
    if n_nodes is None:
        n_nodes = len(json_parser(SOURCE_MODEL)["computing_availability"])
    n = int(rho * n_nodes * cpu_avail // ((sfc_len - 2) * cpu_req))
    return max(n, 1)


def ensure_load_models(rho_list = None, model_file = MODEL_FILE, seed = 0):
    """
        Makes sure one model exists in model_file for each target load factor, generating the missing ones with cost_vs_load.generate_nobel_eu_entry.\n
        Bandwidth availability is set to VNF_BW * n_sfc so that no physical link can ever saturate

        Returns: dict {rho_target: {"model_name":..., "n_sfc":..., "rho":...}} in the order of rho_list
    """
    rho_list = RHO_LIST if rho_list is None else rho_list
    n_nodes = len(json_parser(SOURCE_MODEL)["computing_availability"])
    existing = set()
    if Path(model_file).exists():
        with open(model_file) as file:
            existing = {model["model_name"] for model in json.load(file)["test_models"]}

    models = {}
    for rho_target in rho_list:
        n_sfc = n_sfc_for_rho(rho_target, n_nodes=n_nodes)
        model_name = f"{SOURCE_MODEL}-{n_sfc}SFC"
        rho = compute_rho(n_sfc, VNF_CPU, VNF_MEM, NODE_CPU, NODE_MEM, SFC_LEN, n_nodes)[2]
        if model_name not in existing:
            entry = generate_nobel_eu_entry(n_sfc, cpu_req=VNF_CPU, mem_req=VNF_MEM, bw_req=VNF_BW, cpu_avail=NODE_CPU, mem_avail=NODE_MEM, bw_avail=VNF_BW * n_sfc, sfc_len=SFC_LEN, model_name=model_name, source_model=SOURCE_MODEL, seed=seed)
            append_entry_to_json(entry, model_file)
            existing.add(model_name)
        models[rho_target] = {"model_name": model_name, "n_sfc": n_sfc, "rho": rho}
        print_out(f"rho target {rho_target*100}% -> {n_sfc} SFCs ({n_sfc * (SFC_LEN - 2)} VNFs), actual rho = {rho:.1%}, the model is named '{model_name}'")
    return models


def ensure_prices_csv(prices_csv = PRICES_CSV, model_name = SOURCE_MODEL, start = FETCH_START, end = FETCH_END):
    """
        fetches the price CSV from ENTSO-E if it's not there yet (only the model's countries), and returns it as a dataframe 
    """
    global PRICES_DF # first time I use global in a python program haha, hope it works
    csv_path = Path("data") / prices_csv
    if not csv_path.exists():
        print_out(f"Theres no '{csv_path}', fetching {start.date()} -> {end.date()} from the ENTSO-E API")
        model = json_parser(model_name)
        country_list = list(set(model["node_country"].values()))
        fetch_energy_prices(start=start, end=end, country_list=country_list, csv_file_name=prices_csv)

    df = pd.read_csv(csv_path, sep=",", decimal=".", encoding="utf-8-sig", index_col="datetime", parse_dates=True)
    print_out(f"{len(df)} rows available, from {df.index[0]} to {df.index[-1]} and {len(df.columns)} zones")
    PRICES_DF = df
    return df


def offset_from_datetime(df, date, start_hour = BASELINE_HOUR):
    """
        turns a UTC date + hour into the row offset for NetworkMapping 
    """
    timestamp = pd.Timestamp(f"{date} {start_hour:02d}:00", tz="UTC")
    offset = int(df.index.searchsorted(timestamp))
    assert offset < len(df), f"{timestamp} is out of the range covered by the price CSV"
    return offset


def case_fits(df, N, W, offset, stride = TIME_STRIDE):
    """ 
        the model reads N+W time slots from the offset with a certain stride, this checks that they all exist in the CSV 
    """
    return offset + (N + max(W, 1)) * stride <= len(df)


def price_dispersion(df, model, N, W, offset, stride = TIME_STRIDE):
    """ mean price spread across the model's countries over the scenario """
    countries = list(set(model["node_country"].values()))
    window = df[countries].iloc[offset : offset + (N + max(W, 1)) * stride : stride]
    return float((window.max(axis=1) - window.min(axis=1)).mean())


def build_case_list(df, durations = DURATIONS, start_hours = START_HOURS, date_step_days = DATE_STEP_DAYS, rho_list = None, models = None, baseline_N = BASELINE_N, baseline_date = BASELINE_DATE, baseline_hour = BASELINE_HOUR, baseline_rho = BASELINE_RHO):
    """
        builds the scenario list, one axis at a time around a baseline case (the baseline itself belongs to all four axes, so it's only optimized once) \n
        models is the dict returned by ensure_load_models : it maps each target rho to the generated model that realizes it
        returns cases = [{"case_id":..., "axis":..., "value":..., "label":..., "N":..., "date":..., "hour":..., "offset":..., "model_name":...}, ...]
    """ 
    if rho_list is None:
        rho_list = RHO_LIST
    if models is None:
        models = ensure_load_models(rho_list) 
    assert baseline_rho in models, f"BASELINE_RHO={baseline_rho} must belong to RHO_LIST={rho_list}"
    baseline_model = models[baseline_rho]["model_name"]

    dates = sorted(set(df.index.date))
    dates = dates[:-1]
    if baseline_date is None:
        baseline_date = dates[len(dates)//2]  # the middle of the CSV, so the batch doesn't depend on the span I fetched
    print_out(f"Baseline scenario : N={baseline_N} time slots, starting on {baseline_date} at {baseline_hour:02d}h UTC, rho={models[baseline_rho]['rho']:.1%} ('{baseline_model}')")

    cases = []
    def add_case(axis, value, label, N, date, hour, model_name = None):
        offset = offset_from_datetime(df, date, hour)
        if not case_fits(df, N, max(W_LIST), offset):
            print_out(f"Scenario {axis}={label} cancelled : not enough rows left in the CSV after {date} {hour:02d}h")
            return
        cases.append({"case_id": f"{axis}_{label}", "axis": axis, "value": value, "label": label, "N": N, "date": str(date), "hour": hour, "offset": offset, "model_name": baseline_model if model_name is None else model_name, "model_file": MODEL_FILE})

    for N in durations:
        add_case("duration", N, str(N), N, baseline_date, baseline_hour)
    for date in dates[::date_step_days]:
        add_case("date", pd.Timestamp(date), date.strftime("%m-%d"), baseline_N, date, baseline_hour)
    for hour in start_hours:
        add_case("hour", hour, f"{hour:02d}h", baseline_N, baseline_date, hour)
    for rho_target in rho_list:  # this axis changes the model itself, not the time window
        model = models[rho_target]
        add_case("rho", model["rho"], f"{model['rho']:.0%}", baseline_N, baseline_date, baseline_hour, model_name=model["model_name"])

    return cases


def make_network_mapping(model, W, **kwargs):
    """ 
        exact binary model for W at or below 4, relaxed one for W=0 or W>=5. The exact model gives a tighter bound but its branch-and-bound blows up past W=4 
        Note (TODO?): this was observed on the previous version of the model (1SFC on a simpler model), I might have to check if it's still relevant now, even more considering that we now have a timeout
    """
    if W == 0 or W >= 5:
        from optim_relaxed import NetworkMapping
    else:
        from optim_on_whole_window import NetworkMapping
    return NetworkMapping(model, W=W, time_limit=TIME_LIMIT, **kwargs)


def run_case(case, W, prices_csv = PRICES_CSV, stride = TIME_STRIDE):
    """
        runs one (scenario, W) couple and returns everything the figures need -- W=0 is the static baseline the gain is computed against \n
        the model is carried by the case itself, since the rho axis makes it vary from one scenario to the next
        returns: a dict with the case's parameters and the results 
    """
    model = json_parser(case["model_name"], case.get("model_file", MODEL_FILE)) # case["model_file"] or MODEL_FILE by default if not in it
    network_mapping = make_network_mapping(model, W, S=1, N=case["N"], time_stride=stride, offset=case["offset"], prices_csv=prices_csv)
    network_mapping.plot_graph = lambda *args, **kwargs: None   # to overwrite my plot_graph function, the batch would render thousands of svg files otherwise
    network_mapping.verbose = False

    start_time = time.time()
    network_mapping.run()
    simulation_duration = time.time() - start_time

    costs = network_mapping.cost
    return {
        **case,
        "W": W,
        "overall_cost": float(network_mapping.overall_cost),
        "cost_per_k": [float(c["effective_cost"]) for c in costs],
        "energy_cost": float(sum(float(c["energy_cost"]) for c in costs)),
        "migration_cost": float(sum(float(c.get("migration_cost", 0)) for c in costs)),
        "mean_delay": float(np.mean([float(c["link_delay"]) for c in costs])),
        "migrations": int(np.sum(network_mapping.migrations)),  # empty list for the static model
        "price_dispersion": price_dispersion(PRICES_DF, model, case["N"], W, case["offset"], stride),
        "simulation_duration": simulation_duration,
        "optim_duration": simulation_duration / case["N"],  # average optimization duration per time slot
    }


def run_batch(cases, W_list = W_LIST, batch_name = BATCH_NAME, prices_csv = PRICES_CSV, resume = RESUME):
    """
        runs every (scenario, W) couple, W=0 included, saving after each one so the batch can be resumed if interrupted \n
        Function made with the help of Claude AI
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"batch_{batch_name}.pkl"

    results = []
    if resume and results_path.exists():
        with open(results_path, "rb") as file:
            results = pickle.load(file)
        print_out(f"{len(results)} results loaded from '{results_path.name}', the corresponding scenarios will be skipped")
    done = {(result["case_id"], result["W"]) for result in results}
    computed = {(result["model_name"], result["N"], result["offset"], result["W"]): result for result in results}   # what has already been optimized, whatever the axis it belonged to -- the model is part of the key since the rho axis makes it vary

    todo = [(case, W) for case in cases for W in [0] + list(W_list) if (case["case_id"], W) not in done]
    print_out(f"{len(todo)} optimizations to run\n")
    for index, (case, W) in enumerate(todo):
        print_out(f"===== [{index+1}/{len(todo)}] scenario {case['case_id']} ({case['model_name']}, N={case['N']}, {case['date']} {case['hour']:02d}h) with W={W}")
        key = (case["model_name"], case["N"], case["offset"], W)
        if key in computed: # the baseline scenario belongs to the four axes, so it is optimized once and reused by the others
            result = {**computed[key], **case, "W": W}
            print_out(f"      identical to the scenario '{computed[key]['case_id']}' already optimized, reused as is\n")
        else:
            try:
                result = run_case(case, W, prices_csv=prices_csv)
            except Exception as e: # a single failing scenario must not throw away a batch that has been running for hours
                print_out(f"      /!\\ scenario dropped, the optimization raised : {e}\n")
                continue
            computed[key] = result
            print_out(f"overall cost = {result['overall_cost']:.6f}, {result['migrations']} migrations, {result['simulation_duration']:.1f} s\n")
        results.append(result)
        with open(results_path, "wb") as file:
            pickle.dump(results, file)

    print_out(f"Batch finished, {len(results)} results saved in '{results_path}'")
    return results


def gain_table(results):
    """ 
        reorganizes the results into {axis: {W: [...]}}, with gain(W) added to each entry 
    """
    static_cost = {result["case_id"]: result["overall_cost"] for result in results if result["W"] == 0}
    table = {}
    for result in results:
        if result["W"] == 0:
            continue
        axis, W = result["axis"], result["W"]
        table.setdefault(axis, {}).setdefault(W, []).append({ # setdefault is a nice way to avoid KeyError or having to write 42000 if statments 
            **result,
            "static_cost": static_cost[result["case_id"]],
            "gain": 100 * (static_cost[result["case_id"]] - result["overall_cost"]) / static_cost[result["case_id"]],
        })
    for axis in table:
        for W in table[axis]:
            table[axis][W].sort(key=lambda entry: entry["value"])
    return table


def plot_axis(results, axis, xlabel, title, model_name = BATCH_NAME):
    """ 
        plots overall cost, gain, migrations and solving time for one axis of the batch, and saves the figure (svg and pkl) 
    """
    table = gain_table(results)
    if axis not in table:
        print_out(f"No result for the '{axis}' axis, I skip the figure")
        return
    W_list = sorted(table[axis].keys())
    reference = max(table[axis].values(), key=len)  # the most complete W, in case a scenario had to be dropped for another one
    labels = [entry["label"] for entry in reference]
    position = {label: index for index, label in enumerate(labels)}
    x = np.arange(len(labels))

    def curve(W, key):
        """ (x, y) of one curve, looked up by label so that a missing scenario only leaves a hole instead of shifting the whole curve """
        points = [(position[entry["label"]], entry[key]) for entry in table[axis][W] if entry["label"] in position]
        return [point[0] for point in points], [point[1] for point in points]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    # Overall cost over the whole lifecycle, static model included
    ax = axes[0]
    ax.plot(x, [entry["static_cost"] for entry in reference], marker='o', linestyle='--', color='k', label="Static")
    for W in W_list:
        ax.plot(*curve(W, "overall_cost"), marker='o', label=rf"$W={W}$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Overall cost (EUR)")
    ax.set_title(rf"Overall cost -- {title}")
    ax.legend()
    ax.grid()

    # Relative gain of the dynamic model over the static one
    ax2 = axes[1]
    for W in W_list:
        ax2.plot(*curve(W, "gain"), marker='s', label=rf"$W={W}$")
    ax2.axhline(0, color='k', linewidth=0.8)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r"Gain over the static model (\%)")
    ax2.set_title(rf"Relative saving -- {title}")
    ax2.legend()
    ax2.grid()

    # Number of migrations over the whole lifecycle
    ax3 = axes[2]
    bar_width = 0.8 / len(W_list)
    for j, W in enumerate(W_list):
        x_W, migrations = curve(W, "migrations")
        ax3.bar(np.array(x_W) + (j - (len(W_list) - 1) / 2) * bar_width, migrations, width=bar_width, label=rf"$W={W}$")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(r"Number of migrations")
    ax3.set_title(rf"Migrations -- {title}")
    ax3.legend()
    ax3.grid()

    # Average optimization duration per time slot
    ax4 = axes[3]
    for W in W_list:
        ax4.plot(*curve(W, "optim_duration"), marker='^', label=rf"$W={W}$")
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(r"Optimization duration per time slot (s)")
    ax4.set_title(rf"Solving time -- {title}")
    ax4.legend()
    ax4.grid()

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels([latex_safe(label) for label in labels], rotation=45)

    fig.suptitle(rf"{latex_safe(model_name)} : effect of the {title.lower()} on the gain of the dynamic model")
    add_hyperparams_caption(fig, W_list, reference)
    save_figure(fig, f"gain_vs_{axis}_{model_name}")
    return fig


def plot_summary(results, model_name = BATCH_NAME):
    """ 
        plots the gain distribution over the whole batch and what it correlates with (migrations, price spread), saves the figure (svg + pkl) 
    """
    table = gain_table(results)
    W_list = sorted({W for axis in table for W in table[axis]})
    gains = {W: [entry["gain"] for axis in table for entry in table[axis].get(W, [])] for W in W_list}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.flatten()

    # Distribution of the gain for each W, over every scenario of the batch
    ax = axes[0]
    ax.boxplot([gains[W] for W in W_list], positions=np.arange(len(W_list)))
    ax.set_xticks(np.arange(len(W_list)))
    ax.set_xticklabels([rf"$W={W}$" for W in W_list])
    ax.set_ylabel(r"Gain over the static model (\%)")
    ax.set_title(rf"Distribution of the gain over the {len(gains[W_list[0]])} scenarios")
    ax.grid()

    # Mean gain and standard deviation
    ax2 = axes[1]
    ax2.bar(np.arange(len(W_list)), [np.mean(gains[W]) for W in W_list], yerr=[np.std(gains[W]) for W in W_list], width=0.6, capsize=5, color='tab:blue')
    ax2.set_xticks(np.arange(len(W_list)))
    ax2.set_xticklabels([rf"$W={W}$" for W in W_list])
    ax2.set_ylabel(r"Mean gain (\%)")
    ax2.set_title(r"Mean gain and standard deviation")
    ax2.grid()

    # The gain has to be paid by migrations : this is where we see if it is worth it
    ax3 = axes[2]
    for W in W_list:
        entries = [entry for axis in table for entry in table[axis].get(W, [])]
        ax3.scatter([entry["migrations"] for entry in entries], [entry["gain"] for entry in entries], label=rf"$W={W}$")
    ax3.set_xlabel(r"Number of migrations")
    ax3.set_ylabel(r"Gain over the static model (\%)")
    ax3.set_title(r"Gain vs number of migrations")
    ax3.legend()
    ax3.grid()

    # And there is nothing to win when every country is at the same price
    ax4 = axes[3]
    for W in W_list:
        entries = [entry for axis in table for entry in table[axis].get(W, [])]
        ax4.scatter([entry["price_dispersion"] for entry in entries], [entry["gain"] for entry in entries], label=rf"$W={W}$")
    ax4.set_xlabel(r"Mean price spread accross the countries (EUR/MWh)")
    ax4.set_ylabel(r"Gain over the static model (\%)")
    ax4.set_title(r"Gain vs price spread")
    ax4.legend()
    ax4.grid()

    fig.suptitle(rf"{latex_safe(model_name)} : gain of the dynamic model over the whole batch")
    all_entries = [entry for axis in table for W in table[axis] for entry in table[axis][W]]
    add_hyperparams_caption(fig, W_list, all_entries)
    save_figure(fig, f"gain_summary_{model_name}")
    return fig


def print_summary(results):
    """ 
        prints the same summary as plot_summary but as a table, so I have actual numbers to put in the paper in supplement of the moustache plot \n
        This function was generated with Claude AI
    """
    table = gain_table(results)
    W_list = sorted({W for axis in table for W in table[axis]})
    print_out("\n===== Gain of the dynamic model over the static one")
    print_out(f"{'axis':10s} " + " ".join(f"{'W='+str(W):>12s}" for W in W_list))
    for axis in table:
        gains = {W: [entry["gain"] for entry in table[axis].get(W, [])] for W in W_list}
        print_out(f"{axis:10s} " + " ".join(f"{np.mean(gains[W]):11.3f}%" for W in W_list))
    gains = {W: [entry["gain"] for axis in table for entry in table[axis].get(W, [])] for W in W_list}
    print_out(f"{'overall':10s} " + " ".join(f"{np.mean(gains[W]):11.3f}%" for W in W_list))
    print_out(f"{'best case':10s} " + " ".join(f"{np.max(gains[W]):11.3f}%" for W in W_list))
    print_out(f"{'worst case':10s} " + " ".join(f"{np.min(gains[W]):11.3f}%" for W in W_list))


def save_figure(fig, figname = "figure"):
    """ saves the figure as a mere svg file and as an adjustable python pickle file, in simulations_bat/plots """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{figname}.svg", format="svg")
    with open(PLOTS_DIR / f"{figname}.pkl", "wb") as file:
        pickle.dump(fig, file)
    print_out(f"Figure saved as '{figname}.svg' and '{figname}.pkl' in {PLOTS_DIR}")


def show_pkl(figname = "figure"):
    """ reopens one of the pickled figures of the batch, to adjust it without running the simulations again """
    with open(PLOTS_DIR / f"{figname}.pkl", "rb") as file:
        pickle.load(file)
    plt.show()


if __name__ == "__main__":
    prices_df = ensure_prices_csv()
    models = ensure_load_models()   # generates the missing rho models in load_models.json
    cases = build_case_list(prices_df, models=models)

    results = run_batch(cases)

    print_summary(results)
    plot_axis(results, "duration", r"Duration $N$ (time slots)", "Duration")
    plot_axis(results, "date", r"Starting date", "Date")
    plot_axis(results, "hour", r"Starting hour (UTC)", "Starting hour")
    plot_axis(results, "rho", r"Load factor $\rho$", "Load factor")
    plot_summary(results)
    plt.show()

    # show_pkl(f"gain_summary_{BATCH_NAME}")
