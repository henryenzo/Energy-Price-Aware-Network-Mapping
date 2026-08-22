import matplotlib.pyplot as plt, pickle, glob
plt.rcParams["text.usetex"] = True

for p in glob.glob("plots/simulations/cost_curve_W_nobel-eu-8SFC_N24_*.pkl"):
    fig = pickle.load(open(p, "rb"))
    fig.set_size_inches(14, 9)
    fig.set_layout_engine("constrained")
    fig.savefig(p.replace(".pkl", ".svg"), format="svg")