def set_default_configs(plt, seaborn=None):
    # assuming in the figsize height = 5
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = plt.rcParams['xtick.labelsize']
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.25
    if seaborn is not None:
        seaborn.set_style("whitegrid")
