import pandas as pd
import seaborn as sns

from src.config import BLD


def load_losses():
    """Load losses from bld folder for each project specification."""
    path = BLD / "evaluations"
    project_names = [p.stem for p in path.glob("*")]
    losses = {name: pd.read_csv(path / name / "losses.csv") for name in project_names}
    return losses


def mae_plot(losses):
    """Make plot from losses data frame."""
    plot = sns.lineplot(x="n_obs", y="mae", hue="model", data=losses)
    fig = plot.get_figure()
    ax = fig.get_axes()[0]
    fig.set_size_inches(11.7, 8.27)
    ax.legend(loc="upper right")
    return fig


def save_fig(path, fig):
    """Save figure to bld path."""
    fig.savefig(path / "mae_plot.png", bbox_inches="tight")


def main():
    losses_dict = load_losses()
    for project_name, losses in losses_dict.items():
        fig = mae_plot(losses)
        path = BLD / "figures" / project_name
        path.mkdir(parents=True, exist_ok=True)
        save_fig(path, fig)


if __name__ == "__main__":
    main()
