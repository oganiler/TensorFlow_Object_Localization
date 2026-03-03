# utils.py - shared utility functions
from .imports import get_np, get_plt

def simple_plot(title="Simple Plot"):
    """Create and show a simple sine plot."""
    np = get_np()
    plt = get_plt()
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='sin(x)', color='tab:blue')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('simple_plot.png', dpi=150)
    plt.show()

def plot_training_history(history):
    """Plot loss and accuracy from a Keras training history."""
    plt = get_plt()

    metrics = [k for k in history.history.keys() if not k.startswith('val_')]
    num_plots = len(metrics)

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.plot(history.history[metric], label=f'train {metric}')
        if f'val_{metric}' in history.history:
            ax.plot(history.history[f'val_{metric}'], label=f'val {metric}')
        ax.set_title(metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()