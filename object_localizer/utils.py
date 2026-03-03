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