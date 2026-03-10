"""Visualize the difference between Concatenate and Separate Outputs architectures."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Colors
COLOR_BBOX = '#E74C3C'      # red
COLOR_CLASS = '#2ECC71'      # green
COLOR_OBJ = '#3498DB'        # blue
COLOR_SHARED = '#95A5A6'     # gray
COLOR_CONCAT = '#F39C12'     # orange
COLOR_BG = '#FAFAFA'

def draw_neuron(ax, x, y, color, radius=0.18, label=None, fontsize=8):
    circle = plt.Circle((x, y), radius, color=color, ec='black', linewidth=1.2, zorder=5)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=6)

def draw_connections(ax, x1_list, y1_list, x2_list, y2_list, color='gray', alpha=0.15, linewidth=0.5):
    for x1, y1 in zip(x1_list, y1_list):
        for x2, y2 in zip(x2_list, y2_list):
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, zorder=1)

def draw_layer_bracket(ax, x, y_top, y_bottom, label, side='left'):
    """Draw a bracket with label on the side of a layer."""
    offset = -0.6 if side == 'left' else 0.6
    ha = 'right' if side == 'left' else 'left'
    ax.annotate('', xy=(x + offset, y_top + 0.15), xytext=(x + offset, y_bottom - 0.15),
                arrowprops=dict(arrowstyle='-', color='black', linewidth=1.5))
    ax.text(x + offset - (0.15 if side == 'left' else -0.15),
            (y_top + y_bottom) / 2, label,
            ha=ha, va='center', fontsize=9, fontstyle='italic')

# ============================================================
# LEFT PANEL: Concatenate approach
# ============================================================
ax1.set_xlim(-2, 8)
ax1.set_ylim(-1, 9)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('With Concatenate\n(Single Output Layer)', fontsize=14, fontweight='bold', pad=15)

# Dense(128) shared layer — show 6 representative neurons
shared_x = 1.5
shared_neurons_y = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
shared_neurons_x = [shared_x] * len(shared_neurons_y)

for y in shared_neurons_y:
    draw_neuron(ax1, shared_x, y, COLOR_SHARED)

# Dots to indicate more neurons
ax1.text(shared_x, 0.7, '...', ha='center', va='center', fontsize=16, fontweight='bold', color='gray')
ax1.text(shared_x, 7.3, '...', ha='center', va='center', fontsize=16, fontweight='bold', color='gray')
ax1.text(shared_x, -0.1, 'Dense(128)\nshared', ha='center', va='center', fontsize=10, fontweight='bold')

# Single Dense(8) output layer
output_x = 5.5
# bbox outputs (4)
bbox_ys = [1.0, 2.0, 3.0, 4.0]
# class outputs (3)
class_ys = [5.0, 6.0, 7.0]
# objectness output (1)
obj_ys = [8.0]

all_output_ys = bbox_ys + class_ys + obj_ys
all_output_xs = [output_x] * len(all_output_ys)

# Draw ALL connections from ALL shared neurons to ALL outputs (same gray - one weight matrix)
draw_connections(ax1, shared_neurons_x, shared_neurons_y, all_output_xs, all_output_ys,
                color='#555555', alpha=0.12, linewidth=0.7)

# Draw output neurons
for y in bbox_ys:
    draw_neuron(ax1, output_x, y, COLOR_BBOX)
for y in class_ys:
    draw_neuron(ax1, output_x, y, COLOR_CLASS)
for y in obj_ys:
    draw_neuron(ax1, output_x, y, COLOR_OBJ)

# Labels for outputs
ax1.text(output_x + 0.5, bbox_ys[0], ' bbox[0]', va='center', fontsize=8, color=COLOR_BBOX)
ax1.text(output_x + 0.5, bbox_ys[1], ' bbox[1]', va='center', fontsize=8, color=COLOR_BBOX)
ax1.text(output_x + 0.5, bbox_ys[2], ' bbox[2]', va='center', fontsize=8, color=COLOR_BBOX)
ax1.text(output_x + 0.5, bbox_ys[3], ' bbox[3]', va='center', fontsize=8, color=COLOR_BBOX)
ax1.text(output_x + 0.5, class_ys[0], ' class[0]', va='center', fontsize=8, color=COLOR_CLASS)
ax1.text(output_x + 0.5, class_ys[1], ' class[1]', va='center', fontsize=8, color=COLOR_CLASS)
ax1.text(output_x + 0.5, class_ys[2], ' class[2]', va='center', fontsize=8, color=COLOR_CLASS)
ax1.text(output_x + 0.5, obj_ys[0], ' obj', va='center', fontsize=8, color=COLOR_OBJ)

# Weight matrix label
ax1.text(3.5, 0.0, 'Single weight matrix\n128 × 8 = 1,024 weights\n(all outputs mixed)',
         ha='center', va='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3CD', edgecolor='#F0AD4E'))

ax1.text(output_x, -0.6, 'Dense(8)\nCustom Loss', ha='center', va='center', fontsize=10, fontweight='bold')

# ============================================================
# RIGHT PANEL: Separate Outputs approach
# ============================================================
ax2.set_xlim(-2, 8)
ax2.set_ylim(-1, 9)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('With Separate Outputs\n(Independent Output Layers)', fontsize=14, fontweight='bold', pad=15)

# Dense(128) shared layer — same position
shared_x2 = 1.5
for y in shared_neurons_y:
    draw_neuron(ax2, shared_x2, y, COLOR_SHARED)

ax2.text(shared_x2, 0.7, '...', ha='center', va='center', fontsize=16, fontweight='bold', color='gray')
ax2.text(shared_x2, 7.3, '...', ha='center', va='center', fontsize=16, fontweight='bold', color='gray')
ax2.text(shared_x2, -0.1, 'Dense(128)\nshared', ha='center', va='center', fontsize=10, fontweight='bold')

# Three separate output layers
output_x2 = 5.5

# bbox outputs (4) - separate Dense(4)
bbox_ys2 = [1.0, 2.0, 3.0, 4.0]
# class outputs (3) - separate Dense(3)
class_ys2 = [5.0, 6.0, 7.0]
# objectness output (1) - separate Dense(1)
obj_ys2 = [8.0]

# Draw connections with DIFFERENT colors for each head
draw_connections(ax2, [shared_x2]*len(shared_neurons_y), shared_neurons_y,
                [output_x2]*len(bbox_ys2), bbox_ys2,
                color=COLOR_BBOX, alpha=0.2, linewidth=0.7)

draw_connections(ax2, [shared_x2]*len(shared_neurons_y), shared_neurons_y,
                [output_x2]*len(class_ys2), class_ys2,
                color=COLOR_CLASS, alpha=0.2, linewidth=0.7)

draw_connections(ax2, [shared_x2]*len(shared_neurons_y), shared_neurons_y,
                [output_x2]*len(obj_ys2), obj_ys2,
                color=COLOR_OBJ, alpha=0.25, linewidth=0.7)

# Draw output neurons
for y in bbox_ys2:
    draw_neuron(ax2, output_x2, y, COLOR_BBOX)
for y in class_ys2:
    draw_neuron(ax2, output_x2, y, COLOR_CLASS)
for y in obj_ys2:
    draw_neuron(ax2, output_x2, y, COLOR_OBJ)

# Labels for outputs
ax2.text(output_x2 + 0.5, bbox_ys2[0], ' bbox[0]', va='center', fontsize=8, color=COLOR_BBOX)
ax2.text(output_x2 + 0.5, bbox_ys2[1], ' bbox[1]', va='center', fontsize=8, color=COLOR_BBOX)
ax2.text(output_x2 + 0.5, bbox_ys2[2], ' bbox[2]', va='center', fontsize=8, color=COLOR_BBOX)
ax2.text(output_x2 + 0.5, bbox_ys2[3], ' bbox[3]', va='center', fontsize=8, color=COLOR_BBOX)
ax2.text(output_x2 + 0.5, class_ys2[0], ' class[0]', va='center', fontsize=8, color=COLOR_CLASS)
ax2.text(output_x2 + 0.5, class_ys2[1], ' class[1]', va='center', fontsize=8, color=COLOR_CLASS)
ax2.text(output_x2 + 0.5, class_ys2[2], ' class[2]', va='center', fontsize=8, color=COLOR_CLASS)
ax2.text(output_x2 + 0.5, obj_ys2[0], ' obj', va='center', fontsize=8, color=COLOR_OBJ)

# Separator lines between output groups
ax2.plot([output_x2 - 0.4, output_x2 + 0.4], [4.5, 4.5], 'k--', linewidth=0.8, alpha=0.4)
ax2.plot([output_x2 - 0.4, output_x2 + 0.4], [7.5, 7.5], 'k--', linewidth=0.8, alpha=0.4)

# Layer labels for each head
ax2.text(output_x2 + 1.8, 2.5, 'Dense(4)\nsigmoid\nMSE loss',
         ha='center', va='center', fontsize=8, fontweight='bold', color=COLOR_BBOX,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor=COLOR_BBOX))
ax2.text(output_x2 + 1.8, 6.0, 'Dense(3)\nsoftmax\nSparse CCE',
         ha='center', va='center', fontsize=8, fontweight='bold', color=COLOR_CLASS,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor=COLOR_CLASS))
ax2.text(output_x2 + 1.8, 8.0, 'Dense(1)\nsigmoid\nBCE loss',
         ha='center', va='center', fontsize=8, fontweight='bold', color=COLOR_OBJ,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF5FB', edgecolor=COLOR_OBJ))

# Weight matrix labels
ax2.text(3.5, 0.0, '3 separate weight matrices\n128×4 + 128×3 + 128×1\n= 512 + 384 + 128 = 1,024 weights\n(same total, but independent)',
         ha='center', va='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#D5F5E3', edgecolor='#27AE60'))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_BBOX, edgecolor='black', label='Bounding Box (4)'),
    mpatches.Patch(facecolor=COLOR_CLASS, edgecolor='black', label='Class (3)'),
    mpatches.Patch(facecolor=COLOR_OBJ, edgecolor='black', label='Objectness (1)'),
    mpatches.Patch(facecolor=COLOR_SHARED, edgecolor='black', label='Shared Layer'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('architecture_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved to architecture_comparison.png")
plt.show()
