"""Visualize: Separate Outputs (shared path) vs Split Architecture (independent paths)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 16))

# Colors
COLOR_BBOX = '#E74C3C'
COLOR_CLASS = '#2ECC71'
COLOR_OBJ = '#3498DB'
COLOR_SHARED = '#95A5A6'
COLOR_VGG = '#8E44AD'       # purple for VGG
COLOR_MIXED = '#F39C12'     # orange for mixed gradients

def draw_neuron(ax, x, y, color, radius=0.16, label=None, fontsize=7):
    circle = plt.Circle((x, y), radius, color=color, ec='black', linewidth=1.0, zorder=5)
    ax.add_patch(circle)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=6)

def draw_connections(ax, x1_list, y1_list, x2_list, y2_list, color='gray', alpha=0.15, linewidth=0.5):
    for x1, y1 in zip(x1_list, y1_list):
        for x2, y2 in zip(x2_list, y2_list):
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth, zorder=1)

def draw_layer_box(ax, x, y, w, h, color, label, fontsize=8):
    rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                     boxstyle='round,pad=0.1', facecolor=color, edgecolor='black',
                                     linewidth=1.2, alpha=0.3, zorder=0)
    ax.add_patch(rect)
    ax.text(x, y + h/2 + 0.2, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

# ============================================================
# LEFT PANEL: Separate Outputs with SHARED hidden layers
# ============================================================
ax1.set_xlim(-1.5, 10)
ax1.set_ylim(-2, 11)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Separate Outputs\n(Shared Hidden Path)', fontsize=14, fontweight='bold', pad=15)

# --- VGG block ---
vgg_x = 0.5
vgg_ys = [3.0, 4.0, 5.0, 6.0, 7.0]
for y in vgg_ys:
    draw_neuron(ax1, vgg_x, y, COLOR_VGG)
ax1.text(vgg_x, 2.2, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(vgg_x, 7.8, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(vgg_x, 1.3, 'VGG16\n(6×6×512)', ha='center', fontsize=9, fontweight='bold')

# --- Flatten ---
flat_x = 2.5
flat_ys = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
for y in flat_ys:
    draw_neuron(ax1, flat_x, y, COLOR_SHARED)
ax1.text(flat_x, 1.7, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(flat_x, 8.3, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(flat_x, 0.8, 'Flatten\n(18,432)', ha='center', fontsize=9, fontweight='bold')
# VGG → Flatten connections
draw_connections(ax1, [vgg_x]*len(vgg_ys), vgg_ys, [flat_x]*len(flat_ys), flat_ys,
                color=COLOR_VGG, alpha=0.1, linewidth=0.5)

# --- Dense(256) shared ---
d256_x = 4.3
d256_ys = [3.0, 4.0, 5.0, 6.0, 7.0]
for y in d256_ys:
    draw_neuron(ax1, d256_x, y, COLOR_SHARED)
ax1.text(d256_x, 2.2, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(d256_x, 7.8, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(d256_x, 1.3, 'Dense(256)\nshared', ha='center', fontsize=9, fontweight='bold')
# Flatten → Dense(256)
draw_connections(ax1, [flat_x]*len(flat_ys), flat_ys, [d256_x]*len(d256_ys), d256_ys,
                color=COLOR_MIXED, alpha=0.1, linewidth=0.5)

# --- Dense(128) shared ---
d128_x = 6.0
d128_ys = [3.5, 4.5, 5.5, 6.5]
for y in d128_ys:
    draw_neuron(ax1, d128_x, y, COLOR_SHARED)
ax1.text(d128_x, 2.7, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(d128_x, 7.3, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax1.text(d128_x, 1.3, 'Dense(128)\nshared', ha='center', fontsize=9, fontweight='bold')
# Dense(256) → Dense(128)
draw_connections(ax1, [d256_x]*len(d256_ys), d256_ys, [d128_x]*len(d128_ys), d128_ys,
                color=COLOR_MIXED, alpha=0.12, linewidth=0.5)

# --- Output heads ---
out_x = 8.0
bbox_ys = [2.0, 3.0, 4.0, 5.0]
class_ys = [6.5, 7.5, 8.5]
obj_ys = [10.0]

# Connections: all shared → all outputs (mixed gradients)
draw_connections(ax1, [d128_x]*len(d128_ys), d128_ys, [out_x]*len(bbox_ys), bbox_ys,
                color=COLOR_BBOX, alpha=0.15, linewidth=0.6)
draw_connections(ax1, [d128_x]*len(d128_ys), d128_ys, [out_x]*len(class_ys), class_ys,
                color=COLOR_CLASS, alpha=0.15, linewidth=0.6)
draw_connections(ax1, [d128_x]*len(d128_ys), d128_ys, [out_x]*len(obj_ys), obj_ys,
                color=COLOR_OBJ, alpha=0.2, linewidth=0.6)

for y in bbox_ys:
    draw_neuron(ax1, out_x, y, COLOR_BBOX)
for y in class_ys:
    draw_neuron(ax1, out_x, y, COLOR_CLASS)
for y in obj_ys:
    draw_neuron(ax1, out_x, y, COLOR_OBJ)

# Output labels
ax1.text(out_x + 0.4, 3.5, 'bbox (4)\nMSE', va='center', fontsize=8, color=COLOR_BBOX, fontweight='bold')
ax1.text(out_x + 0.4, 7.5, 'class (3)\nSparse CCE', va='center', fontsize=8, color=COLOR_CLASS, fontweight='bold')
ax1.text(out_x + 0.4, 10.0, 'obj (1)\nBCE', va='center', fontsize=8, color=COLOR_OBJ, fontweight='bold')

# Gradient flow annotation
ax1.annotate('', xy=(2.5, 9.5), xytext=(6.0, 9.5),
            arrowprops=dict(arrowstyle='<->', color=COLOR_MIXED, linewidth=2))
ax1.text(4.25, 9.9, 'ALL gradients flow through\nthese shared layers (mixed)', ha='center', fontsize=8,
         color=COLOR_MIXED, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3CD', edgecolor=COLOR_MIXED))

# ============================================================
# RIGHT PANEL: Split Architecture (independent paths)
# ============================================================
ax2.set_xlim(-1.5, 10)
ax2.set_ylim(-2, 11)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Split Architecture\n(Independent Paths from VGG)', fontsize=14, fontweight='bold', pad=15)

# --- VGG block ---
vgg_x2 = 0.5
vgg_ys2 = [3.0, 4.0, 5.0, 6.0, 7.0]
for y in vgg_ys2:
    draw_neuron(ax2, vgg_x2, y, COLOR_VGG)
ax2.text(vgg_x2, 2.2, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax2.text(vgg_x2, 7.8, '...', ha='center', fontsize=14, color='gray', fontweight='bold')
ax2.text(vgg_x2, 1.3, 'VGG16\n(6×6×512)', ha='center', fontsize=9, fontweight='bold')

# ==================== LOCALIZATION PATH (top, red) ====================
# Background box for localization path
loc_bg = mpatches.FancyBboxPatch((1.5, -0.5), 6.8, 6.2,
                                   boxstyle='round,pad=0.2', facecolor=COLOR_BBOX,
                                   edgecolor=COLOR_BBOX, linewidth=2, alpha=0.06, zorder=0)
ax2.add_patch(loc_bg)

# Flatten
flat_x2 = 2.5
flat_ys2 = [1.0, 2.0, 3.0, 4.0]
for y in flat_ys2:
    draw_neuron(ax2, flat_x2, y, COLOR_BBOX, radius=0.14)
ax2.text(flat_x2, 0.2, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(flat_x2, 4.8, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(flat_x2, -0.8, 'Flatten\n(18,432)', ha='center', fontsize=8, fontweight='bold', color=COLOR_BBOX)

# VGG → Flatten
draw_connections(ax2, [vgg_x2]*len(vgg_ys2), vgg_ys2, [flat_x2]*len(flat_ys2), flat_ys2,
                color=COLOR_BBOX, alpha=0.12, linewidth=0.5)

# Dense(256) for bbox
d256_bbox_x = 4.0
d256_bbox_ys = [1.5, 2.5, 3.5]
for y in d256_bbox_ys:
    draw_neuron(ax2, d256_bbox_x, y, COLOR_BBOX, radius=0.14)
ax2.text(d256_bbox_x, 0.7, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(d256_bbox_x, 4.3, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(d256_bbox_x, -0.3, 'Dense(256)', ha='center', fontsize=8, fontweight='bold', color=COLOR_BBOX)
draw_connections(ax2, [flat_x2]*len(flat_ys2), flat_ys2, [d256_bbox_x]*len(d256_bbox_ys), d256_bbox_ys,
                color=COLOR_BBOX, alpha=0.12, linewidth=0.5)

# Dense(128) for bbox
d128_bbox_x = 5.5
d128_bbox_ys = [2.0, 3.0]
for y in d128_bbox_ys:
    draw_neuron(ax2, d128_bbox_x, y, COLOR_BBOX, radius=0.14)
ax2.text(d128_bbox_x, 1.2, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(d128_bbox_x, 3.8, '...', ha='center', fontsize=12, color=COLOR_BBOX, fontweight='bold')
ax2.text(d128_bbox_x, 0.3, 'Dense(128)', ha='center', fontsize=8, fontweight='bold', color=COLOR_BBOX)
draw_connections(ax2, [d256_bbox_x]*len(d256_bbox_ys), d256_bbox_ys,
                [d128_bbox_x]*len(d128_bbox_ys), d128_bbox_ys,
                color=COLOR_BBOX, alpha=0.15, linewidth=0.5)

# bbox output
bbox_out_x = 7.5
bbox_out_ys = [1.5, 2.5, 3.5, 4.5]
for y in bbox_out_ys:
    draw_neuron(ax2, bbox_out_x, y, COLOR_BBOX, radius=0.14)
ax2.text(bbox_out_x + 0.4, 3.0, 'bbox (4)\nMSE', va='center', fontsize=8, color=COLOR_BBOX, fontweight='bold')
draw_connections(ax2, [d128_bbox_x]*len(d128_bbox_ys), d128_bbox_ys,
                [bbox_out_x]*len(bbox_out_ys), bbox_out_ys,
                color=COLOR_BBOX, alpha=0.2, linewidth=0.6)

# ==================== CLASSIFICATION PATH (bottom, green/blue) ====================
# Background box for classification path
cls_bg = mpatches.FancyBboxPatch((1.5, 5.5), 6.8, 5.0,
                                   boxstyle='round,pad=0.2', facecolor=COLOR_CLASS,
                                   edgecolor=COLOR_CLASS, linewidth=2, alpha=0.06, zorder=0)
ax2.add_patch(cls_bg)

# GAP
gap_x = 2.5
gap_ys = [7.0, 8.0]
for y in gap_ys:
    draw_neuron(ax2, gap_x, y, COLOR_CLASS, radius=0.14)
ax2.text(gap_x, 6.2, '...', ha='center', fontsize=12, color=COLOR_CLASS, fontweight='bold')
ax2.text(gap_x, 8.8, '...', ha='center', fontsize=12, color=COLOR_CLASS, fontweight='bold')
ax2.text(gap_x, 9.6, 'GAP\n(512)', ha='center', fontsize=8, fontweight='bold', color=COLOR_CLASS)

# VGG → GAP
draw_connections(ax2, [vgg_x2]*len(vgg_ys2), vgg_ys2, [gap_x]*len(gap_ys), gap_ys,
                color=COLOR_CLASS, alpha=0.12, linewidth=0.5)

# Dense(128) for classification
d128_cls_x = 4.5
d128_cls_ys = [7.0, 8.0]
for y in d128_cls_ys:
    draw_neuron(ax2, d128_cls_x, y, COLOR_CLASS, radius=0.14)
ax2.text(d128_cls_x, 6.2, '...', ha='center', fontsize=12, color=COLOR_CLASS, fontweight='bold')
ax2.text(d128_cls_x, 8.8, '...', ha='center', fontsize=12, color=COLOR_CLASS, fontweight='bold')
ax2.text(d128_cls_x, 9.6, 'Dense(128)', ha='center', fontsize=8, fontweight='bold', color=COLOR_CLASS)
draw_connections(ax2, [gap_x]*len(gap_ys), gap_ys, [d128_cls_x]*len(d128_cls_ys), d128_cls_ys,
                color=COLOR_CLASS, alpha=0.15, linewidth=0.5)

# class output
cls_out_x = 7.0
cls_out_ys = [6.5, 7.5, 8.5]
for y in cls_out_ys:
    draw_neuron(ax2, cls_out_x, y, COLOR_CLASS, radius=0.14)
ax2.text(cls_out_x + 0.4, 7.5, 'class (3)\nSparse CCE', va='center', fontsize=8, color=COLOR_CLASS, fontweight='bold')
draw_connections(ax2, [d128_cls_x]*len(d128_cls_ys), d128_cls_ys,
                [cls_out_x]*len(cls_out_ys), cls_out_ys,
                color=COLOR_CLASS, alpha=0.2, linewidth=0.6)

# objectness output
obj_out_x = 7.0
obj_out_ys = [10.0]
for y in obj_out_ys:
    draw_neuron(ax2, obj_out_x, y, COLOR_OBJ, radius=0.14)
ax2.text(obj_out_x + 0.4, 10.0, 'obj (1)\nBCE', va='center', fontsize=8, color=COLOR_OBJ, fontweight='bold')
draw_connections(ax2, [d128_cls_x]*len(d128_cls_ys), d128_cls_ys,
                [obj_out_x]*len(obj_out_ys), obj_out_ys,
                color=COLOR_OBJ, alpha=0.25, linewidth=0.6)

# Gradient flow annotations
ax2.text(4.5, 5.1, 'MSE gradients ONLY\n(localization path)', ha='center', fontsize=8,
         color=COLOR_BBOX, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC', edgecolor=COLOR_BBOX))

ax2.text(4.5, 10.5, 'CCE + BCE gradients ONLY\n(classification path)', ha='center', fontsize=8,
         color=COLOR_CLASS, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor=COLOR_CLASS))

# VGG shared annotation
ax2.text(0.5, 0.3, 'VGG backbone:\nonly shared weights\n(frozen or fine-tuned)', ha='center', fontsize=8,
         color=COLOR_VGG, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', edgecolor=COLOR_VGG))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_VGG, edgecolor='black', label='VGG16 Backbone (shared)'),
    mpatches.Patch(facecolor=COLOR_SHARED, edgecolor='black', label='Shared Hidden Layers'),
    mpatches.Patch(facecolor=COLOR_BBOX, edgecolor='black', label='Localization Path (Flatten → bbox)'),
    mpatches.Patch(facecolor=COLOR_CLASS, edgecolor='black', label='Classification Path (GAP → class + obj)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('split_architecture_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved to split_architecture_comparison.png")
plt.show()
