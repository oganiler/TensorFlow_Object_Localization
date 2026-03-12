"""Visualize the SSD-style multi-scale architecture with VGG16 backbone.

Shows how block3_pool, block4_pool, block5_pool feature maps each feed
independent prediction heads, then concatenate into unified outputs.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(24, 14))
ax.set_xlim(-2.5, 22)
ax.set_ylim(-2.5, 14)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('SSD-Style Multi-Scale Architecture\n'
             'VGG16 Backbone with 3 Feature Pyramid Levels',
             fontsize=16, fontweight='bold', pad=20)

# ===================== Colors =====================
COLOR_VGG = '#8E44AD'
COLOR_SCALE3 = '#E67E22'    # orange — block3 (small objects, fine grid)
COLOR_SCALE4 = '#16A085'    # teal — block4 (medium objects)
COLOR_SCALE5 = '#2C3E50'    # dark blue — block5 (large objects, coarse grid)
COLOR_BBOX = '#E74C3C'
COLOR_CLASS = '#2ECC71'
COLOR_OBJ = '#3498DB'
COLOR_CONCAT = '#F39C12'

# Scale metadata
SCALES = [
    {'color': COLOR_SCALE3, 'name': 'block3_pool', 'grid': '25x25',
     'ch': '256', 'stride': 'stride 8', 'slots': '625', 'detect': 'small objects'},
    {'color': COLOR_SCALE4, 'name': 'block4_pool', 'grid': '12x12',
     'ch': '512', 'stride': 'stride 16', 'slots': '144', 'detect': 'medium objects'},
    {'color': COLOR_SCALE5, 'name': 'block5_pool', 'grid': '6x6',
     'ch': '512', 'stride': 'stride 32', 'slots': '36', 'detect': 'large objects'},
]

# ===================== Helpers =====================
def draw_box(ax, x, y, w, h, color, text, fontsize=8, alpha=0.25, text_color=None):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.1', facecolor=color, edgecolor='black',
        linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color or 'black', zorder=6)

def draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.5, style='->', rad=0):
    cs = f'arc3,rad={rad}' if rad else 'arc3,rad=0'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                linewidth=lw, connectionstyle=cs,
                                shrinkA=4, shrinkB=4),
                zorder=4)

# ===================== Layout: Y positions (top to bottom) =====================
# VGG flows top-to-bottom. Branch points at block3/4/5 go RIGHT to scale paths.
# This eliminates crossing arrows since VGG order matches path order.

y_input = 13.0
y_block1 = 11.8
y_block2 = 10.6
y_block3 = 9.0     # <-- branches right to scale path 1
y_block4 = 5.8     # <-- branches right to scale path 2
y_block5 = 2.6     # <-- branches right to scale path 3

SCALE_Y = [y_block3, y_block4, y_block5]

# X positions for the pipeline stages
x_vgg = 0.0
x_feat = 3.5       # feature map label box
x_conv1 = 6.5      # Conv2D(256)
x_conv2 = 9.5      # Conv2D(128)
x_heads = 12.5     # 1x1 conv heads
x_reshape = 15.5   # per-scale output shapes
x_concat = 19.5    # final concatenation

# ===================== VGG16 Backbone =====================
# Background rectangle
vgg_rect = mpatches.FancyBboxPatch(
    (x_vgg - 1.0, y_block5 - 1.5), 2.0, y_input - y_block5 + 3.0,
    boxstyle='round,pad=0.15', facecolor=COLOR_VGG, edgecolor=COLOR_VGG,
    linewidth=2, alpha=0.08, zorder=0)
ax.add_patch(vgg_rect)

# Title
ax.text(x_vgg, y_input + 1.0, 'VGG16 Backbone', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=COLOR_VGG)

# VGG blocks (top to bottom)
vgg_blocks = [
    (y_input,  'Input\n200x200x3'),
    (y_block1, 'block1_pool\n100x100'),
    (y_block2, 'block2_pool\n50x50'),
    (y_block3, 'block3_pool\n25x25x256'),
    (y_block4, 'block4_pool\n12x12x512'),
    (y_block5, 'block5_pool\n6x6x512'),
]

for by, blabel in vgg_blocks:
    draw_box(ax, x_vgg, by, 1.8, 1.0, COLOR_VGG, blabel,
             fontsize=7, alpha=0.3, text_color=COLOR_VGG)

# Vertical flow arrows between VGG blocks
for i in range(len(vgg_blocks) - 1):
    y1 = vgg_blocks[i][0] - 0.55
    y2 = vgg_blocks[i+1][0] + 0.55
    draw_arrow(ax, x_vgg, y1, x_vgg, y2, color=COLOR_VGG, lw=1.5)

# ===================== Per-scale paths =====================
for i, (path_y, s) in enumerate(zip(SCALE_Y, SCALES)):
    sc = s['color']

    # Background band for this scale's path
    band = mpatches.FancyBboxPatch(
        (x_feat - 1.2, path_y - 1.5), x_reshape - x_feat + 4.0, 3.0,
        boxstyle='round,pad=0.15', facecolor=sc, edgecolor=sc,
        linewidth=1.5, alpha=0.06, zorder=0)
    ax.add_patch(band)

    # --- Branch arrow from VGG block to feature map ---
    draw_arrow(ax, x_vgg + 1.0, path_y, x_feat - 0.9, path_y,
               color=sc, lw=2.5)

    # --- Feature map box ---
    draw_box(ax, x_feat, path_y, 1.6, 1.6, sc,
             f'{s["name"]}\n({s["grid"]}x{s["ch"]})\n{s["stride"]}',
             fontsize=7, alpha=0.3, text_color=sc)

    # --- Conv2D(256) + BN ---
    draw_box(ax, x_conv1, path_y, 1.4, 1.4, sc,
             'Conv2D(256)\n3x3 + BN\n+ Dropout',
             fontsize=7, alpha=0.2, text_color=sc)
    draw_arrow(ax, x_feat + 0.9, path_y, x_conv1 - 0.8, path_y,
               color=sc, lw=1.5)

    # --- Conv2D(128) + BN ---
    draw_box(ax, x_conv2, path_y, 1.4, 1.4, sc,
             'Conv2D(128)\n3x3 + BN\n+ Dropout',
             fontsize=7, alpha=0.2, text_color=sc)
    draw_arrow(ax, x_conv1 + 0.8, path_y, x_conv2 - 0.8, path_y,
               color=sc, lw=1.5)

    # --- 1x1 Conv prediction heads (3 neurons: bbox, class, obj) ---
    head_dy = 0.7
    heads = [
        (path_y + head_dy, COLOR_BBOX, 'bbox'),
        (path_y,           COLOR_CLASS, 'cls'),
        (path_y - head_dy, COLOR_OBJ,  'obj'),
    ]
    for hy, hc, hlabel in heads:
        circle = plt.Circle((x_heads, hy), 0.25, color=hc, ec='black',
                             linewidth=1.0, zorder=5)
        ax.add_patch(circle)
        ax.text(x_heads, hy, hlabel, ha='center', va='center',
                fontsize=6, fontweight='bold', color='white', zorder=6)

    # Arrow: conv2 -> heads (fan out)
    for hy, _, _ in heads:
        draw_arrow(ax, x_conv2 + 0.8, path_y, x_heads - 0.3, hy,
                   color=sc, lw=1.0)

    # Label under heads
    ax.text(x_heads, path_y - 1.3, '1x1 Conv', ha='center',
            fontsize=7, fontweight='bold', color='gray')

    # --- Per-scale Reshape output labels ---
    output_labels = [
        (path_y + head_dy, COLOR_BBOX, f'bbox ({s["grid"]}, 4)'),
        (path_y,           COLOR_CLASS, f'cls  ({s["grid"]}, 3)'),
        (path_y - head_dy, COLOR_OBJ,  f'obj  ({s["grid"]}, 1)'),
    ]
    for oy, oc, olabel in output_labels:
        draw_arrow(ax, x_heads + 0.3, oy, x_reshape - 0.7, oy,
                   color=oc, lw=1.0)
        ax.text(x_reshape, oy, olabel, ha='center', va='center',
                fontsize=7, fontweight='bold', color=oc,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=oc, alpha=0.9))

    # --- Scale info label on far right of band ---
    ax.text(x_reshape + 2.2, path_y, f'{s["slots"]} slots\n{s["detect"]}',
            ha='center', va='center', fontsize=9, fontweight='bold', color=sc,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=sc, alpha=0.9))

# ===================== Concatenation column =====================
concat_rect = mpatches.FancyBboxPatch(
    (x_concat - 0.9, y_block5 - 1.5), 1.8, y_block3 - y_block5 + 3.0,
    boxstyle='round,pad=0.15', facecolor=COLOR_CONCAT, edgecolor=COLOR_CONCAT,
    linewidth=2, alpha=0.1, zorder=0)
ax.add_patch(concat_rect)
ax.text(x_concat, y_block3 + 1.8, 'Concatenate\n(axis=1)', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=COLOR_CONCAT)

# Final output boxes inside concat column
final_outputs = [
    (y_block3, COLOR_BBOX,  'bbox_output\n(805, 4)'),
    (y_block4, COLOR_CLASS, 'class_output\n(805, 3)'),
    (y_block5, COLOR_OBJ,  'objectness_output\n(805, 1)'),
]
for fy, fc, flabel in final_outputs:
    draw_box(ax, x_concat, fy, 1.5, 1.2, fc, flabel,
             fontsize=8, alpha=0.5, text_color=fc)

# Arrows from each scale's reshape outputs to concat
for path_y in SCALE_Y:
    for dy, col in [(0.7, COLOR_BBOX), (0.0, COLOR_CLASS), (-0.7, COLOR_OBJ)]:
        # Find target Y in concat column (bbox->top, class->mid, obj->bottom)
        if dy > 0:
            target_y = y_block3
        elif dy == 0:
            target_y = y_block4
        else:
            target_y = y_block5
        ax.annotate('',
                    xy=(x_concat - 0.9, target_y),
                    xytext=(x_reshape + 0.7, path_y + dy),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    linewidth=0.8, alpha=0.4,
                                    connectionstyle='arc3,rad=0'),
                    zorder=3)

# ===================== Legend =====================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_VGG, edgecolor='black', alpha=0.5,
                   label='VGG16 Backbone (shared)'),
    mpatches.Patch(facecolor=COLOR_SCALE3, edgecolor='black', alpha=0.5,
                   label='block3_pool (25x25, stride 8)'),
    mpatches.Patch(facecolor=COLOR_SCALE4, edgecolor='black', alpha=0.5,
                   label='block4_pool (12x12, stride 16)'),
    mpatches.Patch(facecolor=COLOR_SCALE5, edgecolor='black', alpha=0.5,
                   label='block5_pool (6x6, stride 32)'),
    mpatches.Patch(facecolor=COLOR_BBOX, edgecolor='black', alpha=0.7,
                   label='Bounding Box (MSE)'),
    mpatches.Patch(facecolor=COLOR_CLASS, edgecolor='black', alpha=0.7,
                   label='Classification (Sparse CCE)'),
    mpatches.Patch(facecolor=COLOR_OBJ, edgecolor='black', alpha=0.7,
                   label='Objectness (BCE)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
           frameon=True, fancybox=True, shadow=True)

# ===================== Total slots annotation =====================
ax.text(10.5, -1.5,
        'Total: 625 + 144 + 36 = 805 anchor slots  |  '
        '3 independent prediction heads  |  '
        'Cross-scale NMS at inference',
        ha='center', va='center', fontsize=10, fontweight='bold',
        color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F9FA',
                  edgecolor='#CCC', linewidth=1))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('multiscale_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("Saved to multiscale_architecture.png")
plt.show()
