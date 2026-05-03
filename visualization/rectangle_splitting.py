import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm


def plot_rectangle_splitting(
    rs_solver,
    title="Rectangle Splitting Pareto Front",
    obj1_name: str | None = None,
    obj2_name: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in rs_solver.final_rectangles:
        width = r.z2.x - r.z1.x
        height = r.z1.y - r.z2.y
        bottom_left = (r.z1.x, r.z2.y)

        rect_patch = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=1.5,
            edgecolor="gray",
            facecolor="lightgray",
            alpha=0.4,
        )
        ax.add_patch(rect_patch)

    pareto_set = sorted(
        [res.point for res in rs_solver.pareto_set], key=lambda pt: pt.x
    )
    pareto_xs = [p.x for p in pareto_set]
    pareto_ys = [p.y for p in pareto_set]
    ax.step(pareto_xs, pareto_ys, where="post", color="black", alpha=0.8, label="Final Pareto Front")
    ax.scatter([], [], color="white", s=80, edgecolors="black", label="Pareto Optimal Solutions")


    history = rs_solver.points_history
    if not history:
        return
    max_gen = max(gen for _, gen in history)
    cmap = cm.get_cmap("plasma")
    for pt, gen in history:
        color = cmap(gen / max_gen) if max_gen > 0 else cmap(0)
        if pt in pareto_set:
            # specifying zorder to draw points on top of the pareto front line
            ax.scatter(pt.x, pt.y, color=color, s=80, edgecolors="black", zorder=5)
        else:
            ax.scatter(pt.x, pt.y, color=color, s=60, zorder=5)


    if obj1_name:
        ax.set_xlabel(f"Objective 1: {obj1_name}", fontsize=12)
    else: 
        ax.set_xlabel("Objective 1", fontsize=12)

    if obj2_name:
        ax.set_ylabel(f"Objective 2: {obj2_name}", fontsize=12)
    else:
        ax.set_ylabel("Objective 2", fontsize=12)
    
    ax.set_title(title, fontsize=16)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_gen))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Solution Generation", fontsize=12)

    ax.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
