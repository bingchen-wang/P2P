import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import prince
from collections import defaultdict


def get_singular_value_spectrum(matrix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Return singular values and normalized squared values from binary matrix.
    Rows = questions, Columns = agents.
    """
    matrix = matrix.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any').dropna(axis=1, how='any')
    X = matrix.values
    X_centered = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    s_squared = s ** 2
    probs = s_squared / s_squared.sum()
    return s, probs

def compute_effective_rank(matrix: pd.DataFrame) -> float:
    """
    Compute entropy-based effective rank using singular value spectrum.
    """
    _, probs = get_singular_value_spectrum(matrix)
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return np.exp(entropy)

def plot_singular_value_spectrum(
    s: np.ndarray,
    probs: np.ndarray,
    effective_rank: float = None,
    cumulative: bool = True,
    save_path: str = None,
    title_prefix: str = ''
):
    """
    Plot the singular value spectrum and optionally the cumulative explained variance,
    side by side if cumulative=True.

    Args:
        s: Array of singular values.
        probs: Normalized squared singular values (explained variance proportions).
        effective_rank: Optional float, to annotate vertical cutoff line.
        cumulative: Whether to plot cumulative explained variance (default: True).
        save_path: If provided, saves plot(s) to this path (adds suffix).
        title_prefix: String prefix for plot titles (e.g., batch name).
    """
    with sns.axes_style("whitegrid", rc={"grid.linestyle": "--", "grid.linewidth": 0.3, "grid.alpha": 1}), \
         sns.plotting_context("notebook", font_scale=1.15):

        if cumulative:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
            axes = [axes]

        # Plot singular values
        ax = axes[0]
        ax.plot(np.arange(1, len(s) + 1), s, marker='o', color='tab:blue',
                label='Singular values', alpha=0.5, markersize=5)
        if effective_rank:
            ax.axvline(effective_rank, color='red', linestyle='--',
                       label=f'Effective rank ≈ {effective_rank:.2f}')
        ax.set_title(f'{title_prefix}Singular Value Spectrum')
        ax.set_xlabel('Component index')
        ax.set_ylabel('Singular value')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Plot cumulative explained variance
        if cumulative:
            ax = axes[1]
            ax.plot(np.arange(1, len(probs) + 1), np.cumsum(probs),
                    marker='o', color='tab:green', alpha=0.5, markersize=5)
            if effective_rank:
                ax.axvline(effective_rank, color='red', linestyle='--',
                           label=f'Effective rank ≈ {effective_rank:.2f}')
            ax.axhline(0.9, color='gray', linestyle=':', label='90% variance')
            ax.set_title(f'{title_prefix}Cumulative Explained Variance')
            ax.set_xlabel('Component index')
            ax.set_ylabel('Cumulative variance proportion')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig
    
def get_template_from_mode(mode_tuple, templates):
    """Infer template type from a mode tuple using heuristic logic with safe access to templates."""
    keys = list(mode_tuple)

    # Direct matches
    if keys == ['core']:
        return 'core'
    elif keys == ['survey']:
        return 'survey'

    # Optional sections
    question_keys = set(templates.get('question', []))
    thematic_keys = set(templates.get('thematic', []))
    theoretical_keys = set(templates.get('theoretical', []))

    # Question patch logic
    if len(keys) == 1 and keys[0] in question_keys:
        return 'pure_question_patch'
    elif len(keys) > 1 and any(k in question_keys for k in keys):
        return 'mixed_patch'

    # Core + thematic or theoretical
    if len(keys) == 2 and 'core' in keys:
        other = [k for k in keys if k != 'core'][0]
        if other in thematic_keys:
            return 'thematic'
        elif other in theoretical_keys:
            return 'theoretical'

    return 'unknown'

def assign_template_color_palette(modes_by_template):
    """
    Assign visually distinct categorical colors to modes grouped by template.

    Args:
        modes_by_template (dict): {template_name: list of mode tuples}

    Returns:
        dict: {mode: hex color}
    """
    base_palettes = {
        'core': "Greys",
        'thematic': "Blues",
        'theoretical': "Purples",
        'survey': "Reds",
        'pure_question_patch': "Oranges",
        'mixed_patch': "Greens",
        'unknown': "pastel"
    }

    color_mapping = {}

    for template, modes in modes_by_template.items():
        n = len(modes)
        palette_name = base_palettes.get(template, "tab20")

        # Fall back to 'husl' for larger mode groups to ensure contrast
        if n <= 10:
            palette = sns.color_palette(palette_name, n_colors=n)
        else:
            palette = sns.color_palette("husl", n_colors=n)

        for mode, color in zip(sorted(modes), palette):
            color_mapping[mode] = mcolors.to_hex(color)

    return color_mapping

def compute_explained_inertia(mca) -> list:
    """
    Computes the proportion of inertia explained by each component in an MCA object.

    Args:
        mca (prince.MCA): A fitted MCA instance.

    Returns:
        list: A list of explained inertia values (as proportions) per component.
    """
    if not hasattr(mca, 'eigenvalues_'):
        raise AttributeError("MCA object has no 'eigenvalues_'. Make sure it's fitted properly.")

    eigenvalues = mca.eigenvalues_
    total_inertia = sum(eigenvalues)
    if total_inertia == 0:
        raise ValueError("Total inertia is zero. Cannot compute proportions.")
    
    return [eig / total_inertia for eig in eigenvalues]

def plot_mca_agents_by_template_grouped(
    df: pd.DataFrame,
    endowments,
    attribute_bank: dict,
    title: str = "MCA Projection of Agent Response Patterns",
    save_path: str = None,
    figsize: tuple = (10, 6),
    seed = 101
):
    """
    Plots agents in MCA space, color-coded by mode and grouped by template.

    Args:
        df: question × agent DataFrame (categorical or binary)
        endowments: EndowmentManager with .get_mode_by_eid(eid)
        attribute_bank: Parsed YAML dictionary with 'templates' key
        title: Title of the plot
        save_path: Optional path to save the plot
        figsize: Plot size
        seed: random_state
    """
    df_agents = df.T.astype(str)
    mca = prince.MCA(n_components=2, random_state=seed)
    mca_coords = mca.fit_transform(df_agents)
    explained_inertia = compute_explained_inertia(mca)

    eids = list(df.columns)
    modes = [endowments.get_mode_by_eid(eid) for eid in eids]
    templates = attribute_bank["templates"]
    mode_template_map = {mode: get_template_from_mode(mode, templates) for mode in set(modes)}

    # Group modes by template
    modes_by_template = defaultdict(list)
    for mode in set(modes):
        template = mode_template_map[mode]
        modes_by_template[template].append(mode)

    # Assign colors
    color_mapping = assign_template_color_palette(modes_by_template)

    # Assign markers
    template_markers = {
        'core': 'o',
        'thematic': 's',
        'theoretical': '^',
        'survey': 'v',
        'pure_question_patch': 'D',
        'mixed_patch': 'P',
        'unknown': 'X',
    }

    # Assemble plot data
    plot_df = mca_coords.copy()
    plot_df.columns = ["MCA 1", "MCA 2"]
    plot_df["eid"] = eids
    plot_df["mode"] = modes
    plot_df["template"] = [mode_template_map[mode] for mode in modes]
    plot_df["color"] = [color_mapping[mode] for mode in modes]

    # Plot
    plt.figure(figsize=figsize)
    for template in sorted(modes_by_template.keys()):
        marker = template_markers.get(template, 'o')
        for mode in sorted(modes_by_template[template]):
            subset = plot_df[plot_df["mode"] == mode]
            label = " + ".join(mode)
            plt.scatter(subset["MCA 1"], subset["MCA 2"], color=color_mapping[mode],
                        alpha=0.6, edgecolors='none', s=30, marker=marker, label=label)

    # Format
    plt.title(title)
    plt.xlabel(f"MCA Component 1 ({explained_inertia[0]*100:.1f}% inertia)")
    plt.ylabel(f"MCA Component 2 ({explained_inertia[1]*100:.1f}% inertia)")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Legend grouped by template
    handles, labels = plt.gca().get_legend_handles_labels()
    grouped = defaultdict(list)
    for h, l in zip(handles, labels):
        matching_modes = [m for m in mode_template_map if " + ".join(m) == l]
        if matching_modes:
            template = mode_template_map[matching_modes[0]]
            grouped[template].append((l, h))
    legend_items = []

    ordered_templates = ['core', 'thematic', 'theoretical', 'survey', 'pure_question_patch', 'mixed_patch', 'unknown']
    for template in ordered_templates:
        if template not in grouped:
            continue
        legend_items.append(plt.Line2D([0], [0], linestyle="none", label = rf"$\mathbf{{{template.replace('_', r'\,').upper()}}}$", color='black'))
        for label, handle in grouped[template]:
            legend_items.append(handle)
    plt.legend(legend_items, [l.get_label() if hasattr(l, 'get_label') else l for l in legend_items],
               loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    plt.tight_layout()
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return plot_df, explained_inertia

def plot_mca_agents_by_template_grouped_simulation(
    df: pd.DataFrame,
    endowments,
    attribute_bank: dict,
    title: str = "MCA Projection of Agent Response Patterns",
    save_path: str = None,
    figsize: tuple = (10, 6),
    seed = 101
):
    """
    Plots agents in MCA space, color-coded by mode and grouped by template.
    Ground truth agents are highlighted with black edges and larger size.

    Args:
        df: question × agent DataFrame (categorical or binary)
        endowments: EndowmentManager with .get_mode_by_eid(eid) and .get_ground_truth_eids()
        attribute_bank: Parsed YAML dictionary with 'templates' key
        title: Title of the plot
        save_path: Optional path to save the plot
        figsize: Plot size
        seed: random_state
    """
    # 1. MCA projection
    df_agents = df.T.astype(str)
    mca = prince.MCA(n_components=2, random_state=seed)
    mca_coords = mca.fit_transform(df_agents)
    explained_inertia = compute_explained_inertia(mca)

    # 2. Mode and template extraction
    eids = list(df.columns)
    modes = [endowments.get_mode_by_eid(eid) for eid in eids]
    templates = attribute_bank["templates"]
    mode_template_map = {mode: get_template_from_mode(mode, templates) for mode in set(modes)}

    # 3. Group modes by template
    modes_by_template = defaultdict(list)
    for mode in set(modes):
        template = mode_template_map[mode]
        modes_by_template[template].append(mode)

    # 4. Assign color and marker by template
    color_mapping = assign_template_color_palette(modes_by_template)
    template_markers = {
        'core': 'o',
        'thematic': 's',
        'theoretical': '^',
        'survey': 'v',
        'pure_question_patch': 'D',
        'mixed_patch': 'P',
        'unknown': 'X',
    }

    # 5. Build plot data
    plot_df = mca_coords.copy()
    plot_df.columns = ["MCA 1", "MCA 2"]
    plot_df["eid"] = eids
    plot_df["mode"] = modes
    plot_df["template"] = [mode_template_map[mode] for mode in modes]
    plot_df["color"] = [color_mapping[mode] for mode in modes]

    # Identify ground truth agents
    ground_truth_eids = set(endowments.get_eids_by_role('ground_truth'))
    plot_df["is_ground_truth"] = plot_df["eid"].isin(ground_truth_eids)

    # 6. Plot
    plt.figure(figsize=figsize)
    for template in sorted(modes_by_template.keys()):
        marker = template_markers.get(template, 'o')
        for mode in sorted(modes_by_template[template]):
            subset = plot_df[plot_df["mode"] == mode]
            label = " + ".join(mode)

            # Plot proxy agents
            subset_proxy = subset[~subset["is_ground_truth"]]
            plt.scatter(subset_proxy["MCA 1"], subset_proxy["MCA 2"],
                        color=color_mapping[mode], alpha=0.6, edgecolors='none',
                        s=30, marker=marker, label=label)

            # Plot ground truth agents (highlighted)
            subset_gt = subset[subset["is_ground_truth"]]
            plt.scatter(subset_gt["MCA 1"], subset_gt["MCA 2"],
                        color=color_mapping[mode], alpha=0.9, edgecolors='black',
                        linewidths=1.2, s=60, marker=marker)

    # 7. Format axes
    plt.title(title)
    plt.xlabel(f"MCA Component 1 ({explained_inertia[0]*100:.1f}% inertia)")
    plt.ylabel(f"MCA Component 2 ({explained_inertia[1]*100:.1f}% inertia)")
    plt.grid(True, linestyle="--", alpha=0.5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 8. Legend grouped by template
    handles, labels = plt.gca().get_legend_handles_labels()
    grouped = defaultdict(list)
    for h, l in zip(handles, labels):
        matching_modes = [m for m in mode_template_map if " + ".join(m) == l]
        if matching_modes:
            template = mode_template_map[matching_modes[0]]
            grouped[template].append((l, h))

    legend_items = []

    # Optional: add legend entry for ground truth
    legend_items.append(plt.Line2D([0], [0], marker='o', linestyle='none',
                                   label='Ground Truth Agent',
                                   markerfacecolor='gray', markeredgecolor='black',
                                   markersize=8, markeredgewidth=1.2))

    ordered_templates = ['core', 'thematic', 'theoretical', 'survey', 'pure_question_patch', 'mixed_patch', 'unknown']
    for template in ordered_templates:
        if template not in grouped:
            continue
        # Add group label
        legend_items.append(plt.Line2D([0], [0], linestyle="none", label=rf"$\mathbf{{{template.replace('_', r'\,').upper()}}}$", color='black'))
        for label, handle in grouped[template]:
            legend_items.append(handle)

    plt.legend(legend_items,
               [l.get_label() if hasattr(l, 'get_label') else l for l in legend_items],
               loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    # 9. Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return plot_df, explained_inertia



def plot_mca_agents_simulation(
    df: pd.DataFrame,
    endowments,
    title: str = "MCA Projection of Agent Response Patterns (Simulation)",
    save_path: str = None,
    figsize: tuple = (10, 6),
    seed = 101
):
    """
    Plots agents in MCA space, highlighting ground truth vs. proxy agents.

    Args:
        df: question × agent DataFrame (categorical or binary)
        endowments: EndowmentManager with .get_eids_by_role('ground_truth')
        title: Title of the plot
        save_path: Optional path to save the plot
        figsize: Plot size
        seed: random_state
    """
    # 1. MCA projection
    df_agents = df.T.astype(str)
    mca = prince.MCA(n_components=2, random_state=seed)
    mca_coords = mca.fit_transform(df_agents)
    explained_inertia = compute_explained_inertia(mca)

    # 2. Assign EIDs and role
    eids = list(df.columns)
    ground_truth_eids = set(endowments.get_eids_by_role('ground_truth'))

    plot_df = mca_coords.copy()
    plot_df.columns = ["MCA 1", "MCA 2"]
    plot_df["eid"] = eids
    plot_df["role"] = plot_df["eid"].apply(lambda eid: "ground_truth" if eid in ground_truth_eids else "proxy")

    # 3. Plot
    plt.figure(figsize=figsize)

    proxy_df = plot_df[plot_df["role"] == "proxy"]
    gt_df = plot_df[plot_df["role"] == "ground_truth"]

    plt.scatter(proxy_df["MCA 1"], proxy_df["MCA 2"],
                c='lightgray', s=30, alpha=0.5, label="Proxy Agents", edgecolors='none')
    
    plt.scatter(gt_df["MCA 1"], gt_df["MCA 2"],
                c='red', s=60, alpha=0.9, label="Ground Truth Agents",
                edgecolors='black', linewidths=1.2)

    # 4. Format
    plt.title(title)
    plt.xlabel(f"MCA Component 1 ({explained_inertia[0]*100:.1f}% inertia)")
    plt.ylabel(f"MCA Component 2 ({explained_inertia[1]*100:.1f}% inertia)")
    plt.grid(True, linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.tight_layout()

    # 5. Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return plot_df, explained_inertia