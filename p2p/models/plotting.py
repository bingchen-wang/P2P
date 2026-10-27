import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

def plot_lasso_diagnostics(diagnostics, best_alpha, strategy="cv"):
    """
    Plot Lasso MSE diagnostics for either cross-validation or hold-out validation.

    Parameters:
        diagnostics: dict returned by `fit_lasso_model`, containing relevant MSE data
        best_alpha: float, the selected regularization parameter
        strategy: str, "cv" or "holdout" (used for branching)
    """
    with sns.axes_style("whitegrid", rc={"grid.linestyle": "--", "grid.linewidth": 0.3, "grid.alpha": 1}), \
         sns.plotting_context("notebook", font_scale=1.15):
        if strategy == "cv":
            mean_mse = diagnostics['mean_mse']
            std_mse = diagnostics['std_mse']
            alpha_vals = diagnostics['alphas']

            fig, ax = plt.subplots(figsize=(6, 4))

            ax.plot(alpha_vals, mean_mse, marker='o', linestyle='-', linewidth=2,
                    color='#26567e', label='Mean CV MSE')
            ax.fill_between(alpha_vals,
                            mean_mse - std_mse,
                            mean_mse + std_mse,
                            alpha=0.5,
                            color='#ccd5dd',
                            label='± 1 STD')

            ax.axvline(best_alpha, linestyle='--', color='#f50000', linewidth=2, alpha=0.6,
                    label=f'Best $\\alpha = {best_alpha:.2e}$')

            ax.set_xscale('log')
            ax.set_xlabel("$\\alpha$ (log scale)")
            ax.set_ylabel("MSE (CV mean ± std)")
            ax.set_title("Cross-Validated MSE by Lasso Alpha ($\\alpha$)")
            ax.legend()
            fig.tight_layout()

            return fig

        elif strategy == "holdout":
            train_errors = diagnostics['train_errors']
            val_errors = diagnostics['val_errors']
            alphas = diagnostics['alphas']

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

            # Train MSE
            axes[0].plot(alphas, train_errors, marker='o', linestyle='-', linewidth=2,
                        color='#26567e', label='Train MSE')
            axes[0].axvline(best_alpha, color='#f50000', linestyle='--', linewidth=2, alpha=0.6)
            axes[0].set_xscale('log')
            axes[0].set_title("Train MSE")
            axes[0].set_xlabel("$\\alpha$ (log scale)")
            axes[0].set_ylabel("MSE")
            axes[0].grid(True)

            # Validation MSE
            axes[1].plot(alphas, val_errors, marker='o', linestyle='-', linewidth=2,
                        color='#77ab59', label='Validation MSE')
            axes[1].axvline(best_alpha, color='#f50000', linestyle='--', linewidth=2, alpha=0.6,
                            label=f'Best $\\alpha = {best_alpha:.2e}$')
            axes[1].set_xscale('log')
            axes[1].set_title("Validation MSE")
            axes[1].set_xlabel("$\\alpha$ (log scale)")
            axes[1].grid(True)

            # Unified legend below
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles += h
                labels += l

            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                    ncol=3, frameon=False)

            plt.suptitle("Lasso Regression: MSE by $\\alpha$ (Train vs Validation)", fontsize=14, y=0.91)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            return fig

def plot_elasticnet_diagnostics(diagnostics, best_alpha, best_l1_ratio, style="2D"):
    if style == "2D":
        return _plot_elasticnet_heatmap(diagnostics, best_alpha, best_l1_ratio)
    elif style == "3D":
        return _plot_elasticnet_3d_static(diagnostics, best_alpha, best_l1_ratio)
    elif style == "interactive":
        return _plot_elasticnet_3d_interactive(diagnostics, best_alpha, best_l1_ratio)
    else:
        raise ValueError(f"Unknown plot style: {style}. Choose from '2D', '3D', or 'interactive'.")

def _plot_elasticnet_heatmap(diagnostics, best_alpha, best_l1_ratio):
    mean_mse = diagnostics["mean_mse"]
    alphas = diagnostics["alphas"]
    l1_ratios = diagnostics["l1_ratios"]

    best_idx = np.unravel_index(np.argmin(mean_mse), mean_mse.shape)
    best_val = mean_mse[best_idx]

    with sns.axes_style("whitegrid", rc={"grid.linestyle": "--", "grid.linewidth": 0.3, "grid.alpha": 1}), \
         sns.plotting_context("notebook", font_scale=1.15):

        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(
            mean_mse,
            xticklabels=np.round(alphas, 3),
            yticklabels=np.round(l1_ratios, 2),
            cmap="YlGnBu",
            cbar_kws={'label': 'Mean CV MSE'},
            annot=False
        )

        ax.plot(best_idx[1] + 0.5, best_idx[0] + 0.5, 'ro', markersize=8, label='Best')
        ax.text(best_idx[1] + 0.5, best_idx[0] + 1, f'{best_val:.4f}', color='red',
                ha='center', va='center', fontweight='bold')

        xtick_step = 5
        n_alphas = len(alphas)
        tick_indices = list(range(0, n_alphas, xtick_step))
        if (n_alphas - 1) not in tick_indices:
            tick_indices.append(n_alphas - 1)

        ax.set_xticks([i + 0.5 for i in tick_indices])
        ax.set_xticklabels([f"{alphas[i]:.3f}" for i in tick_indices], rotation=45, ha='right')
        ax.set_xlabel("alpha")
        ax.set_ylabel("l1_ratio")
        ax.set_title("ElasticNetCV Cross-Validated MSE Heatmap")
        ax.legend()
        plt.tight_layout()
        return plt.gcf()

def _plot_elasticnet_3d_static(diagnostics, best_alpha, best_l1_ratio):
    mean_mse = diagnostics["mean_mse"]
    alphas = diagnostics["alphas"]
    l1_ratios = diagnostics["l1_ratios"]
    A, L = np.meshgrid(alphas, l1_ratios)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        np.log10(A), L, mean_mse,
        cmap='YlGnBu', edgecolor=None, linewidth=0.5, alpha=0.75, antialiased=True
    )

    best_idx = np.unravel_index(np.argmin(mean_mse), mean_mse.shape)
    best_mse = mean_mse[best_idx]

    ax.scatter(np.log10(best_alpha), best_l1_ratio, best_mse, color='red', s=50, label='Best')
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("l1_ratio")
    ax.set_zlabel("Mean CV MSE")
    ax.set_title("Cross-Valid MSE by Elastic Net penalty parameter ($\\alpha$) and l1_ratio")
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Mean CV MSE')
    plt.tight_layout()
    return fig

def _plot_elasticnet_3d_interactive(diagnostics, best_alpha, best_l1_ratio):
    mean_mse = diagnostics["mean_mse"]
    std_mse = diagnostics["std_mse"]
    alphas = diagnostics["alphas"]
    l1_ratios = diagnostics["l1_ratios"]
    A, L = np.meshgrid(np.log10(alphas), l1_ratios)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=mean_mse, x=A, y=L,
        colorscale='YlGnBu', name='Mean MSE', colorbar=dict(title='Mean CV MSE'),
        showscale=True, hoverinfo='x+y+z', legendgroup='mean', showlegend=True
    ))
    fig.add_trace(go.Surface(
        z=mean_mse + std_mse, x=A, y=L,
        surfacecolor=mean_mse + std_mse, colorscale='Greys', opacity=0.3,
        showscale=False, name='+1 SD', legendgroup='band', showlegend=True, hoverinfo='skip'
    ))
    fig.add_trace(go.Surface(
        z=mean_mse - std_mse, x=A, y=L,
        surfacecolor=mean_mse - std_mse, colorscale='Greys', opacity=0.3,
        showscale=False, name='-1 SD', legendgroup='band', showlegend=True, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[np.log10(best_alpha)], y=[best_l1_ratio], z=[mean_mse.min()],
        mode='markers+text', marker=dict(size=6, color='red', symbol='circle'),
        text=["Best"], textposition="top center", name='Best', showlegend=True
    ))
    fig.update_layout(
        title=dict(text='ElasticNetCV Error Surface with ±1 SD', x=0.5),
        scene=dict(
            xaxis_title='log10(alpha)', yaxis_title='l1_ratio', zaxis_title='Mean CV MSE'
        ),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='gray', borderwidth=1),
        width=900, height=700, margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()
    return fig