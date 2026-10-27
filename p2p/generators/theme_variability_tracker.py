from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.colors as mcolors

from copy import deepcopy

class ThemeVariabilityTracker:
    """
    Tracks and updates the entropy of survey question responses and the overall
    response diversity (variability) of agent modes across update steps.

    This tracker maintains a running log of agent responses by question and by mode,
    calculates normalized entropy per question, computes weighted variability per mode,
    and generates adaptive sampling weights based on response diversity.

    It supports plotting of question entropy trajectories, theme variability over time,
    and current entropy levels to assist in diagnosing diversity and guiding active sampling.
    """
    def __init__(self, questions: list[dict], smoothing: float = 0.1):
        """
        Initializes the tracker with survey metadata and internal storage structures.

        Args:
            questions (list[dict]): List of question dictionaries. Each should include:
                                    - 'id': unique identifier
                                    - 'code_to_answer': mapping of response codes to labels
            smoothing (float): Smoothing factor ∈ [0, 1] added to variability scores to
                            encourage exploration of under-sampled modes.
        """
        self.qid_order = [q["id"] for q in questions]
        self.qid_num_options = {q["id"]: len(q.get("code_to_answer", {})) for q in questions}
        self.category_vocab = {
            q["id"]: list(q.get("code_to_answer", {}).values()) for q in questions
        }

        self.smoothing = smoothing

        self.theme_responses = defaultdict(list)  # mode_tuple -> List[List[str]]
        self.question_response_log = defaultdict(list)  # qid -> List[str]
        self.theme_variability = {}  # mode_tuple -> float

        self.question_entropy_history = defaultdict(list)  # qid -> List[float]
        self.theme_variability_history = defaultdict(list)  # mode -> List[float]

        # Useful for plots
        self.mode_first_seen = {}  # mode_tuple -> update index (int)
        self.theme_update_log = defaultdict(list)  # mode_tuple -> List[int]
        self.update_step = 0       # global update step counter

        self.aggregate_entropy_trajectory = []  # track aggregate group entropy throughout the generation

        self.mode_entropy_history = defaultdict(list)     # mode -> List[float]
        self.mode_entropy_log = defaultdict(list)         # mode -> List[int]


    @property
    def tracked_qids(self):
        """
        Returns:
            list[str]: The list of question IDs being tracked.
        """
        return self.qid_order

    @staticmethod
    def categorical_entropy(counts: list[int]) -> float:
        """
        Computes normalized entropy for a categorical distribution.

        Args:
            counts (list[int]): Counts of each category label.

        Returns:
            float: Normalized entropy ∈ [0, 1], with 0 = pure, 1 = uniform.
        """
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy  # Normalize to [0, 1]

    def compute_entropy(self, qid, responses):
        """
        Computes normalized entropy for a specific question using its response history.

        Args:
            qid (str): Question ID.
            responses (list[str]): List of response strings.

        Returns:
            float: Normalized entropy for the given question and responses.
        """
        counter = Counter(responses)
        vocab = self.category_vocab[qid]
        counts = [counter[c] for c in vocab]
        return self.categorical_entropy(counts)

    def update_from_records(self, agent_records: list[dict]):
        """
        Updates internal response logs and entropy/variability metrics based on a batch of responses.

        Args:
            agent_records (list[dict]): List of agent records. Each must contain:
                - 'mode': tuple of attribute labels
                - 'responses': list of answers aligned with self.qid_order
        """
        themed_batch = defaultdict(list)
        for record in agent_records:
            mode = record["mode"]
            responses = record["responses"]

            themed_batch[mode].append(responses)

            for qid, val in zip(self.qid_order, responses):
                if val is not None:
                    self.question_response_log[qid].append(val)

        self._update_theme_variability(themed_batch)

        # Log raw average entropy per mode (Note this is not variability score)
        global_entropies = self.compute_global_question_entropy()
        for mode in themed_batch:
            responses = np.array(self.theme_responses[mode])
            if responses.shape[0] < 2:
                avg_entropy = 0.0
            else:
                q_entropies = []
                for qid, col in zip(self.qid_order, responses.T):
                    q_entropies.append(self.compute_entropy(qid, col))
                avg_entropy = float(np.mean(q_entropies))
            self.mode_entropy_history[mode].append(avg_entropy)
            self.mode_entropy_log[mode].append(self.update_step)


        # Log first-seen step & update step
        for mode in themed_batch:
            if mode not in self.mode_first_seen:
                self.mode_first_seen[mode] = self.update_step
            self.theme_update_log[mode].append(self.update_step)

        self.update_step += 1

        global_entropy = self.compute_global_question_entropy()
        for qid, h in global_entropy.items():
            self.question_entropy_history[qid].append(h)

        # Update the aggregate group entropy
        avg_entropy = np.mean(list(global_entropy.values()))
        self.aggregate_entropy_trajectory.append(avg_entropy)
    
    def filter_records(self, agent_records, survey):
        """
        Filters agent responses to retain only those for tracked question IDs.

        Args:
            agent_records (list[dict]): List of agent records with full response lists.
            survey (Survey): Survey object with all questions.

        Returns:
            list[dict]: Filtered agent records with responses only for tracked questions.
        """
        tracked_qids = self.tracked_qids
        qid_indices = [i for i, q in enumerate(survey.questions) if q["id"] in tracked_qids]
        filtered = deepcopy(agent_records)
        for record in filtered:
            record["responses"] = [record["responses"][i] for i in qid_indices]
        return filtered    

    def _update_theme_variability(self, themed_batch):
        """
        Internal method to update theme-wise response logs and variability scores.

        Args:
            themed_batch (dict): mode_tuple -> list of response lists (per agent)
        """
        for mode, new_responses in themed_batch.items():
            self.theme_responses[mode].extend(new_responses)
            theme_variability = self.compute_theme_variability(mode)
            self.theme_variability[mode] = theme_variability
            self.theme_variability_history[mode].append(theme_variability)

    def compute_global_question_entropy(self):
        """
        Computes the normalized entropy for each tracked question based on all responses.

        Returns:
            dict: qid -> normalized entropy
        """
        entropy = {}
        for qid, responses in self.question_response_log.items():
            entropy[qid] = self.compute_entropy(qid, responses)
        return entropy

    def compute_inverse_entropy_weights(self):
        """
        Computes normalized inverse entropy weights over questions for use in
        weighted variability scoring.

        Returns:
            dict: qid -> normalized inverse entropy weight (sums to 1)
        """
        global_entropies = self.compute_global_question_entropy()
        weights = {q: 1 / (h + 1e-6) for q, h in global_entropies.items()}
        total = sum(weights.values())
        return {q: w / total for q, w in weights.items()}

    def compute_theme_variability(self, mode):
        """
        Computes the weighted average entropy across all tracked questions
        for a given agent mode.

        Args:
            mode (tuple): Mode label (e.g. ('core', 'economics'))

        Returns:
            float: Variability score ∈ [0, 1]
        """
        responses = self.theme_responses[mode]
        if len(responses) < 2:
            return 0.0

        matrix = np.array(responses)
        weights = self.compute_inverse_entropy_weights()

        qid_to_entropy = {}
        for qid, col in zip(self.qid_order, matrix.T):
            qid_to_entropy[qid] = self.compute_entropy(qid, col)

        w = np.array([weights.get(qid, 0.0) for qid in self.qid_order])
        h = np.array([qid_to_entropy[qid] for qid in self.qid_order])
        return np.dot(w, h)

    def get_scores(self):
        """
        Computes variability scores with optional smoothing.

        Returns:
            dict: mode_tuple -> smoothed variability score
        """
        return {
            mode: (1 - self.smoothing) * score + self.smoothing
            for mode, score in self.theme_variability.items()
        }

    def get_softmax_weights(self, temperature=0.3):
        """
        Converts theme variability scores to softmax probabilities for adaptive sampling.

        Args:
            temperature (float): Softmax temperature. 0 = greedy, ↑ = smoother.

        Returns:
            dict: mode_tuple -> sampling probability
        """
        scores = self.get_scores()
        if not scores:
            return {}

        keys = list(scores.keys())
        values = np.array([scores[k] for k in keys])

        if temperature == 0:
            max_idx = np.argmax(values)
            one_hot = np.zeros_like(values)
            one_hot[max_idx] = 1.0
            return dict(zip(keys, one_hot.tolist()))

        scaled = values / temperature
        probs = np.exp(scaled - np.max(scaled))
        probs /= probs.sum()
        return dict(zip(keys, probs))

    def get_low_entropy_questions(self, top_k=3) -> list[str]:
        """
        Returns a list of top_k questions with the lowest normalized entropy.

        Args:
            top_k (int): Number of low-entropy questions to return.

        Returns:
            list[str]: Question IDs with lowest entropy.
        """
        entropies = self.compute_global_question_entropy()
        if not entropies:
            return []

        sorted_qids = sorted(entropies.items(), key=lambda x: x[1])  # sort by increasing entropy
        return [qid for qid, _ in sorted_qids[:top_k]]

    # Plotting toolkit

    def plot_question_entropy_trajectories(self, base_height=6, width_per_step=1, save_path = None):
        """
        Plots the entropy trajectory over time for each tracked question,
        color-coded by trend (rising, falling, stable).

        Args:
            base_height (int): Height of plot in inches.
            width_per_step (int): Width scaling factor per update step.
            save_path (str): Path for the saved figure. Default is None.
        """
        def classify_trend(history):
            if len(history) < 2:
                return "stable"
            slope = (history[-1] - history[0]) / len(history)
            std_dev = np.std(history)
            if slope > 0.01 and std_dev > 0.02:
                return "rising"
            elif slope < -0.01 and std_dev > 0.02:
                return "falling"
            else:
                return "stable"

        trend_colors = {"rising": "green", "falling": "red", "stable": "gray"}
        trend_alphas = {"rising": 1, "falling": 1, "stable": 0.25}
        trend_counts = {"rising": 0, "falling": 0, "stable": 0}

        # Determine max number of steps and set dynamic figsize
        max_steps = max(len(h) for h in self.question_entropy_history.values())
        figsize = (max(6, width_per_step * max_steps), base_height)

        fig, ax = plt.subplots(figsize=figsize)
        for qid, history in self.question_entropy_history.items():
            trend = classify_trend(history)
            trend_counts[trend] += 1
            ax.plot(history, label=qid, color=trend_colors[trend], alpha=trend_alphas[trend])

        ax.set_xlabel("Update Step")
        ax.set_ylabel("Normalized Entropy")
        ax.set_title("Entropy Trajectories by Question")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(range(max_steps))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(False)

        custom_lines = [
            Line2D([0], [0], color=trend_colors["rising"], lw=2, alpha=trend_alphas["rising"],
                label=f'Rising ({trend_counts["rising"]})'),
            Line2D([0], [0], color=trend_colors["falling"], lw=2, alpha=trend_alphas["falling"],
                label=f'Falling ({trend_counts["falling"]})'),
            Line2D([0], [0], color=trend_colors["stable"], lw=2, alpha=trend_alphas["stable"],
                label=f'Stable ({trend_counts["stable"]})'),
        ]
        ax.legend(
            handles=custom_lines,
            title="Entropy Trend",
            title_fontproperties={'weight': 'bold'},
            loc="lower center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=3,
            frameon=False
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_theme_variability_trajectories(
        self,
        attribute_bank: dict,
        base_height=6,
        width_per_step=1,
        save_path=None
    ):
        """
        Plots variability trajectories for each mode over update steps,
        color-coded by template group using palettes from the attribute bank.

        Args:
            attribute_bank (dict): YAML-style config of attribute templates.
            base_height (int): Height of plot.
            width_per_step (int): Width scaling factor per update step.
            save_path (str): Path for the saved figure. Default is None.
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as ticker
        from matplotlib.lines import Line2D
        from collections import defaultdict
        import numpy as np

        used_modes = list(self.theme_variability_history.keys())

        # Infer templates per mode
        mode_template_map = {
            mode: self._get_template_from_mode(mode, attribute_bank["templates"])
            for mode in used_modes
        }

        # Group modes by template
        modes_by_template = defaultdict(list)
        for mode, template in mode_template_map.items():
            modes_by_template[template].append(mode)

        # Assign consistent color mapping
        color_mapping = self._assign_template_color_palette(modes_by_template)

        # Assign linestyles
        linestyles = self._assign_template_linestyles(modes_by_template)

        # Determine figure layout
        labels = ["+".join(mode) for mode in used_modes]
        legend_ncol, legend_height = self._compute_legend_layout(labels)
        max_steps = self.update_step
        fig_width = max(12, width_per_step * max_steps)
        fig_height = base_height + legend_height
        figsize = (fig_width, fig_height)

        # Setup plot
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[base_height, legend_height], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        handles = {}

        for mode in used_modes:
            history = self.theme_variability_history[mode]
            label = "+".join(mode)
            color = color_mapping[mode]
            linestyle = linestyles[mode]

            # Pad y-values with NaN for missing steps
            y_vals = [np.nan] * max_steps
            for step, score in zip(self.theme_update_log[mode], history):
                y_vals[step] = score

            ax.plot(range(max_steps), y_vals, label=label, color=color, linestyle = linestyle, linewidth=2.3)
            handles[mode] = Line2D([0], [0], color=color, linestyle = linestyle, linewidth=2.3, label=label)

        # Axis formatting
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Variability Score")
        ax.set_title("Theme Variability Trajectories")
        ax.set_xticks(range(max_steps))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Group legend items by template
        grouped_handles = defaultdict(list)
        for mode in used_modes:
            template = mode_template_map[mode]
            label = "+".join(mode)
            grouped_handles[template].append((label, handles[mode]))

        ordered_templates = ["core", "thematic", "theoretical", "survey", "pure_question_patch", "mixed_patch", "unknown"]
        legend_items = []
        for template in ordered_templates:
            if template not in grouped_handles:
                continue
            header = rf"$\mathbf{{{template.replace('_', r'\,').upper()}}}$"
            legend_items.append(Line2D([0], [0], linestyle="none", label=header, color="black"))
            legend_items.extend([h for _, h in grouped_handles[template]])

        legend_labels = [h.get_label() for h in legend_items]

        # Render legend
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")
        legend_ax.legend(
            legend_items,
            legend_labels,
            loc="center",
            ncol=legend_ncol,
            frameon=False,
            fontsize="small"
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


    def plot_theme_variability_trajectories_legacy(self, attribute_bank: dict, base_height=6, width_per_step=1, save_path = None):
        """
        Plots variability trajectories for each mode over update steps,
        color-coded by template group using palettes from the attribute bank.

        Args:
            attribute_bank (dict): YAML-style config of attribute templates.
            base_height (int): Height of plot.
            width_per_step (int): Width scaling factor per update step.
            save_path (str): Path for the saved figure. Default is None.
        """

        # Assign a base palette per template group
        template_base_palettes = {
            "core": sns.light_palette("black", n_colors=10, reverse=True),
            "thematic": sns.color_palette("Set2", n_colors=10),
            "theoretical": sns.color_palette("Dark2", n_colors=10),
            "survey": sns.color_palette("flare", n_colors=10),
            "question": sns.color_palette("crest", n_colors=10),
            "unknown": sns.light_palette("gray", n_colors=10, reverse=True),
        }

        # Build theme → template group mapping
        theme_to_template = {}
        for group, themes in attribute_bank["templates"].items():
            for theme in themes:
                theme_to_template[theme] = group
        theme_to_template["core"] = "core"  # fallback

        used_themes = sorted({
            mode[1] if len(mode) > 1 else mode[0]
            for mode in self.theme_variability_history
        })

        # Assign colors
        template_theme_counts = {k: 0 for k in template_base_palettes}
        theme_to_color = {}

        for theme in used_themes:
            group = theme_to_template.get(theme, "unknown")
            color_list = template_base_palettes[group]
            color_idx = template_theme_counts[group]
            color = color_list[color_idx % len(color_list)]
            template_theme_counts[group] += 1
            theme_to_color[theme] = color

        # Prepare labels and layout info
        labels = ["+".join(mode) for mode in self.theme_variability_history]
        legend_ncol, legend_height = self._compute_legend_layout(labels)

        # Setup plot
        max_steps = self.update_step
        fig_width = max(12, width_per_step * max_steps)
        fig_height = base_height + legend_height
        figsize = (fig_width, fig_height)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[base_height, legend_height], hspace=0.3)
        
        # Main plot
        ax = fig.add_subplot(gs[0])
        handles = []

        for mode, history in self.theme_variability_history.items():
            theme = mode[1] if len(mode) > 1 else mode[0]
            label = "+".join(mode)
            color = theme_to_color[theme]

            # Construct y-values with NaN padding for missing steps
            y_vals = [np.nan] * max_steps
            for step, score in zip(self.theme_update_log[mode], history):
                y_vals[step] = score

            ax.plot(range(max_steps), y_vals, label=label, color=color, linewidth=2)
            handles.append(Line2D([0], [0], color=color, label=label))

        # Final formatting
        ax.set_xlabel("Update Step")
        ax.set_ylabel("Variability Score")
        ax.set_title("Theme Variability Trajectories")

        ax.set_xticks(range(max_steps))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")
        legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=legend_ncol,
            frameon=False,
            title="Themes",
            title_fontproperties={'weight': 'bold'},
            fontsize="small"
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_entropy_bar_chart(self, sort=True, figsize=(16, 5), top_n=None, save_path = None):
        """
        Plots a bar chart of normalized entropy across all tracked questions.

        Args:
            sort (bool): Whether to sort bars by entropy.
            figsize (tuple): Size of the plot.
            top_n (int or None): If set, limits display to top-N questions by entropy.
            save_path (str): Path for the saved figure. Default is None.
        """
        entropies = self.compute_global_question_entropy()
        data = [{"qid": qid, "normalized_entropy": entropies[qid]} for qid in self.qid_order]
        df = pd.DataFrame(data)

        if sort:
            df = df.sort_values(by="normalized_entropy", ascending=True)

        if top_n is not None:
            df = df.tail(top_n)

        plt.figure(figsize=figsize)
        plt.bar(df["qid"], df["normalized_entropy"], color="skyblue")
        plt.title("Normalized Entropy per Question" + (" (Sorted)" if sort else ""))
        plt.xlabel("Question ID")
        plt.ylabel("Normalized Entropy")
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _compute_legend_layout(self, labels: list[str], row_height: float = 0.45):
        """
        Computes legend layout parameters (number of columns and required height)
        based on the number and length of labels.

        Args:
            labels (list[str]): List of legend labels.
            row_height (float): Height per legend row in inches (default = 0.45).

        Returns:
            tuple[int, float]: (legend_ncol, legend_height)
        """
        if not labels:
            return 1, 1.0  # Fallback

        max_label_len = max(len(label) for label in labels)
        
        # Adjust column count based on label length
        if max_label_len > 60:
            legend_ncol = 2
        elif max_label_len > 40:
            legend_ncol = 3
        elif max_label_len > 25:
            legend_ncol = 4
        else:
            legend_ncol = 5

        legend_rows = int(np.ceil(len(labels) / legend_ncol))
        legend_height = max(1.0, legend_rows * row_height)
        return legend_ncol, legend_height
    
    def _assign_template_linestyles(self, modes_by_template: dict) -> dict:
        """
        Assigns distinct linestyles per mode within each template group.

        Args:
            modes_by_template (dict): {template: list of mode tuples}

        Returns:
            dict: {mode_tuple: linestyle string}
        """
        base_styles = ['solid', 'dashed', 'dotted', 'dashdot', (0, (1, 1)), (0, (3, 1, 1, 1))]

        mode_to_style = {}
        for template, mode_list in modes_by_template.items():
            for i, mode in enumerate(sorted(mode_list)):
                mode_to_style[mode] = base_styles[i % len(base_styles)]
        return mode_to_style

    def _get_template_from_mode(self, mode_tuple, attribute_bank):
        """Infer template type from a mode tuple using heuristic logic."""
        keys = list(mode_tuple)
        if keys == ['core']:
            return 'core'
        elif keys == ['survey']:
            return 'survey'
        elif len(keys) == 1 and keys[0] in attribute_bank.get('question', {}):
            return 'pure_question_patch'
        elif len(keys) > 1 and any(k in attribute_bank.get('question', {}) for k in keys):
            return 'mixed_patch'
        elif len(keys) == 2 and 'core' in keys:
            other = [k for k in keys if k != 'core'][0]
            if other in attribute_bank['thematic']:
                return 'thematic'
            elif other in attribute_bank['theoretical']:
                return 'theoretical'
        return 'unknown'

    def _assign_template_color_palette(self, modes_by_template):
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
    
    def get_persistent_mode_entropy_trajectories(self) -> dict:
        """
        Returns:
            dict: {mode_tuple: list of entropy values across all steps, 
                padded with NaN before first appearance, forward-filled after}
        """
        max_step = self.update_step
        full_trajectories = {}

        for mode in self.mode_entropy_history:
            steps = self.mode_entropy_log[mode]
            values = self.mode_entropy_history[mode]

            # Build full timeline
            full = [np.nan] * max_step

            last_val = None
            for s in range(max_step):
                if s in steps:
                    idx = steps.index(s)
                    last_val = values[idx]
                full[s] = last_val if last_val is not None else np.nan

            full_trajectories[mode] = full

        return full_trajectories
    
    def get_entropy_trajectories(self):
        """Return aggregate, per-mode, and per-question trajectories.

        Returns:
            A mapping with ``aggregate``, ``by_mode``, and ``by_question``
            trajectories.
        """
        return {
            "aggregate": self.aggregate_entropy_trajectory,
            "by_mode": self.get_persistent_mode_entropy_trajectories(),
            "by_question": self.question_entropy_history
        }
