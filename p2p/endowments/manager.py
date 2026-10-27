import csv
import warnings
import numpy as np
import random
from collections import Counter, defaultdict
from typing import Optional
from copy import deepcopy


def _categorical_entropy(values, vocab, normalize=True):
    counts = [Counter(values).get(value, 0) for value in vocab]
    total = sum(counts)
    if total == 0:
        return 0.0
    probabilities = [
        count / total
        for count in counts
        if count > 0
    ]
    entropy = -sum(
        probability * np.log2(probability)
        for probability in probabilities
    )
    maximum = np.log2(len(vocab)) if len(vocab) > 1 else 1.0
    return entropy / maximum if normalize else entropy


class Endowments:
    """
    A structured container and access layer for managing a pool of LLM agent endowments.

    Each endowment defines a persona used to condition agent behaviour in simulation or empirical experiments.

    Attributes:
        csv_path (str): Path to the endowment CSV file
        endowments (list[dict]): List of all endowment entries, each a dictionary with:
            - eid (str): Unique identifier
            - endow_text (str): Identity-conditioning prompt passed to LLM
            - role (str): Either 'ground_truth' or 'proxy'
            - weight (float or None): Importance weight (sampling or regression)
        index (dict): Maps eid to endowment dictionary for fast lookup

    Key Methods:
        - load(cls, path): Class method for instantiating from CSV
        - get_endowments(): Return all endowment entries
        - get_endowment_by_eid(eid): Lookup a single endowment
        - get_endowments_by_role(role): Filter by role ('ground_truth' or 'proxy')
        - get_eids_by_role(role): List eids matching a role
        - update_weights(weights): Partially update weights by eid
        - save(path): Save the endowments back to a CSV file
    """

    def __init__(self, csv_path = None):
        self.csv_path = csv_path
        self.endowments = self._load_csv() if csv_path else []
        self.index = {e["eid"]: e for e in self.endowments}
    
    @classmethod
    def load(cls, path):
        """
        Load an Endowments instance from a CSV file.
        This method can be extended for validation, logging, or alternative formats.
        """
        return cls(path)

    def _load_csv(self):
        """Load endowments from a CSV file."""
        if not self.csv_path:
            raise ValueError("csv_path is not set. Cannot load endowments.")
        endowments = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    weight = float(row["weight"]) if "weight" in row and row["weight"].strip() != "" else None
                except ValueError:
                    weight = None             
                endowments.append({
                    "eid": row["eid"].strip(),
                    "endow_text": row["endow_text"].strip(),
                    "role": row.get("role", "proxy").strip(),
                    "weight": weight
                })
        return endowments
    
    def get_endowments(self):
        return self.endowments
    
    def get_endowment_by_eid(self, eid):
        return self.index.get(eid)
    
    def get_endowments_by_role(self, role):
        if role not in {"ground_truth", "proxy"}:
            raise ValueError(f"Invalid role: {role}. Only 'ground_truth' and 'proxy' are allowed.")
        return [e for e in self.endowments if e["role"] == role]
    
    def get_eids_by_role(self, role):
        entries = self.get_endowments_by_role(role) if role else self.endowments
        return [e["eid"] for e in entries]
    
    def update_weights(self, weights):
        """
        Update weights for a subset of endowments.

        Parameters:
            weights (dict): Mapping from eid (str) to new weight (float or None)

        Behaviour:
            - Only updates entries whose eid is present in the weights dict
            - Ignores eids that do not exist in the current pool
            - Supports partial updating (e.g., just ground_truth or just proxy)
        """
        for eid, w in weights.items():
            if eid in self.index:
                self.index[eid]["weight"] = w

    def initialize_ground_truth_weights(self, seed = None, save = False, save_path = None):
        """
        Randomly initialize normalized weights for 'ground_truth' endowments.

        Args:
            seed (int, optional): Random seed for reproducibility.
            save (bool): Whether to save the updated weights to CSV.
            save_path (str, optional): Path to save the updated CSV. Defaults to self.csv_path.
        
        Raises:
            ValueError: If no 'ground_truth' endowments exist.
            Warning: If weights already exist and save=False.
        """
        if not any(e.get('role', '').strip()=='ground_truth' for e in self.endowments):
            raise ValueError("The endowments pool does not contain ground_truth endowments.")
        
        if any(e.get('role', '').strip() == 'ground_truth' and str(e.get('weight', '')).strip() for e in self.endowments) and not save:
            warnings.warn("Overwriting existing weight assignments for ground_truth endowments.")       

        if seed is not None:
            random.seed(seed)

        ground_truth_eids = self.get_eids_by_role('ground_truth')
        n = len(ground_truth_eids)
        raw_weights = [random.uniform(0, 1) for _ in range(n)]
        total = sum(raw_weights)
        normalized_weights = [w / total for w in raw_weights]

        weights_map = dict(zip(ground_truth_eids, normalized_weights))
        self.update_weights(weights_map)

        if save:
            self._save_weights_csv(save_path)


    def assign_roles(self, method = 'manual', role_map = None, proxy_ratio = 1, seed = None, save = False, save_path = None):
        
        if any(e.get('role', '').strip() for e in self.endowments) and not save:
            warnings.warn("Overwriting the roles for endowments that already contains role assignments.")        
        
        if method == 'from_map':
            if not role_map:
                raise ValueError("You must provide role_map when method = 'from_map'.")
            for e in self.endowments:
                e['role'] = role_map.get(e['eid'], 'proxy') # default to proxy 
        elif method == 'manual':
            if seed is not None:
                random.seed(seed)
            total = len(self.endowments)
            indices = list(range(total))
            random.shuffle(indices)

            proxy_end = int(proxy_ratio * total)

            for i, idx in enumerate(indices):
                if i < proxy_end:
                    self.endowments[idx]['role'] = 'proxy'
                else:
                    self.endowments[idx]['role'] = 'ground_truth'
        else:
            raise ValueError("Unknown role assignment method. Use 'manual' or 'from_map'.")
        
        if save:
            self._save_roles_csv(save_path)
            self.report_role_assignment()

    def get_role_counts(self):
        return Counter(e.get('role', 'unspecified') for e in self.endowments)

    def report_role_assignment(self, show_percentage=True):
        """
        Prints a summary of the role distribution among survey questions.

        Parameters:
        - show_percentage (bool): If True, also print percentages.
        """
        total = len(self.endowments)
        role_counts = self.get_role_counts()

        print("Role Assignment Summary:")
        print("-" * 30)
        for role_name, count in role_counts.items():
            if show_percentage:
                percent = (count / total) * 100
                print(f"{role_name:<12} {count:>4} ({percent:5.1f}%)")
            else:
                print(f"{role_name:<12} {count:>4}")
        print("-" * 30)
        print(f"Total endowments: {total}")

    def save(self, filepath):

        path = filepath if filepath else self.csv_path

        keys = ["eid", "endow_text", "role", "weight"]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.endowments)

    def _save_roles_csv(self, save_path=None):
        self.save(filepath=save_path)
    
    def _save_weights_csv(self, save_path=None):
        self.save(filepath=save_path)
    
    def __len__(self):
        return len(self.endowments)
    
    @classmethod
    def from_endowment_list(cls, endowment_list):
        obj = cls(csv_path=None)
        obj.endowments = endowment_list
        obj.index = {e["eid"]: e for e in endowment_list}
        return obj
    

    def clone_with_fraction(self, fraction: float = 1.0, seed: int = 101):
        """
        Returns a new ActiveEndowments object with a sampled subset of endowments,
        preserving the proxy/ground_truth ratio.

        Args:
            fraction (float): Fraction of endowments to retain (0 < fraction ≤ 1).
            seed (int): Random seed for reproducibility.

        Returns:
            ActiveEndowments: A new object with sampled endowments.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")

        random_state = random.Random(seed)

        role_groups = defaultdict(list)
        for e in self.endowments:
            role_groups[e["role"]].append(e)

        sampled = []
        for role, entries in role_groups.items():
            k = max(1, int(len(entries) * fraction))
            sampled.extend(random_state.sample(entries, k=k))

        return self.__class__.from_endowment_list(sampled)
    
    def clone_with_proxy_fraction(self, fraction: float = 1.0, seed: int = 101):
        """
        Returns a new Endowments object with a subsampled set of proxy agents,
        while retaining all ground_truth agents. The sampling preserves the proxy
        structure but shrinks its size.

        Args:
            fraction (float): Fraction of proxy endowments to retain (0 < fraction ≤ 1).
            seed (int): Random seed for reproducibility.

        Returns:
            Endowments: A new object with modified endowments.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be in (0, 1]")

        random_state = random.Random(seed)

        proxies = [e for e in self.endowments if e["role"] == "proxy"]
        ground_truth = [e for e in self.endowments if e["role"] == "ground_truth"]

        k = max(1, int(len(proxies) * fraction))
        sampled_proxies = random_state.sample(proxies, k=k)

        new_endowments = ground_truth + sampled_proxies

        return self.__class__.from_endowment_list(new_endowments)

    def renormalize_ground_truth_weights(self):
        """
        Re-normalizes the weights of ground truth endowments so they sum to 1.

        Preserves relative weights across ground truth agents.
        Raises an error if total weight is zero.

        This is useful after subsampling endowments, since dropped agents may
        invalidate the original normalization.
        """
        gt_endowments = self.get_endowments_by_role("ground_truth")

        if any(e.get("weight") is None for e in gt_endowments):
            warnings.warn("Some ground truth endowments have missing weights; treating as 0.")

        total = sum(e.get("weight", 0) for e in gt_endowments)

        if total == 0:
            raise ValueError("Total ground truth weight is zero. Cannot renormalize.")

        for e in gt_endowments:
            e["weight"] = (e.get("weight", 0) / total)

    @staticmethod
    def compute_entropy(endowment_list, responses, survey, normalize=True):
        """
        Computes average normalized entropy across survey questions for a given list of endowments.

        Args:
            endowment_list (list[dict]): List of endowments, each with an 'eid'.
            responses (Responses): Object providing get_agent_vector(eid).
            survey (Survey): Survey object with question metadata.
            normalize (bool): Whether to normalize entropy ∈ [0,1].

        Returns:
            float: Average entropy across all answered questions.
        """
        qids = [q["id"] for q in survey.questions]
        qid_to_vocab = {q["id"]: list(q.get("code_to_answer", {}).values()) for q in survey.questions}

        responses_by_qid = {qid: [] for qid in qids}
        for e in endowment_list:
            agent_vector = responses.get_agent_vector(e["eid"])
            for qid in qids:
                val = agent_vector.get(qid)
                if val is not None:
                    responses_by_qid[qid].append(val)

        q_entropies = [
            _categorical_entropy(
                responses_by_qid[qid],
                qid_to_vocab[qid],
                normalize=normalize,
            )
            for qid in qids if responses_by_qid[qid]
        ]

        return np.mean(q_entropies) if q_entropies else 0.0

class ActiveEndowments(Endowments):
    """
    Subclass of Endowments that supports mode-aware experiments.
    Each endowment is expected to include:
    - 'mode': a tuple of theme labels, e.g., ('core', 'economics')
    - 'attributes': list of attribute names used during generation
    """

    def _load_csv(self):
        endowments = []
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    weight = float(row["weight"]) if "weight" in row and row["weight"].strip() != "" else None
                except ValueError:
                    weight = None

                # Parse mode and attributes
                raw_mode = row.get("mode", "").strip()
                mode_tuple = tuple(part.strip() for part in raw_mode.split("+") if part)

                raw_attrs = row.get("attributes", "").strip()
                attr_list = [a.strip() for a in raw_attrs.split(";") if a]

                endowments.append({
                    "eid": row["eid"].strip(),
                    "endow_text": row["endow_text"].strip(),
                    "role": row.get("role", "proxy").strip(),
                    "weight": weight,
                    "mode": mode_tuple,
                    "attributes": attr_list
                })
        return endowments
    
    @classmethod
    def from_endowment_list(cls, endowment_list):
        obj = cls(csv_path=None)
        obj.endowments = endowment_list
        obj.index = {e["eid"]: e for e in endowment_list}
        return obj

    def add_batch(self, new_endowments):
        for e in new_endowments:
            if e["eid"] in self.index:
                warnings.warn(f"Duplicate eid '{e['eid']}' detected. Overwriting existing entry.")
            self.endowments.append(e)
            self.index[e["eid"]] = e

    def get_mode_by_eid(self, eid):
        e = self.get_endowment_by_eid(eid)
        if e is None or "mode" not in e:
            raise KeyError(f"Endowment {eid} is missing 'mode'")
        return e["mode"]

    def get_attributes_by_eid(self, eid):
        e = self.get_endowment_by_eid(eid)
        return e.get("attributes", [])

    def group_by_mode(self):
        grouped = defaultdict(list)
        for e in self.endowments:
            grouped[e.get("mode", ())].append(e)
        return dict(grouped)

    def save(self, filepath):
        path = filepath if filepath else self.csv_path
        keys = ["eid", "endow_text", "role", "weight", "mode", "attributes"]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for e in self.endowments:
                writer.writerow({
                    "eid": e["eid"],
                    "endow_text": e["endow_text"],
                    "role": e.get("role", "proxy"),
                    "weight": e.get("weight", None),
                    "mode": "+".join(e.get("mode", [])),
                    "attributes": ";".join(e.get("attributes", []))
                })

    def get_entropy_by_mode(self, responses, survey, normalize=True, top_k=None):
        """
        Computes normalized entropy for each mode based on responses from agents
        sharing that mode.

        Args:
            responses (Responses): An instance with agent responses (must implement get_agent_vector).
            survey (Survey): Survey object with question metadata.
            normalize (bool): Whether to return normalized entropy scores.
            top_k (int or None): If set, only return the top_k highest-entropy modes.

        Returns:
            list[tuple]: Sorted list of (mode, entropy score, number of endowments), descending by entropy.
        """
        mode_to_eids = self.group_by_mode()
        qids = [q["id"] for q in survey.questions]
        qid_to_vocab = {q["id"]: list(q.get("code_to_answer", {}).values()) for q in survey.questions}

        mode_entropy = {}
        for mode, endowments in mode_to_eids.items():
            eids = [e["eid"] for e in endowments]
            mode_responses = {qid: [] for qid in qids}

            for eid in eids:
                agent_resp = responses.get_agent_vector(eid)
                for qid in qids:
                    val = agent_resp.get(qid)
                    if val is not None:
                        mode_responses[qid].append(val)

            q_entropies = [
                _categorical_entropy(
                    mode_responses[qid],
                    qid_to_vocab[qid],
                    normalize=normalize,
                )
                for qid in qids if mode_responses[qid]
            ]
            avg_entropy = np.mean(q_entropies) if q_entropies else 0.0
            mode_entropy[mode] = (avg_entropy, len(eids))

        # Sort by entropy (descending)
        sorted_modes = sorted(
            [(mode, entropy, count) for mode, (entropy, count) in mode_entropy.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_modes[:top_k] if top_k else sorted_modes
    
    
    def clone_with_proxy_modes(
        self,
        included_modes: list[str],
        n_proxies: int = 30,
        seed: Optional[int] = None
    ) -> "ActiveEndowments":
        """
        Clone a subset of the endowments retaining:
        - All ground truth agents.
        - A random subset of proxy agents from the included_modes (default: 30).

        Args:
            included_modes (list[str]): Modes to include when sampling proxy agents.
            n_proxies (int): Number of proxy agents to sample (default: 30).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            ActiveEndowments: New object containing ground truth + sampled proxy agents.
        """

        # 1. Retrieve ground truth agents
        gt_endowments = [
            deepcopy(e)
            for e in self.get_endowments_by_role("ground_truth")
        ]

        # 2. Filter eligible proxy agents by mode
        eligible_proxies = [
            e for e in self.get_endowments_by_role("proxy")
            if e.get("mode") in included_modes
        ]

        # 3. Sample proxy agents
        if seed is not None:
            random.seed(seed)

        sampled_proxies = random.sample(eligible_proxies, min(n_proxies, len(eligible_proxies)))
        proxy_endowments = [deepcopy(e) for e in sampled_proxies]

        # 4. Return new endowment object
        return self.__class__.from_endowment_list(
            endowment_list=gt_endowments + proxy_endowments
        )
