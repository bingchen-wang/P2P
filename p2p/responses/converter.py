from p2p.survey.converter import Survey, BinaryExtendedSurvey
from p2p.responses.cleaner import ResponseCleaner
from typing import List, Dict
from collections import defaultdict, Counter
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from copy import deepcopy

class Responses:
    """
    Container class for loading, accessing, and saving a matrix of survey responses.
    Supports matrix-style CSV input or list-of-dict input, and handles answer-to-code mapping.

    Attributes:
        output_format (str): Determines how responses are stored. 
            - 'code': responses are mapped to numeric or symbolic codes.
            - 'answer': responses are kept as raw (cleaned) answer text.
        survey (Survey): Survey object containing questions and associated metadata.
        questions (dict): Mapping from question ID to metadata (extracted from the survey).
        responses (dict): Nested dictionary where responses[qid][eid] = value (code or answer).
        clean (bool): If True, apply normalization and heuristic cleaning to free-form agent responses.
        cleaner (ResponseCleaner): Optional cleaner instance to use for response normalization and disambiguation.
            If None and `clean=True`, a default ResponseCleaner will be instantiated using the survey.
    """  
    def __init__(self, source, survey, output_format = "code", clean=True, cleaner= None):
        """
        Initialize the Responses object.

        Args:
            source (str or list): Source of responses, either a matrix-formatted CSV file path 
                or a flat list of dictionaries with 'qid', 'eid', and 'answer' keys.
            survey (Survey): A Survey instance defining question metadata and coding scheme.
            output_format (str): Format to store responses: 
                - 'code' (default): use mapped answer codes.
                - 'answer': store cleaned free-form text answers.
            clean (bool): Whether to apply response cleaning and normalization (default: True).
            cleaner (ResponseCleaner, optional): An optional custom ResponseCleaner instance.
                If None, a default cleaner is created using the provided survey.
        """
        if output_format not in {"code", "answer"}:
            raise ValueError("output_format must be 'code' or 'answer'.")
        self.output_format = output_format
        self.source_path = source if isinstance(source, str) else None
        self.survey = survey
        self.questions = {q["id"]: q for q in self.survey.questions}
        self.clean = clean
        self.cleaner = cleaner or ResponseCleaner(survey = survey)
        self.clone = False
        
        # Process responses according to their source format
        if isinstance(source, str):
            self.responses = self._load_from_csv(source)
        elif isinstance(source, list):
            self.responses = self._load_from_list(source)
        else:
            raise ValueError("Unsupported response source type.")

    def _load_from_csv(self, csv_path):
        """
        Internal method to load response matrix from a CSV file.

        Args:
            csv_path (str): Path to matrix-formatted CSV.

        Returns:
            dict: Nested response dictionary keyed by qid and eid.
        """        
        matrix = defaultdict(dict)
        with open(csv_path, newline = '', encoding = 'utf-8') as f:
            reader =  csv.reader(f)
            header = next(reader)
            eid_cols = header[1:]

            for row in reader:
                if not row:
                    continue
                qid = row[0].strip()
                if qid not in self.questions:
                    continue
                for eid, raw_answer in zip(eid_cols, row[1:]):
                    self._process_row(qid, eid.strip(), raw_answer.strip(), matrix)
        return matrix

    def _load_from_list(self, response_list):
        """
        Internal method to load response matrix from a list of dicts.

        Args:
            response_list (list): List of dictionaries with keys: 'qid', 'eid', 'answer'.

        Returns:
            dict: Nested response dictionary keyed by qid and eid.
        """        
        matrix = defaultdict(dict)
        for row in response_list:
            qid = row["qid"].strip()
            eid = row["eid"].strip()
            raw_answer = row["answer"].strip()
            if qid not in self.questions:
                continue
            self._process_row(qid, eid, raw_answer, matrix)
        return matrix
    
    def _process_row(self, qid, eid, raw_answer, matrix):
        """
        Internal method to normalize and store a response entry.

        Args:
            qid (str): Question ID.
            eid (str): Endowment ID (agent or respondent).
            raw_answer (str): Original answer text.
            matrix (dict): Nested response matrix to populate.
        """
        if self.clean and self.cleaner:
            raw_answer = self.cleaner.clean(raw_answer, qid)        
        if self.output_format == "answer":
            matrix[qid][eid] = raw_answer
        else:
            mapping = self.questions[qid].get("answer_to_code", {})
            code = mapping.get(self._normalize(raw_answer))
            if code is not None:
                matrix[qid][eid] = code
    
    def _normalize(self, text):
        """
        Normalize an answer string for mapping.

        Args:
            text (str): Raw text answer.

        Returns:
            str: Lowercased, stripped answer with trailing punctuation removed.
        """        
        return text.strip().lower().rstrip('.')
    
    def get(self, qid, eid):
        """
        Retrieve the response of a specific endowment to a specific question.

        Args:
            qid (str): Question ID.
            eid (str): Endowment ID.

        Returns:
            str or None: Encoded or raw response, or None if not found.
        """
        return self.responses.get(qid, {}).get(eid, None)
    
    def get_question_vector(self, qid):
        """
        Get all responses to a specific question.

        Args:
            qid (str): Question ID.

        Returns:
            dict: Mapping from eid to response value.
        """        
        return self.responses.get(qid, {})
    
    def get_agent_vector(self, eid):
        """
        Get all responses given by a specific agent.

        Args:
            eid (str): Endowment ID.

        Returns:
            dict: Mapping from qid to response value.
        """        
        return {qid: agents[eid] for qid, agents in self.responses.items() if eid in agents}
    
    def get_matrix_by_split(
        self,
        split: str,
        survey=None,
        dropna: bool = False
    ) -> pd.DataFrame:
        """
        Returns a [qid × eid] matrix of responses for a given question split.

        Args:
            split (str): Split name (e.g., 'train', 'valid', 'test').
            survey (Survey, optional): Defaults to self.survey if not specified.
            dropna (bool): If True, drop rows with any missing values.

        Returns:
            pd.DataFrame: Response matrix for the specified question split.
        """
        if survey is None:
            survey = self.survey

        qids = [q["id"] for q in survey.get_questions_by_split(split)]
        all_eids = sorted({eid for q in self.responses.values() for eid in q})

        data = []
        for qid in sorted(qids):
            row = {eid: self.responses[qid].get(eid, None) for eid in all_eids}
            row["qid"] = qid
            data.append(row)

        df = pd.DataFrame(data).set_index("qid")

        if dropna:
            df = df.dropna()

        return df

    def sample_fraction(self, fraction: float, seed: int = 101):
        """
        Samples a fraction of questions while preserving the original split proportions.
        
        Args:
            fraction (float): Fraction of total questions to retain (0 < fraction <= 1).
            seed (int): Random seed for reproducibility.
        
        Returns:
            Responses: A new Responses object containing only the sampled questions.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be between 0 and 1")

        random.seed(seed)

        # Step 1: Group questions by split
        splits = ["train", "valid", "test"]
        questions_by_split = {
            split: self.survey.get_questions_by_split(split) for split in splits
        }

        # Step 2: Determine number to sample from each split
        sampled_qids = []
        for split, qlist in questions_by_split.items():
            qids = [q["id"] for q in qlist]
            n_to_sample = max(1, int(len(qids) * fraction)) if qids else 0
            sampled_qids.extend(random.sample(qids, n_to_sample))

        # Step 3: Subset the question list and response dict
        new_questions = [deepcopy(self.questions[qid]) for qid in sampled_qids if qid in self.questions]
        new_responses = {qid: deepcopy(self.responses[qid]) for qid in sampled_qids if qid in self.responses}

        # Step 4: Create a new Survey and Responses instance
        new_survey = Survey.from_questions(new_questions)
        new_obj = Responses(
            source=[],  # dummy, bypassed by manual assignment below
            survey=new_survey,
            output_format=self.output_format,
            clean=self.clean,
            cleaner=self.cleaner,
        )
        new_obj.responses = new_responses
        new_obj.questions = {q["id"]: q for q in new_questions}
        new_obj.clone = True
        return new_obj
    
    def sample_trainvalid_fraction(self, fraction: float, seed: int = 101):
        """
        Samples a fraction of questions from the train and valid splits while keeping
        the test split fully intact. Preserves original train/valid proportions.

        Args:
            fraction (float): Fraction of train+valid questions to retain (0 < fraction <= 1).
            seed (int): Random seed for reproducibility.

        Returns:
            Responses: A new Responses object containing the sampled questions.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be between 0 and 1")

        random.seed(seed)

        # Step 1: Get questions by split
        train_qs = self.survey.get_questions_by_split("train")
        valid_qs = self.survey.get_questions_by_split("valid")
        test_qs = self.survey.get_questions_by_split("test")

        # Step 2: Sample from train and valid
        sampled_qids = []
        for split_qs in [train_qs, valid_qs]:
            qids = [q["id"] for q in split_qs]
            n_to_sample = max(1, int(len(qids) * fraction)) if qids else 0
            sampled_qids.extend(random.sample(qids, n_to_sample))

        # Step 3: Always include all test questions
        sampled_qids.extend(q["id"] for q in test_qs)

        # Step 4: Subset questions and responses
        new_questions = [deepcopy(self.questions[qid]) for qid in sampled_qids if qid in self.questions]
        new_responses = {qid: deepcopy(self.responses[qid]) for qid in sampled_qids if qid in self.responses}

        # Step 5: Construct new Survey and Responses object
        new_survey = Survey.from_questions(new_questions)
        new_obj = Responses(
            source=[],  # dummy placeholder
            survey=new_survey,
            output_format=self.output_format,
            clean=self.clean,
            cleaner=self.cleaner,
        )
        new_obj.responses = new_responses
        new_obj.questions = {q["id"]: q for q in new_questions}
        new_obj.clone = True
        return new_obj

    def save(self, path):
        """
        Save the response matrix to a matrix-style CSV file.

        Args:
            path (str): Output file path.
        """
        # Collect all unique eids across all questions
        all_eids = sorted({eid for responses in self.responses.values() for eid in responses})
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["qid"] + all_eids)

            # Write one row per qid
            for qid in sorted(self.responses.keys()):
                row = [qid]
                for eid in all_eids:
                    val = self.responses[qid].get(eid, "")
                    row.append(val)
                writer.writerow(row)
    
    @classmethod
    def from_agent_records(cls, agent_records: list[dict], survey, 
                           output_format: str = "answer", clean: bool = True, cleaner=None):
        """
        Construct a Responses object from a list of agent records.

        Args:
            agent_records (list[dict]): Each record should have:
                - "agent_id": str, the EID
                - "responses": list of answers (aligned with survey.questions)
            survey (Survey): Survey instance (with questions metadata).
            output_format (str): "answer" (default) or "code".
            clean (bool): Whether to apply cleaning and normalization.
            cleaner (ResponseCleaner, optional): Optional custom cleaner.

        Returns:
            Responses: A populated Responses instance.
        """
        qids = [q["id"] for q in survey.questions]
        flat_records = []
        for record in agent_records:
            eid = record["agent_id"]
            for qid, answer in zip(qids, record["responses"]):
                flat_records.append({
                    "eid": eid,
                    "qid": qid,
                    "answer": answer if answer is not None else ""
                })

        return cls(flat_records, survey, output_format=output_format, clean=clean, cleaner=cleaner)

    def clone_with_subset(self, subset_qids: list, subset_survey = None):
        """
        Returns a new Responses object restricted to a subset of question IDs.

        Args:
            subset_qids (list[str]): List of question IDs to retain.
            subset_survey (Survey, optional): A survey object that contains only the selected questions.
                                              If not provided, the existing survey will be filtered.

        Returns:
            Responses: A new Responses instance with restricted question-response matrix.
        """

        new = deepcopy(self)
        new.responses = {
            qid: ans_dict for qid, ans_dict in self.responses.items()
            if qid in subset_qids
        }

        if subset_survey is not None:
            new.survey = subset_survey
            new.questions = {q["id"]: q for q in subset_survey.questions}
        else:
            new.questions = {
                qid: q for qid, q in self.questions.items()
                if qid in subset_qids
            }
            # subset survey manually if needed
            new.survey = self.survey.__class__.from_questions(
                questions=list(new.questions.values()),
                config=self.survey.config,
                csv_path=self.survey.csv_path,
                config_path=self.survey.config_path,
            )

        new.clone = True
        return new
        
    def clone_with_agents(self, subset_eids: set[str]):
        """
        Returns a new Responses object restricted to a subset of agent (EID) IDs.

        Args:
            subset_eids (set[str]): Set of agent EIDs to retain.

        Returns:
            Responses: A new Responses instance with restricted agent-response matrix.
        """
        new = deepcopy(self)

        # Filter response matrix to include only selected agents
        new.responses = {
            qid: {eid: val for eid, val in resp_dict.items() if eid in subset_eids}
            for qid, resp_dict in self.responses.items()
        }

        new.clone = True
        return new

    def to_format(self, output_format: str):
        """
        Return a new Responses object with the same data but a different output_format.
        """
        flat_records = [
            {"eid": eid, "qid": qid, "answer": ans}
            for qid, eid_dict in self.responses.items()
            for eid, ans in eid_dict.items()
        ]
        return Responses(flat_records, self.survey,
                        output_format=output_format,
                        clean=self.clean,
                        cleaner=self.cleaner)

    def __len__(self):
        """
        Returns:
            int: Number of unique agents (EIDs) in the response matrix.
        """
        return len({eid for q in self.responses.values() for eid in q})
    

class BinaryExtendedResponses(Responses):
    """
    Extension of the Responses class for binary-extended surveys.

    Automatically maps multiclass responses into binary format by expanding each
    original question into one binary sub-question per answer option.

    Attributes:
        original_questions_by_id (dict): Mapping of original (pre-expansion) question IDs to metadata.
        clean (bool): Whether to apply response cleaning and normalization (inherited from Responses).
        cleaner (ResponseCleaner): Instance used for cleaning answers (inherited from Responses).
    """

    def __init__(self, source, survey, output_format="code", clean=True, cleaner = None):    
        """
        Initialize the BinaryExtendedResponses object.

        Args:
            source (str or list): Source of responses. Either a path to a matrix-style CSV file
                or a list of dictionaries with 'qid', 'eid', and 'answer'.
            survey (BinaryExtendedSurvey): Survey instance with original and binary-expanded questions.
            output_format (str): Format to store responses:
                - 'code': (default) store 1/0 indicating binary match.
                - 'answer': store 'True'/'False' as strings.
            clean (bool): Whether to apply cleaning and normalization to raw answers before mapping.
            cleaner (ResponseCleaner, optional): Optional custom cleaner to use. If None and `clean` is True,
                a default cleaner is instantiated using the original survey.
        """
        if not hasattr(survey, "original_questions"):
            raise ValueError("BinaryExtendedResponses requires a BinaryExtendedSurvey instance.")
        
        self.original_questions_by_id = {
            q["id"]: q for q in survey.original_questions
        }

        cleaner = cleaner or ResponseCleaner(survey = Survey.from_questions(survey.original_questions))

        super().__init__(source, survey, output_format, clean, cleaner)
        self.clone = False

    def _load_from_csv(self, csv_path):
        """
        Load and transform responses from a matrix-style CSV for binary questions.

        Args:
            csv_path (str): Path to response CSV.

        Returns:
            dict: Nested binary response dictionary keyed by binary_qid and eid.
        """
        matrix = defaultdict(dict)
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            eid_cols = header[1:]

            for row in reader:
                if not row:
                    continue
                qid = row[0].strip()
                if qid not in self.original_questions_by_id:
                    continue
                for eid, raw_answer in zip(eid_cols, row[1:]):
                    self._process_row(qid, eid.strip(), raw_answer.strip(), matrix)
        return matrix
    
    def _load_from_list(self, response_list):
        """
        Load and transform responses from a list of raw response dicts.

        Args:
            response_list (list): List of raw responses with 'qid', 'eid', and 'answer'.

        Returns:
            dict: Nested binary response dictionary.
        """
        matrix = defaultdict(dict)
        for row in response_list:
            qid = row["qid"].strip()
            eid = row["eid"].strip()
            raw_answer = row["answer"].strip()

            if qid not in self.original_questions_by_id:
                continue

            self._process_row(qid, eid, raw_answer, matrix)
        return matrix


    def _process_row(self, qid, eid, raw_answer, matrix):
        """
        Expand one multiclass response into multiple binary responses.

        Args:
            qid (str): Original (non-binary) question ID.
            eid (str): Endowment ID.
            raw_answer (str): Original raw answer string.
            matrix (dict): Binary response matrix to populate.
        """
        if self.clean and self.cleaner:
            og_raw = raw_answer
            raw_answer = self.cleaner.clean(raw_answer, qid)         
        norm_answer = self._normalize(raw_answer)
        output_format = self.output_format

        original_question = self.original_questions_by_id.get(qid)
        if original_question is None:
            return

        code_to_answer = original_question.get("code_to_answer", {})
        if len(code_to_answer) < 2:
            return super()._process_row(qid, eid, raw_answer, matrix)

        answer_to_code = original_question.get("answer_to_code", {})
        selected_code = answer_to_code.get(norm_answer)
        if selected_code is None:
            print(f"[WARN] Could not map answer for QID={qid}, EID={eid}, Raw_answer = '{og_raw}', Cleaned_answer='{raw_answer}', Norm='{norm_answer}'")
            return

        for binary_q in self.survey.questions:
            if binary_q.get("base_id") != qid:
                continue

            binary_qid = binary_q["id"]
            binary_code = binary_q["base_code"]

            is_match = str(selected_code) == str(binary_code)
            value = (
                "True" if is_match else "False"
            ) if output_format == "answer" else (
                "1" if is_match else "0"
            )
            matrix[binary_qid][eid] = value
        
        return
    
    def clone_with_subset(self, subset_qids: list[str], subset_survey: BinaryExtendedSurvey = None):
        """
        Clone a BinaryExtendedResponses object with only a subset of binary question IDs.

        Args:
            subset_qids (list[str]): List of binary question IDs to retain.
            subset_survey (BinaryExtendedSurvey, optional): A filtered survey instance.
                If None, it will be inferred from the current survey.

        Returns:
            BinaryExtendedResponses: New response object with restricted binary questions.
        """
        from copy import deepcopy

        if subset_survey is None:
            # Subset survey using clone_with_subset
            subset_survey = self.survey.__class__.clone_with_subset(self.survey, [
                q for q in self.survey.questions if q["id"] in subset_qids
            ])

        # Deepcopy and filter responses
        new_responses = deepcopy(self.responses)
        new_responses = {qid: resp for qid, resp in new_responses.items() if qid in subset_qids}

        # Update questions metadata
        new_questions = {qid: q for qid, q in self.questions.items() if qid in subset_qids}

        # Create new instance
        new_obj = self.__class__(
            source=[],  # dummy
            survey=subset_survey,
            output_format=self.output_format,
            clean=self.clean,
            cleaner=self.cleaner
        )
        new_obj.responses = new_responses
        new_obj.questions = new_questions
        new_obj.original_questions_by_id = {
            k: v for k, v in self.original_questions_by_id.items()
            if any(q["base_id"] == k for q in subset_survey.questions)
        }
        new_obj.clone = True
        return new_obj
    

class ResponseUtils:
    @staticmethod
    def analyze_missing_mappings(code_responses, answer_responses, split: str = None, verbose = True):
        """
        Uses pandas to analyze missing or unmapped coded responses.

        Args:
            code_responses (Responses): Responses with output_format='code'.
            answer_responses (Responses): Responses with output_format='answer'.
            split (str or None): Which survey split to use (e.g., 'train').
                                If None, combines all splits: 'train', 'valid', 'test'.
            verbose (bool): If True, print analysis.

        Returns:
            Tuple:
            - problematic_answers_by_qid: dict of {qid: [list of problematic answers]}
            - agent_count: int (number of unique agents with ≥1 missing across splits)
            - question_count: int (number of unique questions with ≥1 missing across splits)
            - code_to_answer_by_qid: dict of {qid: code_to_answer}
            - output_string: text analysis report
        """
        splits_to_check = [split] if split else ["train", "valid", "test"]
        problematic_answers_by_qid = defaultdict(set)
        all_missing_agents = set()
        all_missing_questions = set()
        code_to_answer_by_qid = {}

        for sp in splits_to_check:
            df_code = code_responses.get_matrix_by_split(sp)
            df_answer = answer_responses.get_matrix_by_split(sp)

            mask = df_code.isna()
            for qid, row in mask.iterrows():
                for eid, is_missing in row.items():
                    if is_missing:
                        ans = df_answer.at[qid, eid]
                        if pd.notna(ans):
                            problematic_answers_by_qid[qid].add(ans)
                            all_missing_agents.add(eid)
                            all_missing_questions.add(qid)

        # Final formatting
        problematic_answers_by_qid = {
            qid: sorted(list(v)) for qid, v in problematic_answers_by_qid.items()
        }
        code_to_answer_by_qid = {
            qid: code_responses.questions[qid].get("code_to_answer", {})
            for qid in problematic_answers_by_qid
        }

        agent_count = len(all_missing_agents)
        question_count = len(all_missing_questions)

        lines = [
            "=== Missing Code Mapping Summary ===\n",
            f"Total questions with unmapped responses: {question_count}",
            f"Total agents affected (with at least one unmapped response): {agent_count}",
            "\n--- Problematic Answers by Question ---"
        ]

        # Loop through all QIDs with unmapped responses
        for qid, answers in problematic_answers_by_qid.items():
            code_map = code_to_answer_by_qid.get(qid, {})
            lines.append(f"- QID: {qid}\n")
            lines.append(f"  Unmapped Answers: {answers}")
            if code_map:
                lines.append(f"  Existing code_to_answer mapping: {code_map}")
            lines.append("")

        output_string = "\n".join(lines)
        if verbose:
            print(output_string)


        return problematic_answers_by_qid, agent_count, question_count, code_to_answer_by_qid, output_string
    
    @staticmethod
    def plot_response_distribution(responses, qid, top_k: int = None, title: str = None, sort_by: str = "code_order"):
        """
        Plots a horizontal histogram of response distribution for a given question,
        treating NaN (missing) as a valid category.

        If `responses` is in 'code' format, this function reconstructs the answer distribution
        using the full response matrix to detect unmapped answers (NaN in code but agent responded).

        Args:
            responses (Responses): A Responses instance.
            qid (str): The question ID to visualize.
            top_k (int, optional): Limit to top-k most frequent answers (for readability).
            title (str, optional): Custom title for the plot.
            sort_by (str): 'frequency', 'label', or 'code_order'
        """
        # Handle code vs answer output
        output_fortmat = getattr(responses, "output_format", None)
        if output_fortmat == "code":
            split = responses.questions[qid].get("split")
            if split:
                df = responses.get_matrix_by_split(split)
                row = df.loc[qid]
                series = row.fillna("NaN")
            else:
                series = pd.Series(responses.get_question_vector(qid)).fillna("[MISSING]")
        else:
            series = pd.Series(responses.get_question_vector(qid)).fillna("[MISSING]")

        counts = series.value_counts()

        if top_k is not None:
            counts = counts.head(top_k)

        # Sort logic
        if sort_by == "label":
            counts = counts.sort_index()
        elif sort_by == "frequency":
            counts = counts.sort_values(ascending=True)
        elif sort_by == "code_order":
            code_to_answer = responses.questions[qid].get("code_to_answer", {})
            if output_fortmat == "answer":
                all_labels = list(code_to_answer.values())
            else:
                all_labels = [str(k) for k in code_to_answer.keys()]

            if output_fortmat == "answer":
                answer_to_rank = {ans: i for i, ans in enumerate(code_to_answer.values())}

                def sort_key(ans):
                    return answer_to_rank.get(ans, float("inf"))
            else:
                code_to_rank = {str(code): i for i, code in enumerate(code_to_answer.keys())}

                def sort_key(code):
                    return code_to_rank.get(str(code), float("inf"))
            counts = counts.reindex(all_labels, fill_value=0)
            counts = counts.sort_index(key=lambda idx: [sort_key(val) for val in idx])
        else:
            raise ValueError("sort_by must be one of: 'frequency', 'label', 'code_order'")

        # Compute entropy from raw (unsorted) counts
        entropy_val = ResponseUtils.categorical_entropy(series.value_counts().values.tolist())

        ax = counts.plot(
            kind="barh",
            figsize=(8, 0.5 * len(counts) + 1),
            color="#588c73",
            edgecolor="black"
        )

        for i, val in enumerate(counts.values):
            ax.text(val + 0.5, i, str(val), va="center")

        ax.set_xlabel("Frequency")
        entropy_str = f" (Entropy = {entropy_val:.2f})"
        question_text = responses.questions[qid].get("question", "")
        ax.set_title(title or f"Response Distribution for QID: {qid}{entropy_str}\n{question_text}")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_question_entropy_by_split(responses, split=None, sort=True, figsize=(16, 5), top_n=None, save_path=None):
        """
        Plots entropy of response distributions across questions, color-coded by question split.

        Args:
            responses (Responses): A Responses instance (usually with output_format='code' or 'answer').
            split (str or list[str], optional): Restrict to specific split(s) ('train', 'valid', 'test').
            sort (bool): Whether to sort bars by entropy.
            figsize (tuple): Size of the plot.
            top_n (int or None): Show only top-N highest-entropy questions.
            save_path (str or None): If set, save plot to this path; otherwise show it.
        """
        if isinstance(split, str):
            split = [split]

        qid_entropy = {}
        qid_split = {}

        SPLIT_COLOR_MAP = {
        "train": "#87CEEB",  
        "valid": "#9ed670", 
        "test": "#f9d62e",    
        "unknown": "#8172B3"
        }

        # Compute entropy for all or filtered questions
        for qid, meta in responses.questions.items():
            q_split = meta.get("split", "unknown")
            if split and q_split not in split:
                continue
                
            # Determine vocab source based on output format
            if responses.output_format == "code":
                raw_vocab = meta.get("code_to_answer", {}).keys()
            elif responses.output_format == "answer":
                raw_vocab = meta.get("code_to_answer", {}).values()
            else:
                raise ValueError(f"Unsupported output_format: {responses.output_format}")
            
            vocab = [str(v) for v in raw_vocab]
            if not vocab:
                continue

            vec = pd.Series(responses.get_question_vector(qid)).dropna().astype(str)
            freqs = vec.value_counts(normalize=True)
            counts = vec.value_counts()
            full_counts = [counts.get(cat, 0) for cat in vocab]
            total = sum(full_counts)
            if total == 0:
                entropy = 0.0
            else:
                probs = [c / total for c in full_counts if c > 0]
                entropy = -sum(p * np.log2(p) for p in probs)
                entropy /= np.log2(len(vocab))  # normalize

            qid_entropy[qid] = entropy
            qid_split[qid] = q_split

        df = pd.DataFrame({
            "qid": list(qid_entropy.keys()),
            "normalized_entropy": list(qid_entropy.values()),
            "split": [qid_split[qid] for qid in qid_entropy]
        })

        if sort:
            df = df.sort_values(by="normalized_entropy", ascending=True)

        if top_n is not None:
            df = df.tail(top_n)

        df["color"] = df["split"].map(SPLIT_COLOR_MAP).fillna(SPLIT_COLOR_MAP["unknown"])
        # Plot
        fig, ax = plt.subplots(figsize = figsize)
        bars = ax.bar(df["qid"], df["normalized_entropy"], color=df["color"])
        ax.set_title("Normalized Question Entropy" + (f" (Filtered by: {split})" if split else ""))
        ax.set_xlabel("Question ID")
        ax.set_ylabel("Normalized Entropy")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["qid"], rotation=90)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.margins(x=0.01)
        # Legend
        split_labels = df["split"].unique()
        handles = [plt.Line2D([0], [0], color=SPLIT_COLOR_MAP[s], lw=6) for s in split_labels]
        ax.legend(
            handles,
            split_labels,
            title="Split",
            loc="upper left",
            frameon=True
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def categorical_entropy(counts: list[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts if c > 0]
        entropy = -sum(p * np.log2(p) for p in probs)
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy
    
    @staticmethod
    def aggregate_weighted_responses(responses: Responses, agent_weights: dict[str, float]) -> dict[str, dict[str, float]]:
        """
        Aggregates agent responses using lasso weights into categorical distributions.

        Args:
            responses (Responses): A Responses object containing agent answers.
            agent_weights (dict[str, float]): Mapping from agent ID (eid) to model-assigned weight.

        Returns:
            dict[str, dict[str, float]]: A nested dictionary mapping:
                question_id -> {category_value -> weighted_probability}
        """
        result = {}

        for qid, answer_dict in responses.responses.items():
            counts = {}
            total_weight = 0.0

            for eid, ans in answer_dict.items():
                if eid not in agent_weights:
                    continue
                weight = agent_weights[eid]
                total_weight += weight
                counts[ans] = counts.get(ans, 0.0) + weight

            if total_weight > 0:
                probs = {ans: w / total_weight for ans, w in counts.items()}
            else:
                probs = {}

            result[qid] = probs

        return result
    
    @staticmethod
    def count_valid_endowments(responses: Responses, split: str = None) -> int:
        """
        Count the number of agents (EIDs) with valid responses to all questions in the given split.

        Args:
            responses (Responses): Responses instance with mapped answers (typically output_format='code').
            split (str or None): Question split to check ('train', 'valid', 'test').
                                If None, uses all questions.

        Returns:
            int: Number of valid agents with no missing response for any question in the split.
        """
        if split:
            df = responses.get_matrix_by_split(split)
        else:
            all_qids = sorted(responses.responses.keys())
            all_eids = sorted({eid for q in responses.responses.values() for eid in q})
            data = []
            for qid in all_qids:
                row = {eid: responses.responses[qid].get(eid, None) for eid in all_eids}
                row["qid"] = qid
                data.append(row)
            df = pd.DataFrame(data).set_index("qid")

        # Count agent columns (EIDs) with no missing responses
        valid_eids = [eid for eid in df.columns if not df[eid].isnull().any()]
        return len(valid_eids)
    
    @staticmethod
    def save_answer_distributions(responses: Responses, path: str, normalize: bool = True):
        """
        Generate a summary JSON file of answer distributions per question.

        Args:
            responses (Responses): A Responses object (code or answer format).
            path (str): Path to save the JSON output.
            normalize (bool): If True, convert counts to proportions.
        """
        summary = {}

        for qid, ans_dict in responses.responses.items():
            # Count frequencies of answers across agents
            counts = Counter(ans_dict.values())
            total = sum(counts.values())

            # Get mapping for this question (e.g. {1: "Yes", 2: "No"})
            code_to_answer = responses.questions[qid].get("code_to_answer", {})

            # Build distribution with answer labels
            if normalize and total > 0:
                dist = {
                    code_to_answer.get(code, str(code)): v / total
                    for code, v in counts.items()
                }
            else:
                dist = {
                    code_to_answer.get(code, str(code)): int(v)
                    for code, v in counts.items()
                }


            summary[qid] = dist

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary