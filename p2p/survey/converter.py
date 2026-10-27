import ast
import csv
import yaml
import copy
import random
import warnings
from collections import Counter

class Survey:
    """Load, manage, split, and save survey questions.

    The YAML schema maps the logical ``question_id``, ``question_text``,
    ``code_to_answer``, and optional ``split`` fields to CSV column names.

    Attributes:
        csv_path: Path to the survey CSV.
        config: Loaded YAML field mapping.
        questions: Parsed question dictionaries.  Each record contains an
            ID, question text, code/answer mappings, and an optional split.
    """

    def __init__(self, csv_path, config_path):
        self.csv_path = csv_path
        self.config_path = config_path
        self.config = self._load_yaml(config_path)
        self.questions = self._load_csv()
        self.clone = False

    def _load_yaml(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def _load_csv(self):
        questions = []    
        with open(self.csv_path, newline = '', encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid_field = self.config['question_id']
                qtext_field = self.config['question_text']
                mapping_field = self.config['code_to_answer']
                split_field = self.config.get('split', 'split')

                id = row.get(qid_field, '').strip()
                qtext = row.get(qtext_field, '').strip()
                mapping_str = row.get(mapping_field, '{}')

                try:
                    code_to_answer = ast.literal_eval(mapping_str)
                except Exception as e:
                    print(f"Failed to parse mapping for {id}: {e}")
                    code_to_answer = {}                

                answer_to_code = {self._normalize(v): str(k) for k, v in code_to_answer.items()}

                split = row.get(split_field, '').strip()

                questions.append({
                    'id': id,
                    'question': qtext,
                    'code_to_answer': code_to_answer,
                    'answer_to_code': answer_to_code,
                    'split': split
                })
        return questions
    
    def _normalize(self, text):
        return text.strip().lower().rstrip('.')
    
    def get_questions(self):
        return self.questions
    
    def get_prompt_text(self, qdict):
        # qdict is a dictionary item of the questions attribute
        opts = ", ".join([f"[{k}] {v}" for k, v in qdict['code_to_answer'].items()])
        return f"{qdict['question']} ({opts})"
    
    def get_question_by_id(self, qid):
        return next((q for q in self.questions if q['id'] == qid), None)
    
    def split_questions(self, method = 'manual', split_map = None, train_ratio=0.7, valid_ratio = 0.15, seed = None, save = False, save_path = None):
        """
        Assigns 'split' attribute to each question.

        Parameters:
        - method: "manual" or "from_map"
        - split_map: dict mapping question IDs to split labels
        - train_ratio, valid_ratio: used only if method="manual"
        - seed: random seed for reproducibility
        - save: if True, write the split info back to a CSV
        - save_path: custom output path; defaults to original csv_path if None
        """
        
        if any(q.get('split') for q in self.questions) and not save:
            warnings.warn("You are re-splitting a survey that already contains split assignments.")

        if method == 'from_map':
            if not split_map:
                raise ValueError("You must provide split_map when method = 'from_map'.")
            for q in self.questions:
                q['split'] = split_map.get(q['id'], 'train') # default to train 
        elif method == 'manual':
            if seed is not None:
                random.seed(seed)
            total = len(self.questions)
            indices = list(range(total))
            random.shuffle(indices)

            train_end = int(train_ratio * total)
            valid_end = train_end + int(valid_ratio * total)

            for i, idx in enumerate(indices):
                if i < train_end:
                    self.questions[idx]['split'] = 'train'
                elif i < valid_end:
                    self.questions[idx]['split'] = 'valid'
                else:
                    self.questions[idx]['split'] = 'test'
        else:
            raise ValueError("Unknown split method. Use 'manual' or 'from_map'.")
        
        if save:
            self._save_split_csv(save_path)
            self.report_splits()
    
    def save(self, filepath=None):
        """
        Saves the current survey state (including splits if present) to a CSV.
        Column names are derived from the YAML config. If splits exist but no
        split column is specified in the config, a column named 'split' is added.
        """

        output_path = filepath if filepath else self.csv_path

        # Load column names from config
        id_field = self.config['question_id']
        text_field = self.config['question_text']
        mapping_field = self.config['code_to_answer']
        split_field = self.config.get('split', 'split')  # default to 'split'

        # Decide whether to include the split column
        include_split = any(q.get('split') is not None for q in self.questions)

        # Define CSV header
        fieldnames = [id_field, text_field, mapping_field]
        if include_split:
            fieldnames.append(split_field)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for q in self.questions:
                row = {
                    id_field: q['id'],
                    text_field: q['question'],
                    mapping_field: str(q['code_to_answer']),
                }
                if include_split:
                    row[split_field] = q.get('split') or ''
                writer.writerow(row)

    
    def get_questions_by_split(self, split_name):
        return [q for q in self.questions if q.get('split') == split_name]

    def get_split_counts(self):
        return Counter(q.get('split', 'unspecified') for q in self.questions)
    
    def get_option_order_text(self, qid):
        q = self.get_question_by_id(qid)
        mapping = q["code_to_answer"]            # {"A": "Very important", ...}
        # order of texts according to code order
        return [mapping[k] for k in mapping.keys()]

    def report_splits(self, show_percentage=True):
        """
        Prints a summary of the split distribution among survey questions.

        Parameters:
        - show_percentage (bool): If True, also print percentages.
        """
        total = len(self.questions)
        split_counts = self.get_split_counts()

        print("Split Summary:")
        print("-" * 30)
        for split_name, count in split_counts.items():
            if show_percentage:
                percent = (count / total) * 100
                print(f"{split_name:<12} {count:>4} ({percent:5.1f}%)")
            else:
                print(f"{split_name:<12} {count:>4}")
        print("-" * 30)
        print(f"Total questions: {total}")

    def _save_split_csv(self, save_path=None):
        self.save(filepath=save_path)

    @classmethod
    def from_questions(cls, questions, config=None, csv_path=None, config_path=None):
        """
        Instantiate a Survey object from a list of preloaded questions.

        Args:
            questions (list): List of question dicts. Must include keys like 'id', 'question', 'code_to_answer', etc.
            config (dict, optional): YAML-style config dict specifying field names.
            csv_path (str, optional): Placeholder CSV path, if needed for compatibility.
            config_path (str, optional): Placeholder config path.

        Returns:
            Survey: An instantiated Survey object with preloaded questions.
        """
        obj = cls.__new__(cls)
        obj.csv_path = csv_path
        obj.config_path = config_path
        obj.config = config or {
            'question_id': 'id',
            'question_text': 'question',
            'code_to_answer': 'code_to_answer',
            'split': 'split'
        }
        obj.questions = questions
        return obj
    
    @classmethod
    def clone_with_subset(cls, existing_survey, subset_questions):
        """
        Clone a Survey object but restrict it to a subset of questions.

        Args:
            existing_survey (Survey): The original Survey instance.
            subset_questions (list): List of question dicts to retain.

        Returns:
            Survey: A cloned Survey object containing only the subset.
        """
        obj = cls.__new__(cls)
        obj.csv_path = existing_survey.csv_path
        obj.config_path = existing_survey.config_path
        obj.config = existing_survey.config
        obj.questions = subset_questions
        obj.clone = True
        return obj



class BinaryExtendedSurvey(Survey):
    def __init__(self, csv_path, config_path):
        super().__init__(csv_path, config_path)
        self.original_questions = copy.deepcopy(self.questions)
        self.binary_to_original_map = {}
        self.original_to_binary_map = {}
        self.questions = self._expand_to_binary(self.questions)
        self.clone = False

    def _expand_to_binary(self, questions):
        binary_questions = []
        for q in questions:
            code_to_answer = q.get("code_to_answer", {})
            answer_to_code = q.get("answer_to_code", {})
            n_options = len(code_to_answer)

            if n_options <= 1:
                # Malformed or meaningless — keep as-is
                binary_questions.append(q)
                continue

            for idx, (code, answer) in enumerate(code_to_answer.items()):
                binary_id = f"{q['id']}_{idx}"
                code_to_answer_bin = {1: "True", 0: "False"}
                answer_to_code_bin = {
                    self._normalize(v): str(k) for k, v in code_to_answer_bin.items()
                }

                binary_q = {
                    "id": binary_id,
                    "question": f"{q['question']} ({answer})",
                    "code_to_answer": code_to_answer_bin,
                    "answer_to_code": answer_to_code_bin,
                    "split": q.get("split"),
                    "base_id": q["id"],
                    "base_code": code,
                    "base_answer": answer,
                }
                binary_questions.append(binary_q)
                self.binary_to_original_map[binary_id] = {
                    "original_id": q["id"],
                    "original_code": code,
                    "original_answer": answer,
                }
                
                # Add to original-to-binary map
                self.original_to_binary_map.setdefault(q["id"], []).append(binary_id)

        return binary_questions
    
    def reverse_map_question(self, binary_id):
        return self.binary_to_original_map.get(binary_id, None)
    
    def get_binary_variants(self, original_id):
        """
        Return list of binary QIDs derived from a given original question ID.
        """
        return self.original_to_binary_map.get(original_id, [])
    
    @classmethod
    def from_survey(cls, survey):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(survey.csv_path, survey.config_path)
        obj.original_questions = copy.deepcopy(survey.questions)
        obj.binary_to_original_map = {}
        obj.original_to_binary_map = {}
        obj.questions = obj._expand_to_binary(obj.original_questions)
        return obj
    
    @classmethod
    def clone_with_subset(cls, existing_survey, subset_questions):
        """
        Clone a BinaryExtendedSurvey but restrict to a subset of binary questions.
        Preserves original question mappings.
        """
        obj = cls.__new__(cls)
        obj.csv_path = existing_survey.csv_path
        obj.config_path = existing_survey.config_path
        obj.config = existing_survey.config
        obj.original_questions = copy.deepcopy(existing_survey.original_questions)

        obj.questions = subset_questions

        obj.binary_to_original_map = {
            k: v for k, v in existing_survey.binary_to_original_map.items()
            if k in {q["id"] for q in subset_questions}
        }
        obj.original_to_binary_map = {
            k: [qid for qid in v if qid in obj.binary_to_original_map]
            for k, v in existing_survey.original_to_binary_map.items()
        }

        obj.clone = True

        return obj
