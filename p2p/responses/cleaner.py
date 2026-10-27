import re
from rapidfuzz import fuzz

class ResponseCleaner:
    def __init__(self, survey, verbose=False, canonical=True):
        """Map free-text answers onto a question's options.

        Args:
            survey: Survey providing each question's options.
            verbose: Print answers that match no option.
            canonical: Return an accepted answer as the option's own text
                (default). With ``False``, an answer that matches an option
                only after normalisation (case, surrounding space, trailing
                full stop) is returned as the agent wrote it, so it is
                scored verbatim.
        """
        self.survey = survey
        self.verbose = verbose
        self.canonical = canonical

    def _normalize(self, s):
        return s.strip().lower().rstrip(".")

    def _denormalize(self, norm_str, code_to_answer):
        """Convert normalized string back to its canonical form in code_to_answer."""
        for val in code_to_answer.values():
            if self._normalize(val) == norm_str:
                return val
        return norm_str  # fallback

    def _norm_option_map(self, code_to_answer):
        return {self._normalize(v): v for v in code_to_answer.values()}

    def _ensure_str_keys(self, d):
        return {str(k): v for k, v in d.items()}

    def clean(self, response: str, question_id: str) -> str:
        question = self.survey.get_question_by_id(question_id)
        if not question:
            if self.verbose:
                print(f"[Warning] Question ID {question_id} not found.")
            return response

        code_to_answer = self._ensure_str_keys(question.get("code_to_answer", {}))
        answer_to_code = question.get("answer_to_code", {})
        norm_response = self._normalize(response)

        # 1. Exact match to known answer (normalized). Return the option's own
        # text: scoring reads answers as text, so "Neither ..." must become the
        # option "neither ..." here, as every later step already does.
        if norm_response in answer_to_code:
            if not self.canonical:
                return response
            return self._denormalize(norm_response, code_to_answer)

        # 2. Exact match to a known code (with or without brackets)
        match = re.match(r"^\s*\[?([A-Za-z0-9_.-]+)\]?\s*$", response)
        if match:
            code = match.group(1)
            if code in code_to_answer:
                return code_to_answer[code]

        # 3. Remove prefixed code like "[A] Answer", "1.0 - Very happy"
        cleaned = re.sub(r"^\s*\[?[A-Za-z0-9_.-]+\]?\s*[:\-–]?\s*", "", response).strip()
        norm_cleaned = self._normalize(cleaned)
        if norm_cleaned in answer_to_code:
            return self._denormalize(norm_cleaned, code_to_answer)

        # 4. Substring match fallback, sorted by descending length to prioritize specific matches
        norm_options = sorted(self._norm_option_map(code_to_answer).items(), key=lambda x: -len(x[0]))
        for norm_opt, original in norm_options:
            if norm_opt in norm_response or norm_opt in norm_cleaned:
                return original

        # 5. Attempt to fix malformed float-like response like '1.  0' → '1.0'
        try:
            as_number = float(norm_response.replace(" ", ""))
            as_str = str(as_number)
            if as_str in code_to_answer:
                return code_to_answer[as_str]
        except ValueError:
            pass

        # 6. Fuzzy match fallback
        norm_keys = list(answer_to_code.keys())
        scores = [(opt, fuzz.partial_ratio(norm_response, opt)) for opt in norm_keys]
        scores = sorted(scores, key=lambda x: -x[1])
        if scores and scores[0][1] > 95:  # threshold
            best_match = scores[0][0]
            return self._denormalize(best_match, code_to_answer)

        if self.verbose:
            print(f"[Uncleaned] '{response}' (QID: {question_id}) — no match.")
        return response