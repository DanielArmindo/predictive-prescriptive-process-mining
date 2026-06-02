from collections import defaultdict, Counter
from dataclasses import dataclass
import math
import re
from typing import Optional, Any
from itertools import product
from .TemplateHandler import TemplateHandler


INPUT_HISTORY = Optional[list[Optional[str]]]

@dataclass
class MatchResult:
    strategy: str
    confidence: float
    match: list[Optional[str]]

class ContentPredictModel:
    def __init__(self, columnContent: str, columnNumActivity: str, confidence_predict: float = 0.5, min_count: int = 0):
        # Save the counts at the end of each view
        self.view_to_templates = defaultdict(Counter)
        # Stores the most frequently viewed templates associated with each activity
        self.full_views = defaultdict()
        self.min_count = min_count
        self.is_fitted = False
        self.dataset_regex: None | TemplateHandler = None
        self.columnContent = columnContent
        self.columnNumActivity = columnNumActivity
        self.confidence_predict =  confidence_predict

    def normalize_activity(self, act: str) -> str:
        act = re.sub(r'\s+', ' ', act)
        act = re.sub(r'\s*\.\s*', '.', act.strip())
        return act

    def _parse_activity(self, act_str: str) -> tuple[str, Optional[str]]:
        act_str = self.normalize_activity(act_str)
        if '.' in act_str:
            parts = act_str.split('.')
            main = parts[0]
            sub = '.'.join(parts[1:])
            return (main, sub)
        return (act_str, None)

    def _extract_all_logical_views_with_last_idx(
        self, papel_seq: list[str], activity_seq: list[str]
    ) -> list[tuple[list[str], int]]:
        if len(papel_seq) != len(activity_seq):
            raise ValueError("papel_seq e activity_seq devem ter mesmo comprimento")

        subprocesses = defaultdict(list)
        main_positions = {}

        parsed = []
        for idx, act in enumerate(activity_seq):
            main, sub = self._parse_activity(str(act))
            parsed.append((main, sub))
            if sub is None:
                main_positions[main] = idx
            else:
                subprocesses[main].append((sub, idx))

        for main in subprocesses:
            subprocesses[main].sort(key=lambda x: [
                int(part) if part.isdigit() else part
                for part in x[0].split('.')
            ])

        splits = []
        i = 0
        while i < len(parsed):
            main, sub = parsed[i]
            if sub is None:
                if main in subprocesses and subprocesses[main]:
                    splits.append(subprocesses[main])
                    last_sub_idx = max(idx for (_, idx) in subprocesses[main])
                    i = last_sub_idx + 1
                    continue
            i += 1

        if not splits:
            return [(papel_seq, len(papel_seq) - 1)]

        all_combinations = list(product(*splits))
        views_with_last_idx = []

        for choice in all_combinations:
            new_view = []
            new_last_idx = -1
            i = 0
            split_iter = iter(choice)
            while i < len(parsed):
                main, sub = parsed[i]
                if sub is None and main in subprocesses and subprocesses[main]:
                    _, chosen_idx = next(split_iter)
                    new_view.append(papel_seq[chosen_idx])
                    new_last_idx = chosen_idx
                    last_sub_idx = max(idx for (_, idx) in subprocesses[main])
                    i = last_sub_idx + 1
                else:
                    new_view.append(papel_seq[i])
                    new_last_idx = i
                    i += 1

            if new_view:
                views_with_last_idx.append((new_view, new_last_idx))

        return views_with_last_idx


    async def get_templates(self, type_dataset: str, content: dict[str, list[str]], templates: None | str = None):
        model_content = TemplateHandler(type_dataset)
        await model_content.fit(content) if templates == None else await model_content.fit_manual(templates, content)
        self.dataset_regex = model_content


    def fit(self, event_log):
        if self.dataset_regex == None:
            raise RuntimeError("Modelo sem templates !")

        for _, trace in enumerate(event_log):
            events = list(trace)
            if not events:
                continue

            # Prefix construction
            template_history = []
            for prefix_len in range(1, len(events) + 1):
                prefix_events = events[:prefix_len]
                papel_seq = [ev.get("concept:name", "Unknown") for ev in prefix_events]
                activity_seq = [ev.get(self.columnNumActivity, "") for ev in prefix_events]

                papel_seq = [str(p).strip() for p in papel_seq]
                activity_seq = [str(a).strip() for a in activity_seq]

                try:
                    views_with_last_idx = self._extract_all_logical_views_with_last_idx(papel_seq, activity_seq)
                except Exception:
                    views_with_last_idx = [(papel_seq, len(papel_seq) - 1)]

                # All sorted views
                for view_papeis, last_idx in views_with_last_idx:
                    if not view_papeis or last_idx < 0 or last_idx >= len(prefix_events):
                        continue

                    view_tuple = tuple(view_papeis)
                    event = prefix_events[last_idx]
                    content = event.get(self.columnContent)

                    if content is None:
                        continue

                    try:
                        if not content in self.dataset_regex.template_map.keys():
                            template_history.append(None if math.isnan(content) else content)
                            continue

                        template = self.dataset_regex.template_map[content]
                        template_history.append(template)

                        # Store the template for this view to make it available for prediction
                        self.view_to_templates[view_tuple][template] += 1
                        self.full_views.setdefault(view_tuple, []).append(template_history.copy())

                    except Exception as e:
                        continue

        # Removes views that do not meet the minimum number of occurrences
        to_remove = [v for v, c in self.view_to_templates.items() if sum(c.values()) < self.min_count]
        for v in to_remove:
            del self.view_to_templates[v]

        for keys, values in self.full_views.items():
            cleaned = [list(x) for x in set(tuple(v) for v in values)]
            self.full_views[keys] = cleaned

        self.is_fitted = True

    def predict_templates(self, users_sequence: list[str], history_content: INPUT_HISTORY, k: int = 3) -> list[dict[str, Any]]:
        if not self.is_fitted:
            raise RuntimeError("Modelo sem treino !")

        normal_predict = self.__predict_normal(users_sequence, k)

        predict_by_content = self.__predict_by_history(users_sequence, history_content)
        if predict_by_content != None:
            return self.__choose_templates(predict_by_content, normal_predict)

        return normal_predict

    async def fill_template(
        self,
        users_sequence: list[str],
        past_contents: list[str],
    ) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Modelo sem treino !")

        # Get a prediction with parameter types
        template_predictions = self.predict_templates(users_sequence, past_contents, k=1)
        if not template_predictions:
            return {
                "template": "<NO_TEMPLATE>",
                "template_filled": "<NO_TEMPLATE>",
                "patterns": {},
            }

        prediction = template_predictions[0]
        template = prediction["template"]

        '''
        Use to store all parameters by index
        patterns = {
            "type_parameter": type,
            "index": idx,
            "values": [str]
        }

        1. Identify a way to extract parameters from text in multiple contexts
        2. After extracting data create filled template by substituting values only for parameters which appear one time
        2.1 If no parameter is found replace by placeholder
        3. Return value as output
        '''

        parameters, metadata = await self.dataset_regex.model_extractor.content_extract(template, past_contents)

        template = self.__fill_template(template, parameters)

        return {
            "template": prediction["template"],
            "template_filled": template,
            "patterns": parameters,
            "metadata_resolution": metadata
        }

    def __fill_template(self, template: str, content: dict) -> str:
        """
        Substitui cada <*> no template com base nas regras:
        
        - 0 values  -> <type_parameter>
        - 1 value   -> substitui pelo valor
        - >1 values -> mantém <*>
        """

        resultado = template
        placeholders = list(re.finditer(r"<\*>", template))
        offset = 0

        for i, match in enumerate(placeholders, start=1):

            if i not in content:
                continue

            param = content.get(i, {})
            raw_values = param.get("values") or []

            # Removes duplicates while preserving the order and removes "None" entries
            values = list(dict.fromkeys(v for v in raw_values if v is not None))

            if len(values) == 0:
                replacement = f"<{param.get('type_parameter', 'param')}>"

            elif len(values) == 1:
                replacement = str(values[0])

            else:
                continue

            start = match.start() + offset
            end = match.end() + offset

            resultado = resultado[:start] + replacement + resultado[end:]
            offset += len(replacement) - len("<*>")

        return resultado

    # Predicts templates based on past data (by comparing whether the input templates match the historical data)
    def __predict_by_history(self, users_sequence: list[str], content: INPUT_HISTORY)-> Optional[list[MatchResult]]:
        if content == None or len(content) == 0:
            return None

        tuple_users = tuple(users_sequence)
        templates, _ = self.dataset_regex.model_extractor._similary_templates(content)

        # Templates in the history
        users_templates_history = []

        # INFO: TEST
        # users_templates_history.append(fails[0])

        for key in content:
            if key in templates:
                users_templates_history.append(templates[key]['template_text'])

            if key is None:
                users_templates_history.append(None)


        if not users_templates_history:
            return None

        best_conf = 0
        best_matches = []  # List of candidates with the highest approval ratings

        try:
            candidates = self.full_views[tuple_users]
        except KeyError:
            return None

        for candidate in candidates:

            # Strategy 1: ORDERED (subsequence)
            match_count = 0
            j = 0
            for item in users_templates_history:
                while j < len(candidate) and candidate[j] != item:
                    j += 1
                if j < len(candidate):
                    match_count += 1
                    j += 1
            ordered_conf = match_count / max(len(candidate), len(users_templates_history))

            # Strategy 2: DISORGANIZED
            intersection = len(set(candidate) & set(users_templates_history))
            unordered_conf = intersection / max(len(candidate), len(users_templates_history))

            # Choose the best strategy
            if ordered_conf >= unordered_conf:
                conf = ordered_conf
                strategy = "ordered"
            else:
                conf = unordered_conf
                strategy = "unordered"

            # Update the list of top picks
            if conf > best_conf:
                best_conf = conf
                append_item = MatchResult(strategy, conf,candidate)
                best_matches = [append_item]
            elif conf == best_conf:
                append_item = MatchResult(strategy, conf,candidate)
                best_matches.append(append_item)

        if best_conf >= self.confidence_predict:
            return best_matches

        return None

    # Prediction for standard templates without historical data
    def __predict_normal(self, users_sequence: list[str], k: int) -> list[dict]:
        # We need to check if the sequence exists in the trained views
        # The views are generated for each prefix in the training process
        # Let's try to match the current sequence with stored views

        normalized = [self.normalize_activity(u) for u in users_sequence]
        L = len(normalized)

        for start in range(L):
            candidate_view = tuple(normalized[start:])
            counter = self.view_to_templates.get(candidate_view)
            if counter:
                total = sum(counter.values())
                results = []

                # Process templates in order of probability
                for tmpl, cnt in counter.most_common(k):
                    prob = cnt / total

                    results.append({
                        "template": tmpl,
                        "prob": prob,
                    })

                    if len(results) >= k:
                        break

                return results
        
        return []

    # Use the set of views with associated content to determine which scenario is most likely to occur
    def __choose_templates(self, items: list[MatchResult], normal_predict: list[dict]) -> list[dict[str, Any]]:
        if len(items) == 1 and items[0].match[-1] == None:
            return []

        if len(items) == 1:
            return [{
                "template": items[0].match[-1],
                "prob": 1,
            }]

        results = []

        # Collect the possible templates from the items
        possible_templates = {
            item.match[-1] for item in items if item.match and item.match[-1] is not None
        }

        for item in normal_predict:
            template = item["template"]
            prob = item["prob"]

            if template in possible_templates:
                results.append({
                    "template": template,
                    "prob": prob
                })

        # Sort by most likely
        results.sort(key=lambda x: x["prob"], reverse=True)

        # Normalize probabilities
        total_prob = sum(r["prob"] for r in results)

        if total_prob > 0:
            for r in results:
                r["prob"] = r["prob"] / total_prob

        return results
