import re
from typing import Optional
from rapidfuzz import fuzz
from core.llm import GeminiTemplates, OllamaActivities, OllamaContextParser
from copy import deepcopy
import pandas as pd
import io
import re
from .ExtractContentModel import ExtractContentModel


ICONTENTMAP = dict[str, list[str]]

class TemplateHandler:
    def __init__(self, dataset: str, similarity: float = 69, min_similarity: float = 49) -> None:
        self.template_map = {}
        self.model_extractor: Optional[ExtractContentModel] = None
        self.is_fitted = False
        self.type_dataset = dataset
        self.similarity = similarity
        self.min_similarity = min_similarity


    async def fit(self, content: ICONTENTMAP) -> None:
        if not content:
            raise ValueError("Content not passed an argument...")

        templates = {}

        # 1. Remove templates was only single words
        tmp = {}
        for activity, content_list in content.items():
            for item in content_list:
                if self._remove_single_words(templates,item):
                    if activity not in tmp or not isinstance(tmp[activity], list): 
                        tmp[activity] = [item]
                    else:
                        tmp[activity].append(item)


        # Clean the dataset by removing the static text
        for act, content_list in tmp.items():
            for item in content_list:
                content[act].remove(item)


        # 2. Handle content with ollama to verify which activities are static/automatable
        # Filter the content by activity to avoid repeating the same template
        acts_ollama = self._activities_length(content)
        handle_cloud = await self._second_phase(acts_ollama, content)


        # 3. The rest of the logs are sent to GEMINI to analize by activity are generate the templates
        handle_content = self._preliminary_ollama(handle_cloud)
        model_gemini = GeminiTemplates()
        cloud_response = await model_gemini.parse_logs(handle_content)


        # 3.1 Handle GEMINI response
        for activity, content_list in content.items():
            for item in content_list:
                similarity = {}
                if activity in cloud_response.keys() and cloud_response[activity] is not None:
                    for source in cloud_response[activity]:
                        template = source['template_text']
                        prob = fuzz.ratio(item, source['template_text'])
                        similarity[template] = prob

                    max_item, max_value = max(similarity.items(), key=lambda item: item[1])
                    if max_value > self.similarity:
                        templates[item] = max_item


        # 4. Check whether a second round of GEMINI is necessary for some texts
        unhandle_text = []
        for activity, content_list in content.items():
             if activity in cloud_response:
                for item in content_list:
                    if item not in templates.keys():
                        unhandle_text.append(item)


        # 4.1 Verify in templates var if any template fit with unhandle texts
        await self._handle_texts_without_templates(unhandle_text, templates, cloud_response)

        # 4.2 If necessary run a second phase to extract the remaining fails

        # 5. Save the final content into class (single-words texts)
        for act, list_itens in tmp.items():
            for item in list_itens:
                if not (item in templates):
                    templates[item] = item


        # Store metadata from templates
        self.model_extractor = ExtractContentModel(self._format_gemini_response(cloud_response))

        self.template_map = templates
        self.is_fitted = True


    # Fit with manual templates from csv
    async def fit_manual(self, content: str, dataset: ICONTENTMAP):
        if not content.strip():
            raise ValueError("Content not passed an argument...")

        # Read templates from csv an fill the following var
        templates = {}
        # Var resposible for map the original content with the templates
        map_templates = {}

        content_2 = io.StringIO(content)
        df = pd.read_csv(content_2, sep=";")
        # print(df.dtypes)
        activity_content = df.groupby("Atividade")

        # 1. Extract content from csv grouped by activity collumn
        for act , group in activity_content:
            templates[act] = []
            for _, item in group.iterrows():
                new_item = {}
                template = str(item["Template"])
                if not template.strip():
                    continue

                if item.iloc[2:].notna().any() == False:
                    new_item['is_templated'] = False
                    new_item['variable_blocks'] = []
                    new_item['template_text'] = item['Template']
                    templates[act].append(new_item)
                    continue

                new_item['is_templated'] = True
                
                blocks = (item.iloc[2:].dropna()).tolist()

                formated_template = template
                new_blocks = []
                cont = 1
                for value, type_value in zip(blocks[::2], blocks[1::2]):
                    new_blocks.append({
                        "position": cont,
                        "examples": [value],
                        "inferred_type" : type_value
                    })
                    cont += 1
                    # regex = fr"(?<!\d){re.escape(value)}(?!\d)"
                    regex = fr"\b{re.escape(value)}\b"
                    formated_template = re.sub(regex, "<*>", formated_template)

                new_item['variable_blocks'] = new_blocks
                new_item['template_text'] = formated_template
                templates[act].append(new_item)


        # 2. Construct map_templates
        for activity, content_list in dataset.items():
            for item in content_list:
                similarity = {}
                if activity in templates.keys():
                    for source in templates[activity]:
                        template = source['template_text']
                        prob = fuzz.ratio(item, source['template_text'])
                        similarity[template] = prob

                    max_item, max_value = max(similarity.items(), key=lambda item: item[1])
                    if max_value > self.similarity:
                        map_templates[item] = max_item


        unhandle_text = []
        for activity, content_list in dataset.items():
             if activity in templates:
                for item in content_list:
                    if item not in templates.keys():
                        unhandle_text.append(item)


        # 2.1 Verify in templates var if any template fit with unhandle texts
        await self._handle_texts_without_templates(unhandle_text, map_templates, templates)

        self.model_extractor = ExtractContentModel(self._format_gemini_response(templates))

        self.template_map = map_templates
        self.is_fitted = True


    async def _second_phase(self, acts_ollama: list[str], content: dict) -> dict[str, list[str]]:
        request_ollama = {}
        for act in acts_ollama:
            if act in content.keys():
                request_ollama[act] = content[act]

        ollama_model = OllamaActivities()
        response = await ollama_model.parse_logs(request_ollama)


        # 1 Handle response from ollama - Which activities are static
        automatable_ollama = {}
        final_response = {}
        for key, data in response.items():
            for act in acts_ollama:
                for value in content[act]:
                    if key == value:
                        if act not in automatable_ollama.keys():
                            automatable_ollama[act] = []
                        automatable_ollama[act].append(data['is_static'])

        for act, itens in automatable_ollama.items():
            if any(obj == False for obj in itens):
                final_response[act] = False
                continue

            final_response[act] = True


        # 2 Process activities which will be handle by GEMINI
        handle_cloud = {}
        for act in content.keys():
            if act not in acts_ollama:
                handle_cloud[act] = content[act]

        for act, is_static in final_response.items():
            if not is_static:
                handle_cloud[act] = content[act]

        return handle_cloud


    # Extract the content that consists of a single word
    def _remove_single_words(self, arr: dict, value: str, threshold:int=90) -> bool:
        formated_value = value.strip()
        full_regex = self._default_regex_patterns()

        cond1 = full_regex.search(formated_value)
        cond2 = " " in formated_value
        cond3 = not formated_value

        if cond1 or cond2 or cond3:
            return False

        # Store in the final template array
        exists = False
        for key in arr.keys():
            if key == value:
                exists = True
                break

        if exists:
            return False

        for key, item in arr.items():
            if fuzz.partial_ratio(key, value) >= threshold:
                arr[value] = item
                return True

        arr[value] = value 
        return True


    # Default regular expressions for any dataset
    def _default_regex_patterns(self) -> re.Pattern:
        patterns = [
            # 1. Currency
            r"\b\d+(?:[.,]\d+)*\s*[€$£¥R\$]|"
            r"[€$£¥R\$]\s*\d+(?:[.,]\d+)*|"
            r"\b\d+(?:[.,]\d+)*\s+(?:USD|EUR|R\$|US\$)\b|"
            r"\b(?:USD|EUR|R\$|US\$)\s+\d+(?:[.,]\d+)*\b",
            # 2. Alphanumeric IDs with suffixes
            r"\b[A-Z]{2,5}[._]?\d+(?:[.,_]\d+){1,}\b",
            r"\b[A-Z]{2,5}(?:[._][A-Z0-9]+){2,}\b",
        ]

        return re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE)

    
    def _remove_similar_texts(self, strings: list[str], threshold:int=90):
        result = []

        for s in strings:
            is_similar = False

            for r in result:
                score = fuzz.partial_ratio(s, r)
                if score >= threshold:
                    is_similar = True
                    break

            if not is_similar:
                result.append(s)

        return result


    def _preliminary_ollama(self, content: ICONTENTMAP) -> ICONTENTMAP:
        new_content = {}
        for activity, content_list in content.items():
            already = [item for item in content_list if item.strip() != ""]
            tmp_content = self._remove_similar_texts(already)
            new_content[activity] = tmp_content

        return {act: list(contents) for act, contents in new_content.items()}


    def _activities_length(self, content: ICONTENTMAP, min_count:int=10) -> list[str]:
        return_value = []
        for act, cnt_list in content.items():
            len_act = len(cnt_list)
            if min_count == None or (isinstance(min_count, int) and min_count >= len_act):
                return_value.append(act)

        return return_value


    async def _handle_texts_without_templates(self, itens: list[str], templates: dict, temporary: dict[str, dict]) -> None:
        # 1. First we are using fuzz to check similarity by caracters in text
        tmp2 = []
        for item in itens:
            prob = self.similarity
            target_item = None
            
            for template in templates.values():
                target_prob = fuzz.ratio(item, template)
                if target_prob > prob:
                    target_item = template
                    prob = target_prob

            if target_item is not None:
                templates[item] = target_item
                tmp2.append(item)

        for item in tmp2:
            itens.remove(item)

        # 2. Now we are using the ollama with "all-MiniLM-L6-v2" library to check similarity by context
        model_context = OllamaContextParser()
        retrive_itens = {}
        familiarity_data: list[tuple[str, str, str]] = []
        for item in itens:
            retrive_itens[item] = []
            for key, value in templates.items():
                familiarity_data.append((item, value, key))
                
        response_context = await model_context.parse_context(familiarity_data)
        for key, prob in response_context.items():
            item, _, target_key = key
            if prob > self.similarity:
                retrive_itens[item].append(target_key)


        for key, value_arr in retrive_itens.items():
            if len(value_arr) == 0:
                continue

            similarity = {}
            for item in value_arr:
                template = templates[item]
                prob = fuzz.ratio(key, template)
                similarity[template] = prob
            
            max_item, max_prob = max(similarity.items(), key=lambda item: item[1])
            if max_prob > self.min_similarity:
                templates[key] = max_item
                itens.remove(key)


        # 3. TEMPORARY (raw response with templates) with the content loaded by the response GEMINI
        retrive_itens = {}
        familiarity_data: list[tuple[str, str, str]] = []
        for item in itens:
            retrive_itens[item] = []
            for value_temp in [x for arr in temporary.values() for x in arr]:
                familiarity_data.append((item, value_temp['template_text'], value_temp['template_text']))

        response_context = await model_context.parse_context(familiarity_data)
        for key, prob in response_context.items():
            item, _, target_key = key
            if prob > self.min_similarity:
                retrive_itens[item].append(target_key)


        for key, value_arr in retrive_itens.items():
            if len(value_arr) == 0:
                continue

            similarity = {}
            for item in value_arr:
                prob = fuzz.ratio(key, item)
                similarity[item] = prob
            
            max_item, max_prob = max(similarity.items(), key=lambda item: item[1])
            if max_prob > self.min_similarity:
                templates[key] = max_item
                itens.remove(key)


    def _format_gemini_response(self, items: dict) -> list:
        join_content = []

        for activity in items.values():
            for template in activity:
                new_item = deepcopy(template)
                if 'template_id' in new_item:
                    del new_item['template_id']
                join_content.append(new_item)

        return join_content
