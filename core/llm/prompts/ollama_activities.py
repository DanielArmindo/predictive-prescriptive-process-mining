prompt = """
Analyze Portuguese text logs to detect if they contain variable parameters (making them templates) or are completely static.

CRITICAL FIRST CHECK (apply BEFORE all other rules):
→ If text contains ANY of these EXACT substrings: ".099." OR ".2025." OR "PC." OR "NE." OR 6+ consecutive digits → IMMEDIATELY output {"is_static": false}
→ NEVER ignore visible IDs. If you SEE ".099." or "PC." physically in the text → false.

You will analyse several texts and must classify them at the end as false or true, i.e. if a rule in a text is detected as a parameter, you immediately classify it as false.

RULES TO DETECT PARAMETERS (answer "false" if ANY found):
- Alphanumeric sequences with separators (., -, /, :) containing BOTH letters AND numbers in fixed positions (e.g., X999.2025.0000001)
    - Unique IDs like "PCxxx.xxxx.xxxxxxx", "NE99.2025.xxxxxxx"
- Proper nouns in ALL CAPS or mixed case that are NOT common Portuguese words
    - Organization/person names (e.g., "JUVENTUDE DESPORTIVA DO LIS")
- Monetary values

STATIC TEXT (answer "true" ONLY if don't match with any rules above):
- Contains ONLY fixed phrases with NO replaceable parts
- No identifiers, names, numbers, or variable fields
- Exactly reproducible in every instance

IGNORE (do not consider as parameters):
- Small numbers like "3", "113", "6" without surrounding dots/letters
- Fixed legal references like "artigo 113.º" that never change

OUTPUT FORMAT (STRICT):
{{
    "is_static": true | false,
}}

NEVER explain, comment or add text. ONLY output valid JSON.
"""
