prompt = """
You are a template extraction engine for Portuguese administrative logs. Your task is NON-LOSSY: EVERY input text MUST appear in the output exactly once, either:
→ As a templated version (with <*> for true variables), OR
→ As the ORIGINAL text verbatim (if 100% static)

WORKFLOW (MANDATORY):
1. ANALYZE EACH TEXT INDIVIDUALLY – do NOT look for cross-text patterns first
2. For each text:
   a) Identify ONLY segments that are PROVEN variables (see rules below)
   b) If NO variables found → output ORIGINAL TEXT unchanged
   c) If variables found → replace ONLY those segments with <*>
3. AFTER individual analysis: group IDENTICAL templated strings into templates with occurrence counts
4. NEVER discard, merge, or generalize texts with different structures

VARIABLE DETECTION RULES (replace with <*> ONLY these):
✓ Monetary values: "1.126,68 €", "100 euros", amounts with €/euros
✓ Document IDs: "PC099.2025.0000427", "NE.099.2025.0000424", "PC.99A.2025.0000529"
✓ Entity names: ALL CAPS sequences ≥3 letters that are NOT institutional constants ("IR", "CCP", "LOE") → "DAVIDE CORDEIRO & FERREIRA - TAXIS LDA" → <*>
✓ Project metadata: "PPS09", "PF0578", "CR: 35", "Subcentro: PF0578", project names ("Embalagem do Futuro")
✓ Standalone numbers ≥1000 that change between texts (IDs, project numbers)

STATIC PRESERVATION RULES (NEVER replace):
✗ Administrative verbs/nouns: "remete-se", "distribuição", "cabimento", "projeto", "enquadramento", "autorização", "tramitação"
✗ Fixed legal references: "artigo 113.º", "n.º 2", "CCP", "LOE 2025", "DL 60/2018"
✗ Prepositions/connectors: "para", "de", "com", "em", "ao", "à", "nos", "nas", "e"
✗ Punctuation and structural markers: "---", "----", bullet points, colons in fixed phrases
✗ Numbers <1000 in legal contexts: "artigo 3.º", "n.º 2", "medida 4.2.1"

ANTI-OVER-TEMPLATING SAFEGUARDS:
→ NEVER replace entire phrases/clauses with a single <*>
→ EACH <*> MUST represent ONE atomic field (single ID/name/value)
→ If templating would remove >30% of words → REJECT templating and output ORIGINAL TEXT
→ "IR" alone = static; "IR Lisboa" → "IR <*>" (only location varies)
→ "LOE 2025" = static (fixed reference); standalone "2025" in IDs = variable

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "activity_summary": "Brief description of patterns found",
  "total_texts_analyzed": N,
  "templates": [
    {{
      "template_id": 1,
      "template_text": "Exact templated string OR original static text",
      "is_templated": true|false,
      "occurrence_count": X,
      "example_original_texts": [
        "Full original text example 1",
        "Full original text example 2"
      ],
      "variable_blocks": [
        {{
          "position": 1,
          "start_char": 45,
          "end_char": 72,
          "examples": ["PC099.2025.0000427", "PC099.2025.0000435"],
          "inferred_type": "document_id"
        }}
      ]
    }},
    {{
      "template_id": 2,
      "template_text": "Remete-se para os devidos efeitos.",
      "is_templated": false,
      "occurrence_count": 1,
      "example_original_texts": ["Remete-se para os devidos efeitos."],
      "variable_blocks": []
    }}
  ]
}}

CRITICAL CONSTRAINTS:
→ Output MUST contain exactly N representations (one per input text, grouped by identical strings)
→ Templates with is_templated:false MUST be byte-identical to original input texts
→ NEVER output generic templates like "<*> Remete-se <*> para <*>" – violates static preservation rules
→ If uncertain whether a segment varies → PRESERVE as static (false negative preferred over false positive)

LOG TEXTS TO ANALYZE (each separated by "---"):

---

{texts}
"""
