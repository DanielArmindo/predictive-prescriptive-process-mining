import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from rapidfuzz import fuzz

@dataclass
class SlotResult:
    value: Optional[str]
    confidence: float
    status: str  # 'extracted' (>=85), 'rejected' (<85), 'partial' (anchor failed)
    method: str
    start_char: int
    end_char: int

class DynamicDeterministicExtractor:
    def __init__(self, template_obj: Dict, debug: bool = False):
        self.debug = debug
        self.template_text = template_obj.get("template_text", "")
        self.variable_blocks = template_obj.get("variable_blocks", [])
        
        # Alphanumeric patterns (letters and numbers take precedence)
        self.patterns = [
            r"[A-Z0-9]{2,}(?:[._-][A-Z0-9]+)+", 
            r"\d+(?:[\.\-]\d+)+",
            r"\d{4,}" 
        ]
        self.combined_regex = re.compile("|".join(f"({p})" for p in self.patterns), re.IGNORECASE)
        self.slots_config = self._learn_anchors()

    def _log(self, message: str):
        if self.debug: print(f"[DEBUG] {message}")

    def _learn_anchors(self) -> List[Dict]:
        parts = self.template_text.split("<*>")
        configs = []

        for i, block in enumerate(self.variable_blocks):
            # When the number of variable blocks in the metadata does not match the text obtained
            if i >= len(parts) - 1:
                break

            prefix = parts[i]
            suffix = parts[i+1] if i+1 < len(parts) else ""
            
            start_words = re.findall(r"[\w.º/:]+", prefix)
            a_in = start_words[-1] if start_words else ""
            
            suffix_clean = suffix.strip()
            end_match = re.search(r"^(\w+|[.;,])", suffix_clean)
            a_out = end_match.group(1) if end_match else ""

            example = str(block["examples"][0])
            is_numeric = any(c.isdigit() for c in example)
            is_small = is_numeric and len(re.sub(r'\D', '', example)) <= 3

            configs.append({
                "position": block["position"],
                "anchor_start": a_in, "anchor_end": a_out,
                "is_alphanumeric": is_numeric, "is_small_number": is_small,
                "examples": block["examples"]
            })
        return configs

    def _calc_confidence(self, val: str, examples: List[str]) -> float:
        v_norm = re.sub(r'[.\-_]', '', val).upper()
        max_s = 0
        for ex in examples:
            ex_norm = re.sub(r'[.\-_]', '', str(ex)).upper()
            s = fuzz.partial_ratio(v_norm, ex_norm)
            if s > max_s: max_s = s
        return float(max_s)

    def process(self, input_text: str) -> Dict[int, List[SlotResult]]:
        results = {config["position"]: [] for config in self.slots_config}
        self._log(f"\n--- Iniciando Varrimento Global ---")

        for config in self.slots_config:
            pos = config["position"]
            a_in = config["anchor_start"]
            a_out = config["anchor_end"]

            # Alphanumeric Strategy: Global Search
            if config["is_alphanumeric"] and not config["is_small_number"]:
                for match in self.combined_regex.finditer(input_text):
                    val = match.group(0)
                    conf = self._calc_confidence(val, config["examples"])
                    status = "extracted" if conf >= 85 else "rejected"
                    
                    results[pos].append(SlotResult(
                        value=val, confidence=conf, status=status,
                        method="global_regex", start_char=match.start(), end_char=match.end()
                    ))

            # Strategy for organizations/small businesses: anchor scanning
            else:
                current_search_pos = 0
                while True:
                    # 1. Find Next Start Anchor
                    idx_start = input_text.lower().find(a_in.lower(), current_search_pos)
                    if idx_start == -1: break
                    
                    start_content = idx_start + len(a_in)
                    lookahead = input_text[start_content:]
                    
                    extracted_val = None
                    conf = 0.0
                    status = "not_found"
                    end_offset = 0

                    if a_out:
                        # 2. Try End Anchor (Strict)
                        end_pattern = fr"\b{re.escape(a_out)}\b" if a_out.isalnum() else re.escape(a_out)
                        end_match = re.search(end_pattern, lookahead, re.IGNORECASE)
                        
                        if end_match:
                            extracted_val = lookahead[:end_match.start()].strip()
                            conf = 100.0
                            status = "extracted"
                            end_offset = start_content + end_match.end()
                        else:
                            # 3. Fallback: Partial Capture (up to the delimiter)
                            partial = re.split(r'[\s.;\n]', lookahead.strip())[0]
                            extracted_val = partial
                            conf = 50.0
                            status = "partial"
                            end_offset = start_content + len(partial) + 1
                    else:
                        # No fixed end anchor
                        partial = re.split(r'[\s.;\n]', lookahead.strip())[0]
                        extracted_val = partial
                        conf = 100.0
                        status = "extracted"
                        end_offset = start_content + len(partial)

                    if extracted_val:
                        results[pos].append(SlotResult(
                            value=extracted_val, confidence=conf, status=status,
                            method="anchor_scan", start_char=idx_start, end_char=end_offset
                        ))
                    
                    current_search_pos = end_offset if end_offset > current_search_pos else idx_start + 1

        return results
