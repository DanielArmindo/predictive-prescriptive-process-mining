from collections import defaultdict, Counter
import numpy as np
import re


class model:
    def __init__(self, columnNumActivity, order=10):
        self.event_log = None
        self.columnNumActivity = columnNumActivity
        self.order = order
        self.context_counts = [defaultdict(Counter) for _ in range(order)]  # only for parallel (compound tokens)
        self.sequential_context_counts = [defaultdict(Counter) for _ in range(order)]  # only for sequential
        self.unigram_counts = Counter()
        self.vocab = set()
        self.start_counts = Counter()
        self.total_unigrams = 0
        self.activity_metrics = {}
        self.divergence_points = {}

        # For managing parallel and sequential processes
        self.first_papeis_in_subprocess = set()
        self.papeis_que_precedem_split = set()

    def normalize_activity(self, act):
        if not isinstance(act, str):
            act = str(act)
        act = re.sub(r'\s+', ' ', act)
        act = re.sub(r'\s*\.\s*', '.', act.strip())
        return act

    def _parse_activity(self, act_str):
        act_str = self.normalize_activity(act_str)
        if '.' in act_str:
            main, sub = act_str.split('.', 1)
            try:
                sub = int(sub)
            except ValueError:
                pass
            return (main, sub)
        return (act_str, None)

    def _extract_logical_views_for_papel(self, papel_seq, activity_seq):
        if len(papel_seq) != len(activity_seq):
            raise ValueError("papel_seq e activity_seq devem ter mesmo comprimento")

        parsed_activities = [self._parse_activity(a) for a in activity_seq]
        subprocesses = defaultdict(list)
        main_positions = {}

        for idx, (main_id, sub_step) in enumerate(parsed_activities):
            if sub_step is None:
                main_positions[main_id] = idx
            else:
                subprocesses[main_id].append((sub_step, idx))

        for main_id in subprocesses:
            subprocesses[main_id].sort(key=lambda x: x[0])

        views = []
        for main_id, steps in subprocesses.items():
            if main_id not in main_positions:
                continue
            main_idx = main_positions[main_id]
            before = [papel_seq[i] for i in range(main_idx)]
            sub_papeis = [papel_seq[idx] for (_, idx) in steps]
            last_sub_idx = max(idx for (_, idx) in steps)
            after = [papel_seq[i] for i in range(last_sub_idx + 1, len(papel_seq))]
            views.append(before + sub_papeis + after)

        if not subprocesses:
            views.append(papel_seq)

        return views

    def fit(self, sequences, event_log, df):
        self.event_log = event_log
        all_views = []

        for trace_idx, (case_id_from_seq, papel_seq) in enumerate(sequences):
            trace = event_log[trace_idx]
            case_id_from_log = trace.attributes.get("concept:name")
            activity_seq = []
            corrected_papel_seq = []

            for ev_idx, event in enumerate(trace):
                act_val = event.get(self.columnNumActivity)
                if act_val is None:
                    raise KeyError(f"Evento {ev_idx} do caso {case_id_from_log} não tem '{self.columnNumActivity}'")
                papel_val = papel_seq[ev_idx]
                if str(act_val).upper() == "END":
                    papel_val = "Fim"
                activity_seq.append(act_val)
                corrected_papel_seq.append(papel_val)

            views = self._extract_logical_views_for_papel(corrected_papel_seq, activity_seq)
            for view in views:
                if view and view[-1] != "Fim":
                    view = list(view) + ["Fim"]
                all_views.append((case_id_from_log, view))

        # Update basic counts using logical views
        for _, seq in all_views:
            if seq:
                self.start_counts[seq[0]] += 1
            self.vocab.update(seq)
            for i, next_a in enumerate(seq):
                self.unigram_counts[next_a] += 1
                self.total_unigrams += 1
                for n in range(2, self.order + 1):
                    if i - (n - 1) < 0:
                        break
                    ctx = tuple(seq[i - (n - 1): i])
                    self.context_counts[n - 1][ctx][next_a] += 1

        # Generate composite tokens for all splits (not just the first one)
        for trace_idx in range(len(event_log)):
            if trace_idx >= len(sequences):
                continue
            trace = event_log[trace_idx]
            _, papel_seq_raw = sequences[trace_idx]
            papel_seq = [self.normalize_activity(p) for p in papel_seq_raw]
            act_seq = [ev.get(self.columnNumActivity, "") for ev in trace]
            L = len(act_seq)
            parsed = [self._parse_activity(a) for a in act_seq]

            i = 0
            while i < L:
                main_id, sub_step = parsed[i]
                if sub_step is None:
                    # Find immediate subprocesses
                    next_main = L
                    for j in range(i + 1, L):
                        if parsed[j][1] is None:
                            next_main = j
                            break
                    first_papeis_by_sub = {}
                    for j in range(i + 1, next_main):
                        sub_root, step_val = parsed[j]
                        if sub_root not in first_papeis_by_sub:
                            if str(step_val).strip() in ("1", "1.0"):
                                first_papeis_by_sub[sub_root] = papel_seq[j]
                            elif sub_root not in first_papeis_by_sub:
                                first_papeis_by_sub[sub_root] = papel_seq[j]

                    if len(first_papeis_by_sub) >= 2:
                        token = tuple(sorted(set(first_papeis_by_sub.values())))
                        # Collect statistics for inference
                        self.papeis_que_precedem_split.add(papel_seq[i])
                        for p in token:
                            self.first_papeis_in_subprocess.add(p)

                        # Inject only into context_counts (parallel mode)
                        for ctx_size in range(1, min(self.order, i + 2)):
                            if i - ctx_size + 1 >= 0:
                                ctx = tuple(papel_seq[i - ctx_size + 1:i + 1])
                                self.context_counts[ctx_size][ctx][token] += 1
                                self.vocab.add(token)
                i += 1

        # Expanded sequential views (only for sequential_context_counts)
        expanded_views = []
        for trace_idx, (case_id_from_seq, papel_seq) in enumerate(sequences):
            trace = event_log[trace_idx]
            activity_seq = [ev.get(self.columnNumActivity) for ev in trace]
            papel_seq = [str(p).strip() for p in papel_seq]
            parsed = [self._parse_activity(a) for a in activity_seq]
            L = len(parsed)

            i = 0
            while i < L:
                main_id, sub_step = parsed[i]
                if sub_step is None:
                    next_main = L
                    for j in range(i + 1, L):
                        if parsed[j][1] is None:
                            next_main = j
                            break
                    subs_by_root = defaultdict(list)
                    for j in range(i + 1, next_main):
                        root, step = parsed[j]
                        subs_by_root[root].append((step, j))

                    if len(subs_by_root) >= 2:
                        base_prefix = papel_seq[:i+1]
                        suffix = papel_seq[next_main:]
                        for root, steps in subs_by_root.items():
                            steps_sorted = sorted(steps, key=lambda x: x[0])
                            sub_papeis = [papel_seq[idx] for (_, idx) in steps_sorted]
                            view = base_prefix + sub_papeis + suffix
                            if view and view[-1] != "Fim":
                                view = view + ["Fim"]
                            expanded_views.append((case_id_from_seq, view))
                        i = next_main
                        continue
                i += 1

            if not expanded_views or expanded_views[-1][0] != case_id_from_seq:
                orig = papel_seq + (["Fim"] if papel_seq and papel_seq[-1] != "Fim" else [])
                expanded_views.append((case_id_from_seq, orig))

        # Inject only into sequential_context_counts
        for _, seq in expanded_views:
            self.vocab.update(seq)
            for i, next_a in enumerate(seq):
                self.unigram_counts[next_a] += 1
                self.total_unigrams += 1
                for n in range(2, self.order + 1):
                    if i - (n - 1) < 0:
                        break
                    ctx = tuple(seq[i - (n - 1): i])
                    self.sequential_context_counts[n - 1][ctx][next_a] += 1

        # End
        self._fit_metrics_activities(df)
        self.vocab.add("Fim")

    def _fit_metrics_activities(self, df):
        for act, group in df.groupby("concept:name"):
            durations = (group["time:timestamp"] - group["start_timestamp"]).dt.total_seconds()
            self.activity_metrics[act] = {
                "mean_time": float(durations.mean()) if not durations.isna().all() else 0.0,
                "std_time": float(durations.std()) if len(durations) > 1 else 0.0,
                "sla_risk": (float(durations.quantile(0.95)) / float(durations.mean()) - 1) if durations.mean() > 0 else 0,
                "rework_rate": group.duplicated(subset=["case:concept:name", "concept:name"]).mean(),
            }

    def _topk_from_counts(self, counts: Counter, k: int):
        total = sum(counts.values())
        if total == 0:
            return []
        items = [(act, cnt / total) for act, cnt in counts.items() if cnt > 0]
        items.sort(key=lambda x: -x[1])
        return items[:k]

    def _infer_mode(self, normalized_prefix):
        """Infer mode from prefix of papéis only."""
        if not normalized_prefix:
            return "parallel"

        last = normalized_prefix[-1]

        # Heuristic 1: person's name (space + uppercase letter) → likely a subprocess
        if isinstance(last, str) and ' ' in last and any(c.isupper() for c in last.split()[0]):
            return "sequential"

        # Heuristic 2: The last role is typically the first in a subprocess
        if last in self.first_papeis_in_subprocess:
            return "sequential"

        # Heuristic 3: The last role is typically dealt before the split
        if last in self.papeis_que_precedem_split:
            return "parallel"

        # Fallback
        return "sequential"

    def predict_next(self, prefix, k=5, in_subprocess=None):
        """
        Previsão inteligente com suporte a múltiplos splits.

        Args:
            prefix: lista de papéis (ex: ["A", "B", "Ana Ferreira"])
            k: número de sugestões
            in_subprocess:
                None → modo automático (recomendado)
                True → força modo sequencial (dentro de subprocesso)
                False → força modo paralelo (fluxo principal)
        """
        if self.event_log is None:
            raise RuntimeError("Chama fit() primeiro.")

        normalized_prefix = [self.normalize_activity(a) for a in prefix]

        unknown = [a for a in normalized_prefix if a not in self.vocab]
        if unknown:
            return [{"error": "Unknown papéis: " + ", ".join(unknown)}]

        # Select mode
        if in_subprocess is None:
            mode = self._infer_mode(normalized_prefix)
        elif in_subprocess:
            mode = "sequential"
        else:
            mode = "parallel"

        # Try the main mode
        result = self._predict_in_mode(normalized_prefix, k, mode)

        # Inteligent fallback
        if not result or ("error" in result[0]):
            alt_mode = "parallel" if mode == "sequential" else "sequential"
            result = self._predict_in_mode(normalized_prefix, k, alt_mode)

        return result

    def _predict_in_mode(self, normalized_prefix, k, mode):
        # Special case: empty prefix → use start_counts
        if len(normalized_prefix) == 0:
            preds = self._topk_from_counts(self.start_counts, k)
            return [
                {"activity": list(a), "prob": p} if isinstance(a, tuple) else {"activity": a, "prob": p}
                for a, p in preds
            ] if preds else [{"error": "No start activities observed."}]

        # Select counts
        counts_list = self.sequential_context_counts if mode == "sequential" else self.context_counts

        # Normal backoff
        for ctx_size in range(self.order - 1, 0, -1):
            if len(normalized_prefix) >= ctx_size:
                ctx = tuple(normalized_prefix[-ctx_size:])
                if ctx in counts_list[ctx_size]:
                    counts = counts_list[ctx_size][ctx]

                    if mode == "sequential":
                        filtered = Counter({a: c for a, c in counts.items() if isinstance(a, str)})
                    else:
                        filtered = counts

                    if filtered:
                        preds = self._topk_from_counts(filtered, k)
                        return [
                            {"activity": list(a), "prob": p} if isinstance(a, tuple) else {"activity": a, "prob": p}
                            for a, p in preds
                        ]

        if mode == "sequential":
            uni = Counter({a: c for a, c in self.unigram_counts.items() if isinstance(a, str)})
        else:
            uni = self.unigram_counts

        preds = self._topk_from_counts(uni, k)
        return [
            {"activity": list(a), "prob": p} if isinstance(a, tuple) else {"activity": a, "prob": p}
            for a, p in preds
        ]

    # Auxiliary methods
    def predict_next_with_context(self, prefix, k=5, in_subprocess=None):
        base_preds = self.predict_next(prefix, k, in_subprocess)
        if "error" in base_preds[0]:
            return base_preds

        relevant_events = [
            ev for trace in self.event_log
            for ev in trace
            if ev.get("concept:name") in prefix
        ]

        times = [(ev["time:timestamp"] - ev["start_timestamp"]).total_seconds()
                 for ev in relevant_events]
        resources = [ev["concept:name"] for ev in relevant_events]

        ctx = {
            "mean_exec_time_so_far": np.mean(times) if times else 0,
            "unique_resources": len(set(resources)),
            "num_reworks": len(prefix) - len(set(prefix)),
        }

        enriched = []
        for pred in base_preds:
            act = pred["activity"]
            prob = pred["prob"]
            if isinstance(act, list):
                times_vals = []
                sla_risks = []
                for a in act:
                    m_a = self.activity_metrics.get(a, {})
                    if m_a.get("mean_time"):
                        times_vals.append(m_a["mean_time"])
                    if m_a.get("sla_risk"):
                        sla_risks.append(m_a["sla_risk"])
                mean_time = np.mean(times_vals) if times_vals else 0.0
                sla_risk = np.mean(sla_risks) if sla_risks else 0.0
                m = {"mean_time": mean_time, "sla_risk": sla_risk}
            else:
                m = self.activity_metrics.get(act, {})

            adj_prob = prob * max(0.1, 1 - ctx["num_reworks"] * 0.05)
            risk_score = (m.get("sla_risk", 0) +
                          (ctx["mean_exec_time_so_far"] / m.get("mean_time", 1) if m.get("mean_time", 1) > 0 else 0)) / 2
            enriched.append({
                "activity": act,
                "prob": adj_prob,
                "expected_time": m.get("mean_time", 0.0),
                "sla_risk": risk_score,
                "rework_risk": ctx["num_reworks"] / max(1, len(prefix)),
                "expected_process_completion_time":
                    ctx["mean_exec_time_so_far"] + m.get("mean_time", 0.0),
            })
        return enriched
