import numpy as np
from dataclasses import asdict
from core.constants import MetadataColumnsFlowModel
from core.models import ModelEdoc


class model(ModelEdoc):
    def __init__(self, columnNumActivity, order=10, metadataColumns: MetadataColumnsFlowModel | None = None):
        super().__init__(columnNumActivity, order)
        # Columns for additional metrics
        self.metadataColumns = metadataColumns

    
    def __isValidMetadata(self) -> bool:
        if self.metadataColumns is None:
            return False

        # Convert to dictionary
        data = asdict(self.metadataColumns)

        # Checks if any field is empty or None
        for value in data.values():
            if value is None or value == "":
                return False

        return True


    # Fit global metrics associated to predicted activities
    def _fit_metrics_activities(self, df):
        if not self.__isValidMetadata():
            for act, group in df.groupby("concept:name"):
                durations = (group["time:timestamp"] - group["start_timestamp"]).dt.total_seconds()
                self.activity_metrics[act] = {
                    "mean_time": float(durations.mean()) if not durations.isna().all() else 0.0,
                    "std_time": float(durations.std()) if len(durations) > 1 else 0.0,
                    "sla_risk": (float(durations.quantile(0.95)) / float(durations.mean()) - 1) if durations.mean() > 0 else 0,
                    "rework_rate": group.duplicated(subset=["case:concept:name", "concept:name"]).mean(),
                }
        else:
            assert self.metadataColumns is not None
            for act, group in df.groupby("concept:name"):
                columnReceptionTimestamp = self.metadataColumns.columnReceptionTimestamp
                columnYear = self.metadataColumns.columnYear
                humanResources = self.metadataColumns.humanResources
                durations = (group["time:timestamp"] -
                             group["start_timestamp"]).dt.total_seconds()
                waits = (group["start_timestamp"] -
                         group[columnReceptionTimestamp]).dt.total_seconds()
                self.activity_metrics[act] = {
                    "mean_time": durations.mean(),
                    "std_time": durations.std(),
                    "waiting_ratio": (waits.mean() / durations.mean()) if durations.mean() > 0 else 0,
                    "throughput_rate": len(group) / df[columnYear].nunique(),
                    "handover_count": group[humanResources].nunique() / len(group),
                    # proxies:
                    "sla_risk": (durations.quantile(0.95) / durations.mean()) - 1,
                    "rework_rate": group.duplicated(subset=["case:concept:name", "concept:name"]).mean(),
                }


    def __predict_extra_normal(self, base_preds, prefix):
        assert self.event_log is not None
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


    def __predict_extra_metadata(self, base_preds, prefix):
        ctx = self._get_case_context(prefix)
        enriched = []
        for pred in base_preds:
            act = pred["activity"]
            prob = pred["prob"]
            m = self.activity_metrics.get(act, {})
            adj_prob = prob * (1 - ctx["num_reworks"]*0.05)
            risk_score = (m.get("sla_risk", 0) +
                          ctx["mean_exec_time_so_far"]/m.get("mean_time", 1))/2
            enriched.append({
                "activity": act,
                "prob": adj_prob,
                "expected_time": m.get("mean_time"),
                "sla_risk": risk_score,
                "waiting_ratio": m.get("waiting_ratio"),
                "rework_risk": ctx["num_reworks"]/max(1, len(prefix)),
                "expected_process_completion_time":
                    ctx["mean_exec_time_so_far"] + m.get("mean_time", 0),
            })
        return enriched


    def _get_case_context(self, prefix):
        assert self.metadataColumns is not None
        assert self.event_log is not None
        humanResources = self.metadataColumns.humanResources
        columnPhase = self.metadataColumns.columnPhase
        acts = prefix
        relevant_events = [
            ev for tr in self.event_log for ev in tr if ev["concept:name"] in acts]
        times = [(ev["time:timestamp"] - ev["start_timestamp"]
                  ).total_seconds() for ev in relevant_events]
        resources = [ev[humanResources] for ev in relevant_events]
        fases = [ev[columnPhase] for ev in relevant_events]

        return {
            "mean_exec_time_so_far": np.mean(times) if times else 0,
            "unique_resources": len(set(resources)),
            "num_reworks": len(acts) - len(set(acts)),
            "fases_concluidas": len(set(fases)),
        }


    def fit(self, sequences, event_log, df):
        super().fit(sequences, event_log, df)


    # Predict next activity with metrics of ongoing process
    def predict_next_with_context(self, prefix, k=5, in_subprocess=None):
        if self.event_log is None:
            return [{"error": "Modelo não treinado !"}]

        base_preds = self.predict_next(prefix, k, in_subprocess)
        if "error" in base_preds[0]:
            return base_preds

        if self.__isValidMetadata():
            return self.__predict_extra_metadata(base_preds, prefix)
        else:
            return self.__predict_extra_normal(base_preds, prefix)
