"""Routing policy: turn noisy per-window drift labels into stable model-switch decisions."""

ROUTES = {
    "normal": "fraud-detection-model",
    "A": "fraud-spec-A",
    "B": "fraud-spec-B",
    "C": "fraud-spec-C",
    "U": "fraud-universal",
}


class RoutingPolicy:
    def __init__(self, confirm=2, recover=3):
        self.confirm, self.recover = confirm, recover
        self.state = "normal"
        self._cand, self._n, self._clean = None, 0, 0

    def _reset_candidate(self):
        self._cand, self._n = None, 0

    def _candidate(self, label):
        if label == self._cand:
            self._n += 1
        else:
            self._cand, self._n = label, 1
        if self._n >= self.confirm:
            self.state = label
            self._reset_candidate()
            self._clean = 0
            return "switch"
        return "pending"

    def update(self, label):
        event = "none"
        if self.state == "normal":
            if label == "normal":
                self._reset_candidate()
            else:
                event = self._candidate(label)
        else:
            if label == self.state:
                self._reset_candidate()
                self._clean = 0
            elif label == "normal":
                self._reset_candidate()
                self._clean += 1
                if self._clean >= self.recover:
                    self.state, self._clean, event = "normal", 0, "recover"
            else:
                event = self._candidate(label)
                self._clean = 0
        return {"event": event, "state": self.state, "route": ROUTES[self.state]}