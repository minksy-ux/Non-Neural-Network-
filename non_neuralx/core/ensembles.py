from collections import Counter
from typing import Iterable, Sequence


class SimpleVotingEnsemble:
    """Minimal majority-vote helper for combining routing/model outputs."""

    def predict(self, labels: Sequence[str]) -> str:
        if not labels:
            raise ValueError("labels must be non-empty")
        return Counter(labels).most_common(1)[0][0]

    def batch_predict(self, labels_per_item: Iterable[Sequence[str]]) -> list[str]:
        return [self.predict(labels) for labels in labels_per_item]
