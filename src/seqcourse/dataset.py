from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._palette import default_colors
from ._utils import ensure_object_matrix, normalize_weights, stable_unique, trailing_length


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    data: np.ndarray
    alphabet: tuple[str, ...]
    state_labels: tuple[str, ...]
    state_colors: tuple[str, ...]
    weights: np.ndarray
    time_labels: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    missing_state: str = "__MISSING__"
    missing_label: str = "Missing"
    missing_color: str = "#7f7f7f"
    void_state: str = "__VOID__"
    void_label: str = "Void"
    void_color: str = "#f5f5f5"

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.uint16)
        if data.ndim != 2:
            raise ValueError("'data' must be a two-dimensional uint16 matrix.")
        if len(self.alphabet) == 0:
            raise ValueError("'alphabet' must contain at least one state.")
        if len(self.state_labels) != len(self.alphabet):
            raise ValueError("'state_labels' must match the alphabet length.")
        if len(self.state_colors) != len(self.alphabet):
            raise ValueError("'state_colors' must match the alphabet length.")
        if data.shape[0] != len(self.weights):
            raise ValueError("'weights' must match the number of sequences.")
        if data.shape[0] != len(self.sequence_ids):
            raise ValueError("'sequence_ids' must match the number of sequences.")
        if data.shape[1] != len(self.time_labels):
            raise ValueError("'time_labels' must match the sequence width.")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))

    @classmethod
    def from_wide(
        cls,
        data: object,
        *,
        alphabet: list[str] | tuple[str, ...] | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
        time_labels: list[str] | tuple[str, ...] | None = None,
        missing_values: set[object] | None = None,
        void_values: set[object] | None = None,
        missing_state: str = "__MISSING__",
        missing_label: str = "Missing",
        missing_color: str = "#7f7f7f",
        void_state: str = "__VOID__",
        void_label: str = "Void",
        void_color: str = "#f5f5f5",
    ) -> "SequenceDataset":
        values, row_labels, column_labels = ensure_object_matrix(data)
        missing_values = set() if missing_values is None else set(missing_values)
        void_values = set() if void_values is None else set(void_values)

        discovered: list[str] = []
        for item in values.flat:
            if pd.isna(item):
                continue
            if item in missing_values or item in void_values:
                continue
            discovered.append(str(item))
        normalized_alphabet = tuple(alphabet) if alphabet is not None else stable_unique(discovered)
        if not normalized_alphabet:
            raise ValueError("Unable to infer an alphabet from the provided data.")

        if labels is None:
            state_labels = normalized_alphabet
        else:
            state_labels = tuple(labels)

        if colors is None:
            state_colors = default_colors(len(normalized_alphabet))
        else:
            state_colors = tuple(colors)

        encoded = np.zeros(values.shape, dtype=np.uint16)
        mapping = {state: index + 1 for index, state in enumerate(normalized_alphabet)}
        missing_code = len(normalized_alphabet) + 1

        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                item = values[row_index, column_index]
                if pd.isna(item) or item in missing_values:
                    encoded[row_index, column_index] = missing_code
                    continue
                if item in void_values:
                    encoded[row_index, column_index] = 0
                    continue
                key = str(item)
                if key not in mapping:
                    raise ValueError(f"Encountered unknown state {item!r} not present in the alphabet.")
                encoded[row_index, column_index] = mapping[key]

        resolved_weights = normalize_weights(weights, encoded.shape[0])
        resolved_time_labels = tuple(time_labels) if time_labels is not None else column_labels

        return cls(
            data=encoded,
            alphabet=normalized_alphabet,
            state_labels=state_labels,
            state_colors=state_colors,
            weights=resolved_weights,
            time_labels=resolved_time_labels,
            sequence_ids=row_labels,
            missing_state=missing_state,
            missing_label=missing_label,
            missing_color=missing_color,
            void_state=void_state,
            void_label=void_label,
            void_color=void_color,
        )

    @classmethod
    def from_spell(
        cls,
        data: pd.DataFrame,
        *,
        id_col: str = "id",
        state_col: str = "state",
        start_col: str = "start",
        end_col: str | None = "end",
        duration_col: str | None = None,
        weight_col: str | None = None,
        alphabet: list[str] | tuple[str, ...] | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        colors: list[str] | tuple[str, ...] | None = None,
        missing_values: set[object] | None = None,
        missing_state: str = "__MISSING__",
        missing_label: str = "Missing",
        missing_color: str = "#7f7f7f",
        void_state: str = "__VOID__",
        void_label: str = "Void",
        void_color: str = "#f5f5f5",
    ) -> "SequenceDataset":
        if end_col is None and duration_col is None:
            raise ValueError("Provide either 'end_col' or 'duration_col' when constructing from spell data.")
        frame = data.copy()
        if end_col is None:
            frame["__end__"] = frame[start_col] + frame[duration_col] - 1
            end_name = "__end__"
        else:
            end_name = end_col

        ids = list(dict.fromkeys(frame[id_col].tolist()))
        min_start = int(frame[start_col].min())
        max_end = int(frame[end_name].max())
        width = max_end - min_start + 1

        wide = np.full((len(ids), width), void_state, dtype=object)
        weights = np.ones(len(ids), dtype=float)
        id_to_row = {item: position for position, item in enumerate(ids)}

        for row in frame.sort_values([id_col, start_col, end_name]).itertuples(index=False):
            row_dict = row._asdict()
            sequence_index = id_to_row[row_dict[id_col]]
            start = int(row_dict[start_col]) - min_start
            stop = int(row_dict[end_name]) - min_start + 1
            if stop <= start:
                raise ValueError("Spell intervals must have positive length.")
            if np.any(wide[sequence_index, start:stop] != void_state):
                raise ValueError("Overlapping spell intervals are not supported.")
            wide[sequence_index, start:stop] = row_dict[state_col]
            if weight_col is not None:
                weights[sequence_index] = float(row_dict[weight_col])

        columns = [str(value) for value in range(min_start, max_end + 1)]
        frame_wide = pd.DataFrame(wide, index=[str(item) for item in ids], columns=columns)
        return cls.from_wide(
            frame_wide,
            alphabet=alphabet,
            labels=labels,
            colors=colors,
            weights=weights,
            time_labels=columns,
            missing_values=missing_values,
            void_values={void_state},
            missing_state=missing_state,
            missing_label=missing_label,
            missing_color=missing_color,
            void_state=void_state,
            void_label=void_label,
            void_color=void_color,
        )

    @property
    def n_sequences(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_positions(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_states(self) -> int:
        return len(self.alphabet)

    @property
    def void_code(self) -> int:
        return 0

    @property
    def missing_code(self) -> int:
        return self.n_states + 1

    @property
    def has_missing(self) -> bool:
        return bool(np.any(self.data == self.missing_code))

    def states(self, *, with_missing: bool = False) -> tuple[str, ...]:
        if with_missing and self.has_missing:
            return self.alphabet + (self.missing_state,)
        return self.alphabet

    def labels(self, *, with_missing: bool = False) -> tuple[str, ...]:
        if with_missing and self.has_missing:
            return self.state_labels + (self.missing_label,)
        return self.state_labels

    def colors(self, *, with_missing: bool = False) -> tuple[str, ...]:
        if with_missing and self.has_missing:
            return self.state_colors + (self.missing_color,)
        return self.state_colors

    def sequence_lengths(self) -> np.ndarray:
        return np.asarray([trailing_length(row, void_code=self.void_code) for row in self.data], dtype=int)

    def take(self, rows: list[int] | np.ndarray) -> "SequenceDataset":
        indices = np.asarray(rows, dtype=int)
        return SequenceDataset(
            data=self.data[indices],
            alphabet=self.alphabet,
            state_labels=self.state_labels,
            state_colors=self.state_colors,
            weights=self.weights[indices],
            time_labels=self.time_labels,
            sequence_ids=tuple(self.sequence_ids[index] for index in indices),
            missing_state=self.missing_state,
            missing_label=self.missing_label,
            missing_color=self.missing_color,
            void_state=self.void_state,
            void_label=self.void_label,
            void_color=self.void_color,
        )

    def to_wide(
        self,
        *,
        missing_value: Any = pd.NA,
        void_value: Any = None,
    ) -> pd.DataFrame:
        decoded = np.empty(self.data.shape, dtype=object)
        for code, state in enumerate(self.alphabet, start=1):
            decoded[self.data == code] = state
        decoded[self.data == self.void_code] = self.void_state if void_value is None else void_value
        decoded[self.data == self.missing_code] = missing_value
        return pd.DataFrame(decoded, index=self.sequence_ids, columns=self.time_labels)

    def to_spell(
        self,
        *,
        id_name: str = "id",
        state_name: str = "state",
        start_name: str = "start",
        end_name: str = "end",
        duration_name: str = "duration",
        include_missing: bool = True,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        state_lookup = {index + 1: state for index, state in enumerate(self.alphabet)}
        if include_missing and self.has_missing:
            state_lookup[self.missing_code] = self.missing_state

        for row_index, row in enumerate(self.data):
            current_code = None
            current_start = 0
            for position, code in enumerate(row):
                if code == self.void_code:
                    if current_code is not None:
                        records.append(
                            {
                                id_name: self.sequence_ids[row_index],
                                state_name: state_lookup[current_code],
                                start_name: current_start + 1,
                                end_name: position,
                                duration_name: position - current_start,
                            }
                        )
                        current_code = None
                    continue
                if code == self.missing_code and not include_missing:
                    continue
                if current_code is None:
                    current_code = int(code)
                    current_start = position
                    continue
                if code != current_code:
                    records.append(
                        {
                            id_name: self.sequence_ids[row_index],
                            state_name: state_lookup[current_code],
                            start_name: current_start + 1,
                            end_name: position,
                            duration_name: position - current_start,
                        }
                    )
                    current_code = int(code)
                    current_start = position
            if current_code is not None:
                records.append(
                    {
                        id_name: self.sequence_ids[row_index],
                        state_name: state_lookup[current_code],
                        start_name: current_start + 1,
                        end_name: trailing_length(row, void_code=self.void_code),
                        duration_name: trailing_length(row, void_code=self.void_code) - current_start,
                    }
                )
        return pd.DataFrame.from_records(records)

    def most_frequent_index(self, *, weighted: bool = True) -> int:
        scores: dict[tuple[int, ...], float] = {}
        first_seen: dict[tuple[int, ...], int] = {}
        weights = self.weights if weighted else np.ones(self.n_sequences, dtype=float)
        for index, row in enumerate(self.data):
            key = tuple(row.tolist())
            scores[key] = scores.get(key, 0.0) + float(weights[index])
            first_seen.setdefault(key, index)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
        return first_seen[ranked[0][0]]
