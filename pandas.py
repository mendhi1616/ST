import math
from typing import Any, Dict, List, Iterable

class Series:
    def __init__(self, data: List[Any]):
        self.data = list(data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return Series([v for v, flag in zip(self.data, idx) if flag])
        return self.data[idx]

    def __eq__(self, other):
        return [v == other for v in self.data]

    def __gt__(self, other):
        return [v > other for v in self.data]

    def __lt__(self, other):
        return [v < other for v in self.data]

    def __ge__(self, other):
        return [v >= other for v in self.data]

    def __le__(self, other):
        return [v <= other for v in self.data]

    def dropna(self):
        return Series([v for v in self.data if v is not None])

    def median(self):
        if not self.data:
            return 0
        sorted_vals = sorted(self.data)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2:
            return sorted_vals[mid]
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2

    def unique(self):
        seen = []
        for v in self.data:
            if v not in seen:
                seen.append(v)
        return seen

    def to_list(self):
        return list(self.data)

class _ILoc:
    def __init__(self, df: 'DataFrame'):
        self.df = df

    def __getitem__(self, idx):
        row = self.df._get_row(idx)
        return row

class Grouped:
    def __init__(self, df: 'DataFrame', by: str):
        self.df = df
        self.by = by

    def __getitem__(self, col: str):
        self.col = col
        return self

    def transform(self, func):
        groups: Dict[Any, List[int]] = {}
        by_values = self.df.columns[self.by]
        for i, v in enumerate(by_values):
            groups.setdefault(v, []).append(i)

        result = [None] * len(by_values)
        col_values = self.df.columns[self.col]
        for key, idxs in groups.items():
            subset = [col_values[i] for i in idxs]
            transformed = func(Series(subset))
            # transformed can be list/Series; align length
            if isinstance(transformed, Series):
                transformed = transformed.data
            for offset, i in enumerate(idxs):
                try:
                    result[i] = transformed[offset]
                except Exception:
                    result[i] = transformed
        return Series(result)

class DataFrame:
    def __init__(self, data: Any = None, columns: List[str] = None):
        if data is None:
            self.columns = {}
        elif isinstance(data, list):
            # list of dicts
            self.columns = {}
            for row in data:
                for key in row:
                    self.columns.setdefault(key, []).append(row.get(key))
                # fill missing keys with None
                for key in self.columns:
                    if key not in row:
                        self.columns[key].append(None)
        elif isinstance(data, dict):
            self.columns = {k: list(v) for k, v in data.items()}
        else:
            self.columns = {}
        if columns:
            # ensure specified columns exist
            for col in columns:
                self.columns.setdefault(col, [])

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self.columns.get(key, []))
        if isinstance(key, list):
            # boolean mask filtering
            return self._filter_rows(key)
        if isinstance(key, Series):
            return self._filter_rows(list(key.data))
        return None

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            value = value.data
        self.columns[key] = list(value)

    def __len__(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def _filter_rows(self, mask: List[bool]):
        filtered = {}
        for col, values in self.columns.items():
            filtered[col] = [v for v, flag in zip(values, mask) if flag]
        return DataFrame(filtered)

    def _get_row(self, idx):
        row = {}
        for col, values in self.columns.items():
            row[col] = values[idx]
        return row

    @property
    def iloc(self):
        return _ILoc(self)

    @property
    def empty(self):
        return self.__len__() == 0

    def copy(self):
        return DataFrame({k: list(v) for k, v in self.columns.items()})

    def unique(self, col: str):
        return Series(self.columns.get(col, [])).unique()

    def groupby(self, by: str):
        return Grouped(self, by)

    def sort_values(self, by: str, key=None, ascending=True):
        values = self.columns.get(by, [])
        if key:
            sorted_idx = sorted(range(len(values)), key=lambda i: key(values[i]), reverse=not ascending)
        else:
            sorted_idx = sorted(range(len(values)), key=lambda i: values[i], reverse=not ascending)
        sorted_columns = {col: [vals[i] for i in sorted_idx] for col, vals in self.columns.items()}
        return DataFrame(sorted_columns)

    def median(self):
        medians = {}
        for col, values in self.columns.items():
            medians[col] = Series(values).median()
        return medians

    def to_excel(self, path, index=False):
        with open(path, 'w') as f:
            headers = list(self.columns.keys())
            f.write(','.join(headers) + '\n')
            for i in range(len(self)):
                row = [str(self.columns[h][i]) if i < len(self.columns[h]) else '' for h in headers]
                f.write(','.join(row) + '\n')

    def to_csv(self, path, index=False):
        with open(path, 'w') as f:
            headers = list(self.columns.keys())
            f.write(','.join(headers) + '\n')
            for i in range(len(self)):
                row = [str(self.columns[h][i]) if i < len(self.columns[h]) else '' for h in headers]
                f.write(','.join(row) + '\n')

    def iterrows(self):
        for i in range(len(self)):
            yield i, self._get_row(i)

    def __repr__(self):
        return f"DataFrame({self.columns})"
