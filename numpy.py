import copy

uint8 = int
ndarray = object

class SimpleArray:
    def __init__(self, data, dtype=None, shape=None, fill_value=None):
        if shape is not None:
            self.data = None
            self._shape = tuple(shape)
            self._fill_value = fill_value if fill_value is not None else 0
        else:
            self.data = _deep_copy(data)
            self._shape = _shape(self.data)
            self._fill_value = None
        self.dtype = dtype or type(self.data)

    @property
    def shape(self):
        return self._shape

    def copy(self):
        return SimpleArray(self.data if self.data is not None else None,
                           dtype=self.dtype,
                           shape=self._shape,
                           fill_value=self._fill_value)

    def __mul__(self, other):
        if self.data is None:
            return SimpleArray(None, dtype=self.dtype, shape=self._shape, fill_value=self._fill_value * other)
        return SimpleArray(_apply(self.data, lambda x: x * other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, idx):
        return self.tolist()[idx]

    def __setitem__(self, idx, value):
        data = self.tolist()
        data[idx] = value
        self.data = data
        self._fill_value = None

    def __iter__(self):
        return iter(self.tolist())

    def __len__(self):
        return len(self.tolist())

    def tolist(self):
        if self.data is not None:
            return _deep_copy(self.data)
        return _build_filled(self._shape, self._fill_value)


def _shape(data):
    if isinstance(data, list):
        if not data:
            return (0,)
        inner = _shape(data[0])
        return (len(data),) + inner
    else:
        return ()


def _apply(data, func):
    if isinstance(data, list):
        return [_apply(item, func) for item in data]
    else:
        return func(data)


def _deep_copy(data):
    return copy.deepcopy(data)


def _build_filled(shape, value):
    if not shape:
        return value
    if len(shape) == 1:
        return [value for _ in range(shape[0])]
    return [_build_filled(shape[1:], value) for _ in range(shape[0])]


def array(data, dtype=None):
    return SimpleArray(data, dtype=dtype)


def zeros(shape, dtype=None):
    return _fill(shape, 0, dtype)


def ones(shape, dtype=None):
    return _fill(shape, 1, dtype)


def _fill(shape, value, dtype=None):
    return SimpleArray(None, dtype=dtype, shape=shape, fill_value=value)


def fromfile(path, dtype=None):
    try:
        with open(path, 'rb') as f:
            content = f.read()
        return array(list(content), dtype=dtype)
    except Exception:
        return None


def abs(values):
    from pandas import Series  # Local import to avoid circular dependency
    if isinstance(values, SimpleArray):
        return _apply(values.data, __builtins__['abs'])
    if isinstance(values, Series):
        return Series([__builtins__['abs'](v) if v is not None else None for v in values.data])
    return __builtins__['abs'](values)
