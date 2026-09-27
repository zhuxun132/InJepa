"""Local interpreter compatibility without changing tensor semantics."""
import builtins
import sys

_HAS_STRICT_ZIP = sys.version_info >= (3, 10)


def zip_compatible(*iterables, strict=False):
    if _HAS_STRICT_ZIP:
        return builtins.zip(*iterables, strict=strict)
    if not strict:
        return builtins.zip(*iterables)
    iterators = tuple(iter(value) for value in iterables)
    return _strict_rows(iterators)


def _strict_rows(iterators):
    if not iterators:
        return
    while True:
        row = []
        for index, iterator in enumerate(iterators):
            try:
                row.append(next(iterator))
            except StopIteration:
                if index:
                    raise ValueError("zip() argument is shorter than earlier arguments") from None
                sentinel = object()
                for remaining in iterators[1:]:
                    if next(remaining, sentinel) is not sentinel:
                        raise ValueError("zip() argument is longer than the first argument")
                return
        yield tuple(row)
