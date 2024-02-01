from typing import Callable, Collection, Dict, Union


def is_iterable(c) -> bool:
    try:
        iter(c)
    except TypeError:
        return False
    else:
        return True


def membership_fn(c: Collection) -> Callable:
    def is_inside(x):
        return x in c
    return is_inside


def update_keys(d: Dict, key_fn: Callable) -> Dict:
    return {key_fn(k): v for k, v in d.items()}


def update_values(d: Dict, val_fn: Callable) -> Dict:
    return {k: val_fn(v) for k, v in d.items()}


def filter_keys(d: Dict, key_fn: Union[Callable, Collection]) -> Dict:
    if is_iterable(key_fn):
        key_fn = membership_fn(key_fn)
    return {k: v for k, v in d.items() if key_fn(k)}


def filter_values(d: Dict, val_fn: Union[Callable, Collection]) -> Dict:
    if is_iterable(val_fn):
        val_fn = membership_fn(val_fn)
    return {k: v for k, v in d.items() if val_fn(v)}


# ----------------------------------------------------------------------------------------------------------------------
# Multi-dict utilities
# ----------------------------------------------------------------------------------------------------------------------

def same_keys(a: Dict, b: Dict) -> bool:
    return set(a.keys()) == set(b.keys())


def add_dicts(a: Dict, b: Dict) -> Dict:
    assert same_keys(a, b), "Dictionaries must have exact same keys when adding them!"
    return {key: value + b[key] for key, value in a.items()}
