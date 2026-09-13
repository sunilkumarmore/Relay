"""Relay's tests.

A package, so shared helpers can be imported as `tests.helpers` no matter how
pytest is invoked. Without this, `pytest` and `python -m pytest` put different
things on sys.path and only the latter worked.
"""
