"""Leakage-safe, pre-match feature engineering.

Every feature for a match is computed only from matches that kicked off
before it, so the same code works for historical training rows and for
future fixtures (rows with no score yet).
"""
