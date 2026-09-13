"""Distributed task execution — work between an agent's thinking steps.

Inference asks a model a question. A *task* is everything else: multiplying a
block of a matrix, transforming text, reducing a column of numbers. A node that
runs tasks needs no GPU and no model, which is what makes a phone a plausible
provider.

The inference path is untouched by anything in this package.
"""
