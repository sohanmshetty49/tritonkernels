"""Correctness tests for the fused Triton softmax.

Runs on CPU through Triton's interpreter, so CI needs no GPU:
    TRITON_INTERPRET=1 pytest -q
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")  # must be set before triton is imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from triton_softmax import fused_softmax


@pytest.mark.parametrize("rows,cols", [(1, 1), (4, 16), (7, 100), (32, 1000), (5, 1024), (3, 3000)])
def test_matches_torch(rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols)
    torch.testing.assert_close(fused_softmax(x), torch.softmax(x, dim=1), atol=1e-5, rtol=1e-5)


def test_rows_sum_to_one():
    x = torch.randn(16, 777)
    torch.testing.assert_close(fused_softmax(x).sum(dim=1), torch.ones(16), atol=1e-5, rtol=0)


def test_large_values_do_not_overflow():
    # without subtracting the row max, exp(1e4) overflows to inf
    x = torch.full((2, 64), 1e4)
    x[:, 0] = 1e4 + 1
    y = fused_softmax(x)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.softmax(x, dim=1))


def test_rejects_non_contiguous_input():
    x = torch.randn(64, 32).t()  # transposed view, not row contiguous
    with pytest.raises(AssertionError):
        fused_softmax(x)
