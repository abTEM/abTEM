"""Harness tests import abtem from the checkout they live in, as the CLI does."""

from pathlib import Path

from abtem_bench.cli import use_invoking_checkout

CHECKOUT = Path(__file__).resolve().parents[2]

use_invoking_checkout()
