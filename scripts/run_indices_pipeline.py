"""
CLI entry point for building PET, STI, SAPEI, and SCDHI from the combined
monthly climate dataset.
"""

from __future__ import annotations

import argparse
import logging

from componergy.indices.pipeline import main as pipeline_main
