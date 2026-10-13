"""Overlapping-experiments family: joint launch evidence for two concurrent binary experiments.

Two independently randomized experiments are analyzed on their declared
jointly eligible population. Per-unit eligibility, assignment, and exposure
timing are validated before the four joint assignment cells are summarized;
prespecified contrasts of the cell means then give marginal effects under the
experiment's treatment mix, effects conditional on the other experiment's
arm, joint launch effects, and the interaction. Intervals are asymptotic in
the per-cell counts.
"""

from increment._overlap.admission import CellMoments, EligibilityAudit
from increment._overlap.contrasts import CONTRAST_DESCRIPTIONS, CONTRASTS, OverlapContrast
from increment._overlap.design import ExperimentColumns, OverlapDesign, OverlapExperiment
from increment._overlap.refusals import OVERLAP_REFUSALS
from increment.overlap.cells import OverlapCells
from increment.overlap.estimator import (
    ASSIGNMENT_CHECK_ALPHA,
    AssignmentCheck,
    OverlapEffect,
    OverlapReadout,
    estimate_overlap_effects,
)

__all__ = [
    "ASSIGNMENT_CHECK_ALPHA",
    "CONTRASTS",
    "CONTRAST_DESCRIPTIONS",
    "OVERLAP_REFUSALS",
    "AssignmentCheck",
    "CellMoments",
    "EligibilityAudit",
    "ExperimentColumns",
    "OverlapCells",
    "OverlapContrast",
    "OverlapDesign",
    "OverlapEffect",
    "OverlapExperiment",
    "OverlapReadout",
    "estimate_overlap_effects",
]
