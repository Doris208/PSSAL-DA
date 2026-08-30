# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.phy_reliability import PhyReliabilityHead
from mmseg.models.uda.proto_gate import ProtoGate
from mmseg.models.uda.struct_prior_gate import StructPriorConsistencyGate
from mmseg.models.uda.teacher_module import EMATeacher

try:
    from mmseg.models.uda.pssal_supervised import PSSALSupervised
except ModuleNotFoundError:
    PSSALSupervised = None

__all__ = [
    'DACS', 'PhyReliabilityHead',
    'StructPriorConsistencyGate', 'ProtoGate',
    'MaskingConsistencyModule', 'EMATeacher'
]
if PSSALSupervised is not None:
    __all__.append('PSSALSupervised')
