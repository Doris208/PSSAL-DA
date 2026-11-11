"""Shared Structural schedule for training, probes and NumPy diagnostics."""
import math


def struct_gate_strength(iteration, schedule='legacy_warmup_linear', warmup=6000,
                         rampup=4000, initial_strength=.10, full_iter=4000):
    if schedule == 'legacy_warmup_linear':
        return min(1., max(0., (iteration-warmup)/max(float(rampup), 1.)))
    if schedule != 'early_nonzero_linear':
        raise ValueError('Unknown Structural schedule: '+schedule)
    if (not math.isfinite(initial_strength) or not 0 <= initial_strength <= 1 or
            not math.isfinite(full_iter) or full_iter <= 0):
        raise ValueError('Early Structural scheduling requires initial_strength in [0,1] and full_iter > 0')
    return initial_strength+(1-initial_strength)*min(1., max(0., iteration/float(full_iter)))
