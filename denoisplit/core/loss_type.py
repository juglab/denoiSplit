from denoisplit.core.custom_enum import Enum


class LossType:
    Elbo = 0
    ElboWithCritic = 1
    ElboMixedReconstruction = 2
    MSE = 3
    ElboWithNbrConsistency = 4
    ElboSemiSupMixedReconstruction = 5
    ElboCL = 6
    ElboRestrictedReconstruction = 7