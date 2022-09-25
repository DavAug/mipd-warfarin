# Models

This directory contains 2 models of the INR response to warfarin treatment:

1. A semi-mechanistic model introduced in Hamberg AK, Wadelius M, Lindh JD, Dahl ML, Padrini R, Deloukas P,
    Rane A, Jonsson EN. A pharmacometric model describing the relationship
    between warfarin dose and INR response with respect to variations in
    CYP2C9, VKORC1, and age. Clin Pharmacol Ther. 2010 Jun;87(6):727-34.
2. A quantitative systems pharmacology model introduced in Wajima T, Isbister GK, Duffull SB. A
    comprehensive model for the humoral coagulation network in humans. Clin Pharmacol Ther. 2009
    Sep;86(3):290-8.

The parameters of the models are defined in separate CSV files. Wajima et al's
model has to separate sets of model parameters: one from the original publication,
and one introduced in a subsequent publication by Hartmann et al. For the purpose
of this study, we also define a second version of Wajima et al's model, which
also simulates the normalised prothrombin time (INR).

