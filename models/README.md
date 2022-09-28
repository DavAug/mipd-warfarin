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

## Hamberg model and sensitivities

Outputs:
```math
c = \frac{a_c}{v} \quad \text{and} \quad \bar{y} = \bar{y}_0 + \bar{y}_{max}\left( 1 - \frac{f_1 f_2}{2}\right)
```

Pharmacokinetics:
```math
\frac{\mathrm{d}a_d}{\mathrm{d}t} = -k_a a_d + r(t)
```
```math
\frac{\mathrm{d}a_c}{\mathrm{d}t} = k_a a_d - k_e a_c
```

Pharmacodynamics:
```math
\frac{\mathrm{d}f_{i, 1}}{\mathrm{d}t} = k_i\left( 1 - \frac{\kappa c}{c + c_{50}}\right) - k_i f_{i, 1}, \quad i\in \{1, 2\}
```
```math
\frac{\mathrm{d}f_{i, 2}}{\mathrm{d}t} = k_i f_{i, 1} - k_i f_{i, 2}
```
```math
\frac{\mathrm{d}f_{i}}{\mathrm{d}t} = k_i f_{i, 2} - k_i f_{i}
```

Inferred parameters: $(v, k_e, c_{50}, k_1, k_2)$

Sensitivities:

1. with respect to $v$:
```math
\frac{\mathrm{d}c}{\mathrm{d}v} = -\frac{a_c}{v^2}
\quad \text{and} \quad
\frac{\mathrm{d}\bar{y}}{\mathrm{d}v} = -\bar{y}_{max}\left( \frac{f_2}{2}\frac{\mathrm{d}f_1}{\mathrm{d}v} + \frac{f_1}{2}\frac{\mathrm{d}f_2}{\mathrm{d}v} \right)
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_i}{\mathrm{d}v} = k_i \frac{\mathrm{d}f_{2, i}}{\mathrm{d}v} - k_i \frac{\mathrm{d}f_i}{\mathrm{d}v}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{2, i}}{\mathrm{d}v} = k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}v} - k_i \frac{\mathrm{d}f_{2,i}}{\mathrm{d}v}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{1, i}}{\mathrm{d}v} = -k_i\frac{\kappa}{c + c_{50}}\frac{\mathrm{d}c}{\mathrm{d}v} + k_i\frac{\kappa c}{(c + c_{50})^2}\frac{\mathrm{d}c}{\mathrm{d}v} - k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}v}
```

2. with respect to $k_e$:
```math
\frac{\mathrm{d}c}{\mathrm{d}k_e} = \frac{1}{v}\frac{\mathrm{d}a_c}{\mathrm{d}k_e}
\quad \text{and} \quad
\frac{\mathrm{d}\bar{y}}{\mathrm{d}k_e} = -\bar{y}_{max}\left( \frac{f_2}{2}\frac{\mathrm{d}f_1}{\mathrm{d}k_e} + \frac{f_1}{2}\frac{\mathrm{d}f_2}{\mathrm{d}k_e} \right)
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}a_c}{\mathrm{d}k_e} = - a_c - k_e \frac{\mathrm{d}a_c}{\mathrm{d}k_e}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_i}{\mathrm{d}k_e} = k_i \frac{\mathrm{d}f_{2, i}}{\mathrm{d}k_e} - k_i \frac{\mathrm{d}f_i}{\mathrm{d}k_e}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{2, i}}{\mathrm{d}k_e} = k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_e} - k_i \frac{\mathrm{d}f_{2,i}}{\mathrm{d}k_e}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_e} = -k_i\frac{\kappa}{c + c_{50}}\frac{\mathrm{d}c}{\mathrm{d}k_e} + k_i\frac{\kappa c}{(c + c_{50})^2}\frac{\mathrm{d}c}{\mathrm{d}k_e} - k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_e}
```

3. with respect to $c_{50}$:
```math
\frac{\mathrm{d}c}{\mathrm{d}c_{50}} = 0
\quad \text{and} \quad
\frac{\mathrm{d}\bar{y}}{\mathrm{d}c_{50}} = -\bar{y}_{max}\left( \frac{f_2}{2}\frac{\mathrm{d}f_1}{\mathrm{d}c_{50}} + \frac{f_1}{2}\frac{\mathrm{d}f_2}{\mathrm{d}c_{50}} \right)
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_i}{\mathrm{d}c_{50}} = k_i \frac{\mathrm{d}f_{2, i}}{\mathrm{d}c_{50}} - k_i \frac{\mathrm{d}f_i}{\mathrm{d}c_{50}}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{2, i}}{\mathrm{d}c_{50}} = k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}c_{50}} - k_i \frac{\mathrm{d}f_{2,i}}{\mathrm{d}c_{50}}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{1, i}}{\mathrm{d}c_{50}} = k_i\frac{\kappa c}{(c + c_{50})^2} - k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}c_{50}}
```

4. with respect to $k_i$:
```math
\frac{\mathrm{d}c}{\mathrm{d}k_i} = 0
\quad \text{and} \quad
\frac{\mathrm{d}\bar{y}}{\mathrm{d}k_i} = -\bar{y}_{max}\left( \frac{f_2}{2}\frac{\mathrm{d}f_1}{\mathrm{d}k_i} + \frac{f_1}{2}\frac{\mathrm{d}f_2}{\mathrm{d}k_i} \right)
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_i}{\mathrm{d}k_i} = f_{2, i} - f_{i} + k_i \frac{\mathrm{d}f_{2, i}}{\mathrm{d}k_i} - k_i \frac{\mathrm{d}f_i}{\mathrm{d}k_i}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{2, i}}{\mathrm{d}k_i} = f_{1, i} - f_{2, i} + k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_i} - k_i \frac{\mathrm{d}f_{2,i}}{\mathrm{d}k_i}
```
```math
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_i} = \left( 1 - \frac{\kappa c}{c + c_{50}}\right) - f_{i, 1} - k_i \frac{\mathrm{d}f_{1, i}}{\mathrm{d}k_i}
```

