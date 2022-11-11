# Codes for "Unbalanced response to growth variations reshapes the cell fate decision landscape"

Jingwen Zhu, Pan Chu, Xiongfei Fu  20220923

-----------------------------



## Software and packages

* python = 3.7.1

  * **Numeric analysis**

    `numpy`, `scipy`, `pandas`

  * **Data visualization**

    `matplotlib`

  * **Parallel**

    `threading`, `joblib`

  * **C extensions for Python**

    `cython`

    

## Overview

#### Deterministic model for the mutual repressive system

The model of the Toggle was defined in `toggle_dynamic.py`. In the python class `ToggleBasic`, we defined its parameters (properties, attributions) and methods that helps calculating the system's steady-states and evaluating the ODEs.

#### Model for gene expression under fluctuating conditions

Codes for simulating the gene expression dynamics of constitutive expressed gene and the toggle are defined in `empirical_expression_toggle.py` .

#### A deterministic quasi-potential landscape

The function used for calculating the quasi-potential landscape is defined in `toggle_dynamic.py`, and the sample codes are provided in `Toggle_quasilandscape.py`. 

#### A probability potential landscape

`toggle_potential_landscape.py` is used to calculate the probability potential landscape.

