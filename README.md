# Arc Characteristics Model for TIG Welding
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mateotournoud/acm4tw.git/main?urlpath=%2Fdoc%2Ftree%2FArc+Characteristics+Model+for+TIG+Welding.ipynb)

*By Mateo Tournoud.*


This Jupyter Notebook offers the possibility to simulate various plasma properties of TIG welding using relatively simple algebraic expressions, following the workflow proposed by Delgado-Álvarez et al. (2021) for monoatomic gases. These expressions are based on dimensional analysis, with unknown coefficients fitted to the results of both a highly detailed physical model and experimental data. They can be used to calculate the Arc Shape, the Arc Column Characteristics and the Arc–Weld Pool Interactions of a TIG welding process for a given current intensity, arc length, and gas composition.

This Jupyter Notebook and its modules do not need to be downloaded or installed. By clicking the Launch Binder button, the notebook is run in a virtual environment, facilitating the process and ensuring reproducibility.

All steps are implemented using the acm4tw module, written in Python 3.11.9. A brief explanation of the algorithms used, their limitations, and the interpretation of the results is provided in each section. All equations and coefficients used in the model are taken from Delgado-Álvarez et al. (2021).
