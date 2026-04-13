<p align="center">
  <img src="https://raw.githubusercontent.com/Sujal-G-Sanyasi/Cherrypick/main/assets/cherrylogo.jpeg" alt="cherrypick-ml logo" width="1100" height=450>
</p>

-----------------

# cherrypick-ml: A Machine Learning Orchestration and Pipeline Toolkit

| | |
| --- | --- |
| Testing | Structured validation of preprocessing, orchestration, and explainability components [![Tests](https://img.shields.io/badge/tests-passing-green?style=plastic)]() |
| Package | PyPI distribution [![cherrypick-ml](https://img.shields.io/badge/cherrypick--ml-PyPI-blue?style=plastic)](https://pypi.org/project/cherrypick-ml/) [![version](https://img.shields.io/pypi/v/cherrypick-ml?style=plastic)](https://pypi.org/project/cherrypick-ml/)|
| License | [![license](https://img.shields.io/badge/license-MIT-yellow?style=plastic)](LICENSE) |
| downloads | [![PyPI Downloads](https://static.pepy.tech/personalized-badge/cherrypick-ml?period=total&units=NONE&left_color=YELLOWGREEN&right_color=RED&left_text=downloads)](https://pepy.tech/projects/cherrypick-ml) |
---

## What is it?

**cherrypick-ml** is a Python package that provides a unified interface for building, managing, and evaluating machine learning workflows. It integrates preprocessing, anomaly detection, model orchestration, and explainability into a single, modular framework.

The library is designed to simplify real-world machine learning development by reducing repetitive code while maintaining flexibility and transparency in model pipelines.

---

## Contributors and Contributions

<a href="https://github.com/Sujal-G-Sanyasi/Cherrypick/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Sujal-G-Sanyasi/Cherrypick" width="50" />
</a>

<p align="center">
  <a href="https://github.com/Sujal-G-Sanyasi/Cherrypick">
    <img src="https://github-readme-activity-graph.vercel.app/graph?username=Sujal-G-Sanyasi&repo=Cherrypick&theme=github-dark&hide_title=true" width="800" />
  </a>
</p>

---

## Table of Contents

- [Main Features](#main-features)
- [Core Components](#core-components)
- [Where to get it](#where-to-get-it)
- [Documentation](#documentation)

---

## Main Features

cherrypick-ml provides the following core capabilities:

- Automated model orchestration for classification and regression tasks  
- Integrated preprocessing utilities including encoding and missing value handling  
- Outlier detection using statistical method such as Inter quartile range(IQR), Z-score, modified Z-score, Isolation  Forest and Local Outlier Factor based outlier pruning  
- SHAP-based explainability for feature importance and model interpretation  
- Flexible train-test splitting utilities  
- Modular design allowing independent usage of components  
- Designed for practical, real-world machine learning workflows  

---

## Core Components

The library is structured into the following modules:

- **Orchestrator**  
  High-level interface for training, evaluating, and selecting models with explainable visualisation

- **preprocessing**  
  Tools for encoding, imputation, and feature preparation  

- **anomaly**  
  Outlier detection and data pruning utilities  

- **explain**  
  Model explainability using SHAP-based analysis  

- **splits**  
  Utilities for dataset partitioning  

---

## Documentation
Explore the full documentation for **Cherrypick-ML**
[![Docs](https://img.shields.io/badge/docs-online-blue?style=plastic)]([https://cherrypick.readthedocs.io](https://cherrypick-ml.readthedocs.io/en/latest/modules.html))

## Where to get it

The source code is currently hosted on GitHub at:

https://github.com/Sujal-G-Sanyasi/Cherrypick

Binary installers for the latest released version are available at the Python Package Index (PyPI):

```sh
pip install cherrypick-ml
