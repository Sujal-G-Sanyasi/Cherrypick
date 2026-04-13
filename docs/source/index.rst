.. Cherrypick documentation master file, created by
   sphinx-quickstart on Mon Apr 13 22:44:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cherrypick - ML model Orchestration
=============

**Cherrypick** is an automated machine learning (Automated-ML) library that simplifies:

- Data preprocessing
- Model selection
- Outlier removal 
- Explainability

It helps you quickly train, evaluate, and interpret models with minimal effort.

---

Quick Example
-------------

.. code-block:: python
    
    from cherrypick import Orchestrator

    # For Regression:
    cherry = Orchestrator(
                        problem_Statement = 'regression',
                        train = train_reg,
                        test = test_reg,
                        focus_regressor = 'mse',
                        file_dir = 'model_'
                        )
    # Orchestrates the model selection 
    cherry.orchestrate()

    # For Classification
      cherry = Orchestrator(
                          problem_Statement = 'classification',
                          train = train_classify,
                          test = test_classify,
                          focus_classifier = 'f1score',
                          file_dir = 'model_'
                        )
    # Orchestrates the model selection
    cherry.orchestrate()

---

Key Features
------------

-  Automated model selection
-  Built-in explainability (SHAP)
-  Performance evaluation tools
-  Data preprocessing pipeline

---

Installation
------------

Install via pip:

.. code-block:: bash

    pip install cherrypick-ml

---

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

