=============
Release notes
=============

0.5.0 (2024-02-27)
------------------

* [added] Task-API interface. (#35) The Benchmark class allows now to specify one of the supported NLP tasks and handles explanation and evaluation according to the semantic of the task.
* [added] Support to speech models for classification (#36). The library exposes a new SpeechBenchmark class that implements the methodology presentend in `this paper <https://arxiv.org/abs/2309.07733>`_. 
* [deprecated] We deprecated the methods *evaluate_samples* and *show_samples_evaluation_table* since they run basic aggregation / averaging which we decided to leave to the user.


0.4.1 (2022-12-27)
------------------

* [added] Integrated interface to Thermostat datasets and pre-coumpute feature attributions

0.4.0 (2022-09-01)
------------------

* [added] GPU inference for all the supported explainers 
* [added] Batched inference on both CPU and GPU (see our `usage guides <https://ferret.readthedocs.io/en/latest/usage.html>`_)
* New cool-looking `docs <https://ferret.readthedocs.io/en/latest>`_ using `Furo <https://github.com/pradyunsg/furo>`_.

0.1.0 (2022-05-30)
------------------

* First release on PyPI. And a lot of dreams ahead of us.

