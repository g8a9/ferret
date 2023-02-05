ferret documentation
====================

|pypi badge| |docs badge| |demo badge| |youtube badge| |arxiv badge| |downloads badge|


.. |pypi badge| image:: https://img.shields.io/pypi/v/ferret-xai.svg
    :target: https://pypi.python.org/pypi/ferret-xai
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/ferret/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://ferret.readthedocs.io/en/latest/?version=latest

.. |demo badge| image:: https://img.shields.io/badge/HF%20Spaces-Demo-yellow
    :alt: HuggingFace Spaces Demo 
    :scale: 100%
    :target: https://huggingface.co/spaces/g8a9/ferret

.. |youtube badge| image:: https://img.shields.io/badge/youtube-video-red
    :alt: YouTube Video
    :scale: 100%
    :target: https://www.youtube.com/watch?v=kX0HcSah_M4

.. |banner| image:: /_static/banner.png
    :alt: Ferret circular logo with the name to the right
    :scale: 100%
    
.. |arxiv badge| image:: https://img.shields.io/badge/arXiv-2208.01575-b31b1b.svg
    :alt: arxiv preprint
    :scale: 100%
    :target: https://arxiv.org/abs/2208.01575
    
.. |downloads badge| image:: https://pepy.tech/badge/ferret-xai
    :alt: downloads badge
    :scale: 100%
    :target: https://pepy.tech/project/ferret-xai


ferret is Python library for benchmarking interpretability techniques on
Transformers.

Use any of the badges above to test our live demo, view a video demonstration, or explore our technical paper in detail. 

.. warning::
    We are in the process of restructuring and enriching the documentation. As such, you might find WIP pages or missing
    docstrings. We are sorry for the inconvenience.

Installation
------------

To install our latest stable release, run this command in your terminal:

.. code-block:: console

    pip install -U ferret-xai

This is the preferred method to install ferret, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Citation
--------

If you are using ferret for your work, please consider citing us!

.. code-block:: bibtex

    @article{attanasio2022ferret,
      title={ferret: a Framework for Benchmarking Explainers on Transformers},
      author={Attanasio, Giuseppe and Pastor, Eliana and Di Bonaventura, Chiara and Nozza, Debora},
      journal={arXiv preprint arXiv:2208.01575},
      year={2022}
    }


.. Indices and tables
.. ==================
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

Index
-----

.. toctree::
    :maxdepth: 2

    user_guide/index
    api/index
    history