ferret documentation
====================

|pypi badge| |demo badge| |youtube badge| |arxiv badge| |downloads badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/ferret-xai.svg
    :target: https://pypi.python.org/pypi/ferret-xai
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/ferret/badge/?version=latest
    :alt: Documentation Status
    :target: https://ferret.readthedocs.io/en/latest/?version=latest

.. |demo badge| image:: https://img.shields.io/badge/HF%20Spaces-Demo-yellow
    :alt: HuggingFace Spaces Demo 
    :target: https://huggingface.co/spaces/g8a9/ferret

.. |youtube badge| image:: https://img.shields.io/badge/youtube-video-red
    :alt: YouTube Video
    :target: https://www.youtube.com/watch?v=kX0HcSah_M4

.. |banner| image:: /_static/banner.png
    :alt: Ferret circular logo with the name to the right
    
.. |arxiv badge| image:: https://img.shields.io/badge/arXiv-2208.01575-b31b1b.svg
    :alt: arxiv preprint
    :target: https://arxiv.org/abs/2208.01575
    
.. |downloads badge| image:: https://pepy.tech/badge/ferret-xai
    :alt: downloads badge
    :target: https://pepy.tech/project/ferret-xai


ferret is Python library for benchmarking interpretability techniques on
Transformers.

Use any of the badges above to test our live demo, view a video demonstration, or explore our technical paper in detail. 


Installation
------------

To install our latest stable release in default mode (which does not include the depenencies for the speech XAI functionalities), run this command in your terminal:

.. code-block:: console

    pip install -U ferret-xai

If the speech XAI functionalities are needed, then run:

.. code-block:: console

    pip install -U ferret-xai[speech]

At the moment, the speech XAI-related dependencies are the only extra ones, so installing with :code:`ferret-xai[speech]` or :code:`ferret-xai[all]` is equivalent.

These are the preferred methods to install ferret, as they will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


Citation
--------

If you are using ferret for your work, please consider citing us!

.. code-block:: bibtex

    @inproceedings{attanasio-etal-2023-ferret,
        title = "ferret: a Framework for Benchmarking Explainers on Transformers",
        author = "Attanasio, Giuseppe and Pastor, Eliana and Di Bonaventura, Chiara and Nozza, Debora",
        booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
        month = may,
        year = "2023",
        publisher = "Association for Computational Linguistics",
    }

Also, ferret's Speech XAI functionalities are based on

.. code-block:: bibtex

    @misc{pastor2023explaining,
        title " Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features",
        author= "Pastor, Eliana and Koudounas, Alkis and Attanasio, Giuseppe and Hovy, Dirk and Baralis, Elena",
        month = september,
        year = "2023",
        eprint = "2309.07733",
        archivePrefix = "arXiv",
        primaryClass = "cs.CL",
        publisher = "",
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