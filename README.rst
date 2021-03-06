.. raw:: html

    <embed>
        <p align="center">
            <img width="300" src="https://github.com/yngtodd/scene/blob/master/img/scene.png">
        </p>
    </embed>

--------------------------

.. image:: https://badge.fury.io/py/scene.png
    :target: http://badge.fury.io/py/scene

.. image:: https://circleci.com/gh/yngtodd/scene.svg?style=svg
    :target: https://circleci.com/gh/yngtodd/scene

=============================
Scene
=============================

Guessing movie genres from script snippets.

Documentation
--------------
 
For references, tutorials, and examples check out our `documentation`_.

Installation
------------

From Sources:

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/yngtodd/scene

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/yngtodd/scene/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python setup.py install

The repository uses spaCy for tokenization of texts. The easiest way to install SpaCy is through pip:

.. code-block:: console

    pip install -U spacy
    python -m spacy download en

.. _tarball: https://github.com/yngtodd/scene/tarball/master
.. _documentation: https://scene.readthedocs.io/en/latest
