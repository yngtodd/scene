.. highlight:: shell

============
Installation
============


Stable release
--------------

To install scene, run this command in your terminal:

.. code-block:: console

    pip install scene 

This is the preferred method to install scene, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for scene can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/yngtodd/scene

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/yngtodd/scene/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python setup.py install

The repository uses spaCy for the tokenization of texts.

.. code-block:: console

    pip install -U spacy
    python -m spacy download en


.. _Github repo: https://github.com/yngtodd/scene
.. _tarball: https://github.com/yngtodd/scene/tarball/master
