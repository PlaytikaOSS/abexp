[comment]: <> (Modify also docs/installation.rst if change the README.md)
[comment]: <> (Modify also LICENSE.rst if change the README.md)

ABexp
=====

[comment]: <> (Modify also docs/badges.rst if you change the badges)
[comment]: <> (Modify also LICENSE.rst if you change the license)
![alt text](https://img.shields.io/badge/build-passing-brightgreen)
![alt text](https://img.shields.io/badge/docs-passing-brightgreen)
![alt text](https://img.shields.io/badge/coverage-95%25-green)
![alt text](https://img.shields.io/badge/version-0.0.1-blue)
![alt text](https://img.shields.io/badge/license-MIT-blue)

**ABexp**  is a ``Python`` library which aims to support users along the entire end-to-end A/B test experiment flow
(see picture below). It contains A/B testing modules which use both frequentist and bayesian statistical approaches
including bayesian generalized linear model (GLM).

<br/>

![A/B testing experiment flow](https://github.com/PlaytikaOSS/abexp/blob/main/docs/src/img/experiment_flow.png)

<br/>


Installation
------------

This library is distributed on [PyPI](https://pypi.org/project/abexp/) and
can be installed with ``pip``. The latest release is version ``0.0.1``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install abexp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The command above will automatically install all the dependencies listed in ``requirements.txt``. Please visit the
[installation](https://playtikaoss.github.io/abexp/installation.html)
page for more details.

<br/>

Getting started
---------------
A short example, illustrating it use:

~~~~~~~~~~~~~~~
import abexp
~~~~~~~~~~~~~~~

Compute the minimum sample size needed for an A/B test experiment with two variants, so called control and treatment
groups.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from abexp.core.design import SampleSize

c = 0.33  # conversion rate control group
t = 0.31  # conversion rate treatment group

sample_size = SampleSize.ssd_prop(prop_contr=c, prop_treat=t)  # minimum sample size per each group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<br/>

Documentation
-------------
For more information please read the full
[documentation](https://playtikaoss.github.io/abexp/abexp.html)
and
[tutorials](https://playtikaoss.github.io/abexp/tutorials.html).

<br/>

Info for developers
-------------------

The source code of the project is available on [GitHub](https://github.com/playtikaoss/abexp).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ git clone https://github.com/PlaytikaOSS/abexp.git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install the library and the dependencies with one of the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ pip install .                        # install library + dependencies
$ pip install .[develop]               # install library + dependencies + developer-dependencies
$ pip install -r requirements.txt      # install dependencies
$ pip install -r requirements-dev.txt  # install developer-dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As suggested by the authors of ``pymc3`` and ``pandoc``, we highly recommend to install these dependencies with
``conda``:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ conda install -c conda-forge pandoc
$ conda install -c conda-forge pymc3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the file ``abexp.whl`` for the installation with ``pip`` run the following command:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ python setup.py sdist bdist_wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the HTML documentation run the following commands:

~~~~~~~~~~~
$ cd docs
$ make html
~~~~~~~~~~~

<br/>

Run tests
---------

Tests can be executed with ``pytest`` running the following commands:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
$ cd tests
$ pytest                                      # run all tests
$ pytest test_testmodule.py                   # run all tests within a module
$ pytest test_testmodule.py -k test_testname  # run only 1 test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<br/>

License
-------

[MIT License](LICENSE)
