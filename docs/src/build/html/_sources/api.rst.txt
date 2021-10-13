=================
API documentation
=================

Architecture
------------

* ``abexp.core`` contains the most relevant modules to run A/B test experiments.

   - :mod:`abexp.core.design`
   - :mod:`abexp.core.planning`
   - :mod:`abexp.core.allocation`
   - :mod:`abexp.core.analysis_frequentist`
   - :mod:`abexp.core.analysis_bayesian`

* ``abexp.statistics`` provides statistical utils used in the modules above.

   - :mod:`abexp.statistics.stats_metrics`
   - :mod:`abexp.statistics.stats_tests`

* ``abexp.visualization`` contains modules to display A/B test results.

    - :mod:`abexp.visualization.analysis_plots`

API
---

Please visit the full :ref:`API list <modindex>` for details.
