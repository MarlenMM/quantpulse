"""Section 32's forward test: the app trading its own published ratings.

The backtest can only rank one of the seven categories, because five of them
have weeks of stored history rather than years (`analysis/backtest.py`, and the
limitation stated on the Track Record page). Forward testing is the way round
that -- it *accumulates* the history instead of needing it to already exist,
and a record decided before the outcome is known cannot be fitted to it.

`alpaca` talks to Alpaca's paper endpoint and is the only module here that
touches a network. `strategy` decides what to hold, as pure functions.
"""
