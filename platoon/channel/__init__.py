"""
:mod:`channel` -- Platoon's communication backend
=================================================

.. module:: channel
   :platform: Unix
   :synopsis: Contains controller and worker modules which compose Platoon's
              communication architecture.

This file serves as a backwards compatibility layer for Platoon v0.5.0.

"""
from __future__ import absolute_import
from .worker import Worker
from .controller import Controller
