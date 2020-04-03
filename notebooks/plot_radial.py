#!/usr/bin/env python

from LLC_Membranes.timeseries.coordinate_trace import CoordinateTrace
from LLC_Membranes.llclib import file_rw

res = 'MET'

trace = file_rw.load_object('trajectories/%s_trace.pl' % res)

trace.plot_trace(2)

