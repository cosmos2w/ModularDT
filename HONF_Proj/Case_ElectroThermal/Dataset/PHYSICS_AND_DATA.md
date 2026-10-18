# HET-MF physical and data contract

The native sheet is the unit square in dimensionless fixed reference units. The saved environment consists of 12 × 12 control records for electrical sheet conductance `s`, thermal sheet conductance `kappa`, and cooling `beta`; the solver reconstructs positive graded fields by log-bilinear interpolation with border clamping. Contact records contain center, radius, total conductance `gamma`, and terminal voltage `u`. Contacts have normalized finite-area quartic footprints and insulating lateral boundaries.

The reference first solves the finite-volume electrical system, then deposits its unique-face Joule energy and finite-contact loss into the sheet and solves the thermal system. Raw arrays include potential, temperature rise, separate sheet/contact heat, and oriented face current/heat flux. Processed fields use `[potential, jx_cell, jy_cell, temperature_rise, q_total]` in native `[y,x]` order. Cell current reconstructions are visualization or model targets, not substitutes for face-flux balance. Source labels, solver diagnostics, and hashes are excluded from model inputs.

The pilot is development-exposed and is not an independent formal TEST set. There is no HONF training, learned graph, inverse optimization, physical-unit calibration, or approved formal dataset in this case.
