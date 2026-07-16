# External data

The SWaT dataset is NOT redistributed here.

Obtain SWaT.A1 & A2 (Dec. 2015) from iTrust, Singapore University of Technology
and Design (access on request): https://itrust.sutd.edu.sg/itrust-labs_datasets/

Place the merged CSV here, then:
  python -m asqdt.scripts.rebuild_swat_traces dataset/<merged>.csv
  python -m asqdt.scripts.run_swat_validation

Expected after reconstruction: 491,354 normal rows; attack trace of 455,365 rows
containing 54,621 attack-labeled seconds across 35 contiguous scenarios.
