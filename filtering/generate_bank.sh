#!/bin/bash

pycbc_geom_nonspinbank \
	--pn-order threePointFivePN \
	--f-low 20 --f-upper 2048 \
	--delta-f 0.0625 \
       	--min-match 0.965 \
	--min-mass1 1.0 \
	--min-mass2 1.0 \
	--max-mass1 3.0 \
	--max-mass2 3.0 \
	--verbose \
	--output-file BNS_paper_bank_v2.hdf \
       	--sample-rate 4096 \
	--psd-model aLIGOZeroDetHighPower \
	--hdf-store True
