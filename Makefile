all: compute_container.sif attacker_container.sif
SHELL:=/bin/bash

%.sif: %.def
ifdef OAR_NODEFILE
	module load apptainer;sudo-g5k $$(which apptainer) -d build $@ $<;
else
	singularity -d build --fakeroot $@ $<  
endif

clean:
	rm *.sif