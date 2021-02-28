#How to use:
# To compile all 3 implementations (test.cpp, global.cu, shared.cu), I just use
#	make all
# To run the CUDA implementations, and not the test.cpp, I use the syntax:
#	make run im_dim patch_dim image_name validation_flag, where:
#		im_dim->		The dimension of my image
#		patch_dim->	 	The dimension of my patch
#		image_name->	The name of the image I want to run (.csv in the proper form). If I want a random image, then I pass the word random as the image_name, otherwise I pass the string to the file.
#		validation_flag->Flag to know whether the user wants to compare this result to the C result. If flag==1 then compare. It also returns the time for the C implementation
# To run the test.cpp, I use ./test.out im_dim patch_dim image_name
#	example: make run 128 7 random 1 -> It runs with a random image, with the dimensions stated, and compares the implementations with the C implementation.
#			 ./test.out 64 7 house.csv	It run the test using the house.csv file.

# The run command would be too confusing to use, if I passed the opportunity to give all of the parameters in the command line (they are too many). The user
# can only change the ones that actually have an effect on the running time.


CXX=g++
NVCC=nvcc 
CFLAGS= -O3

default: all

global:global.cu
	$(NVCC) $(CFLAGS) -o global.out global.cu

shared:shared.cu
	$(NVCC) $(CFLAGS)  -o shared.out shared.cu

test:test.cpp
	$(CXX) $(CFLAGS)  -o test.out test.cpp

all: test global shared

ifeq (run, $(firstword $(MAKECMDGOALS)))

  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

run:
	./global.out $(RUN_ARGS)
	./shared.out $(RUN_ARGS)


.PHONY: clean

clean:
	rm -f test.out global.out shared.out



