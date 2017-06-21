ARCH=sm_50

OPT =-g
CU_OPT=--cuda-gpu-arch=$(ARCH) -O2

CXXFLAGS:=$(shell llvm-config --cxxflags) -I/usr/local/cuda/include -g
LDFLAGS:=$(shell llvm-config --ldflags) $(shell llvm-config --libs engine codegen analysis scalaropts core irreader mc asmprinter bitreader selectiondag support target transformutils vectorize option nvptx) -ltinfo -lz -lffi -ldl -lm -L/usr/local/cuda/lib64 -lcuda -lcudart

bfs: bfs.o kernel.o KernelFunction.o
	g++ -pthread $(CXXFLAGS) -o bfs bfs.o kernel.o KernelFunction.o $(LDFLAGS)

KernelFunction.o : KernelFunction.cpp
	clang $(OPT) $(CXXFLAGS) -c -o KernelFunction.o KernelFunction.cpp

kernel.bc: bfs.cu kernel.cu kernel2.cu
	clang $(CU_OPT) $(CXXFLAGS) --cuda-device-only -c -emit-llvm -o kernel.bc bfs.cu

bfs.o: bfs.cu
	clang $(CU_OPT) --cuda-host-only -fPIC $(CXXFLAGS) -c -o bfs.o bfs.cu

kernel.o: kernel.bc
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 kernel.bc kernel.o