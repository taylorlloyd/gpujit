ARCH=sm_50

OPT =-g
CU_OPT=--cuda-gpu-arch=$(ARCH) -O2

CXXFLAGS:=$(shell llvm-config --cxxflags) -I/usr/local/cuda/include -g
LDFLAGS:=$(shell llvm-config --ldflags) $(shell llvm-config --libs engine codegen analysis scalaropts core irreader mc asmprinter bitreader selectiondag support target transformutils vectorize option nvptx) -ltinfo -lz -lffi -ldl -lm -lcuda

link-bc: main.o kernel.o KernelFunction.o
	g++ -pthread $(CXXFLAGS) -o link-bc main.o kernel.o KernelFunction.o $(LDFLAGS)

KernelFunction.o : KernelFunction.cpp
	clang $(OPT) $(CXXFLAGS) -c -o KernelFunction.o KernelFunction.cpp

kernel.bc: kernel.cu
	clang $(CU_OPT) --cuda-device-only -c -emit-llvm -o kernel.bc kernel.cu

main.o: main.cpp
	clang $(OPT) -fPIC $(CXXFLAGS) -c -o main.o main.cpp

kernel.o: kernel.bc
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 kernel.bc kernel.o
