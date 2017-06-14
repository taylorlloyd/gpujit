ARCH=sm_50

OPT =-O2
CU_OPT=--cuda-gpu-arch=$(ARCH) -O2

CXXFLAGS:=$(shell llvm-config --cxxflags)
LDFLAGS:=$(shell llvm-config --ldflags) $(shell llvm-config --libs engine bitreader irreader scalaropts)

link-bc: main.o kernel.o
	g++ -pthread $(CXXFLAGS) -o link-bc main.o kernel.o $(LDFLAGS) -lffi -ldl -lm

kernel.bc: kernel.cu
	clang $(CU_OPT) --cuda-device-only -c -emit-llvm -o kernel.bc kernel.cu

main.o: main.cpp
	clang $(OPT) -fPIC $(CXXFLAGS) -c -o main.o main.cpp

kernel.o: kernel.bc
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 kernel.bc kernel.o
