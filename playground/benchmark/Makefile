ARCH=$(shell ../../util/native_gpu/gpu_native_arch)

OPT =-g
CU_OPT=--cuda-gpu-arch=$(ARCH) -O2

CXXFLAGS:=$(shell llvm-config --cxxflags) -I/usr/local/cuda/include -g -pthread
CXXFLAGS:= -I/usr/local/include  -fPIC -fvisibility-inlines-hidden -Werror=date-time -std=c++11 -Wall -W -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -pedantic -Wno-long-long -Wdelete-non-virtual-dtor -Wno-comment -ffunction-sections -fdata-sections -O2 -g -DNDEBUG  -fno-exceptions -DLLVM_BUILD_GLOBAL_ISEL -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/usr/local/cuda/include -g -pthread
LDFLAGS:=$(shell llvm-config --ldflags) $(shell llvm-config --libs engine codegen analysis scalaropts core irreader mc asmprinter bitreader selectiondag support target transformutils vectorize option nvptx) -ltinfo -lz -ldl -lm -L/usr/local/cuda/lib64 -lcuda -lcudart -lnvToolsExt

bfs: bfs.o kernel.o KernelFunction.o Assumption.o
	g++ -pthread $(CXXFLAGS) -o bfs bfs.o kernel.o KernelFunction.o Assumption.o $(LDFLAGS)

KernelFunction.o : KernelFunction.cpp KernelFunction.h Assumption.h
	clang $(OPT) $(CXXFLAGS) -c -o KernelFunction.o KernelFunction.cpp

Assumption.o : Assumption.h Assumption.cpp
	clang $(OPT) $(CXXFLAGS) -c -o Assumption.o Assumption.cpp

kernel.bc: bfs.cu kernel.cu kernel2.cu
	clang $(CU_OPT) $(CXXFLAGS) --cuda-device-only -c -emit-llvm -o kernel.bc bfs.cu

bfs.o: bfs.cu
	clang $(CU_OPT) --cuda-host-only -fPIC $(CXXFLAGS) -c -o bfs.o bfs.cu

kernel.o: kernel.bc
	objcopy --input binary --output elf64-x86-64 --binary-architecture i386 kernel.bc kernel.o
