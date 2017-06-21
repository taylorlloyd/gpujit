#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "KernelFunction.h"

// These will be injected by objcopy
extern char _binary_kernel_bc_start;
extern char _binary_kernel_bc_end;

int main() {
    printf("Start: %lX, size: %ld bytes\n",(size_t) &_binary_kernel_bc_start, ((size_t)&_binary_kernel_bc_end)-((size_t)&_binary_kernel_bc_start));

    char* bitcode = &_binary_kernel_bc_start;
    size_t len = ((size_t)&_binary_kernel_bc_end)-((size_t)&_binary_kernel_bc_start);

    KernelFunction kernel(bitcode, len);

    CUfunction f = kernel.getCUFunction();

    int x = 4;
    void* params[] = {&x};
    CUresult err = cuLaunchKernel(f,1,1,1,1,1,1,0,0, params, NULL);
    if(err != CUDA_SUCCESS) {
        return 1;
    }
    err = cuStreamSynchronize(0);
    if(err != CUDA_SUCCESS) {
        return 2;
    }
    return 0;
}
