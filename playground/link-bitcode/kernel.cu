#include <stdio.h>
extern "C" {
    __global__ void helloWorldKernel(int x);
}

__global__ void helloWorldKernel(int x) {
    printf("Hello World %d\n",x);
    return;
}
