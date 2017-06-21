#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("sm_%d%d\n", prop.major, prop.minor);
}
