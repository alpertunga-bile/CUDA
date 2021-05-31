#include <stdio.h>

__global__ void helloFromGpu()
{
	printf("Hello World from GPU!\n");
}

int main()
{
	printf("Hello World from CPU!\n");

	helloFromGpu << <1, 10 >> > ();

	cudaDeviceReset();
	return 0;
}


