#include "include/helper.h"

__global__ void sumDivergence(float* C)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	
	float a, b;
	a = b = 0.0f;

	if ((ix / warpSize) % 2 == 0)
		a = 100.0f;
	else
		b = 200.0f;

	C[ix] = a + b;
}

__global__ void sumEvenOdd(float* C)
{
	unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;

	float a, b;
	a = b = 0.0f;

	if (ix % 2 == 0)
		a = 100.0f;
	else
		b = 200.0f;

	C[ix] = a + b;
}

int main()
{
	Timer timer;
	std::chrono::duration<double> tTime;

	int device = 0;
	cudaDeviceProp deviceProp;
	IfFailed(cudaGetDeviceProperties(&deviceProp, device));
	printf("Using Device %d: %s\n", device, deviceProp.name);
	IfFailed(cudaSetDevice(device));

	int nx = 64;
	int ny = 1;
	int nxy = nx * ny;
	int numBytes = nxy * sizeof(float);

	////////////////////////////////////////////////////////////////////////
	// CPU SIDE 
	////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////
	// GPU SIDE 
	////////////////////////////////////////////////////////////////////////

	int n = 64;
	dim3 block(n, 1);
	dim3 grid((nx + block.x - 1) / block.x, 1);

	float* d_C;
	IfFailed(cudaMalloc((void**)&d_C, numBytes));

	timer.start();
	sumDivergence << <grid, block >> > (d_C);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();
	printf("sumDivergence<<<(%d, %d), (%d, %d)>>> %d x %d Time: %f s\n", grid.x, grid.y, block.x, block.y, nx, ny, tTime.count());

	timer.start();
	sumEvenOdd << <grid, block >> > (d_C);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();
	printf("sumEvenOdd<<<(%d, %d), (%d, %d)>>> %d x %d Time: %f s\n", grid.x, grid.y, block.x, block.y, nx, ny, tTime.count());

	cudaFree(d_C);

	cudaDeviceReset();

	return 0;
}