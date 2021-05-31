#include "helper.h"

void initialData(int* ip, int size)
{
	for (int i = 0; i < size; i++)
		ip[i] = i;
}

void printMatrix(int* C, const int nx, const int ny)
{
	int* ic = C;
	printf("\nMatrix: %d x %d\n", nx, ny);

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

__global__ void printThreadIndex(int* A, const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global_index %2d ival %2d\n",
		threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main()
{
	int device = 0;
	cudaDeviceProp deviceProp;
	IfFailed(cudaGetDeviceProperties(&deviceProp, device));
	printf("Using Device %d: %s\n", device, deviceProp.name);
	IfFailed(cudaSetDevice(device));

	int nx = 8;
	int ny = 6;
	int nxy = nx * ny;
	int numBytes = nxy * sizeof(float);

	int* h_A;
	h_A = (int*)malloc(numBytes);

	initialData(h_A, nxy);
	printMatrix(h_A, nx, ny);

	int* d_A;
	IfFailed(cudaMalloc((void**)&d_A, numBytes));

	IfFailed(cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice));

	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	printThreadIndex << <grid, block >> > (d_A, nx, ny);
	cudaDeviceSynchronize();

	cudaFree(d_A);
	free(h_A);

	cudaDeviceReset();

	return 0;
}