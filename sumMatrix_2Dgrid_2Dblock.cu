#include "helper.h"

void initialData(float* ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < size; i++)
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumMatrixHost(float* A, float* B, float* C, const int nx, const int ny)
{
	float* ia = A;
	float* ib = B;
	float* ic = C;

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx; ib += nx; ic += nx;
	}
}

__global__ void sumMatrixDevice(float* A, float*B, float*C,const int nx, const int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;

	if (ix < nx && iy < ny)
		C[idx] = A[idx] + B[idx];
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

	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int numBytes = nxy * sizeof(float);

	////////////////////////////////////////////////////////////////////////
	// CPU SIDE 
	////////////////////////////////////////////////////////////////////////

	float* h_A, *h_B, *h_ref, *d_ref;
	h_A = (float*)malloc(numBytes);
	h_B = (float*)malloc(numBytes);
	h_ref = (float*)malloc(numBytes);
	d_ref = (float*)malloc(numBytes);

	initialData(h_A, nxy);
	initialData(h_B, nxy);

	memset(h_ref, 0, numBytes);
	memset(d_ref, 0, numBytes);

	timer.start();
	sumMatrixHost(h_A, h_B, h_ref, nx, ny);
	tTime = timer.elapsedTime();
	printf("sumMatrixHost %d x %d Time: %f s\n", nx, ny, tTime.count());

	////////////////////////////////////////////////////////////////////////
	// GPU SIDE 
	////////////////////////////////////////////////////////////////////////

	int dimx = 16;
	int dimy = 16;
	dim3 block(dimx, dimy);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	float* d_A, float* d_B, float* d_C;
	IfFailed(cudaMalloc((void**)&d_A, numBytes));
	IfFailed(cudaMalloc((void**)&d_B, numBytes));
	IfFailed(cudaMalloc((void**)&d_C, numBytes));

	IfFailed(cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice));
	IfFailed(cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice));

	timer.start();
	sumMatrixDevice << <grid, block >> > (d_A, d_B, d_C, nx, ny);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();
	printf("sumMatrixDevice<<<(%d, %d), (%d, %d)>>> %d x %d Time: %f s\n", grid.x, grid.y, block.x, block.y, nx, ny, tTime.count());

	IfFailed(cudaMemcpy(d_ref, d_C, numBytes, cudaMemcpyDeviceToHost));

	checkResult(h_ref, d_ref, nxy);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_ref);
	free(d_ref);

	cudaDeviceReset();

	return 0;
}