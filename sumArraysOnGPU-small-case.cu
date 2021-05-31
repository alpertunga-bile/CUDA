#include "helper.h"

void initialData(float* ip, int size)
{
	time_t t;
	srand((unsigned) time(&t));
	
	for (int i = 0; i < size; i++)
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void sumArrayHost(float* A, float* B, float* C, int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
	}
}

__global__ void sumArrayDevice(float* A, float* B, float* C, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) C[i] = A[i] + B[i];
}

int main()
{
	Timer timer;

	// setup device
	int device = 0;
	cudaSetDevice(device);

	int numElem = 1 << 24;
	printf("Vector Size: %d\n", numElem);

	// malloc host memory
	size_t numBytes = numElem * sizeof(float);

	float* h_A, * h_B, * h_ref, *d_ref;
	h_A = (float*)malloc(numBytes);
	h_B = (float*)malloc(numBytes);
	h_ref = (float*)malloc(numBytes);
	d_ref = (float*)malloc(numBytes);

	// initialize data on host side
	initialData(h_A, numElem);
	initialData(h_B, numElem);

	memset(h_ref, 0, numBytes);
	memset(d_ref, 0, numBytes);

	// malloc device global memory
	float* d_A, * d_B, * d_C;
	IfFailed( cudaMalloc((float**)&d_A, numBytes) );
	IfFailed( cudaMalloc((float**)&d_B, numBytes) );
	IfFailed( cudaMalloc((float**)&d_C, numBytes) );

	// transfer data form host to device
	IfFailed(cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice));
	IfFailed(cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice));

	// invoke kernel at host side
	int iLen = 256;
	dim3 block(iLen);
	dim3 grid( (numElem + block.x - 1) / block.x );

	timer.start();
	sumArrayDevice << <grid, block >> > (d_A, d_B, d_C, numElem);
	IfFailed(cudaDeviceSynchronize());
	std::chrono::duration<double> tTime = timer.elapsedTime();

	printf("sumArrayGPU<<<%d, %d>>> Time elapsed %f sn\n", grid.x, block.x, tTime.count());

	IfFailed(cudaMemcpy(d_ref, d_C, numBytes, cudaMemcpyDeviceToHost));

	sumArrayHost(h_A, h_B, h_ref, numElem);

	checkResult(h_ref, d_ref, numElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_ref);
	free(d_ref);

	return 0;
}