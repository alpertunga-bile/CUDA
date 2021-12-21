#include "include/helper.h"

int recursiveReduce(int* data, const int size)
{
	if (size == 1)
		return data[0];

	const int stride = size / 2;

	for (int i = 0; i < stride; i++)
	{
		data[i] += data[i + stride];
	}

	return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x;

	if (tid >= n) return;

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceNeighboredLess(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x;

	if (idx >= n) return;

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		// convert tid into local array index
		int index = 2 * stride * tid;
		if (index < blockDim.x)
			localData[index] += localData[index + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceInterleaved(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x;

	if (idx >= n) return;

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceUnrolling2(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 2+ threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x * 2;

	if (idx + blockDim.x < n) 
		inData[idx] += inData[idx + blockDim.x];

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceUnrolling4(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x * 4;

	if (idx + blockDim.x * 3 < n)
	{
		int  a1 = inData[idx];
		int  a2 = inData[idx + blockDim.x];
		int  a3 = inData[idx + 2 * blockDim.x];
		int  a4 = inData[idx + 3 * blockDim.x];
		inData[idx] = a1 + a2 + a3 + a4;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceUnrolling8(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x * 8;

	if (idx + blockDim.x * 7 < n)
	{
		int  a1 = inData[idx];
		int  a2 = inData[idx + blockDim.x];
		int  a3 = inData[idx + 2 * blockDim.x];
		int  a4 = inData[idx + 3 * blockDim.x];
		int  a5 = inData[idx + 4 * blockDim.x];
		int  a6 = inData[idx + 5 * blockDim.x];
		int  a7 = inData[idx + 6 * blockDim.x];
		int  a8 = inData[idx + 7 * blockDim.x];
		inData[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceUnrollingWarps8(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x * 8;

	if (idx + blockDim.x * 7 < n)
	{
		int  a1 = inData[idx];
		int  a2 = inData[idx + blockDim.x];
		int  a3 = inData[idx + 2 * blockDim.x];
		int  a4 = inData[idx + 3 * blockDim.x];
		int  a5 = inData[idx + 4 * blockDim.x];
		int  a6 = inData[idx + 5 * blockDim.x];
		int  a7 = inData[idx + 6 * blockDim.x];
		int  a8 = inData[idx + 7 * blockDim.x];
		inData[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (tid < stride)
			localData[tid] += localData[tid + stride];

		__syncthreads();
	}

	if (tid < 32)
	{
		volatile int* vmem = localData;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

__global__ void reduceCompleteUnrollingWarps8(int* inData, int* outData, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

	// iterate through blockIdx
	int* localData = inData + blockIdx.x * blockDim.x * 8;

	if (idx + blockDim.x * 7 < n)
	{
		int  a1 = inData[idx];
		int  a2 = inData[idx + blockDim.x];
		int  a3 = inData[idx + 2 * blockDim.x];
		int  a4 = inData[idx + 3 * blockDim.x];
		int  a5 = inData[idx + 4 * blockDim.x];
		int  a6 = inData[idx + 5 * blockDim.x];
		int  a7 = inData[idx + 6 * blockDim.x];
		int  a8 = inData[idx + 7 * blockDim.x];
		inData[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512) 
		localData[tid] += localData[tid + 512];
	
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) 
		localData[tid] += localData[tid + 256];
	
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) 
		localData[tid] += localData[tid + 128];
	
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) 
		localData[tid] += localData[tid + 64];
	
	__syncthreads();

	if (tid < 32)
	{
		volatile int* vmem = localData;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0)
		outData[blockIdx.x] = localData[0];
}

int main()
{
	Timer timer;
	std::chrono::duration<double> tTime;
	bool equal = false;

	// setup device
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	std::cout << "Device: " << dev << " " << deviceProp.name << std::endl;

	int size = 1 << 24;
	std::cout << "Array size: " << size << std::endl;

	int blockSize = 512;
	dim3 block(blockSize, 1);
	dim3 grid((size + block.x - 1) / block.x, 1);
	std::cout << "Kernel: <<<" << grid.x << ", " << block.x << ">>>" << std::endl;

	////////////////////////////////////////////////////////////////////////
	// CPU SIDE 
	////////////////////////////////////////////////////////////////////////

	// allocate host memory
	size_t bytes = size * sizeof(int);
	int* h_inData = (int*)malloc(bytes);
	int* h_outData = (int*)malloc(grid.x * sizeof(int));
	int* temp = (int*)malloc(bytes);

	initialDataInt(h_inData, size);

	memcpy(temp, h_inData, bytes);

	timer.start();
	const int cpu_sum = recursiveReduce(temp, size);
	tTime = timer.elapsedTime();

	std::cout << "Recursive Reduce CPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	////////////////////////////////////////////////////////////////////////
	// GPU SIDE 
	////////////////////////////////////////////////////////////////////////

	int* d_inData = NULL;
	int* d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	// | --------------------------------------------------------- |
	//   reduceNeighbored
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceNeighbored << <grid, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Neighbored GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	
	int gpu_sum = 0;

	for (int i = 0; i < grid.x; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;

	// | --------------------------------------------------------- |
	//   reduceNeighboredLess
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceNeighboredLess << <grid, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Neighbored without Warp Divergence GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceInterleaved
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceInterleaved << <grid, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Interleaved GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceUnrolling2
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);


	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceUnrolling2 << <grid.x/2, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Unrolling2 GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x / 2; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceUnrolling4
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceUnrolling4 << <grid.x/4, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Unrolling4 GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x / 4 * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x / 4; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceUnrolling8
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceUnrolling8 << <grid.x/8, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce Unrolling8 GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceUnrollingWarps8
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceUnrollingWarps8 << <grid.x / 8, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce UnrollingWarps8 GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	// | --------------------------------------------------------- |
	//   reduceCompleteUnrollingWarps8
	// | --------------------------------------------------------- |

	cudaFree(d_inData);
	cudaFree(d_outData);

	d_inData = NULL;
	d_outData = NULL;

	cudaMalloc((void**)&d_inData, bytes);
	cudaMalloc((void**)&d_outData, grid.x * sizeof(int));

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_inData, h_inData, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	timer.start();
	reduceCompleteUnrollingWarps8 << <grid.x / 8, block >> > (d_inData, d_outData, size);
	cudaDeviceSynchronize();
	tTime = timer.elapsedTime();

	std::cout << "Reduce CompleteUnrollingWarps8 GPU Summation for " << size << " bytes in " << tTime.count() << " s" << std::endl;

	cudaMemcpy(h_outData, d_outData, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);

	gpu_sum = 0;

	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += h_outData[i];

	if (gpu_sum == cpu_sum)
		equal = true;
	else
		equal = false;

	if (equal)
	{
		printf("\033[0;32m");
		printf("Sums matched");
		printf("\033[0m");
		printf("\n");
	}
	else
	{
		printf("\033[0;31m");
		printf("Sums do not match\n");
		printf("\033[0m");
		printf("CPU Sum: %d != GPU Sum: %d", cpu_sum, gpu_sum);
	}
	
	free(h_inData);
	free(h_outData);
	free(temp);

	cudaFree(d_inData);
	cudaFree(d_outData);

	cudaDeviceReset();

	return 0;
}