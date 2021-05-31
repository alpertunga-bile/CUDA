#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

class Timer
{
public:
	void start()
	{
		startTime = std::chrono::system_clock::now();
	}

	std::chrono::duration<double> elapsedTime()
	{
		endTime = std::chrono::system_clock::now();
		return endTime - startTime;
	}

private:
	std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
};

void IfFailed(cudaError_t, int);

template<typename T>
void checkResult(T*, T*, const int N);

template<typename T>
void printMatrix(T*, const int, const int);

template<typename T>
void printMatrix(T* C, const int nx, const int ny)
{
	T* ic = C;
	printf("\nMatrix: %d x %d\n", nx, ny);

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%3f", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

template<typename T>
void checkResult(T* host_ref, T* gpu_ref, const int N)
{
	double epsilon = 1.0E-8;
	bool match = true;
	
	for (int i = 0; i < N; i++)
	{
		if (abs(host_ref[i] - gpu_ref[i]) > epsilon)
		{
			match = false;
			printf("\033[0;31m");
			printf("Arrays do not match\n");
			printf("\033[0m");
			printf("host %5.2f gpu %5.2f at current %d\n", host_ref[i], gpu_ref[i], i);
			break;
		}
	}

	if (match)
	{
		printf("\033[0;32m");
		printf("Arrays matched");
		printf("\033[0m");
	}
}

void IfFailed(cudaError_t error)
{
	if (error != cudaSuccess)
	{
		printf("ERROR::Memory Management::File %s::Line %d\n", __FILE__, __LINE__);
		printf("code: %d, reason: %s\n", error,cudaGetErrorString(error));
		exit(1);
	}
}