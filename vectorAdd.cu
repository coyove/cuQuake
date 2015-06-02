/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#define OMEGANUM 99840
#define TIMENUM 512

//__constant__ float gOmega[OMEGANUM];
//__constant__ float gTime[OMEGANUM];

__global__ void genARS(const float *inputOmega, float *outputARS, int numElems){
	const float PI = 3.14159265358;
	const float A = 0.0484;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElems){
		float omega = inputOmega[i];
		float t = PI * 2 / omega;
		if (t < 0.1)
			outputARS[i] = A * (5.5 * t + 0.45);
		else if (t < 0.35)
			outputARS[i] = A;
		else
			outputARS[i] = A * (float)(0.35 / t);
	}
}

__global__ void genWRS(const float *inputARS, const float *inputOmega, float *outputWRS, float duration, int numElems){
	const float PI = 3.14159265358;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElems){
		outputWRS[i] = 0.05 * (inputARS[i] * inputARS[i]) / (-1 * (PI * inputOmega[i]) * log(-PI * log(0.85) / (inputOmega[i] * duration)));
		if (outputWRS[i] < 0) outputWRS[i] = 0;
	}
	
}

__global__ void setupPRNG(curandState *state){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(731, idx, 0, &state[idx]);
}

__global__ void genASW(const float *inputWRS, const float *inputOmega, const float *inputTime, float *outputASW, int numTimeElems, int numOmegaElems, curandState *state){
	const float PI = 3.14159265358;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numTimeElems){
		float t = inputTime[i];
		float sum = 0;
		for (int j = 0; j < numOmegaElems; ++j){
			float r = curand_uniform(state + j) * PI * 2;
			sum += sqrt(4 * 0.01 * inputWRS[j]) * __cosf(inputOmega[j] * t + r);
		}
		outputASW[i] = sum;
	}

}

float regenARS(const float inputTime, const float *inputASW, int numTimeElems){
	const float PI = 3.14159265358;

	float angfreq = 2 * PI / inputTime;
	float deltime = 0.0390625;
	float dampr = 0.05;
	
	float a = angfreq * sqrt(1 - dampr * dampr) * deltime;
	float b = dampr * angfreq * deltime;
	float d = sqrt(1 - dampr * dampr);
	float e = exp(-b);
	float a11 = e * (dampr * sin(a) / d + cos(a));
	float a12 = e * sin(a) / angfreq / d;
	float a21 = e * (-angfreq) * sin(a) / d;
	float a22 = e * (-dampr * sin(a) / d + cos(a));
	float b11 = e * ((2 * dampr * dampr - 1 + b)*sin(a) / pow(angfreq, 2) / a + (2 * dampr + angfreq * deltime)*cos(a) / pow(angfreq, 3) / deltime) - 2 * dampr / pow(angfreq, 3) / deltime;
	float b12 = e * ((1 - 2 * dampr * dampr) * sin(a) / pow(angfreq, 2) / a - 2 * dampr *cos(a) / pow(angfreq, 3) / deltime) - 1 / pow(angfreq, 2) + 2 * dampr / pow(angfreq, 3) / deltime;
	float b21 = e * ((-dampr - angfreq*deltime)*sin(a) / angfreq / a - cos(a) / pow(angfreq, 2) / deltime) + 1 / pow(angfreq, 2) / deltime;
	float b22 = e * (dampr *sin(a) / angfreq / a + cos(a) / pow(angfreq, 2) / deltime) - 1 / pow(angfreq, 2) / deltime;

	float *tempdis = new float[numTimeElems];
	float *tempvel = new float[numTimeElems];
	float *tempacc = new float[numTimeElems];

	tempdis[0] = tempvel[0] = tempacc[0] = 0;

	float max = 0;

	for (int k = 1; k < numTimeElems; ++k){
		tempdis[k] = a11 * tempdis[k - 1] + a12 * tempvel[k - 1] + b11 * inputASW[k - 1] + b12 * inputASW[k];
		tempvel[k] = a21 * tempdis[k - 1] + a22 * tempvel[k - 1] + b21 * inputASW[k - 1] + b22 * inputASW[k];
		tempacc[k] = -(2 * dampr * angfreq * tempvel[k] + angfreq * angfreq * tempdis[k]);
		if (abs(tempacc[k]) > max) max = abs(tempacc[k]);
	}

	return max;
}

/**
* Host main routine
*/

void assertError(cudaError_t err, char *prompt){
	if (err != cudaSuccess)
	{
		fprintf(stderr, prompt, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

float envelopeF(float t, float v){
	if (t <= 2)
		return v * (t * t / 4);
	else if (t >= 16)
		return v * exp(-(2.5 / (0.4 * 20)) * (t - 16));
	else
		return v;
}

float findMax(float *A, int l){
	float max = 0;
	for (int i = 0; i < l; ++i)
		if (abs(A[i]) > max) 
			max = abs(A[i]);
	return max;
}

void queryCUDACard(){
	int deviceCount, device;
	int gpuDeviceCount = 0;
	struct cudaDeviceProp properties;
	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
	if (cudaResultCode != cudaSuccess)
		deviceCount = 0;
	/* machines with no GPUs can still report one emulation device */
	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) /* 9999 means emulation only */
			if (device == 0)
			{
				printf("multiProcessorCount %d\n", properties.multiProcessorCount);
				printf("maxThreadsPerMultiProcessor %d\n", properties.maxThreadsPerMultiProcessor);
			}
	}
}

#define PREPARE_VAR(name, size) float *name = NULL; \
								assertError(cudaMalloc((void **)&name, size), "Failed to allocate device var (error code %s)!\n"); \

/*int main(void)
{
	queryCUDACard();

    cudaError_t err = cudaSuccess;
	curandState *rState;
	cudaMalloc(&rState, sizeof(curandState));
	setupPRNG<<<1, 1>>>(rState);

    // Print the vector length to be used, and compute its size
	int numOmegaElems = OMEGANUM;
	int numTimeElems = TIMENUM;
	float dOmega = 0.01;
	float dTime = (20.0 / (float)numTimeElems);

	size_t size = numOmegaElems * sizeof(float);
	size_t size2 = numTimeElems * sizeof(float);
    //printf("[Vector addition of %d elements]\n", numElements);

	float *hTime = new float[size];
    float *hOmega = new float[size];
	float *hOTime = new float[size2];

    // Initialize the host input vectors
	for (int i = 0; i < numOmegaElems; ++i){
		hOmega[i] = dOmega * (i + 1);
		hTime[i] = 3.14159265358 * 2 / hOmega[i];
	}

	for (int i = 0; i < numTimeElems; ++i)
		hOTime[i] = dTime * (i + 1);
		
	PREPARE_VAR(inputOmega, size);
	PREPARE_VAR(inputTime, size2);
	PREPARE_VAR(zARS, size); float *hARS = new float[size];
	PREPARE_VAR(zWRS, size);
	PREPARE_VAR(zASW, size); float *hASW = new float[size2];

    printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(inputOmega, hOmega, size, cudaMemcpyHostToDevice);
	assertError(err, "Failed to copy from host to device (error code %s)!\n");

	err = cudaMemcpy(inputTime, hOTime, size2, cudaMemcpyHostToDevice);
	assertError(err, "Failed to copy from host to device (error code %s)!\n");

    // Launch the Vector Add CUDA Kernel
    int TPB = 256;
	int blocksPerGrid = (numOmegaElems + TPB - 1) / TPB;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, TPB);
    genARS<<<blocksPerGrid, TPB>>>(inputOmega, zARS, numOmegaElems);
	genWRS<<<blocksPerGrid, TPB>>>(zARS, inputOmega, zWRS, 20, numOmegaElems);
	genASW<<<TPB, (numTimeElems + TPB - 1) / TPB>>>(zWRS, inputOmega, inputTime, zASW, numTimeElems, numOmegaElems, rState);

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(hARS, zARS, size, cudaMemcpyDeviceToHost);
	assertError(err, "Failed to copy ARS from device to host (error code %s)!\n");
	err = cudaMemcpy(hASW, zASW, size2, cudaMemcpyDeviceToHost);
	assertError(err, "Failed to copy ASW from device to host (error code %s)!\n");

    printf("Done\n");

	ofstream output, sgs, newars, ars;
	string tttt = "D:\\wave.dat";
	output.open(tttt);
	//sgs.open("D:\\test.sgs");
	newars.open("D:\\newars.dat");
	ars.open("D:\\ars.dat");

	printf("Output to gnuplot\n");
	output << "#x y" << endl;
	//sgs << "*SGSw\n*TITLE, zzz\n*TITLE, \n*X-AXIS, Time(sec)\n*Y-AXIS, Ground Accel. (g)\n*UNIT&TYPE, GRAV, ACCEL\n*FLAGS, 0, 0\n*DATA\n";

	for (int i = 0; i < numTimeElems; ++i)	
		hASW[i] = envelopeF(hOTime[i], hASW[i]);
	
	float maxV = findMax(hASW, numTimeElems);
	//printf("%f\n", maxV);
	for (int i = 0; i < numTimeElems; ++i){
		float v = hASW[i] / maxV * 0.0218;
		
		output << hOTime[i] << " " << v << endl;
		//sgs << hOTime[i] << "," << v << endl;
		hASW[i] = v;
	}
	//sgs << "*ENDDATA\n";
	output.close();
	//sgs.close();

	ars << "#x y" << endl;
	for (int i = 0; i < numOmegaElems; ++i){
		ars << hTime[i] << " " << hARS[i] << endl;
	}
	ars.close();

	float *hARS2 = new float[size2];
	for (int i = 0; i < numTimeElems; ++i){
		hARS2[i] = regenARS(hOTime[i], hASW, numTimeElems);
	}

	newars << "#x y" << endl;
	float *dd = new float[numTimeElems];
	for (int i = 0; i < numTimeElems; ++i){
		newars << hOTime[i] << " " << hARS2[i] << endl;
		
		float tom = 3.1415926 * 2 / hOTime[i];
		dd[i] = hARS2[i] - hARS[(int)(tom / dOmega)];

	}
	newars.close();

	float avg = 0, sum = 0;

	for (int i = 0; i < numTimeElems; ++i)
		sum += dd[i];

	avg = sum / numTimeElems;
	sum = 0;
	for (int i = 0; i < numTimeElems; ++i)
		sum += pow(dd[i] - avg, 2);

	avg = sum / numTimeElems * 1000000;

	printf("S: %f\n", avg);
    // Free device global memory
    err = cudaFree(inputOmega);
	err = cudaFree(inputTime);
    err = cudaFree(zARS);
	err = cudaFree(zWRS);
	err = cudaFree(zASW);
    free(hOmega);
	free(hTime);
	free(hOTime);
	free(hARS2);
    free(hARS);

    err = cudaDeviceReset();
	assertError(err, "Failed to deinitialize the device! error=%s\n");

    printf("Done\n");
	//system("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" D:\\output.plt");
    return 0;
}*/

float cuWave(string outputFile, string outputARS1, string outputARS2)
{
	cudaError_t err = cudaSuccess;
	curandState *rState;
	cudaMalloc(&rState, sizeof(curandState));
	setupPRNG << <1, 1 >> >(rState);

	// Print the vector length to be used, and compute its size
	int numOmegaElems = OMEGANUM;
	int numTimeElems = TIMENUM;
	float dOmega = 0.01;
	float dTime = (20.0 / (float)numTimeElems);

	size_t size = numOmegaElems * sizeof(float);
	size_t size2 = numTimeElems * sizeof(float);
	//printf("[Vector addition of %d elements]\n", numElements);

	float *hTime = new float[size];
	float *hOmega = new float[size];
	float *hOTime = new float[size2];

	// Initialize the host input vectors
	for (int i = 0; i < numOmegaElems; ++i){
		hOmega[i] = dOmega * (i + 1);
		hTime[i] = 3.14159265358 * 2 / hOmega[i];
	}

	for (int i = 0; i < numTimeElems; ++i)
		hOTime[i] = dTime * (i + 1);

	PREPARE_VAR(inputOmega, size);
	PREPARE_VAR(inputTime, size2);
	PREPARE_VAR(zARS, size); float *hARS = new float[size];
	PREPARE_VAR(zWRS, size);
	PREPARE_VAR(zASW, size); float *hASW = new float[size2];

	//printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(inputOmega, hOmega, size, cudaMemcpyHostToDevice);
	assertError(err, "Failed to copy from host to device (error code %s)!\n");

	err = cudaMemcpy(inputTime, hOTime, size2, cudaMemcpyHostToDevice);
	assertError(err, "Failed to copy from host to device (error code %s)!\n");

	// Launch the Vector Add CUDA Kernel
	int TPB = 256;
	int blocksPerGrid = (numOmegaElems + TPB - 1) / TPB;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, TPB);
	genARS << <blocksPerGrid, TPB >> >(inputOmega, zARS, numOmegaElems);
	genWRS << <blocksPerGrid, TPB >> >(zARS, inputOmega, zWRS, 20, numOmegaElems);
	genASW << <TPB, (numTimeElems + TPB - 1) / TPB >> >(zWRS, inputOmega, inputTime, zASW, numTimeElems, numOmegaElems, rState);

	//printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(hARS, zARS, size, cudaMemcpyDeviceToHost);
	assertError(err, "Failed to copy ARS from device to host (error code %s)!\n");
	err = cudaMemcpy(hASW, zASW, size2, cudaMemcpyDeviceToHost);
	assertError(err, "Failed to copy ASW from device to host (error code %s)!\n");

	printf("CUDA Device Done\n");

	ofstream output, sgs, newars, ars;

	output.open(outputFile);
	//sgs.open("D:\\test.sgs");
	newars.open(outputARS2);
	

	//printf("Output to gnuplot\n");
	output << "#x y" << endl;
	//sgs << "*SGSw\n*TITLE, zzz\n*TITLE, \n*X-AXIS, Time(sec)\n*Y-AXIS, Ground Accel. (g)\n*UNIT&TYPE, GRAV, ACCEL\n*FLAGS, 0, 0\n*DATA\n";

	for (int i = 0; i < numTimeElems; ++i)
		hASW[i] = envelopeF(hOTime[i], hASW[i]);

	float maxV = findMax(hASW, numTimeElems);
	//printf("%f\n", maxV);
	for (int i = 0; i < numTimeElems; ++i){
		float v = hASW[i] / maxV * 0.0218;

		output << hOTime[i] << " " << v << endl;
		//sgs << hOTime[i] << "," << v << endl;
		hASW[i] = v;
	}
	//sgs << "*ENDDATA\n";
	output.close();
	//sgs.close();
	if (!outputARS1.empty()){
		ars.open(outputARS1);
		ars << "#x y" << endl;
		for (int i = 0; i < numOmegaElems; ++i){
			ars << hTime[i] << " " << hARS[i] << endl;
		}
		ars.close();
	}

	float *hARS2 = new float[size2];
	for (int i = 0; i < numTimeElems; ++i){
		hARS2[i] = regenARS(hOTime[i], hASW, numTimeElems);
	}

	newars << "#x y" << endl;
	float *dd = new float[numTimeElems];
	for (int i = 0; i < numTimeElems; ++i){
		newars << hOTime[i] << " " << hARS2[i] << endl;

		float tom = 3.1415926 * 2 / hOTime[i];
		dd[i] = hARS2[i] - hARS[(int)(tom / dOmega)];

	}
	newars.close();

	float avg = 0, sum = 0;

	for (int i = 0; i < numTimeElems; ++i)
		sum += dd[i];

	avg = sum / numTimeElems;
	sum = 0;
	for (int i = 0; i < numTimeElems; ++i)
		sum += pow(dd[i] - avg, 2);

	avg = sum / numTimeElems * 1000000;

	//printf("S: %f\n", avg);
	// Free device global memory
	err = cudaFree(inputOmega);
	err = cudaFree(inputTime);
	err = cudaFree(zARS);
	err = cudaFree(zWRS);
	err = cudaFree(zASW);
	free(hOmega);
	free(hTime);
	free(hOTime);
	free(hARS2);
	free(hARS);

	err = cudaDeviceReset();
	assertError(err, "Failed to deinitialize the device! error=%s\n");

	printf("Done\n");
	
	return avg;
}

int main(void){
	cout << "cuQuake" << endl;
	queryCUDACard();

	float *dd = new float[10];

	for (int i = 0; i < 10; ++i){
		string ofn = "D:\\wave_" + to_string(i) + ".dat";
		string ofnars1 = "D:\\ars_" + to_string(i) + ".dat";
		string ofnars2 = "D:\\newars_" + to_string(i) + ".dat";

		if (i==0)
			dd[i] = cuWave(ofn, ofnars1, ofnars2);
		else
			dd[i] = cuWave(ofn, "", ofnars2);
	}

	sort(dd, dd + 10);

	cout << "Best:" << dd[0] << " Worst:" << dd[9] << endl;

	return 0;
}