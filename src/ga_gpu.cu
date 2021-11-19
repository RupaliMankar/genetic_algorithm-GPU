#ifndef GA_GPU_CU
#define GA_GPU_CU

//#include <cuda.h>
//#include "cuda_runtime.h"
//#include <cuda_runtime_api.h>
//#include "device_launch_parameters.h"
#include <stim/cuda/cudatools/error.h>

#include "timer.h"
//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <fstream>

extern Timer timer;


__global__ void kernel_computeSb(float* gpuSb, unsigned int* gpuP, float* gpuM, float* gpuCM, size_t ub, size_t f, size_t p, size_t nC, unsigned int* gpu_nPxInCls){

	size_t i = blockIdx.x * blockDim.x + threadIdx.x;			//gnomeindex in population matrix
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;			//index of feature index from gnome
	size_t gnomeIndx = blockIdx.z * blockDim.z + threadIdx.z;			//if we use 3d grid then it is needed
	
   
	if(gnomeIndx >= p || i >= f || j >= f) return;								//handling segmentation fault
 	
	//form a sb matrix from vector sbVec, multiply each element in matrix with num of pixels in the current class
	//and add it to previous value of between class scatter matrix sb 
	float tempsbval;
	size_t n1;
	size_t n2;
	size_t classIndx;										//class index in class mean matrix
		
	for(size_t c = 0; c < nC; c++){	
		tempsbval = 0;
		classIndx = c * ub;
		n1 = gpuP[gnomeIndx * f + i];						//actual feature index in original feature matrix 
		n2 = gpuP[gnomeIndx * f + j];						//actual feature index in original feature matrix 
		tempsbval = ((gpuCM[classIndx  + n1] - gpuM[n1]) *(gpuCM[classIndx  + n2] - gpuM[n2])) * (float)gpu_nPxInCls[c] ;	
		gpuSb[gnomeIndx * f * f + j * f + i] += tempsbval;
	}	
}


//Compute within class scatter sw (p x f x f) of all gnome features phe(tP x f)
__global__ void kernel_computeSw(float* gpuSw, unsigned int* gpuP, float* gpuCM, float* gpuF, unsigned int* gpuT, size_t ub, size_t f, size_t p, size_t nC, size_t tP){
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;			//gnomeindex in population matrix
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;			//index of feature index from gnome
	size_t gnomeIndx = blockIdx.z * blockDim.z + threadIdx.z;	//total number of individuals
		
	if(gnomeIndx >= p || i >= f || j >= f) return;				//handling segmentation fault
	float tempswval;

	size_t n1 = gpuP[gnomeIndx * f + i];						//actual feature index in original feature matrix 
	size_t n2 = gpuP[gnomeIndx * f + j];						//actual feature index in original feature matrix
	tempswval = 0;
	for(size_t c = 0; c < nC; c++){	
		tempswval = 0;
		for(size_t k = 0; k < tP; k++){
			if(gpuT[k] == (c+1) ){  
				tempswval += ((gpuF[ k * ub + n1] - gpuCM[c * ub  + n1]) * (gpuF[k * ub + n2] - gpuCM[c * ub  + n2]));	
			}
		}
		gpuSw[gnomeIndx * f * f + j * f + i] += tempswval;
	}		
}
	



	//=============================gpu intialization=============================================
	/// Initialize all GPU pointers used in the GA-GPU algorithm
	/// @param gpuP is a pointer to GPU memory location, will point to memory space allocated for the population
	/// @param p is the population size
	/// @param f is the number of desired features
	/// @param gpuCM is a pointer to a GPU memory location, will point to the class mean
	/// @param cpuM is a pointer to the class mean on the CPU
	/// @param gpu_nPxInCls is a pointer to a GPU memory location storing the number of pixels in each class
	/// @param gpu_nPxInCls is a CPU array storing the number of pixels in each class
	/// @param gpuSb is a GPU memory pointer to the between-class scatter matrices
	/// @param gpuSw is a GPU memory pointer to the within-class scatter matrices
	/// @param gpuF is the destination for the GPU feature matrix
	/// @param cpuF is the complete feature matrix on the CPU

	void gpuIntialization(unsigned int** gpuP, size_t p, size_t f,													//variables required for the population allocation
						  float** gpuCM, float* cpuCM, size_t nC, unsigned int ub,
						  float** gpuM, float* cpuM,  unsigned int** gpu_nPxInCls,
						  float** gpuSb, float** gpuSw,
						  float** gpuF, float* cpuF,
						  unsigned int** gpuT, unsigned int* cpuT, size_t tP, unsigned int* cpu_nPxInCls){
        
		HANDLE_ERROR(cudaMalloc(gpuP, p * f * sizeof(unsigned int)));										//allocate space for the population on the GPU

		HANDLE_ERROR(cudaMalloc(gpuCM, nC * ub * sizeof(float)));											//allocate space for the class mean and copy it to the GPU
		HANDLE_ERROR(cudaMemcpy(*gpuCM, cpuCM, nC * ub * sizeof(float), cudaMemcpyHostToDevice));


		HANDLE_ERROR(cudaMalloc(gpuM, ub * sizeof(float)));												//allocate space for the mean of the feature matrix
		HANDLE_ERROR(cudaMemcpy(*gpuM, cpuM, ub * sizeof(float), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMalloc(gpu_nPxInCls, nC * sizeof(unsigned int)));											//number of pixels in each class
		HANDLE_ERROR(cudaMemcpy(*gpu_nPxInCls, cpu_nPxInCls, nC * sizeof(unsigned int), cudaMemcpyHostToDevice));
		
		
		HANDLE_ERROR(cudaMalloc(gpuSb, p * f * f * sizeof(float)));										//allocate memory for sb which is calculated for eery class separately and added together in different kernel
		HANDLE_ERROR(cudaMalloc(gpuSw,  p * f * f * sizeof(float)));
		
		HANDLE_ERROR(cudaMalloc(gpuF, tP * ub * sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(*gpuF, cpuF, tP * ub * sizeof(float), cudaMemcpyHostToDevice));
		
		HANDLE_ERROR(cudaMalloc(gpuT, tP * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMemcpy(*gpuT, cpuT, tP* sizeof(unsigned int), cudaMemcpyHostToDevice));
  
	}	

	//computation on GPU
	/// Initialize all GPU pointers used in the GA-GPU algorithm
	/// @param gpuP is a pointer to GPU memory location, will point to memory space allocated for the population
	/// @param p is the population size
	/// @param f is the number of desired features
	/// @param gpuSb is a GPU memory pointer to the between-class scatter matrices
	/// @param cpuSb is the between-class scatter matrix on the GPU (this function will copy the GPU result there)
	/// @param gpuSw is a GPU memory pointer to the within-class scatter matrices
	/// @param cpuSw is the within-class scatter matrix on the GPU (this function will copy the GPU result there)

	/// @param gpuCM is a pointer to a GPU memory location, will point to the class mean
	/// @param cpuM is a pointer to the class mean on the CPU
	/// @param gpu_nPxInCls is a pointer to a GPU memory location storing the number of pixels in each class
	/// @param gpu_nPxInCls is a CPU array storing the number of pixels in each class	
	
	/// @param gpuF is the destination for the GPU feature matrix
	/// @param cpuF is the complete feature matrix on the CPU
	void gpucomputeSbSw(unsigned int* gpuP, unsigned int* cpuP, size_t p, size_t f,
						 float* gpuSb, float* cpuSb,
						 float* gpuSw, float* cpuSw,
						 float* gpuF, unsigned int* gpuT,float* gpuM, float* gpuCM, 
						 size_t nC, size_t tP, cudaDeviceProp props, size_t gen, size_t gnrtn, size_t ub, unsigned int* gpu_nPxInCls, std::ofstream& profilefile){
		    
    timer.start();
		HANDLE_ERROR(cudaMemcpy(gpuP, cpuP, p * f * sizeof(unsigned int), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemset(gpuSb, 0, p * f * f * sizeof(float)));
				
		//grid configuration of GPU
		size_t threads = (size_t)sqrt(props.maxThreadsPerBlock);
		if(threads > f) threads = f;
		size_t numberofblocksfor_f = (size_t)ceil((float)f/ threads);
		dim3 blockdim((int)threads, (int)threads, 1);
		dim3 griddim((int)numberofblocksfor_f, (int)numberofblocksfor_f, (int)p);									//X dimension blocks will cover all gnomes of the population and each block will have as many gnomes as it can feet 
		//sharedbytes calculation
		size_t sharedBytes = p * f * sizeof(unsigned int);												//copy population to shared memory
		if(props.sharedMemPerBlock < sharedBytes) sharedBytes = props.sharedMemPerBlock;
		
		//launch kernel to compute sb matrix
		kernel_computeSb<<<griddim, blockdim, sharedBytes>>>(gpuSb, gpuP, gpuM, gpuCM, ub, f, p, nC, gpu_nPxInCls);
    cudaDeviceSynchronize();
   
		HANDLE_ERROR(cudaMemcpy(cpuSb, gpuSb, p * f * f * sizeof(float), cudaMemcpyDeviceToHost));		//copy between class scatter from gpu to cpu
		const auto elapsedg1 = timer.time_elapsed();
		if(gen > gnrtn -2){
			std::cout << "Sb gpu time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsedg1).count() << "us" << std::endl;
			profilefile << "Sb gpu time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsedg1).count() << "us" << std::endl;
		}

		timer.start();
		//Compute within class scatter
		HANDLE_ERROR(cudaMemset(gpuSw, 0, p * f * f * sizeof(float)));

		//launch kernel to compute sb matrix
		kernel_computeSw<<<griddim, blockdim>>>(gpuSw, gpuP, gpuCM, gpuF, gpuT, ub, f, p, nC, tP);
    cudaDeviceSynchronize();
		//copy between class scatter from gpu to cpu
		HANDLE_ERROR(cudaMemcpy(cpuSw, gpuSw, p * f * f * sizeof(float), cudaMemcpyDeviceToHost));
		const auto elapsedg2 = timer.time_elapsed();
		if(gen > gnrtn - 2){
			std::cout << "Sw gpu time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsedg2).count() << "us" << std::endl;
			profilefile<< "Sw gpu time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsedg2).count() << "us" << std::endl;
		}

	}
 
 //free all gpu pointers
	void gpuDestroy(unsigned int* gpuP, float* gpuCM, float* gpuM, unsigned int* gpu_nPxInCls, float* gpuSb, float* gpuSw, float* gpuF, unsigned int* gpuT){
 
		HANDLE_ERROR(cudaFree(gpuP));
		HANDLE_ERROR(cudaFree(gpuCM));
		HANDLE_ERROR(cudaFree(gpuM));
		HANDLE_ERROR(cudaFree(gpu_nPxInCls));
		HANDLE_ERROR(cudaFree(gpuSb));
    HANDLE_ERROR(cudaFree(gpuSw));
		HANDLE_ERROR(cudaFree(gpuF));
    HANDLE_ERROR(cudaFree(gpuT));
	}

#endif

