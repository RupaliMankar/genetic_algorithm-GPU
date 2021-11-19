#ifndef GA_GPU_H
#define GA_GPU_H

#include <iostream>
#include <thread>
#include <complex>
//#include <cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "timer.h"

#include "basic_functions.h"
//LAPACKE support for Visual Studio

#ifndef LAPACK_COMPLEX_CUSTOM
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include "lapacke.h"
#endif


#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

//CUDA functions
void gpuIntialization(unsigned int** gpuP, size_t p, size_t f,													//variables required for the population allocation
	float** gpuCM, float* cpuCM, size_t nC, unsigned int ub,
	float** gpuM, float* cpuM, unsigned int** gpu_nPxInCls,
	float** gpuSb, float** gpuSw,
	float** gpuF, float* cpuF,
	unsigned int** gpuT, unsigned int* cpuT, size_t tP, unsigned int* cpu_nPxInCls);
void gpucomputeSbSw(unsigned int* gpuP, unsigned int* cpuP, size_t p, size_t f,
	float* gpuSb, float* cpuSb,
	float* gpuSw, float* cpuSw,
	float* gpuF, unsigned int* T, float* gpuM, float* gpuCM,
	size_t nC, size_t tP, cudaDeviceProp props, size_t gen, size_t gnrtn, size_t ub, unsigned int* gpu_nPxInCls, std::ofstream& profilefile);
void gpuDestroy(unsigned int* gpuP, float* gpuCM, float* gpuM, unsigned int* gpu_nPxInCls, float* gpuSb, float* gpuSw, float* gpuF, unsigned int* gpuT);

struct _fcomplex { float re, im; };
typedef struct _fcomplex fcomplex;

Timer timer;

class ga_gpu {

public:
	float* F;					//pointer to the raw data in host memory
	unsigned int* T;			//pointer to the class labels in host memory
	size_t gnrtn;				//total number of generations
	size_t p;					//population size
	size_t f;					// number of features to be selected

	unsigned int* P;			//pointer to population of current generation genotype matrix (p x f)
	float* S;					//pointer to score(fitness value) of each gnome from current population matric P
	unsigned int* i_guess;		//initial guess of features if mentioined in args add to initial population
	unsigned int ub;			//upper bound for gnome value (maximum feature index from raw feature matrix F)
	unsigned int lb;			//lower bound for gnome value (minimum feature index from raw feature matrix F = 0)
	float uniformRate;
	float mutationRate;
	size_t tournamentSize;  	//number of potential gnomes to select parent for crossover
	bool elitism;        	//if true then passes best gnome to next generation

							//declare gpu pointers
	float* gpuF;				//Feature matrix	
	unsigned int* gpuT;			//target responses of entire feature matrix	
	unsigned int* gpuP;		//population matrix
	unsigned int* gpu_nPxInCls;
	float* gpuCM;				//class mean of entire feature matrix
	float* gpuM;				//total mean of entire feature matrix
	float* gpuSb;				//between class scatter for all individuals of current population
	float* gpuSw;				//within class scatter for all individuals of current population

								//constructor
	ga_gpu() {}

	//==============================generate initial population

		void initialize_population(std::vector<unsigned int> i_guess, bool debug) {
			if (debug) {
				std::cout << std::endl;
				std::cout << "initial populatyion is: " << std::endl;
			}
			
			lb = 0;
			P = (unsigned int*)calloc(p * f, sizeof(unsigned int));					//allcate memory for genetic population(indices of features from F), p number of gnomes of size f
			S = (float*)calloc(p, sizeof(float));							//allcate memory for scores(fitness value) of each gnome from P 

			srand(1);
			//add intial guess to the population if specified by user as a output of other algorithm or by default just random guess
			std::memcpy(P, i_guess.data(), f * sizeof(unsigned int));

			//generate random initial population
			for (size_t i1 = 1; i1 < p; i1++) {
				for (size_t i2 = 0; i2 < f; i2++) {
					P[i1 * f + i2] = rand() % ub + lb;						//select element of gnome as random feature index within lower bound(0) and upper bound(B) 
					if (debug)	std::cout << P[i1 * f + i2] << "\t";
				}
				if (debug) std::cout << std::endl;
			}
		}

		//===================generation of new population==========================================

		size_t evolvePopulation(unsigned int* newPop, float* M, bool debug) {

			//gget index of best gnome in the current population
			size_t bestG_Indx = gIdxbestGnome();
			//-------------(reproduction)-------
			if (elitism) {
				saveGnomeIdx(0, bestG_Indx, newPop);							//keep best gnome from previous generation to new generation
			}
			// ------------Crossover population---------------
			int elitismOffset;
			if (elitism) {
				elitismOffset = 1;
			}
			else {
				elitismOffset = 0;
			}

			//Do crossover for rest of population size
			for (int i = elitismOffset; i <p; i++) {
				//	std::cout<<"crossover of gnome "<<i<<std::endl;
				std::vector<unsigned int>gnome1;
				gnome1.reserve(f);
				gnome1 = tournamentSelection(5);								//select first parent for crossover from tournament selection of 5 gnomes
																				//	displaygnome(gnome1);
				std::vector<unsigned int>gnome2;
				gnome2.reserve(f);
				gnome2 = tournamentSelection(5);								//select first parent for crossover from tournament selection of 5 gnomes
																				//	displaygnome(gnome2);
				std::vector<unsigned int>gnome;
				gnome.reserve(f);
				gnome = crossover(gnome1, gnome2, M);								//Do crossover of above parent gnomes to produce new gnome
																					//	displaygnome(gnome);
				saveGnome(i, gnome, newPop);									//save crosseover result to new population
			}

			//--------------Mutate population------------
			// introduce some mutation in new population
			for (int i = elitismOffset; i <p; i++) {
				//std::cout<<"mutation of gnome"<<std::endl;
				std::vector<unsigned int>gnome;
				gnome.reserve(f);

				for (size_t n = 0; n < f; n++)
					gnome.push_back(newPop[i*f + n]);
				//std::cout<<"\n starting address "<<(&newPop[0] + i*f)<<"\t end address is "<<(&newPop[0] + i*f + f-1) <<std::endl;
				//std::copy((&newPop[0] + i*f), (&newPop[0] + i*f +f-1), gnome.begin());
				//	displaygnome(gnome);
				mutate(gnome);
				//	displaygnome(gnome);
				saveGnome(i, gnome, newPop);								//save new gnome to new population at position i
			}
			return bestG_Indx;
		}

		//============================== functions for population evolution ===========================================================================
		std::vector<unsigned int> tournamentSelection(size_t tSize) {
			// Create a tournament population
			unsigned int* tournamentP = (unsigned int*)malloc(tSize * f * sizeof(unsigned int));
			std::vector<float>tournamentS;

			// For each place in the tournament get a random individual
			for (size_t i = 0; i < tSize; i++) {
				size_t rndmIdx = rand() % p + lb;
				tournamentS.push_back(S[rndmIdx]);
				//for (size_t n = 0; n <f; n++) 
				//tournamentP[i * f + n] = (getGnome(rndmIdx)).at(n);
				std::vector<unsigned int> temp_g(getGnome(rndmIdx));
				std::copy(temp_g.begin(), temp_g.end(), tournamentP + i*f);
			}
			// Get the fittest
			std::vector<unsigned int>fittestgnome;
			fittestgnome.reserve(f);

			//select index of best gnome from fitness score
			size_t bestSIdx = 0;
			for (size_t i = 0; i < tSize; i++) {
				if (tournamentS[i] < tournamentS[bestSIdx])
					bestSIdx = i;    //float check : it was like this b(&idx[i], &idx[j]) but gave me error
			}

			for (size_t n = 0; n < f; n++)
				fittestgnome.push_back(tournamentP[bestSIdx * f + n]);
			return fittestgnome;
		} //end of tournament selection 


		std::vector<unsigned int>  crossover(std::vector<unsigned int>  gnome1, std::vector<unsigned int>  gnome2, float* M) {
			std::vector<unsigned int> gnome;
			for (size_t i = 0; i < f; i++) {
				// Crossover
				float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				if (r <= uniformRate) {
					gnome.push_back(gnome1.at(i));
				}
				else {
					gnome.push_back(gnome2.at(i));
				}
			}

			//check new gnome for all zero bands and duplicated values
			std::vector<unsigned int> gnomeunique;
			int flag = 0;
			std::sort(gnome.begin(), gnome.end()); // 1 1 2 2 3 3 3 4 4 5 5 6 7 
			std::unique_copy(gnome.begin(), gnome.end(), std::back_inserter(gnomeunique));
			/*	if(gnomeunique.size()< gnome.size()){
			flag = 1;
			std::cout<<"gnome:["<<g<<"] "<<"\t duplications are "<< (gnome.size() - gnomeunique.size())<<std::endl;
			}*/
			unsigned int featureband, featureband1, featureband2;
			if (gnomeunique.size() < f) {
				for (size_t k = gnomeunique.size(); k < f; k++) {
					featureband = rand() % ub + lb;
					for (size_t i = 0; i < f; i++) {
						featureband1 = gnome1.at(i);
						featureband2 = gnome2.at(i);
						for (size_t j = 0; j < gnomeunique.size(); j++) {
							if (gnomeunique.at(j) != featureband1) {
								featureband = featureband1;
							}
							else if (gnomeunique.at(j) != featureband2) {
								featureband = featureband2;
							}
							else if (gnomeunique.at(j) == featureband) {
								featureband = rand() % ub + lb;
								while (M[featureband] == 0) {
									featureband = rand() % ub + lb;
								}
							}
						}
					}
					gnomeunique.push_back(featureband);
				}
			}
			//if(flag ==1){
			//	std::cout<<"\n original gnome "<<g<<" are "<<std::endl;
			//	for(int k = 0; k < gnome.size(); k++)
			//		std::cout<<gnome[k]<<"\t";
			//	std::cout<<"\n unique results in cpp for gnome "<<g<<" are "<<std::endl;
			//	for(int k = 0; k < gnomeunique.size(); k++)
			//		std::cout<<gnomeunique[k]<<"\t";
			//}

			return gnomeunique;
		}

		void mutate(std::vector<unsigned int> gnome) {
			for (size_t i = 0; i < f; i++) {
				float LO = (float)0.01;
				float HI = 1;
				float r3 = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
				//if random value is less than mutationRate then mutate this gnome
				if (r3 <= mutationRate) {
					gnome.at(i) = (rand() % ub + lb);
					gnome.push_back(rand() % ub + lb);
				}
			}
		}

		///returns gnome of given index
		std::vector<unsigned int> getGnome(size_t idx) {
			std::vector<unsigned int> gnome;
			gnome.reserve(f);
			//pulling gnome idx from population P
			for (size_t n = 0; n < f; n++)
				gnome.push_back(P[idx * f + n]);
			//memcpy(&gnome[0], P+idx*f, f*sizeof(size_t));
			return gnome;
		}

		//save gnome of index gIdx from previous population at position i in the new population
		void saveGnomeIdx(size_t i, size_t gIdx, unsigned int* newPop) {
			for (size_t n = 0; n < f; n++)
				newPop[i * f + n] = P[gIdx * f + n];
		}

		void saveGnome(size_t idx, std::vector<unsigned int>gnome, unsigned int* newPop) {
			std::copy(gnome.begin(), gnome.end(), newPop + idx*f);
		}

		size_t gIdxbestGnome() {
			//std::cout<<"best gnome indes is: "<<sortSIndx()[0];
			return sortSIndx()[0];
		}

		void displaygnome(std::vector<unsigned int> gnome) {
			std::cout << "\t gnome: ";
			for (int i = 0; i<gnome.size(); ++i)
				std::cout << gnome[i] << ' ';
			std::cout << std::endl;
		}

		//---------------------post processing of score-------------------------------------
		void Snorm() { //normalize gnome scores 
			double s;
			for (size_t i = 0; i < p; i++) {
				s += S[i];			//sum of all gnome score in population
			}
			//std::cout<<"mean Score is: "<<(double) s/p;
			for (size_t i = 0; i <p; i++)
				S[i] = S[i] / s;
		}

		size_t* sortSIndx() {     //sort gnome index according to gnome scores
			//sort indices of score in ascending order (fitness value)
			size_t *idx = (size_t*)malloc(p * sizeof(size_t));	//array to hold sorted gnome index
			for (size_t i = 0; i < p; i++) {					//initialize index array from 1 to p(population size) in an ascending order
				idx[i] = i;
			}

			for (size_t i = 0; i<p; i++) {				//sort gnome indices according to score values using bubble sort 
				for (size_t j = i + 1; j<p; j++) {
					if (S[idx[i]] > S[idx[j]]) {
						std::swap(idx[i], idx[j]);		//float check : it was like this b(&idx[i], &idx[j]) but gave me error
					}
				}
			}

			//display best gnome
            //std::cout << "best fitness value: " << S[idx[0]] << std::endl;
			/*if (S[idx[0]] < 0) {				
				std::cout << "best gnome is " << std::endl;
				for (size_t i = 0; i < f; i++)
					std::cout << P[f * idx[0] + i] << ", ";
				std::cout << std::endl;
			}*/

			return idx; //use as sortSIdx in selection
		}


		//size_t* sortIndx(float* input, size_t size) {
		//	//sort indices of score in ascending order (fitness value)
		//	size_t *idx;
		//	idx = (size_t*)malloc(size * sizeof(size_t));
		//	for (size_t i = 0; i < size; i++)
		//		idx[i] = i;

		//	for (size_t i = 0; i<size; i++) {
		//		for (size_t j = i + 1; j<size; j++) {
		//			if (input[idx[i]] < input[idx[j]]) {
		//				std::swap(idx[i], idx[j]);				   //float check : it was like this b(&idx[i], &idx[j]) but gave me error
		//			}
		//		}
		//	}
		//	return idx; //use as sortSIdx in selection

		//}

		void generateNewP(unsigned int* newPop) {
			//std::memcpy(P, 0 , p * f *sizeof(unsigned int));			   //copy sb of gnome 'g' into bufferarray tempg_s	
			std::memcpy(P, newPop, p * f * sizeof(unsigned int));		   //copy sb of gnome 'g' into bufferarray tempg_s	
		}

		//============================== functions for fitness function  ===========================================================================
		//compute total mean M (1 X B) of all features (tP X B)
		void ttlMean(float* M, size_t tP, size_t B) {
			//std::cout<<"total number of pixels are "<<tP<<std::endl;
			for (int k = 0; k < tP; k++) {								  //total number of pixel in feature matrix
				for (size_t n = 0; n < B; n++) {							  // index of feature in ith gnome
					M[n] += F[k * B + n];
				}
			}
			for (size_t n = 0; n < B; n++)								 //take an avarage of above summation 
				M[n] = M[n] / (float)tP;
		}

		void dispalymean(float* M) {				//display mean
			std::cout << std::endl;
			std::cout << "Total mean of gnome 1 features are is " << std::endl;

			for (size_t i = 0; i < 1; i++) {
				for (size_t j = 0; j < f; j++) {
					size_t index = P[i*f + j];
					std::cout << "feature index " << index << "\t total mean" << M[index] << std::endl;
				}
			}
			std::cout << std::endl;
		}

		//Compute class means cM (p x nC x f) of all gnome features phe(tP x f)
		void classMean(float* cM, size_t tP, size_t nC, size_t B, std::vector<unsigned int> nPxInCls) {
			for (size_t c = 0; c < nC; c++) {									//index of class feature matrix responses
				float* tempcM = (float*)calloc(B, sizeof(float));			//tempcM holds classmean vector for current gnome 'i', class 'c'
				for (size_t k = 0; k < tP; k++) {							//total number of pixel in feature matrix
					if (T[k] == c + 1) {									//class numbers start from 1 not 0
						for (size_t n = 0; n < B; n++) {					//total number of features in a gnome
							tempcM[n] += F[k * B + n];					//add phe value for feature n of class 'c' in  ith gnome
						}
					}
				}
				for (size_t n = 0; n < B; n++)
					cM[c * B + n] = tempcM[n] / (float)nPxInCls[c];			//divide by number of pixels from class 'c'

			}

		}

		//display class mean
		void dispalyClassmean(float* cM, size_t nC) {
			std::cout << std::endl;
			std::cout << "class mean of gnome 1 with total classes " << nC << " is :" << std::endl;
			for (size_t i = 0; i < 1; i++) {
				for (size_t c = 0; c < nC; c++) {
					for (size_t j = 0; j < f; j++) {
						size_t index = P[i*f + j];

						std::cout << "class index: " << c << "\t feature index " << index << "\t  class mean " << cM[c * ub + index] << std::endl;
					}
				}
			}
			std::cout << std::endl;
		}

		//-----------------------------------------between and within class Scattering computation---------------------------------------------------------------
		//computation on CPU
		void cpu_computeSbSw(float* sb, float* sw, float* M, float* cM, size_t nC, size_t tP, std::vector<unsigned int> nPxInCls) {
			timer.start();
			computeSb(sb, M, cM, nC, nPxInCls);					//compute between class scatter on CPU
			const auto elapsed = timer.time_elapsed();
			std::cout << "Sb CPU time " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << "us" << std::endl;

			timer.start();
			computeSw(sw, cM, nC, tP);							//compute within class scatter on CPU
			const auto elapsed1 = timer.time_elapsed();
			std::cout << "Sw CPU time " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count() << "us" << std::endl;
		}

		//display between class scatter
		void displaySb(float* sb) {
			std::cout << "between scatter is  " << std::endl;
			for (size_t g = 0; g<1; g++) {
				std::cout << std::endl;
				for (size_t j = 0; j < f; j++) {								//total number of features in a gnome
					for (size_t k = 0; k < f; k++) {							//total number of features in a gnome	
						std::cout << sb[g * f * f + j * f + k] << "  ";
					}
					std::cout << std::endl;
				}
			}
			std::cout << std::endl;
		}

		//Compute between class scatter sb (p x f x f) of all gnome features phe(tP x f)
		void computeSb(float* sb, float* M, float* cM, size_t nC, std::vector<unsigned int> nPxInCls) {
			float tempsbval;
			size_t n1;
			size_t n2;
			size_t classIndx;										//class index in class mean matrix
			/*std::cout <<"population of computation of cpusb "<< std::endl;
			for (size_t i2 = 0; i2 < f; i2++) {
				std::cout << P[i2] << "\t";
			}*/

			for (size_t gnomeIndx = 0; gnomeIndx < p; gnomeIndx++) {
				for (size_t c = 0; c < nC; c++) {
					for (size_t i = 0; i < f; i++) {
						for (size_t j = 0; j < f; j++) {
							tempsbval = 0;
							classIndx = c * ub;
							n1 = P[gnomeIndx * f + i];						//actual feature index in original feature matrix 
							n2 = P[gnomeIndx * f + j];						//actual feature index in original feature matrix 
						//	std::cout << "i: " << i << " j: " <<j<< " n1: " << n1 << " n2:" << n2 << std::endl;
							tempsbval = ((cM[classIndx + n1] - M[n1]) *(cM[classIndx + n2] - M[n2]));							
							sb[gnomeIndx * f * f + i * f + j] += tempsbval * (float)nPxInCls[c];  // compute tempsb[j][k] element of class 'c' of gnome 'i'
						}
					}
				}
			}

		}

		//Compute within class scatter sw (p x f x f) of all gnome features phe(tP x f)
		void computeSw(float* sw, float* cM, size_t nC, size_t tP) {
			float tempswval;
			size_t n1;
			size_t n2;
			size_t cMclass;										//class index in class mean matrix
			size_t Pg;
			size_t swg;
			size_t pheg;
			for (size_t gnomeIndx = 0; gnomeIndx < p; gnomeIndx++) {
				Pg = gnomeIndx * f;
				swg = gnomeIndx * f * f;
				pheg = gnomeIndx * tP * f;;
				for (size_t c = 0; c < nC; c++) {
					cMclass = c * ub;

					for (size_t k = 0; k < tP; k++) {
						if (T[k] == (c + 1)) {
							for (size_t i = 0; i < f; i++) {
								for (size_t j = 0; j < f; j++) {
									n1 = P[Pg + i];						//actual feature index in original feature matrix 
									n2 = P[Pg + j];						//actual feature index in original feature matrix 

									tempswval = 0;
									tempswval = ((F[k * ub + n1] - cM[cMclass + n1]) * (F[k * ub + n2] - cM[cMclass + n2]));
									//tempswval = ((phe[gnomeIndx * tP * f + k * f + i] - cM[c * ub + P[gnomeIndx * f + i]]) * (phe[gnomeIndx * tP *f + k * f + j] - cM[c * ub + P[gnomeIndx * f + j]]));	
									sw[gnomeIndx * f * f + i * f + j] += tempswval;
								}
							}
						}
					}
				}

			}
		}
		//checking bands with all zeros and replacing duplicated bands in gnome but this function is only for initial population
		//void zerobandcheck(float* M, bool initial) {
		//	for (size_t g = 0; g < p; g++) {										// for each gnome			
		//		for (size_t i = 0; i < f; i++) {									//check each band (feature) index in that gnome
		//			while (M[P[g * f + i]] == 0) {									//if mean of band is zero then replace band index in population
		//				P[g * f + i] = rand() % ub + lb;
		//			}
		//		}
		//		//checking for duplicats in a gnome
		//		std::vector<unsigned int> gnome = getGnome(g);
		//		std::vector<unsigned int> gnomeunique;
		//		int flag = 0;							//flag will be set if gnome has duplicated band (feature) index
		//		std::sort(gnome.begin(), gnome.end());	// 1 1 2 2 3 3 3 4 4 5 5 6 7 
		//		std::unique_copy(gnome.begin(), gnome.end(), std::back_inserter(gnomeunique));		//keep only unique copies of indices and remove duplicate copies
		//		if (gnomeunique.size()< gnome.size()) {
		//			flag = 1;							//set flag for those if there are duplicated indices
		//												//std::cout<<"gnome:["<<g<<"] "<<"\t duplications are "<< (gnome.size() - gnomeunique.size())<<std::endl;
		//		}

		//		//adding extra random feature indices to unique copy of gnome to achive gnome size  = f
		//		if (gnomeunique.size() < f) {
		//			for (size_t k = gnomeunique.size(); k < f; k++) {
		//				unsigned int rnumber = rand() % ub + lb;
		//				//check if this randomaly generated number is already present in that gnome or not
		//				for (size_t j = 0; j < gnomeunique.size(); j++) {
		//					if (gnomeunique.at(j) == rnumber) {				//if new index is duplicated copy of any of previous gnome element replace it with another random number
		//						rnumber = rand() % ub + lb;
		//						j = 0;										//set j = 0 to start checking of duplication of feature index from the first element of gnome
		//					}
		//				}
		//				gnomeunique.push_back(rnumber);						//add feature index to gnomeunique 			
		//			}
		//		}
		//		std::copy(gnomeunique.begin(), gnomeunique.end(), P + g * f);
		//	}
		//}

		//checking bands with all zeros and replacing duplicated bands in gnome
		void zerobandcheck(float* M, bool initialPop) {
			size_t startgnome;
			if (initialPop) {
				startgnome = 0;    //for initial population check all gnomes 
			}
			else {
				startgnome = 1;    //for next generations start gnome check after elite children offset 
			}
			for (size_t g = startgnome; g < p; g++) {							// for each gnome except 		
				
				for (size_t i = 0; i < f; i++) {						//check each band (feature) index in that gnome
					while (M[P[g * f + i]] == 0) {						//if mean of band is zero then replace band index in population
						P[g * f + i] = rand() % ub + lb;
					}
				}
				//checking for duplicats in a gnome
				std::vector<unsigned int> gnome = getGnome(g);			//get current gnome g from population matrix P
				std::vector<unsigned int> gnomeunique;					//array to store only unique band indicies in a genome
				int flag = 0;											//flag will be set if gnome has duplicated band (feature) index
				std::sort(gnome.begin(), gnome.end());					//sort current gnome 
				std::unique_copy(gnome.begin(), gnome.end(), std::back_inserter(gnomeunique));		//remove duplicat copies of band indices and keep only unique in a gnome
				if (gnomeunique.size()< gnome.size()) {
					flag = 1;							//set flag for those if there are duplicated indices
					//std::cout<<"gnome:["<<g<<"] "<<"\t duplications are "<< (gnome.size() - gnomeunique.size())<<std::endl;
				}

				//adding extra random feature indices to unique copy of gnome to achive gnome size  = f
				if (gnomeunique.size() < f) {
					for (size_t k = gnomeunique.size(); k < f; k++) {
						unsigned int rnumber = rand() % ub + lb;
						//check if this randomaly generated number is already present in that gnome or not
						for (size_t j = 0; j < gnomeunique.size(); j++) {
							if (gnomeunique.at(j) == rnumber) {				//if new index is duplicated copy of any of previous gnome element replace it with another random number
								rnumber = rand() % ub + lb;					//generate random number between upper bound and lower bound (ub. lb)
								j = 0;										//set j = 0 to start checking of duplication of feature index from the first element of gnome
							}
						}
						gnomeunique.push_back(rnumber);						//add feature index to gnomeunique 			
					}
				}

				//diplay loop only if gnome has duplicated indices
				//if(flag ==1){
				//	std::cout<<"\n original gnome "<<g<<" are "<<std::endl;
				//	for(int k = 0; k < gnome.size(); k++)
				//		std::cout<<gnome[k]<<"\t";
				//	std::cout<<"\n unique results in cpp for gnome "<<g<<" are "<<std::endl;
				//	for(int k = 0; k < gnomeunique.size(); k++)
				//		std::cout<<gnomeunique[k]<<"\t";
				//}
				std::copy(gnomeunique.begin(), gnomeunique.end(), P + g * f);  //copy new gnome without any duplicate band index at current gnome location
			}
		}

		

		//gpu calling functions
		//gpu initialization (allocating space for all array on GPU)
		void gpuInitializationfrommain(float* cpuM, float* cpuCM, std::vector<unsigned int>cpu_nPxInCls, size_t tP, size_t nC) {
			//	call gpuInitialization(......) with all of the necessary parameters
			gpuIntialization(&gpuP, p, f, &gpuCM, cpuCM, nC, ub, &gpuM, cpuM, &gpu_nPxInCls, &gpuSb, &gpuSw, &gpuF, F, &gpuT, T, tP, &cpu_nPxInCls[0]);
			
		}

		//Computation of between class scatter and within class scatter in GPU
		void gpu_computeSbSw(float* cpuSb, float* cpuSw, size_t nC, size_t tP, cudaDeviceProp props, size_t gen, bool debug, std::ofstream& profilefile) {
			//calling function for SW and Sb computation and passing necessary arrays for computation
     // std::cout<<"gpu function calling"<<std::endl;
			gpucomputeSbSw(gpuP, P, p, f, gpuSb, cpuSb, gpuSw, cpuSw, gpuF, gpuT, gpuM, gpuCM, nC, tP, props, gen, gnrtn, ub, gpu_nPxInCls, profilefile);

			//display computed Sb and Sw if debug is set 
			if (debug) {
				std::cout << "From GA-GPU class: gpu results of Sb sn Sw" << std::endl;
				displayS(cpuSb, f);									//display Sb				
				displayS(cpuSw, f);									//display Sw
				std::cout << std::endl;
			}
		}

		//call function to free gpu pointers
		//free all gpu pointers
		void gpu_Destroy() {
			gpuDestroy(gpuP, gpuCM, gpuM, gpu_nPxInCls, gpuSb, gpuSw, gpuF, gpuT);
		}

		//Write a destructor here
		~ga_gpu() {

			if (F != NULL) 		std::free(F);		//not sure about this as it is only for 2nd constructor 
			if (T != NULL)		std::free(T);		//same as above
			if (P != NULL)		std::free(P);		//not sure about this as it is only for 2nd constructor 
			if (S != NULL)		std::free(S);		//same as above
														//if(i_guess!=NULL) 	std::free(i_guess);     //same as above
														//HANDLE_ERROR(cudaDeviceReset());

		}
	};

#endif
