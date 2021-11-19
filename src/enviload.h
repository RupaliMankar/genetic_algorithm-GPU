#include <iostream>
#include <fstream>
#include <thread>
#include <random>
#include <vector>
//#include <algorithm>

#define NOMINMAX

//stim libraries
#include <stim/envi/envi.h>
#include <stim/image/image.h>
#include <stim/parser/arguments.h>
#include <stim/ui/progressbar.h>
#include <stim/parser/filename.h>
//#include <stim/visualization/colormap.h>
#include <stim/parser/table.h>

std::vector< stim::image<unsigned char> > C;		//2D array used to access each mask C[m][p], where m = mask# and p = pixel#
//loads spectral features into a feature matrix based on a set of class images (or masks)
float* load_features(size_t nC, size_t tP, size_t B, stim::envi E, std::vector< unsigned int > nP){
	float progress = 0;									//initialize the progress bar variable
	unsigned long long bytes_fmat = sizeof(float) * tP * B;							//calculate the number of bytes in the feature matrix
	std::cout<<"totalnumber of samples "<<tP<<std::endl;
	std::cout<<"Allocating space for the feature matrix: "<<tP<<" x "<<B<<" = "<<(float)bytes_fmat/(float)1048576<<"MB"<<std::endl;
	float* F = (float*) malloc(bytes_fmat);								//allocate space for the sifted matrix
	std::cout<<"Loading Training Data ("<<nC<<" classes)"<<std::endl;
	//load all of the training spectra into an array
	unsigned long long F_idx = 0;												//initialize the matrix index to 0
	//unsigned long long R_idx = 0;
	for(unsigned c = 0; c < nC; c++){											//for each class image
		std::cout<<"\tSifting class "<<c+1<<" = "<<nP[c]<<" pixels..."<<std::endl;
	//	std::thread t1 = std::thread(progress_thread_envi, &E);					//start the progress bar thread		
		E.sift((void*)&F[F_idx], C[c].data(), true);							//sift that class into the matrix at the proper location
		F_idx += nP[c] * B;
		progress = (float)(c+1) / (float)nC * 100;
	//	t1.join();
	}
	
		return F;
}

/// Load responses for a Random Forest Classifier
unsigned int*  ga_load_responses(size_t tP, size_t nC, std::vector< unsigned int > nP){
	unsigned int* T = (unsigned int*)malloc(tP*sizeof(unsigned int));						//generate an OpenCV vector of responses
	size_t R_idx = 0;														//index into the response array
	for(size_t c = 0; c < nC; c++){										//for each class image
		for(unsigned long long l = 0; l < nP[c]; l++){						//assign a response for all pixels of class c loaded in the training matrix
			T[R_idx + l] = (unsigned int)c+1;
		}
		R_idx += nP[c];														//increment the response vector index
	}
	return T;
}


//loads the necessary data for training a random forest classifier
std::vector< unsigned int > ga_load_class_images(int argc, stim::arglist args, size_t* nC, size_t* tP){
	if(args["classes"].nargs() < 2){									//if fewer than two classes are specified, there's a problem
		std::cout<<"ERROR: training requires at least two class masks"<<std::endl;
		exit(1);
	}
	std::vector< unsigned int > nP;
	size_t num_images = args["classes"].nargs();						//count the number of class images
	//size_t num_images = args["rf"].nargs();						//count the number of class images
	//std::vector<std::string> filenames(num_images);				//initialize an array of file names to store the names of the images
	std::string filename;										//allocate space to store the filename for an image
	for(size_t c = 0; c < num_images; c++){						//for each image
		filename = args["classes"].as_string(c);;						//get the class image file name			
		stim::image<unsigned char> image(filename);				//load the image
		//push_training_image(image.channel(0), nC, tP, nP);					//push channel zero (all class images are assumed to be single channel)
		C.push_back(image.channel(0));
		unsigned int npixels = (unsigned int)image.channel(0).nnz();
		nP.push_back(npixels);									//push the number of pixels onto the pixel array
		*tP += npixels;											//add to the running total of pixels
		*nC = *nC + 1;
	}

	return nP;
}

void display_PixelfeatureNclass(float* F, unsigned int* T, size_t B, size_t Idx){
	//display code for debug, displaying Idx th pixel from feature matrix F with all features B 
	std::cout<<"class of pixel["<<Idx<<"]" <<"is: "<<T[Idx]<<std::endl;
	std::cout<<"feature["<<Idx<<"] is: "<<std::endl;
	for (size_t i = 0; i< B; i++)
		std::cout<<" "<<F[Idx * B + i];		
}


void display_args(int argc, stim::arglist args){
	std::cout<<"number of arguments "<<argc<<std::endl;
	std::cout<<"arg 0  "<<args.arg(0)<<std::endl;
	std::cout<<"arg 1  "<<args.arg(1)<<std::endl;
}

void display_dataSize(size_t X, size_t Y, size_t B){
	std::cout<<"number of samples "<<X*Y<<std::endl;
	std::cout<<"number of bands "<<B<<std::endl;

}

void display_phe(float* phe, unsigned int* P, size_t p,size_t f, size_t i, size_t j){
	//display code for debug, displaying jth pixel from new feature matrix which is created for gnome i 
	std::cout<<"phe["<<i<<"]["<<j<<"]"<<std::endl;	
	for(unsigned int n = 0; n < f; n++){
		std::cout<<P[i * f + n]; //spectral feature indices from gnome i of current population
		std::cout<<"    "<<phe[i* (p * f) +j * f + n]<<std::endl; //display 100th pixel value corresponding to feature indices in the gnome
		
	}
}


void display_gnome(unsigned int* P,size_t f,size_t gIdx){
	//display code for debug, displaying gnome gIdx of current population, gnome is subset of feature indices
	for (size_t i = 0; i< f; i++)
		std::cout<<" "<<P[gIdx * f + i];	
}

