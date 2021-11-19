#include <iostream>

//stim libraries
#include <stim/envi/envi.h>
#include <stim/image/image.h>
#include <stim/ui/progressbar.h>
#include <stim/parser/filename.h>
#include <stim/parser/table.h>
#include <stim/parser/arguments.h>
//input arguments
stim::arglist args;
#include <fstream>
#include <thread>
#include <random>
#include <vector>
#include <math.h>
#include <limits>

#define NOMINMAX



//GA
#include "ga_gpu.h"
#include "enviload.h"


//envi input file and associated parameters
stim::envi E;										//ENVI binary file object
unsigned int B;										//shortcuts storing the spatial and spectral size of the ENVI image
//mask and class information used for training
//std::vector< stim::image<unsigned char> > C;		//2D array used to access each mask C[m][p], where m = mask# and p = pixel#
std::vector<unsigned int> nP;						//array holds the number of pixels in each mask: nP[m] is the number of pixels in mask m
size_t nC = 0;										//number of classes
size_t tP = 0;										//total number of pixels in all masks: tP = nP[0] + nP[1] + ... + nP[nC]
float* fea;

//ga_gpu class object
ga_gpu ga;
bool debug;
bool binaryClass;
int binClassOne;

//creating struct to pass to thread functions as it limits number of arguments to 3
typedef struct {
	float* S;
	float* Sb;
	float* Sw;
	float* lda;
}gnome;
gnome gnom;

//computing matrix determinant using LU decomposition
template<typename T>
T mtxdeterminant(T* M, size_t r, size_t c) {
	//----DETERMINANT using LU decomposition using LAPACK
	/* DGETRF computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges.
	   The factorization has the form
		A = P * L * U
	   where P is a permutation matrix, L is lower triangular with unit diagonal elements (lower trapezoidal if m > n), and U is upper  triangular (upper trapezoidal if m < n).*/
	int* ipiv = (int*)malloc(sizeof(int) * r);
	memset(ipiv, 0, r * sizeof(int));
	LAPACKE_sgetrf(LAPACK_COL_MAJOR, (int)r, (int)c, M, (int)r, ipiv);

	//determinant of U
	T product = 1;
	for (size_t i = 0; i < r; i++) {
		for (size_t j = 0; j < r; j++) {
			if (i == j) {
				product = product * M[i * r + j];
			}
		}
	}

	//determinant of P
	int j;
	double detp = 1.;
	for (j = 0; j < r; j++) {
		if (j + 1 != ipiv[j]) {
			// j+1 : following feedback of ead : ipiv is from Fortran, hence starts at 1.
			// hey ! This is a transpose !
			detp = -detp;
		}
	}
	T detSw = product * detp * 1; //det(U)*det(P)*det(L)
	if (ipiv != NULL) std::free(ipiv);
	return detSw;

}

void gpuComputeEignS( size_t g, size_t fea){	
	//eigen value computation will return  r = (nC-1) eigen vectors so new projected data will have dimension of r rather than f
	//  std::thread::id this_id = std::this_thread::get_id();
	//	std::cout<<"thread id is "<< this_id<<std::endl;  
	size_t f = fea;
	//std::thread::id g = std::this_thread::get_id();	
	float* LeftEigVectors_a = (float*) malloc(f * f * sizeof(float));			
	float* gSw_a = (float*) malloc(f * f * sizeof(float));						//copy of between class scatter		
	std::memcpy(gSw_a, &gnom.Sw[g * f * f], f * f *sizeof(float));
	if(debug){
		std::cout<<"From Eigen function: Sb and Sw "<<std::endl;
		displayS(gSw_a, f);									//display Sb				
		displayS(&gnom.Sb[g * f * f], f);									//display Sw
		std::cout<<std::endl;
	}	
	
	std::vector<unsigned int> features = ga.getGnome(g);
	std::vector<unsigned int> featuresunique;
	int flag = 0;
	std::sort(features.begin(), features.end()); // 1 1 2 2 3 3 3 4 4 5 5 6 7 
	std::unique_copy(features.begin(), features.end(), std::back_inserter(featuresunique));
	if(featuresunique.size()< features.size()){
		f = featuresunique.size();
	}
				
	size_t r = nC-1;                                         //LDA projected dimension (limited to number of classes - 1  by rank)
	if(r > f){
		r = f;
	}

	int info;
	float* EigenvaluesI_a = (float*)malloc(f * sizeof(float));
	float* Eigenvalues_a = (float*)malloc(f * sizeof(float));
	int *IPIV = (int*) malloc(sizeof(int) * f);
	//computing inverse of matrix Sw
	memset(IPIV, 0, f * sizeof(int));
	LAPACKE_sgetrf(LAPACK_COL_MAJOR, (int)f, (int)f, gSw_a, (int)f, IPIV);		
	// DGETRI computes the inverse of a matrix using the LU factorization computed by DGETRF.
	LAPACKE_sgetri(LAPACK_COL_MAJOR, (int)f, gSw_a, (int)f, IPIV);
	
	float* gSbSw_a  = (float*)calloc(f * f, sizeof(float));
	//mtxMul(gSbSw_a, gSw_a, &gnom.Sb[g * f * f * sizeof(float)], f, f, f,f);
	mtxMul(gSbSw_a, gSw_a, &gnom.Sb[g * f * f], f, f, f,f);
	if(debug){
		std::cout<<"From Eigen function: inverse of sw and ratio of sb and sw (Sb/Sw)";
		displayS(gSw_a, f);			//display inverse of Sw (1/Sw)				
		displayS(gSbSw_a, f);		//display ratio of Sb and Sw (Sb/Sw)
	}
   
   //compute left eigenvectors for current gnome from ratio of between class scatter and within class scatter:  Sb/Sw 
	info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'V', 'N', (int)f, gSbSw_a, (int)f, Eigenvalues_a, EigenvaluesI_a, LeftEigVectors_a, (int)f, 0, (int)f);
	//sort eignevalue indices in descending order							
	size_t* sortedindx = sortIndx(Eigenvalues_a, f);
	//displayS(LeftEigVectors_a, f);					//display Eignevectors (Note these are -1 * matlab eigenvectors does not change fitness score results but keep in mind while projecting data on it)		
	//sorting left eigenvectors  (building forward transformation matrix As)
	for (size_t rowE = 0; rowE < r; rowE++){
		for (size_t colE = 0; colE < f; colE++){
			size_t ind1 = g * r * f + rowE * f + colE;			
			//size_t ind1 =  rowE * f + colE;
			size_t ind2 = sortedindx[rowE] * f + colE;				//eigenvector as row vector			
			gnom.lda[ind1] = LeftEigVectors_a[ind2];
		}
	}
		
	if(debug){
		std::cout<<"Eigenvalues are"<<std::endl;
		for(size_t n = 0 ; n < f; n ++){
			std::cout << Eigenvalues_a[n] << ", " ;
		}
		std::cout<< std::endl;
		std::cout<<"From Eigen function: Eignevector"<<std::endl;
	  				
		std::cout<<"LDA basis is "<<std::endl;
		std::cout << "r is " << r << std::endl;
		for(size_t l = 0 ; l < r; l++){
			for(size_t n = 0 ; n < f; n ++){
				std::cout << gnom.lda[g * l * f + l * f + n] << ", " ;
			}
			std::cout<<std::endl;
		}
			
	}
	//Extract only r eigne vectors as a LDA projection basis
	float* tempgSb = (float*)calloc(r * f, sizeof(float));
	//mtxMul(tempgSb, &gnom.lda[g * r * f * sizeof(float)], &gnom.Sb[g * f * f * sizeof(float)], r, f, f,f);
	//mtxMul(tempgSb, &lda[g * r * f ], gSb, r, f, f,f);
	mtxMul(tempgSb, &gnom.lda[g * r * f], &gnom.Sb[g * f * f], r, f, f,f);
	float* nSb = (float*)calloc(r * r, sizeof(float));
	mtxMultranspose(nSb, tempgSb, &gnom.lda[g * r * f], r, f, r, f);
	
	float* tempgSw = (float*)calloc(r * f, sizeof(float));
	//mtxMul(tempgSw, &gnom.lda[g * r * f * sizeof(float)], &gnom.Sw[g * f * f * sizeof(float)], r, f, f,f);
	mtxMul(tempgSw, &gnom.lda[g * r * f], &gnom.Sw[g * f * f], r, f, f,f);
	float* nSw = (float*)calloc(r * r, sizeof(float));
	mtxMultranspose(nSw, tempgSw, &gnom.lda[g * r * f], r, f, r, f);
		
	//determinant of Sb and Sw
	float detSw = mtxdeterminant(nSw, r, r);
	float detSb = mtxdeterminant(nSb, r, r);

	if (debug) {
		std::cout << "From Eigen function: projected Sb sn Sw" << std::endl;
		displayS(nSb, r);									//display Sb				
		displayS(nSw, r);									//display Sw
		std::cout << std::endl;
		std::cout <<"det(Sb)= "<< detSb<< std::endl;
		std::cout << "det(Sw)= " << detSw << std::endl;
	}
	
	//fisher's ratio from ratio of projected sb and sw
	float fisherRatio = detSb /detSw;
	gnom.S[g] = 1/fisherRatio;
	if (debug) {
		std::cout<<"Score["<<g<<"]: "<< gnom.S[g]<<std::endl;
	
		std::cout << "best gnoem is " << std::endl;
		for (size_t i = 0; i < f; i++)
			std::cout << ga.P[ga.f * g + i] << ", ";
		std::cout << std::endl;
	}

		
	
	if(IPIV!= NULL) std::free(IPIV);	
	if(gSw_a!= NULL) std::free(gSw_a);
	if(gSbSw_a!= NULL) std::free(gSbSw_a);
	if(Eigenvalues_a!= NULL) std::free(Eigenvalues_a);
	if(EigenvaluesI_a!= NULL) std::free(EigenvaluesI_a);
	if(tempgSb!= NULL) std::free(tempgSb);
	if(tempgSw!= NULL) std::free(tempgSw);
	
}	

	
void fitnessFunction( float* sb, float* sw, float* lda, float* M, float* cM, size_t f, cudaDeviceProp props, size_t gen, std::ofstream& profilefile){  
		
		size_t tP = 0;																	//total number of pixels
		std::for_each(nP.begin(), nP.end(), [&] (size_t n) {
			tP += n;
		});		
 		size_t nC = nP.size();													//total number of classes
		
		//--------------Compute between class scatter 		
	
		ga.gpu_computeSbSw(sb, sw, nC, tP, props, gen, debug, profilefile);

		if(debug){
			std::cout<<"From fitness function: gpu results of Sb sn Sw"<<std::endl;
			displayS(sb, ga.f);									//display Sb				
			displayS(sw, ga.f);									//display Sw
			std::cout<<std::endl;
		}				
		
	// -----------------------   Linear discriminant Analysis   --------------------------------------
		gnom.S = ga.S;
		gnom.Sw = sw;
		gnom.Sb = sb;
		gnom.lda = lda;

		//calling function without using threads
		for (size_t i = 0; i<ga.p; i++){			
			//calling function for eigencomputation
			gpuComputeEignS(i, f);
		}	
	
		const auto elapsed1 = timer.time_elapsed();
		if(gen > ga.gnrtn - 2){
			std::cout << "gpu_eigen time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count() << "us" << std::endl;
      profilefile<< "gpu_eigen time "<<std::chrono::duration_cast<std::chrono::microseconds>(elapsed1).count() << "us" << std::endl;
		}

}//end of fitness function

void binaryclassifier(int classnum){
	unsigned int* target = (unsigned int*) calloc(tP, sizeof(unsigned int));
	memcpy(target, ga.T, tP * sizeof(unsigned int));
	for(int i = 0 ; i < tP; i++){
		if(target[i]==classnum){
			ga.T[i] = 1;
			
		}else
			ga.T[i] = 0;
	}
}



void advertisement() {
	std::cout << std::endl;
	std::cout << "=========================================================================" << std::endl;
	std::cout << "Thank you for using the GA-GPU features selection for spectroscopic image!" << std::endl;
	std::cout << "=========================================================================" << std::endl << std::endl;
}

int main(int argc, char* argv[]){
	
//Add the argument options and set some of the default parameters
	args.add("help", "print this help");
	args.section("Genetic Algorithm");
	args.add("features", "select features selection algorithm parameters","10", "number of features to be selected");
	args.add("classes", "image masks used to specify classes", "", "class1.bmp class2.bmp class3.bmp");
	args.add("population", "total number of feature subsets in puplation matrix", "1000");
	args.add("generations", "number of generationsr", "50");
	args.add("initial_guess", "initial guess of featues", "");
	args.add("debug", "display intermediate data for debugging");
	args.add("binary", "Calculate features based on class1 vs. all other classes", "");
	args.add("trim", "this gives wavenumber to use in trim option of siproc which trims all bands from envi file except gagpu selected bands");

	args.parse(argc,argv);                  //parse the command line arguments

//Print the help text if set
	if(args["help"].is_set()){				//display the help text if requested
		advertisement();
		std::cout<<std::endl<<"usage: ga-gpu input_ENVI output.txt --classes class1.bmp class2.bmp ... --option [A B C ...]"<<std::endl;
		std::cout<<std::endl<<std::endl;
		std::cout<<args.str()<<std::endl;
		exit(1);
	}
	if (args.nargs() < 2) {										//if the user doesn't provide input and output files
		std::cout << "ERROR: GA-GPU requires an input (ENVI) file and an output (features, text) file." << std::endl;
		return 1;
	}
	if (args["classes"].nargs() < 2) {							//if the user doesn't specify at least two class images
		std::cout << "ERROR: GA-GPU requires at least two class images to be specified using the --classes option" << std::endl;
		return 1;
	}

	std::string outfile = args.arg(1);							//outfile is text file where bnad index, LDA-basis, wavelength and if --trim option is set then trim wavelengths are set respectively 
  std::string profile_file = "profile_" + outfile ;
  std::ofstream profilefile(profile_file.c_str(), std::ios::out);			//open outfstream for outfile
	
	time_t t_start = time(NULL);								//start a timer for file reading 	
	E.open(args.arg(0), std::string(args.arg(0)) + ".hdr");		//open header file 		
	size_t X = E.header.samples;								//total number of pixels in X dimension
	size_t Y = E.header.lines;									//total number of pixels in Y dimension
	B = (unsigned int)E.header.bands;							//total number of bands (features)
	std::vector<double> wavelengths = E.header.wavelength;		//wavelengths of each band

	if(E.header.interleave != stim::envi_header::BIP){			//this code can only load bip files and hence check that in header file
		std::cout<<"this code works for only bip files. please convert file to bip file"<<std::endl;
		exit(1);												//if file is not bip file exit code execution
	}	
	
///--------------------------Load features--------------------------------------------- 
	nP = ga_load_class_images(argc, args, &nC, &tP);			//load supervised class images
	ga.F = load_features( nC, tP, B, E, nP);					//generate the feature matrix
	ga.T = ga_load_responses(tP, nC, nP);						//load the responses for RF training
	E.close();													//close the hyperspectral file
	time_t t_end = time(NULL);
	std::cout<<"Total time: "<<t_end - t_start<<" s"<<std::endl;	
	
///--------------------------Genetic algorith configurations with defult paramets and from argument values---------------------
	ga.f = args["features"].as_int(0);					//number of features to be selected by user default value is 10
	ga.p = args["population"].as_int(0);				//population size to be selected by user default value is 1000
	ga.gnrtn = args["generations"].as_int(0);			//number of generations to be selected by user default value is 50
	if(args["binary"]) {						//set this option when features are to be selected as binary clas features (class vs stroma) 
		binClassOne = args["binary"].as_int(0);			//sel class number here, if 2 then features are selected for (class-2 vs stroma) 
		//feture selection for class selected by user with user arguments (make it binary class data by making chosen class label as 1 and al other class labels 0 from multiclass data )
		//to select feature for all classes in joint class data using binary class system need to write a script with loop covering all classes
		binaryclassifier(binClassOne);
	}  ///not fully implemented yet

	ga.ub = B;						//upper bound is number of bands (i.e. size of z dimension)  Note: for this particular application and way code is written lower bound is 0 and upper bound is size of z dimension
	ga.uniformRate = 0.5;			//uniform rate is used in crossover
	ga.mutationRate = 0.5f;		//in percentage for mutation operation on gnome
	ga.tournamentSize = 5;			//for crossover best parents are selected from tournament of gnomes  
	ga.elitism = true;				// if it is true then best gnome of current generation is passed to next generation
	//initial guess of population
	ga.i_guess = (unsigned int*) calloc(ga.f, sizeof(unsigned int));		
	debug = args["debug"];				

//==================Generate intial population =================================================
	std::vector<unsigned int> i_guess(ga.f);
	for (size_t b = 0; b < ga.f; b++)     //generate default initial guess 
		i_guess[b] = rand() % B + 0;

	if (args["initial_guess"].is_set()) {//if the user specifies the --initialguess option & provides feature indices as initial guess
		size_t nf = args["initial_guess"].nargs();				//get the number of arguments after initial_guess
		if (nf == 1 || nf == ga.f) {								//check if file with initial guessed indices is given or direct indices are given as argument
			if (nf == 1) {		 //if initial guessed feature indices are given in file										
				std::ifstream in;		//open the file containing the baseline points
				in.open(args["initial_guess"].as_string().c_str());
				if (in.is_open()){	//if file is present and can be opened then read it 
					unsigned int b_ind;
					while (in >> b_ind)  //get each band index and push it into the vector
						i_guess.push_back(b_ind);					
				}
				else 
					std::cout << "cannot open file of initial_guess indices" << std::endl;
			}
			else if (nf == ga.f) {	 //if direct indices are given as argument
				for (size_t b = 0; b < nf; b++)								//for each band given by the user
					i_guess[b] = args["initial_guess"].as_int(b);			//store that band in the i_guess array
			}
		}		
	}

	ga.initialize_population(i_guess, debug);      //initialize first population set for first generation, user can pass manually preferred features from command line 
	//display_gnome(0);	

//------------------Calculate class means and total mean of features----------------------------
	float* M = (float*)calloc( B , sizeof(float));			//total mean of entire feature martix for all features (bands B)
	ga.ttlMean(M, tP, B);									//calculate total mean, ga.F is entire feature matrix, M is mean for all bands B(features)
	if(debug) ga.dispalymean(M);							//if option --debug is used display all bands mean
	
	//display band index of bands with mean zero, this indicates that band has all zero values 
	std::cout<<"Display features indices with zero mean "<<std::endl;
	for(unsigned int i = 0; i < B; i++){
		if(M[i]== 0)
			std::cout<<"\t"<<i;
	}
	std::cout<<std::endl;
//	std::cout << "pixel target is " << ga.T[0] << "  " << ga.T[1] << "  " << ga.T[tP - 2] << "  " << ga.T[tP - 1]<<std::endl;
	float* cM = (float*)calloc(nC * B , sizeof(float));		//cM is nC X B matrix with each row as mean of all samples in one class for all features (bands B)
	ga.classMean(cM, tP, nC, B, nP);						//calculate class mean, ga.F is entire feature matrix, M is mean for all bands B(features)
	if(debug) ga.dispalyClassmean(cM, nC);

//------------------------------------GPU init----------------------------------------------------
	//checking for cuda device 
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	if(count < 1){
		std::cout<<"no cuda device is available"<<std::endl;
		return 1;
	}
	cudaDeviceProp props;
	HANDLE_ERROR(cudaGetDeviceProperties(&props, 0));
	ga.gpuInitializationfrommain(M, cM, nP, tP, nC);


//============================= GA evolution by generations ====================================================
	std::vector<unsigned int> bestgnome;												//holds best gnome after each generation evaluation 
	size_t bestG_Indx;																	//This gives index of best gnome in the current population to get best gnome and its fitness value
	unsigned int* newPop = (unsigned int*) calloc(ga.p * ga.f, sizeof(unsigned int));   //temprory storage of new population
	double* best_S = (double*) calloc (ga.gnrtn, sizeof(double));						//stores fitness value of best gnome at each iteration
	float* lda = (float*) calloc (ga.p * (nC-1) * ga.f, sizeof(float));					//stores LDA basis for each gnome so that we can have best gnome's LDA basis 																	
	float* sb = (float*) calloc( ga.p * ga.f * ga.f , sizeof(float)) ;					//3d matrix for between class scatter (each 2d matrix between class scatter for one gnome) 
	float* sw = (float*) calloc( ga.p * ga.f * ga.f , sizeof(float)) ;					//3d matrix for within class scatter (each 2d matrix within class scatter for one gnome) 
	ga.zerobandcheck(M, true);										//checking bands with all zeros and duplicated bands in a gnome replacing them with other bands avoiding duplication and zero mean 
	ga.zerobandcheck(M, true);										//Repeating zeroband cheack as some of these bands are not replaced in previous run and gave random results
	time_t gpu_timestart = time(NULL);					//start a timer for total evoluation 

	for (size_t gen = 0; gen < ga.gnrtn; gen++){		//for each generation find fitness value of all gnomes in population matrix and generate population for next generation
		//std::cout<<"Generation: "<<gen<<std::endl;
		fitnessFunction(sb, sw, lda, M , cM, ga.f, props, gen, profilefile);		//Evaluate phe(feature matrix for current population) for fitness of all gnomes in current population 
		timer.start();												//start timer for new population generation
		bestG_Indx = ga.evolvePopulation(newPop, M, debug);			//evolve population to generate new generation population
		const auto pop_generation = timer.time_elapsed(); 			// end timer for new population generation
		if(gen >ga.gnrtn -2){
			std::cout << "population evolution time "<<std::chrono::duration_cast<std::chrono::microseconds>(pop_generation).count() << "us" << std::endl;
  			profilefile<<"population evolution time "<<std::chrono::duration_cast<std::chrono::microseconds>(pop_generation).count() << "us" <<std::endl;
		}
						
		best_S[gen] = ga.S[bestG_Indx];								//score of best gnome in current generation 
		bestgnome = ga.getGnome(bestG_Indx);						//Best gnome of current populaation																				
		ga.generateNewP(newPop);									//replace current population with new populaiton in the ga classs object
		ga.zerobandcheck(M, false);										//checking bands with all zeros and duplicated bands in a gnome replacing them with other bands avoiding duplication and zero mean 
		ga.zerobandcheck(M, false);										//Repeating zeroband cheack as some of these bands are not replaced in previous run and gave random results
	}//end generation	  
	
	time_t gpu_timeend = time(NULL);					//end a timer for total evoluation 
	std::cout<<"Total gpu time: "<<gpu_timeend - gpu_timestart<<" s"<<std::endl;
 	profilefile<<"Total gpu time: "<<gpu_timeend - gpu_timestart<<" s"<<std::endl;

//================================ Results of GA ===============================================================
	std::cout<<"best gnome's fitness value is "<<best_S[ga.gnrtn-1]<<std::endl;	
	std::cout<<"best gnome is:	";
	for(size_t i = 0; i < ga.f; i++){
		std::cout<<" "<<(bestgnome.at(i));
   	}
	std::cout<<std::endl;	

	//create a text file to store the LDA stats (features subset and  LDA-basis)
	////format of CSV file is: 1st row - band index, 2nd LDA basis depending on number of classes, 3rd - wavenumber corresponding to band index and it --trim is selected then trim wavnumbersare also given
	std::ofstream csv(outfile.c_str(), std::ios::out);			//open outfstream for outfile
	size_t ldaindx = bestG_Indx * (nC-1) * ga.f ;				//Compute LDA basis index of best gnome
	

 //fitness values of best gnome is 
	csv<<"fitness value for best gnome:  "<<best_S[ga.gnrtn-1]<<std::endl;									//output fitness value of best gnome in last generation
	/*//output gnome i.e. band index of selected featurs
	csv<<(bestgnome.at(0));									//output feature subset
	for(size_t i = 1; i < ga.f; i++)
		csv<<","<<(bestgnome.at(i));
	csv<<std::endl;*/

	//output actual wavelenths corresponding to those band indices 
	csv << "selected wavenumbers:"<<std::endl;
	csv << (wavelengths[bestgnome.at(0)]);
	for (size_t i = 1; i < ga.f; i++)
		csv << ", " << (wavelengths[bestgnome.at(i)]);
	csv << std::endl;

	//output LDA basis of size r X f, r is nC - 1 as LDA projection is rank limited by number of classes - 1
	csv << "LDA basis:" << std::endl;
	for (size_t i = 0; i < nC - 1; i++) {
		csv << lda[ldaindx + i * ga.f];
		for (size_t j = 1; j < ga.f; j++) {
			csv << ", " << lda[ldaindx + i * ga.f + j];
		}
		csv << std::endl;
	}


	if (args["trim"].is_set()) {
		csv << "trim info" << std::endl;
		std::sort(bestgnome.begin(), bestgnome.end());			//sort features index in best gnome 

		std::vector<unsigned int> trimindex(ga.f);				//create a vector to store temprory trim index bounds 
		std::vector<unsigned int> finaltrim_ind;				//create a vector to store final trim index bounds 
		std::vector<unsigned int> trim_wv;						//create a vector to store final trim wavelength bounds 

		//trim index 
		trimindex.push_back(1);									//1st trimming band index is 1
		for (size_t i = 0; i < ga.f; i++) {						// for each feature find its bound indexes 
			trimindex[i * 2] = bestgnome.at(i) - 1;		
			trimindex[i * 2 + 1] = bestgnome.at(i) + 1;			
		}
		trimindex.push_back(B);								    //last bound index is B 		

		//organize trim index
		int k = 0;
		for (size_t i = 0; i < ga.f + 1; i++) {						// find valid pair of trim indices bound excluding adjacent trim indices
			if (trimindex[2 * i] < trimindex[2 * i + 1]) {
				finaltrim_ind.push_back(trimindex[2 * i]);			//this is left bound
				finaltrim_ind.push_back(trimindex[2 * i + 1]);
				k = k + 2;
			}
		}
		//add duplicated trim indices as single index to final trim index
		for (size_t i = 0; i < ga.f + 1; i++) {						//check each pair of trim indices for duplications	
			if (trimindex[2 * i] == trimindex[2 * i + 1]) {			// (duplication caused due to adjacent features)
				finaltrim_ind.push_back(trimindex[2 * i]);			// remove duplicated trim indices replace by one
				k = k + 1;
			}
		}


		////output actual wavelenths corresponding to those trim indices 
		////these wavenumber are grouped in pairs, check each pair, if duplicated numbers are there in pair delete one and keep other as band to trim, if 2nd wavnumber is smaller than 1st in a pair ignore that pair
		////e.g [1, 228, 230, 230, 232, 350,352, 351, 353, 1200] pairas [1(start)-228,230-230, 232-350, 352-351, 353-1200(end)], trimming wavenumbers are [1-228, 230, 233-350, 353-1200]
		
		
		csv << (wavelengths[finaltrim_ind.at(0)]);
		for (size_t i = 1; i < ga.f * + 2 ; i++)
			csv << "," << (wavelengths[finaltrim_ind.at(i)]);
		csv << std::endl;
	} //end trim option


  //free gpu pointers
  ga.gpu_Destroy();

	//free pointers
	std::free(sb);
	std::free(sw);
	std::free(M);
	std::free(cM);
	std::free(best_S);
	std::free(lda);
	std::free(newPop);	
	
}//end main


