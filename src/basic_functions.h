#include <stdio.h>


size_t* sortIndx(float* input, size_t size){
	//sort indices of score in ascending order (fitness value)
	size_t *idx; 
	idx = (size_t*) malloc (size * sizeof (size_t));
	for (size_t i = 0; i < size; i++)
		idx[i] = i;    
		 	
	for (size_t i=0; i<size; i++){
		for (size_t j=i+1; j<size; j++){
			if (input[idx[i]] < input[idx[j]]){
				std::swap (idx[i], idx[j]);    //float check : it was like this b(&idx[i], &idx[j]) but gave me error
			}
		}
	}				 
	return idx; //use as sortSIdx in selection
}

	
template<typename T>
void mtxMul(T* M3, T* M1, T* M2, size_t r1, size_t c1, size_t r2, size_t c2){
   //compute output matrix M3 of size row1 X column2 and data is column major		 
	for(size_t i = 0 ; i <r1; i++){								
		for(size_t j = 0; j< c2; j++){
			T temp = 0;
			for(size_t k = 0; k < c1 ; k++){					//column1 = row2 for matrix multiplication
				temp+= M1[i * c1  + k] * M2[k * c2 + j];			//compute an element of output matrix
			}       
			M3[i * c1 + j] = temp;								//copy an element to output matrix
		}
	}
}

template<typename T>
void mtxMultranspose(T* M3, T* M1, T* M2, size_t r1, size_t c1, size_t r2, size_t c2){
   //compute output matrix M3 of size row1 X column2 and data is column major
	for(size_t i = 0 ; i <r1; i++){								
		for(size_t j = 0; j< r2; j++){		
			T temp = 0;
			for(size_t k = 0; k < c1 ; k++){					//column1 = row2 for matrix multiplication
				temp+= M1[i * c1 + k] * M2[j * c2 + k];			//compute an element of output matrix
			}    
			M3[i * r1 + j] = temp;								//copy an element to output matrix		
		}	
	}		
}



template<typename T>
void mtxOutputFile(std::string filename, T* M, size_t rows, size_t cols){
	std::cout<<"Outputting "<<rows<<" x "<<cols<<" matrix to file: "<<filename<<std::endl;
	ofstream outfile(filename.c_str());										//open a file for writing
	//output the matrix
	for(size_t r = 0; r < rows; r++){
		for(size_t c = 0; c < cols; c++){
			outfile<<M[r * cols + c];
			if(c != cols-1)
				outfile<<",";
		}
		if(r != rows - 1)
			outfile<<std::endl;
	}
	outfile.close();
	//outfile<<"This is a test."<<std::endl;
}

		//display within class scatter
template<typename T>
void displayS(T* sw, size_t f){

	for(size_t g = 0; g<1; g++){
		std::cout<<std::endl;
		for(size_t j = 0; j < f; j++){										//total number of features in a gnome
				for(size_t k = 0; k < f; k++){						    //total number of features in a gnome	
					std::cout<<sw[g*f*f + j*f + k]<<"  ";
		}
	    std::cout<<std::endl;
		}
	}
 	std::cout<<std::endl;
}

//sort eigenvalues from lapacke results
size_t* sortEigenVectorIndx(float* eigenvalue, size_t N){
		//sort indices of score in ascending order (fitness value)
		size_t *idx = (size_t*) malloc (N * sizeof (size_t));
		for (size_t i = 0; i < N; i++)
			idx[i] = i;    
		 	
		for (size_t i=0; i<N; i++){
			for (size_t j=i+1; j<N; j++){
				if (eigenvalue[idx[i]] > eigenvalue[idx[j]]){
					std::swap (idx[i], idx[j]);    //float check : it was like this b(&idx[i], &idx[j]) but gave me error
				}
			}
		}

		std::cout<<"best                                                                                                                                                                                                                                                                      eigenvalue index: "<<eigenvalue[idx[0]]<<std::endl;
 
		return idx; //use as sortSIdx in selection

}
