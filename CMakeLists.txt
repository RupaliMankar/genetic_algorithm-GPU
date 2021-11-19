#Specify the version being used aswell as the language
cmake_minimum_required(VERSION 3.16)

#Name your project here
project(genetic-gpu)

#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")



#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
endif ( MSVC )




#find packages-----------------------------------
find_package(Threads)

#find the X11 package
find_package(X11)

#find the STIM library
find_package(STIM)

#find CUDA, mostly for LA stuff using cuBLAS
find_package(CUDA REQUIRED)

#find Boost for Unix-based file lists
if( CMAKE_COMPILER_IS_GNUCC )
	find_package(Boost COMPONENTS filesystem system)
	if(Boost_FOUND)
		include_directories(${Boost_INCLUDE_DIR})
	else()
		message(FATAL_ERROR "HSIproc requires Boost::filesystem and Boost::system when using GCC")
	endif()
endif()

#find LAPACK and supporting link_libraries
find_package(LAPACKE REQUIRED)

#include include directories
include_directories(${CUDA_INCLUDE_DIRS}
					${LAPACKE_INCLUDE_DIR}
					${STIM_INCLUDE_DIRS}
					"${CMAKE_SOURCE_DIR}/src"
)

#collect all source files
include_directories("${CMAKE_SOURCE_DIR}/src")
file(GLOB GA_GPU_SRC "${CMAKE_SOURCE_DIR}/src/*")


#create an executable file
cuda_add_executable(genetic-gpu
		${GA_GPU_SRC}
)

target_link_libraries(genetic-gpu ${CUDA_LIBRARIES}
						 #${CUDA_CUBLAS_LIBRARIES}
						 #${CUDA_CUFFT_LIBRARIES}
						 ${LAPACKE_LIBRARIES}
						 ${LAPACK_LIBRARIES}
						 ${BLAS_LIBRARIES}
						 ${X11_LIBRARIES}
)


#if Boost is found, set an environment variable to use with preprocessor directives
if(Boost_FILESYSTEM_FOUND)
	target_link_libraries(genetic-gpu ${Boost_FILESYSTEM_LIBRARIES}
					 ${Boost_SYSTEM_LIBRARY}
	)
endif(Boost_FILESYSTEM_FOUND)
