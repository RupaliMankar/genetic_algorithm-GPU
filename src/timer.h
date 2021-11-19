#ifndef GA_GPU_TIMER_H
#define GA_GPU_TIMER_H

#include <chrono>
//#include <thread>
//#include <iostream>

//using namespace std::chrono;

class Timer {
typedef std::chrono::high_resolution_clock Clock;
 
Clock::time_point epoch;
public:
  void start(){
	  epoch = Clock::now();
  }
  Clock::duration time_elapsed() const{
	  return Clock::now() - epoch;
  }
};

#endif


//class Timer {
//	std::chrono::time_point<std::chrono::high_resolution_clock> epoch;
//
//	public:
//	typedef high_resolution_clock Clock;
//	void start(){
//		epoch = Clock::now();
//	}
//	std::chrono::time_point<std::chrono::high_resolution_clock> time_elapsed() const {
//		return Clock::now() - epoch;
//	}
//};