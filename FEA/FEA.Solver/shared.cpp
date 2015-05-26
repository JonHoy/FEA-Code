#include "shared.h"

// Add function definitions here

unsigned long STL_Unique(unsigned long* arrPtr, unsigned long arrLength) {
	std::vector<unsigned long> myVec(&arrPtr[0], &arrPtr[arrLength]);
	std::sort(myVec.begin(), myVec.end());
	std::vector<unsigned long>::iterator endLocation = std::unique(myVec.begin(), myVec.end());
	return (unsigned long)(endLocation - myVec.begin()); 
}
