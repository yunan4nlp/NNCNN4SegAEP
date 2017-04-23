#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization

	int wordHiddenSize;
	int wordContext;
	int wordWindow;
	int windowOutput;
	dtype dropProb;


	//auto generated
	int wordDim;
	int polarityDim;
	int inputSize;
	int labelSize;

public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		wordHiddenSize = opt.wordHiddenSize;
		wordContext = opt.wordcontext;
		dropProb = opt.dropProb;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */