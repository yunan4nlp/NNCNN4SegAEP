#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	LookupTable words; // should be initialized outside
	Alphabet attAlpha;// should be initialized outside
	LookupTable atts;// should be initialized outside
	Alphabet evalCharAlpha;// should be initialized outside
	LookupTable evalChars;// should be initialized outside
	UniParams hidden_linear;
	UniParams eval_char_hidden_linear;
	TriParams seg_att_eval_concat;
	UniParams olayer_linear; // output
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;


public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (words.nVSize <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.wordDim = words.nDim;
		opts.attDim = atts.nDim;
		opts.evalCharDim = evalChars.nDim;
		opts.wordWindow = opts.wordContext * 2 + 1;
		opts.evalCharWindow = opts.evalCharContext * 2 + 1;
		opts.labelSize = labelAlpha.size();
		hidden_linear.initial(opts.wordHiddenSize, opts.wordDim * opts.wordWindow, true, mem);
		eval_char_hidden_linear.initial(opts.evalCharHiddenSize, opts.evalCharDim * opts.evalCharWindow, true, mem);
		seg_att_eval_concat.initial(opts.concatHiddenSize, opts.wordHiddenSize, opts.attDim * 3, opts.evalCharHiddenSize * 3 * 3, true, mem);
		opts.inputSize = opts.concatHiddenSize;
		olayer_linear.initial(opts.labelSize, opts.inputSize, false, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		atts.exportAdaParams(ada);
		evalChars.exportAdaParams(ada);
		hidden_linear.exportAdaParams(ada);
		eval_char_hidden_linear.exportAdaParams(ada);
		seg_att_eval_concat.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&hidden_linear.W, "hidden_linear.W");
		checkgrad.add(&hidden_linear.b, "hidden_linear.b");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */