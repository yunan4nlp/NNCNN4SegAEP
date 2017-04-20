#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Instance.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	RNNBuilder _rnn;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;
	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_rnn.resize(sent_length);
		_avg_pooling.setParam(sent_length);
		_max_pooling.setParam(sent_length);
		_min_pooling.setParam(sent_length);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_word_window.clear();
		_rnn.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
		}
		_rnn.init(&model.rnn_params, opts.dropProb, true, mem);
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		_avg_pooling.init(opts.wordHiddenSize, -1, mem);
		_max_pooling.init(opts.wordHiddenSize, -1, mem);
		_min_pooling.init(opts.wordHiddenSize, -1, mem);
		_concat.init(opts.wordHiddenSize * 3, -1, mem);
		_output.setParam(&model.olayer_linear);
		_output.init(opts.labelSize, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Instance& inst, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = inst.m_segs.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			_word_inputs[i].forward(this, inst.m_segs[i]);
		}
		_word_window.forward(this, getPNodes(_word_inputs, words_num));

		_rnn.forward(this, getPNodes(_word_window._outputs, words_num));
		_max_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_min_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_avg_pooling.forward(this, getPNodes(_rnn._output, words_num));
		_concat.forward(this, &_max_pooling, &_min_pooling, &_avg_pooling);
		_output.forward(this, &_concat);
	}
};

#endif /* SRC_ComputionGraph_H_ */