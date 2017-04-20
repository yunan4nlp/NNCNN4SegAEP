#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Instance.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;
	const static int max_att_size = 10;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	vector<UniNode> _hidden;

	MaxPoolNode _word_max_pooling;


	vector<LookupNode> _att_inputs;
	Node _att_bucket;
	AvgPoolNode _att_avg_pooling;
	MaxPoolNode _att_max_pooling;
	MinPoolNode _att_min_pooling;
	ConcatNode _att_pooling_concat;

	BiNode _concat_seg_att;

	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length, int att_size){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_hidden.resize(sent_length);
		_word_max_pooling.setParam(sent_length);

		_att_inputs.resize(att_size);
		_att_avg_pooling.setParam(att_size);
		_att_max_pooling.setParam(att_size);
		_att_min_pooling.setParam(att_size);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_att_inputs.clear();
		_word_window.clear();
		_hidden.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
			_hidden[idx].setParam(&model.hidden_linear);
			_hidden[idx].init(opts.wordHiddenSize, opts.dropProb, mem);
			_hidden[idx].setFunctions(frelu, drelu);
		}
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		_word_max_pooling.init(opts.wordHiddenSize, -1, mem);

		for (int idx = 0; idx < _att_inputs.size(); idx++) {
			_att_inputs[idx].setParam(&model.atts);
			_att_inputs[idx].init(opts.attDim, opts.dropProb, mem);
		}
		_att_bucket.set_bucket();
		_att_bucket.init(opts.attDim, -1, mem);
		_att_avg_pooling.init(opts.attDim, -1, mem);
		_att_max_pooling.init(opts.attDim, -1, mem);
		_att_min_pooling.init(opts.attDim, -1, mem);
		_att_pooling_concat.init(opts.attDim * 3, -1, mem);

		_concat_seg_att.setParam(&model.seg_att_concat);
		_concat_seg_att.init(opts.concatHiddenSize, opts.dropProb, mem);

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
		int word_num = inst.m_segs.size();
		if (word_num > max_sentence_length)
			word_num = max_sentence_length;
		for (int i = 0; i < word_num; i++) {
			_word_inputs[i].forward(this, inst.m_segs[i]);
		}
		_word_window.forward(this, getPNodes(_word_inputs, word_num));

		for (int i = 0; i < word_num; i++) {
			_hidden[i].forward(this, &_word_window._outputs[i]);
		}
		_word_max_pooling.forward(this, getPNodes(_hidden, word_num));

		int att_num = inst.m_attributes.size();
		if (att_num > max_att_size)
			att_num = max_att_size;
		for (int i = 0; i < att_num; i++) {
			_att_inputs[i].forward(this, inst.m_attributes[i]);
		}


		if (att_num == 0)
			_att_pooling_concat.forward(this, &_att_bucket, &_att_bucket, &_att_bucket);
		else {
			_att_avg_pooling.forward(this, getPNodes(_att_inputs, att_num));
			_att_max_pooling.forward(this, getPNodes(_att_inputs, att_num));
			_att_min_pooling.forward(this, getPNodes(_att_inputs, att_num));
			_att_pooling_concat.forward(this, &_att_avg_pooling, &_att_max_pooling, &_att_min_pooling);
		}

		_concat_seg_att.forward(this, &_word_max_pooling, &_att_pooling_concat);

		_output.forward(this, &_concat_seg_att);
	}
};

#endif /* SRC_ComputionGraph_H_ */