#ifndef SRC_NNCNNLabeler_H_
#define SRC_NNCNNLabeler_H_


#include "N3L.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Classifier {


public:
	unordered_map<string, int> m_word_stats;
	unordered_map<string, int> m_att_stats;
	unordered_map<string, int> m_eval_char_stats;
	unordered_map<string, int> m_polarity_stats;

public:
	Options m_options;

	Driver m_driver;

	Pipe m_pipe;


public:
	Classifier(int memsize);
	virtual ~Classifier();

public:

	int createAlphabet(const vector<Instance>& vecTrainInsts);
	void getGoldAnswer(vector<Instance>& vecInsts);
	int addTestAlpha(const vector<Instance>& vecInsts);


public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	int predict(const Instance& inst, string& output);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCNNLabeler_H_ */
