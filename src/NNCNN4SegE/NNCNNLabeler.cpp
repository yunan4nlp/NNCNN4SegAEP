#include "NNCNNLabeler.h"

#include "Argument_helper.h"

Classifier::Classifier(int memsize) :m_driver(memsize){
	// TODO Auto-generated constructor stub
	srand(0);
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

int Classifier::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0){
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

	int numInstance;

	m_driver._modelparams.labelAlpha.clear();

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_segs;
		const string &label = pInstance->m_label;

		m_driver._modelparams.labelAlpha.from_string(label);
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
		}

		const vector<vector<string> > &eval_chars = pInstance->m_eval_chars;
		int eval_num = eval_chars.size();
		
		for (int i = 0; i < eval_num; i++) {
			int char_size = eval_chars[i].size();
			for (int j = 0; j < char_size; j++) {
				m_eval_char_stats[eval_chars[i][j]]++;
			}
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << numInstance << " " << endl;
	cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
	cout << "Eval num: " << m_eval_char_stats.size() << endl;
	m_driver._modelparams.labelAlpha.set_fixed_flag(true);

	return 0;
}

void Classifier::getGoldAnswer(vector<Instance>& vecInsts){
	int max_size = vecInsts.size();
	for (int idx = 0; idx < max_size; idx++) {
		Instance& curInst = vecInsts[idx];
		const string &orcale = curInst.m_label;
		int numLabel = m_driver._modelparams.labelAlpha.size();
		vector<dtype>& curlabels = curInst.m_gold_answer;
		curlabels.clear();
		for (int j = 0; j < numLabel; ++j) {
			string str = m_driver._modelparams.labelAlpha.from_id(j);
			if (str.compare(orcale) == 0)
				curlabels.push_back(1.0);
			else
				curlabels.push_back(0.0);
		}
	}
}

int Classifier::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;


	for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_segs;
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
		}

		const vector<vector<string> > &eval_chars = pInstance->m_eval_chars;
		int eval_num = eval_chars.size();
		for (int i = 0; i < eval_num; i++) {
			int char_size = eval_chars[i].size();
			for (int j = 0; j < char_size; j++) {
				string currchar = normalize_to_lowerwithdigit(eval_chars[i][j]);
				if (!m_options.evalCharEmbFineTune)m_eval_char_stats[currchar]++;
			}
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	return 0;
}

void Classifier::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	if (trainFile != "")
		m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	createAlphabet(trainInsts);

	getGoldAnswer(trainInsts);
	getGoldAnswer(devInsts);
	getGoldAnswer(testInsts);
	//Ensure that each file in m_options.testFiles exists!
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
		getGoldAnswer(otherInsts[idx]);
	}

	//std::cout << "Training example number: " << trainInsts.size() << std::endl;
	//std::cout << "Dev example number: " << trainInsts.size() << std::endl;
	//std::cout << "Test example number: " << trainInsts.size() << std::endl;

	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestAlpha(otherInsts[idx]);
	}

	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Instance> > otherInstances(otherInsts.size());

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	m_driver._modelparams.evalCharAlpha.initial(m_eval_char_stats, m_options.evalCharCutOff);
	m_driver._modelparams.evalChars.initial(&m_driver._modelparams.evalCharAlpha, m_options.evalCharEmbSize, m_options.evalCharEmbFineTune);
	if (m_options.wordFile != "") {
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	m_driver._hyperparams.setRequared(m_options);
	m_driver.initial();


	dtype bestDIS = 0;

	int inputSize = trainInsts.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;
	static vector<Instance> subInsts;
	int devNum = devInsts.size(), testNum = testInsts.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subInsts.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subInsts.push_back(trainInsts[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subInsts, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_driver.checkgrad(subInsts, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();

		}

		if (devNum > 0) {
			clock_t time_start = clock();
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devInsts.size(); idx++) {
				string result_label;
				predict(devInsts[idx], result_label);

				devInsts[idx].evaluate(result_label, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_label);
					decodeInstResults.push_back(curDecodeInst);
				}
			}
			std::cout << "Dev finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
			std::cout << "dev:" << std::endl;
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				time_start = clock();
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testInsts.size(); idx++) {
					string result_label;
					predict(testInsts[idx], result_label);

					testInsts[idx].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "Test finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherInstances.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idy = 0; idy < otherInstances[idx].size(); idy++) {
					string result_label;
					predict(otherInstances[idx][idy], result_label);

					otherInsts[idx][idy].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Classifier::predict(const Instance& inst, string& output) {
	//assert(features.size() == words.size());
	int labelIdx;
	m_driver.predict(inst, labelIdx);
	output = m_driver._modelparams.labelAlpha.from_id(labelIdx, unknownkey);

	if (output == nullkey){
		std::cout << "predict error" << std::endl;
	}
	return 0;
}

void Classifier::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	int testNum = testInsts.size();
	vector<Instance> testInstResults;
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testInsts.size(); idx++) {
		string result_label;
		predict(testInsts[idx], result_label);
		testInsts[idx].evaluate(result_label, metric_test);
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_label);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}


void Classifier::loadModelFile(const string& inputModelFile) {

}

void Classifier::writeModelFile(const string& outputModelFile) {

}

int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
 	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	Classifier the_classifier(memsize);
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	getchar();
	//test(argv);
	//ah.write_values(std::cout);
}