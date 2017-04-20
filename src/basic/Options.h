#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

	int wordCutOff;
	int attCutOff;
	int evalCharCutOff;
	int maxIter;
	int batchSize;
	dtype adaEps;
	dtype adaAlpha;
	dtype regParameter;
	dtype dropProb;
	dtype polarDropProb;

	int wordEmbSize;
	int wordcontext;
	bool wordEmbFineTune;
	int wordHiddenSize;

	int attEmbSize;
	bool attEmbFineTune;

	int polarityEmbSize;
	int polarityEmbFineTune;
	int polarityHiddenSize;

	int evalCharEmbSize;
	int evalCharContext;
	bool evalCharEmbFineTune;
	int evalCharHiddenSize;

	int concatHiddenSize;


	int verboseIter;
	bool train;
	int maxInstance;
	vector<string> testFiles;
	string outBest;
	bool seg;
	int relu;
	bool saveIntermediate;
	//embedding files
	string wordFile;

	Options() {
		wordCutOff = 0;
		attCutOff = 0;
		evalCharCutOff = 0;
		maxIter = 1000;
		batchSize = 1;
		adaEps = 1e-6;
		adaAlpha = 0.01;
		regParameter = 1e-8;
		dropProb = -1;
		polarDropProb = -1;


		wordEmbSize = 50;
		wordcontext = 2;
		wordEmbFineTune = true;
		wordHiddenSize = 100;


		attEmbSize = 50;
		attEmbFineTune = true;

		polarityEmbSize = 50;
		polarityEmbFineTune = true;
		polarityHiddenSize = 50;

		evalCharEmbSize = 50;
		evalCharContext = 2;
		evalCharEmbFineTune = true;
		evalCharHiddenSize = 50;

		concatHiddenSize = 100;

		verboseIter = 100;
		train = false;
		maxInstance = -1;
		testFiles.clear();
		outBest = "";
		relu = 0;
		seg = false;

		wordFile = "";
		saveIntermediate = true;
	}

	virtual ~Options() {

	}

	void setOptions(const vector<string> &vecOption) {
		int i = 0;
		for (; i < vecOption.size(); ++i) {
			pair<string, string> pr;
			string2pair(vecOption[i], pr, '=');
			if (pr.first == "wordCutOff")
				wordCutOff = atoi(pr.second.c_str());
			if (pr.first == "attCutOff")
				attCutOff == atoi(pr.second.c_str());
			if (pr.first == "evalCharCutOff")
				evalCharCutOff = atoi(pr.second.c_str());
			if (pr.first == "maxIter")
				maxIter = atoi(pr.second.c_str());
			if (pr.first == "batchSize")
				batchSize = atoi(pr.second.c_str());
			if (pr.first == "adaEps")
				adaEps = atof(pr.second.c_str());
			if (pr.first == "adaAlpha")
				adaAlpha = atof(pr.second.c_str());
			if (pr.first == "regParameter")
				regParameter = atof(pr.second.c_str());
			if (pr.first == "dropProb")
				dropProb = atof(pr.second.c_str());
			if (pr.first == "polarDropProb")
				polarDropProb = atof(pr.second.c_str());


			if (pr.first == "wordcontext")
				wordcontext = atoi(pr.second.c_str());
			if (pr.first == "wordEmbSize")
				wordEmbSize = atoi(pr.second.c_str());
			if (pr.first == "wordEmbFineTune")
				wordEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "wordHiddenSize")
				wordHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "attEmbSize")
				attEmbSize = atoi(pr.second.c_str());
			if (pr.first == "attEmbFineTune")
				attEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "polarityEmbSize")
				polarityEmbSize = atoi(pr.second.c_str());
			if (pr.first == "polarityEmbFineTune")
				polarityEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "polarityHiddenSize")
				polarityHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "evalCharEmbSize")
				evalCharEmbSize = atoi(pr.second.c_str());
			if (pr.first == "evalCharContext")
				evalCharContext = atoi(pr.second.c_str());
			if (pr.first == "evalCharEmbFineTune")
				evalCharEmbFineTune = (pr.second == "true") ? true : false;
			if (pr.first == "evalCharHiddenSize")
				evalCharHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "concatHiddenSize")
				concatHiddenSize = atoi(pr.second.c_str());

			if (pr.first == "verboseIter")
				verboseIter = atoi(pr.second.c_str());
			if (pr.first == "train")
				train = (pr.second == "true") ? true : false;
			if (pr.first == "maxInstance")
				maxInstance = atoi(pr.second.c_str());
			if (pr.first == "testFile")
				testFiles.push_back(pr.second);
			if (pr.first == "outBest")
				outBest = pr.second;
			if (pr.first == "relu")
				relu = atoi(pr.second.c_str());
			if (pr.first == "seg")
				seg = (pr.second == "true") ? true : false;

			if (pr.first == "saveIntermediate")
				saveIntermediate = (pr.second == "true") ? true : false;

			if (pr.first == "wordFile")
				wordFile = pr.second;
		}
	}

	void showOptions() {
		std::cout << "wordCutOff = " << wordCutOff << std::endl;
		std::cout << "attCutOff = " << attCutOff << std::endl;
		std::cout << "evalCharCutOff = " << evalCharCutOff << std::endl;
		std::cout << "maxIter = " << maxIter << std::endl;
		std::cout << "batchSize = " << batchSize << std::endl;
		std::cout << "adaEps = " << adaEps << std::endl;
		std::cout << "adaAlpha = " << adaAlpha << std::endl;
		std::cout << "regParameter = " << regParameter << std::endl;
		std::cout << "dropProb = " << dropProb << std::endl;
		std::cout << "polarDropProb = " << polarDropProb << std::endl;

		std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
		std::cout << "wordcontext = " << wordcontext << std::endl;
		std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
		std::cout << "wordHiddenSize = " << wordHiddenSize << std::endl;
		std::cout << "attEmbSize = " << attEmbSize << std::endl;
		std::cout << "attEmbFineTune = " << attEmbFineTune << std::endl;
		std::cout << "polarityEmbSize = " << polarityEmbSize << std::endl;
		std::cout << "polarityEmbFineTune = " << polarityEmbFineTune << std::endl;
		std::cout << "polarityHiddenSize = " << polarityHiddenSize << std::endl;

		std::cout << "evalCharEmbSize = " << evalCharEmbSize << std::endl;
		std::cout << "evalCharContext = " << evalCharContext << std::endl;
		std::cout << "evalCharEmbFineTune = " << evalCharEmbFineTune << std::endl;
		std::cout << "evalCharHiddenSize = " << evalCharHiddenSize << std::endl;

		std::cout << "concatHiddenSize = " << concatHiddenSize << std::endl;

		std::cout << "verboseIter = " << verboseIter << std::endl;
		std::cout << "maxInstance = " << maxInstance << std::endl;
		std::cout << "outBest = " << outBest << std::endl;
		for (int idx = 0; idx < testFiles.size(); idx++) {
			std::cout << "testFile = " << testFiles[idx] << std::endl;
		}
		std::cout << "seg = " << seg << std::endl;

		std::cout << "wordFile = " << wordFile << std::endl;
	}

	void load(const std::string& infile) {
		ifstream inf;
		inf.open(infile.c_str());
		vector<string> vecLine;
		while (1) {
			string strLine;
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (strLine.empty())
				continue;
			vecLine.push_back(strLine);
		}
		inf.close();
		setOptions(vecLine);
	}
};

#endif

