#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		m_segs.clear();
		m_attributes.clear();
		m_evalutions.clear();
		m_eval_chars.clear();
		m_polarity.clear();
		m_label.clear();
	}

	void evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == m_label)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		m_segs = anInstance.m_segs;
		m_attributes = anInstance.m_attributes;
		m_evalutions = anInstance.m_evalutions;
		m_eval_chars = anInstance.m_eval_chars;
		m_polarity = anInstance.m_polarity;
		m_label = anInstance.m_label;
	}

	void assignLabel(const string& resulted_label) {
		m_label = resulted_label;
	}

	int size() const {
		return m_segs.size();
	}

public:
	vector<string> m_segs;
	vector<string> m_attributes;
	vector<string> m_evalutions;
	vector<vector<string> > m_eval_chars;
	string m_polarity;
	string m_label;

	vector<dtype> m_gold_answer;
};

#endif /*_INSTANCE_H_*/
