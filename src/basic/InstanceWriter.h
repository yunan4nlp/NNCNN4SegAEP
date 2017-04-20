#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;

class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
	  if (!m_outf.is_open()) return -1;
		

	  const string &label = pInstance->m_label;

	  m_outf << label << " ";
	  const vector<string> words = pInstance->m_segs;
	  int word_size = words.size();
	  for (int idx = 0; idx < word_size; idx++)
		  m_outf << words[idx] << " ";
		int att_num = pInstance->m_attributes.size();
		const vector<string> atts = pInstance->m_attributes;
		for(int idx = 0; idx < att_num; idx++) {
			m_outf << atts[idx] << " ";
		}
		int eval_num = pInstance->m_evalutions.size();
		const vector<string> evals = pInstance->m_evalutions;
		for(int idx = 0; idx < eval_num; idx++) {
			m_outf << evals[idx] << " ";
		}
		m_outf << pInstance->m_polarity << endl;
	  return 0;
	}
};

#endif

