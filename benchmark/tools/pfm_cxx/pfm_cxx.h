#ifndef _PFM_CXX_H
#define _PFM_CXX_H

#include <string>

class pfm_instance
{
public:
	pfm_instance();
	bool event_encoding(const std::string& event, unsigned int& type, unsigned long long& config);
	~pfm_instance();	
};
#endif
