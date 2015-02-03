#include "pfm_cxx.h"
#include <sys/types.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include <sys/ioctl.h>
#include <err.h>

#include <perfmon/pfmlib_perf_event.h>

pfm_instance::pfm_instance()
{
	int ret = pfm_initialize();
        if (ret != PFM_SUCCESS)
                errx(1, "cannot initialize library: %s", pfm_strerror(ret));
	return;
}

pfm_instance::~pfm_instance()
{
	/* free libpfm resources cleanly */
        pfm_terminate();
}

bool pfm_instance::event_encoding(const std::string& event, unsigned int& type, unsigned long long& config)
{
	int ret;
	struct perf_event_attr attr;
	memset(&attr, 0, sizeof(attr));

	ret = pfm_get_perf_event_encoding(event.c_str(), PFM_PLM3, &attr, NULL, NULL);
        if (ret != PFM_SUCCESS)
	{
		type = 0;
		config = 0;
		return false;
	}

	type = attr.type;
	config = attr.config;

	return true;
}
