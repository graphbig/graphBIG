#include <iostream>
#include <string>
#include <errno.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pfm_cxx.h"

#include <linux/perf_event.h>
#include <asm/unistd.h>

using namespace std;

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                   group_fd, flags);
    return ret;
}

int
main(int argc, char **argv)
{
        struct perf_event_attr attr;
        int fd, ret;
	long long count;

        memset(&attr, 0, sizeof(attr));
        unsigned int type;
        unsigned long long config;

	pfm_instance pfm;
	
        pfm.event_encoding("PERF_COUNT_HW_CACHE_L1D:READ:ACCESS", type, config);
        attr.type = type;
        attr.size = sizeof(struct perf_event_attr);
        attr.config = config;
        /* do not start immediately after perf_event_open() */
        attr.disabled = 1;

	fd = perf_event_open(&attr, 0, -1, -1, 0);
        if (fd < 0)
                err(1, "cannot create event");

        ret = ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        if (ret)
                err(1, "ioctl(enable) failed");

        printf("Fibonacci(%d)=%lu\n", 100, 10000);

        ret = ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        if (ret)
                err(1, "ioctl(disable) failed");

        ret = read(fd, &count, sizeof(count));
	cout<<"count: "<<count<<endl;
	return 0;
}
