// Performance Counter Wrapper Classes
#ifndef _PERF_H
#define _PERF_H

#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <string>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include <string.h>

#ifndef NO_PFM
#include "pfm_cxx.h"
#endif

#define DEFAULT_PERF_GRP_SZ 4

// PERF SYSTEM CALL REFERENCE:

// if "type" is PERF_TYPE_HARDWARE, "config" is one of: 
//----PERF_COUNT_HW_CPU_CYCLES
//----PERF_COUNT_HW_INSTRUCTIONS
//----PERF_COUNT_HW_CACHE_REFERENCES: may include prefetches and coherency messages
//----PERF_COUNT_HW_CACHE_MISSES
//----PERF_COUNT_HW_BRANCH_INSTRUCTIONS
//----PERF_COUNT_HW_BRANCH_MISSES
//----PERF_COUNT_HW_BUS_CYCLES
//----PERF_COUNT_HW_STALLED_CYCLES_FRONTEND (since Linux 3.0)
//----PERF_COUNT_HW_STALLED_CYCLES_BACKEND (since Linux 3.0)
//----PERF_COUNT_HW_REF_CPU_CYCLES (since Linux 3.3)

// if "type" is PERF_TYPE_SOFTWARE, "config" is one of:
//----PERF_COUNT_SW_CPU_CLOCK
//----PERF_COUNT_SW_TASK_CLOCK
//----PERF_COUNT_SW_PAGE_FAULTS
//----PERF_COUNT_SW_CONTEXT_SWITCHES
//----PERF_COUNT_SW_CPU_MIGRATIONS
//----PERF_COUNT_SW_PAGE_FAULTS_MIN
//----PERF_COUNT_SW_PAGE_FAULTS_MAJ
//----PERF_COUNT_SW_ALIGNMENT_FAULTS (since Linux 2.6.33)
//----PERF_COUNT_SW_EMULATION_FAULTS (since Linux 2.6.33)
//----PERF_COUNT_SW_DUMMY (since Linux 3.12)

// if "type" is PERF_TYPE_HW_CACHE, "config" is (perf_hw_cache_id) 
//                                              | (perf_hw_cache_op_id << 8) 
//                                              | (perf_hw_cache_op_result_id << 16)
// where perf_hw_cache_id is:
//----PERF_COUNT_HW_CACHE_L1D
//----PERF_COUNT_HW_CACHE_L1I
//----PERF_COUNT_HW_CACHE_LL
//----PERF_COUNT_HW_CACHE_DTLB
//----PERF_COUNT_HW_CACHE_ITLB
//----PERF_COUNT_HW_CACHE_BPU: branch prediction unit
//----PERF_COUNT_HW_CACHE_NODE (since Linux 3.0): local memory accesses
// and perf_hw_cache_op_id is one of:
//----PERF_COUNT_HW_CACHE_OP_READ
//----PERF_COUNT_HW_CACHE_OP_WRITE
//----PERF_COUNT_HW_CACHE_OP_PREFETCH
// and perf_hw_cache_op_result_id is one of:
//----PERF_COUNT_HW_CACHE_RESULT_ACCESS
//----PERF_COUNT_HW_CACHE_RESULT_MISS






struct read_format {
    unsigned long long value;         /* The value of the event */
    unsigned long long time_enabled;  /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
    unsigned long long time_running;  /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
    unsigned long long id;            /* if PERF_FORMAT_ID */
};

class gBenchPerf_handler
{
public:
    gBenchPerf_handler(unsigned int type=PERF_TYPE_HARDWARE, 
                       unsigned long long config=PERF_COUNT_HW_CPU_CYCLES,
                       int group_fd=-1)
    :_perf(-1),_type(type),_config(config),_group_fd(group_fd),_perf_cnt(0),_multiplexing(false){}

    ~gBenchPerf_handler() { if (_perf != -1) close(_perf); }

    gBenchPerf_handler(const gBenchPerf_handler& rhs)
    {
        _perf = rhs._perf;
        _type = rhs._type;
        _config = rhs._config;
        _group_fd = rhs._group_fd;
        _perf_cnt = rhs._perf_cnt;
        _multiplexing = rhs._multiplexing;
    }
    void set_type(unsigned int type) { _type = type; }
    void set_config(unsigned long long config) { _config = config; }

//        exclude_user   : 1,   /* don't count user */
//        exclude_kernel : 1,   /* don't count kernel */
//        exclude_hv     : 1,   /* don't count hypervisor */
//        exclude_idle   : 1,   /* don't count when idle */
    void open(bool exclude_user, bool exclude_kernel,
              bool exclude_idle, bool exclude_hv=false)
    {
        if (_perf != -1) close(_perf);
        _perf_cnt = 0;
        _multiplexing = false;

        struct perf_event_attr _perf_attr;
        memset(&_perf_attr, 0, sizeof(struct perf_event_attr));
        _perf_attr.type = _type;
        _perf_attr.size = sizeof(struct perf_event_attr);
        _perf_attr.config = _config;
        _perf_attr.disabled = 1;
        _perf_attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | 
                PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID;

        if (exclude_user)   _perf_attr.exclude_user = 1;
        if (exclude_kernel) _perf_attr.exclude_kernel = 1;
        if (exclude_idle)   _perf_attr.exclude_idle = 1;
        if (exclude_hv)     _perf_attr.exclude_hv = 1;
        
        _perf = perf_event_open(&_perf_attr, 0, -1, _group_fd, 0);
        if (_perf == -1)
        {
            std::cout<<"cannot open perf event: type-"<<_perf_attr.type<<"  config-"<<_perf_attr.config<<std::endl;
        }
    }

    void start(void)
    {
        if (_perf == -1) return;
        _multiplexing = false;
        _perf_cnt = 0;

        ioctl(_perf, PERF_EVENT_IOC_RESET, 0);
        ioctl(_perf, PERF_EVENT_IOC_ENABLE, 0);
    }

    unsigned long long stop(void)
    {
        if (_perf == -1) return 0;

        struct read_format ret;
        ioctl(_perf, PERF_EVENT_IOC_DISABLE, 0);
        long int n = read(_perf, &ret, sizeof(struct read_format));

        if (n < 0) 
        {
            std::cout<<"error when stopping perf"<<std::endl;
            return 0;
        }

        if (ret.time_enabled != ret.time_running) _multiplexing = true;
        if (_multiplexing)
            _perf_cnt = ret.value * ((double)ret.time_enabled / (double)ret.time_running);
        else
            _perf_cnt = ret.value;
        return _perf_cnt;
    }

    unsigned long long get_perf_cnt(void) { return _perf_cnt; }
    bool is_multiplexing(void) { return _multiplexing; }

protected:
    long perf_event_open( struct perf_event_attr *hw_event, pid_t pid,
                      int cpu, int group_fd, unsigned long flags )
    {
        int ret;

        ret = syscall( __NR_perf_event_open, hw_event, pid, cpu,
                       group_fd, flags );
        return ret;
    }


    int _perf;    
    unsigned int _type;
    unsigned long long _config;
    int _group_fd;

    unsigned long long _perf_cnt;
    bool _multiplexing;
};




#define GBENCH_PERF_INIT(id) gBenchPerf_full perf_##id; perf_##id.open();
#define GBENCH_PERF_START(id) perf_##id.start();
#define GBENCH_PERF_STOP(id) perf_##id.stop();
#define GBENCH_PERF_PRINT(id) perf_##id.print();

//===============================//
//HW_CPU_CYCLES
//HW_INSTRUCTIONS
//HW_CACHE_REFERENCES
//HW_CACHE_MISSES
//HW_BRANCH_INSTRUCTIONS
//HW_BRANCH_MISSES
//===============================//
//SW_CPU_CLOCK
//SW_TASK_CLOCK
//SW_PAGE_FAULTS
//SW_CONTEXT_SWITCHES
//SW_CPU_MIGRATIONS
//SW_PAGE_FAULTS_MIN
//SW_PAGE_FAULTS_MAJ
//===============================//
//CACHE_<L1D|L1I|LL|DTLB|ITLB|BPU>_<READ|WRITE|PREFETCH>_<ACCESS|MISS>
//===============================//
class gBenchPerf_event
{
public:
    gBenchPerf_event(){}
    gBenchPerf_event(const gBenchPerf_event& rhs)
    {
        _perf_vec = rhs._perf_vec;
        _event_vec = rhs._event_vec;
        _cnt_vec = rhs._cnt_vec;
        _multiplexing_vec = rhs._multiplexing_vec;
        exclude_user = rhs.exclude_user;
        exclude_kernel = rhs.exclude_kernel;
        exclude_idle = rhs.exclude_idle;
        exclude_hv = rhs.exclude_hv;
    }
    gBenchPerf_event(std::vector<std::string>& inputarg, bool call_open=true)
    {
        size_t i=1;
        if (inputarg.size()<2) return;

        exclude_user=false;
        exclude_kernel=false;
        exclude_idle=false;
        exclude_hv=false;

        _event_vec.clear();
        while (true) 
        {
            if (inputarg[i]=="--perf-event") 
            {
                size_t k;
                for (k=i+1;k<inputarg.size();k++) 
                {
                    if (inputarg[k].substr(0,2)=="--") break;
                    _event_vec.push_back(inputarg[k]);
                }

                inputarg.erase(inputarg.begin()+i, inputarg.begin()+k);
            }
            else if (inputarg[i]=="--perf-exclude-user") 
            {
                exclude_user=true;
                inputarg.erase(inputarg.begin()+i);
            }
            else if (inputarg[i]=="--perf-exclude-idle") 
            {
                exclude_idle=true;
                inputarg.erase(inputarg.begin()+i);
            }
            else if (inputarg[i]=="--perf-exclude-kernel") 
            {
                exclude_kernel=true;
                inputarg.erase(inputarg.begin()+i);
            }
            else if (inputarg[i]=="--perf-exclude-hv") 
            {
                exclude_hv=true;
                inputarg.erase(inputarg.begin()+i);
            }
            else
                i++;

            if (i >= inputarg.size()) break;
        }

        _cnt_vec.resize(_event_vec.size(), 0);
        _multiplexing_vec.resize(_event_vec.size(), false);

#ifndef NO_PFM
        pfm_instance pfm;
#endif

        // process event
        for (size_t i=0;i<_event_vec.size();i++)
        {
            unsigned int type=0;
            unsigned long long config=0;

            if (_event_vec[i].substr(0,11)=="PERF_COUNT_") 
            {
                event_switch(_event_vec[i].substr(11),type,config);
                _perf_vec.push_back(gBenchPerf_handler(type, config));
            }
#ifndef NO_PFM
            else if (pfm.event_encoding(_event_vec[i], type, config))
            	_perf_vec.push_back(gBenchPerf_handler(type, config));
#endif
            else
                std::cout<<"wrong event: "<<_event_vec[i]<<std::endl;
        }

        if (call_open)
            open(exclude_user,exclude_kernel,exclude_idle,exclude_hv);

        return;
    }
    gBenchPerf_event(std::string arg)
    {
        event_parser(arg);
    }
    
    gBenchPerf_event& operator=(const gBenchPerf_event& rhs)
    {
        _perf_vec = rhs._perf_vec;
        _event_vec = rhs._event_vec;
        _cnt_vec = rhs._cnt_vec;
        _multiplexing_vec = rhs._multiplexing_vec;
        exclude_user = rhs.exclude_user;
        exclude_kernel = rhs.exclude_kernel;
        exclude_idle = rhs.exclude_idle;
        exclude_hv = rhs.exclude_hv;
        return *this;
    }

    
    void set_arg(std::string arg)
    {
        event_parser(arg);
    }

    void open(bool exclude_user, bool exclude_kernel,
              bool exclude_idle, bool exclude_hv=false)
    {
        for (size_t i=0;i<_perf_vec.size();i++)
        {
            _perf_vec[i].open(exclude_user,exclude_kernel,exclude_idle,exclude_hv);
        }
    }
    void open(int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        size_t start = (group_id == -1)? 0 : group_id*group_size;
        size_t end = (group_id == -1)? _perf_vec.size() : start+group_size;
        if (start >= _perf_vec.size()) return;
        if (end > _perf_vec.size()) end = _perf_vec.size();

        for (size_t i=start;i<end;i++)
        {
            _perf_vec[i].open(exclude_user,exclude_kernel,exclude_idle,exclude_hv);

        }
    }

    void start(int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        size_t start = (group_id == -1)? 0 : group_id*group_size;
        size_t end = (group_id == -1)? _perf_vec.size() : start+group_size;
        if (start >= _perf_vec.size()) return;
        if (end > _perf_vec.size()) end = _perf_vec.size();

        for (size_t i=start;i<end;i++)
        {
            _perf_vec[i].start();
        }
    }

    void stop(int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        size_t start = (group_id == -1)? 0 : group_id*group_size;
        size_t end = (group_id == -1)? _perf_vec.size() : start+group_size;
        if (start >= _perf_vec.size()) return;
        if (end > _perf_vec.size()) end = _perf_vec.size();

        for (size_t i=start;i<end;i++)
        {
            _perf_vec[i].stop();
        }
        for (size_t i=start;i<end;i++)
        {
            _cnt_vec[i] = _perf_vec[i].get_perf_cnt();
            _multiplexing_vec[i] = _perf_vec[i].is_multiplexing();
        }
    }

    void print(void)
    {
        for (size_t i=0;i<_event_vec.size();i++)
        {

            std::cout<<_event_vec[i]<<"\t==>\t"<<_cnt_vec[i];
            if (_multiplexing_vec[i]) std::cout<<"\tMUX";
            std::cout<<std::endl;
        }
    }
    
    unsigned long long event_counter(size_t id)
    {
        if (id >= _cnt_vec.size())
        {
            std::cerr<<"Wrong event id: "<<id<<std::endl;
            return 0;
        }

        return _cnt_vec[id];
    }
    std::string event_name(size_t id)
    {
        if (id >= _cnt_vec.size())
        {
            std::cerr<<"Wrong event id: "<<id<<std::endl;
            return 0;
        }
        return _event_vec[id];
    }
    bool event_mux(size_t id)
    {
        if (id >= _cnt_vec.size()) return false;
        return _multiplexing_vec[id];
    }
    size_t get_event_cnt(void)
    {
        return _cnt_vec.size();
    }
protected:
    //parsing event list arguments
    void event_parser(std::string arguments)
    {
        _event_vec.clear();
        // preprocess 
        size_t next=0;
        while (next != std::string::npos) 
        {
            std::string cell;
            next=csv_nextCell(arguments,",:;",cell,next);
            _event_vec.push_back(cell);
        }
        
        _cnt_vec.resize(_event_vec.size(), 0);
        _multiplexing_vec.resize(_event_vec.size(), false);

        // process event
        for (size_t i=0;i<_event_vec.size();i++)
        {
            unsigned int type;
            unsigned long long config;

            event_switch(_event_vec[i],type,config);
            _perf_vec.push_back(gBenchPerf_handler(type, config));
        }
    }
    size_t csv_nextCell(std::string& line, std::string sepr, std::string& ret, size_t pos=0)
    {
        sepr.append(" ");

        ret.clear();

        size_t head, tail;
        bool in_quotation = false;

        head = line.find_first_not_of(sepr, pos);
        if (head == std::string::npos) 
        {
            ret.clear();
            return std::string::npos;
        }

        if (line[head]=='\"')
        {
            head++;
            in_quotation = true;
        }
        if (in_quotation) 
        {
            size_t prev;
            tail = line.find_first_of('\"', head);
            ret = line.substr(head, tail-head);

            if (tail == (line.size()-1) ) // reach line end
                return std::string::npos;
            
            while (line[tail+1]=='\"') // double quote means a quote mark in fileds
            {
                ret.append("\"");
                prev = tail+2;
                tail = line.find_first_of('\"', prev);
                if (tail == std::string::npos) 
                {   
                    ret.append(line.substr(prev));
                    return std::string::npos;
                }
                ret.append(line.substr(prev, tail-prev));
                if (tail == (line.size()-1)) 
                    return std::string::npos;
            }
            return line.find_first_not_of(sepr, tail+1);
        }
        else
        {
            tail = line.find_first_of(sepr, head);
            if (tail != std::string::npos) 
            {
                ret = line.substr(head, tail - head);
                return line.find_first_not_of(sepr, tail);
            }
            else
            {
                ret = line.substr(head);
                return std::string::npos;
            }
            
        }

        return std::string::npos; // should not reach here 
    }
    void event_switch(std::string ievent, unsigned int & type, unsigned long long & config)
    {
        
        if (ievent=="HW_CPU_CYCLES") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_CPU_CYCLES;
        }
        else if (ievent=="HW_INSTRUCTIONS") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_INSTRUCTIONS;
        }
        else if (ievent=="HW_CACHE_REFERENCES") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_CACHE_REFERENCES;
        }
        else if (ievent=="HW_CACHE_MISSES") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_CACHE_MISSES;
        }
        else if (ievent=="HW_BRANCH_INSTRUCTIONS") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
        }
        else if (ievent=="HW_BRANCH_MISSES") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_BRANCH_MISSES;
        }
        else if (ievent=="HW_BUS_CYCLES") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_BUS_CYCLES;
        }
        // commented out for compatability concern
        // older linux kernel may not support this
        /*else if (ievent=="HW_STALLED_CYCLES_FRONTEND") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
        }
        else if (ievent=="HW_STALLED_CYCLES_BACKEND") 
        {
            type = PERF_TYPE_HARDWARE;
            config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
        }*/
        else if (ievent=="SW_CPU_CLOCK") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_CPU_CLOCK;
        }
        else if (ievent=="SW_TASK_CLOCK") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_TASK_CLOCK;
        }
        else if (ievent=="SW_PAGE_FAULTS") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_PAGE_FAULTS;
        }
        else if (ievent=="SW_CONTEXT_SWITCHES") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_CONTEXT_SWITCHES;
        }
        else if (ievent=="SW_CPU_MIGRATIONS") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_CPU_MIGRATIONS;
        }
        else if (ievent=="SW_PAGE_FAULTS_MIN") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_PAGE_FAULTS_MIN;
        }
        else if (ievent=="SW_PAGE_FAULTS_MAJ") 
        {
            type = PERF_TYPE_SOFTWARE;
            config = PERF_COUNT_SW_PAGE_FAULTS_MAJ;
        }
        else if (ievent.substr(0,9)=="HW_CACHE_")
        {
            type = PERF_TYPE_HW_CACHE;

            unsigned long long cache_id=0;
            unsigned long long cache_op_id=0;
            unsigned long long cache_op_result_id=0;

            std::string hw,op,stat;
            size_t head,tail;

            head = 9;
            tail = ievent.find('_', head);
            hw = ievent.substr(head, tail-head);

            head = tail+1;
            tail = ievent.find('_', head);
            op = ievent.substr(head, tail-head);

            head = tail+1;
            tail = ievent.find('_', head);
            stat = ievent.substr(head, tail-head);

            // cache_id
            if (hw=="L1D")
                cache_id = PERF_COUNT_HW_CACHE_L1D;
            else if (hw=="L1I")
                cache_id = PERF_COUNT_HW_CACHE_L1I;
            else if (hw=="LL")
                cache_id = PERF_COUNT_HW_CACHE_LL;
            else if (hw=="DTLB")
                cache_id = PERF_COUNT_HW_CACHE_DTLB;
            else if (hw=="ITLB")
                cache_id = PERF_COUNT_HW_CACHE_ITLB;
            else if (hw=="BPU")
                cache_id = PERF_COUNT_HW_CACHE_BPU;
            else
                std::cerr<<"Wrong cache type: "<<hw<<std::endl;

            // cache_op_id
            if (op=="READ")
                cache_op_id = PERF_COUNT_HW_CACHE_OP_READ;
            else if (op=="WRITE")
                cache_op_id = PERF_COUNT_HW_CACHE_OP_WRITE;
            else if (op=="PREFETCH")
                cache_op_id = PERF_COUNT_HW_CACHE_OP_PREFETCH;
            else
                std::cerr<<"Wrong cache operation: "<<op<<std::endl;

            // cache_op_result_id
            if (stat=="ACCESS")
                cache_op_result_id = PERF_COUNT_HW_CACHE_RESULT_ACCESS;
            else if (stat=="MISS")
                cache_op_result_id = PERF_COUNT_HW_CACHE_RESULT_MISS;
            else
                std::cerr<<"Wrong cache stat: "<<stat<<std::endl;

            config = cache_id | (cache_op_id<<8) | (cache_op_result_id<<16);
        }
        else
        {
            std::cerr<<"Wrong event type: "<<ievent<<std::endl;
        }
    }

    std::vector<gBenchPerf_handler> _perf_vec;
    std::vector<std::string> _event_vec;
    std::vector<unsigned long long> _cnt_vec;
    std::vector<bool> _multiplexing_vec;
    bool exclude_user;
    bool exclude_kernel;
    bool exclude_idle;
    bool exclude_hv;
};

class gBenchPerf_multi
{
public:
    gBenchPerf_multi(unsigned threadnum, const gBenchPerf_event& rhs)
    {
        _perf_vec.resize(threadnum, rhs);
    }

    void open(unsigned tid, int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        if (tid >= _perf_vec.size()) return; 
        _perf_vec[tid].open(group_id, group_size);
    }
    void start(unsigned tid, int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        if (tid >= _perf_vec.size()) return; 
        _perf_vec[tid].start(group_id, group_size);
    }
    void stop(unsigned tid, int group_id=-1, unsigned group_size=DEFAULT_PERF_GRP_SZ)
    {
        if (tid >= _perf_vec.size()) return; 
        _perf_vec[tid].stop(group_id, group_size);
    }
    void print(void)
    {
        for (size_t c=0;c<_perf_vec[0].get_event_cnt();c++) 
        {
            std::cout<<_perf_vec[0].event_name(c)<<"\t==>\t";
            unsigned long long res=0;
            bool mux=false;
            for (size_t i=0;i<_perf_vec.size();i++) 
            {
                res += _perf_vec[i].event_counter(c);
                mux |= _perf_vec[i].event_mux(c);
            }
            std::cout<<res;
            if (mux) std::cout<<"\tMUX";
            std::cout<<std::endl;
        }
    }
protected:
    std::vector<gBenchPerf_event> _perf_vec;
};

#endif
