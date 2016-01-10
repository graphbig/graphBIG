#ifndef _UBENCH_COMMON_H
#define _UBENCH_COMMON_H

#include "def.h"
#ifndef NO_PERF
#include "perf.h"
#endif
#include <sys/time.h>
#include <math.h>
#include <stdio.h>
#include <limits>
#include <cstring>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <map>
#include <tr1/unordered_map>
#include <set>



#if defined TRAITS_LL
#define TRAITS_TYPE adjlist_list_list_traits

#define TRAITS_D adjlist_list_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::DIRECTED>
#define TRAITS_UD adjlist_list_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::UNDIRECTED>

#elif defined TRAITS_VV
#define TRAITS_TYPE adjlist_vector_vector_traits

#define TRAITS_D adjlist_vector_vector_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::DIRECTED>
#define TRAITS_UD adjlist_vector_vector_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::UNDIRECTED>

#elif defined TRAITS_VL
#define TRAITS_TYPE adjlist_vector_list_traits

#define TRAITS_D adjlist_vector_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::DIRECTED>
#define TRAITS_UD adjlist_vector_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::UNDIRECTED>

#elif defined TRAITS_LV
#define TRAITS_TYPE adjlist_list_vector_traits

#define TRAITS_D adjlist_list_vector_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::DIRECTED>
#define TRAITS_UD adjlist_list_vector_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::UNDIRECTED>

#else
#define TRAITS_TYPE adjlist_list_list_traits

#define TRAITS_D adjlist_list_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::DIRECTED>
#define TRAITS_UD adjlist_list_list_traits<ldbc_vertex_property,ldbc_edge_property,ibmppl::UNDIRECTED>

#endif

//================================================================//
class argument_parser
{
    struct arg_t
    {
        std::string info;
        std::string value;
        bool has_value;
        arg_t(){}
        arg_t(std::string _info, std::string _value, bool _has_value):
            info(_info),value(_value),has_value(_has_value){}
    };
public:
    argument_parser()
    {
        add_arg("dataset", "../../dataset/small", "path of dataset files");
        add_arg("separator", "|,", "separators of csv dataset files");
        add_arg("threadnum", "1", "thread number");
        add_arg("help", "0", "print help info", false);
#ifdef SIM
        add_arg("beginiter","0","sim begin iteration #");
        add_arg("enditer","0","sim end iteration # (0-sim till the end)");
#endif        
    }

    void add_arg(std::string name, std::string _default, std::string info, bool has_value=true)
    {
        _arg[name].info = info;
        _arg[name].value = _default;
        _arg[name].has_value = has_value;
    }
    bool parse(int argc, char* argv[], bool is_open=false)
    {
            std::vector<std::string> arguments;
            for (int i=0;i<argc;i++) 
                arguments.push_back(std::string(argv[i]));

            for (unsigned i=1;i<arguments.size();i++)
            {
                std::string name = arguments[i];
                if (name.size()<=2 || name.substr(0,2)!=std::string("--"))
                {
                    std::cout<<"[ERROR] argument has to start with \'--\'"<<std::endl;
                    return false;
                }

                name = name.substr(2);
                if (_arg.find(name)==_arg.end())
                {
                    std::cout<<"[ERROR] unkown argument: --"<<name<<std::endl;
                    return false;
                }
                else if (_arg[name].has_value)
                {
                    i++;
                    _arg[name].value = arguments[i];
                }
                else
                {
                    _arg[name].value = std::string("1");
                }
            } 

            if (_arg["help"].value!=std::string("0"))
                return false;
            return true;
    }
#ifndef NO_PERF    
    bool parse(int argc, char* argv[], gBenchPerf_event & perf, bool is_open=false)
    {
        std::vector<std::string> arguments;
        for (int i=0;i<argc;i++) 
            arguments.push_back(std::string(argv[i]));

        gBenchPerf_event tmp(arguments, is_open);
        perf = tmp;

        for (unsigned i=1;i<arguments.size();i++)
        {
            std::string name = arguments[i];
            if (name.size()<=2 || name.substr(0,2)!=std::string("--"))
            {
                std::cout<<"[ERROR] argument has to start with \'--\'"<<std::endl;
                return false;
            }

            name = name.substr(2);
            if (_arg.find(name)==_arg.end())
            {
                std::cout<<"[ERROR] unkown argument: --"<<name<<std::endl;
                return false;
            }
            else if (_arg[name].has_value)
            {
                i++;
                _arg[name].value = arguments[i];
            }
            else
            {
                _arg[name].value = std::string("1");
            }
        } 

        if (_arg["help"].value!=std::string("0"))
            return false;
        return true;
    }
#endif    
    void help(void)
    {
        std::cout<<"[Usage]:"<<std::endl;
        std::map<std::string, struct arg_t>::iterator iter;
        for (iter=_arg.begin();iter!=_arg.end();iter++)
        {
            std::cout<<"--"<<iter->first<<":\t"<<iter->second.info<<std::endl;
            if (iter->second.has_value)
                std::cout<<"\t\tdefault - \'"<<iter->second.value<<"\'"<<std::endl;
        }
    }

    bool get_value(std::string name, std::string & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value.clear();
            return false;
        }
        value = _arg[name].value;
        return true;
    }
    bool get_value(std::string name, double & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value = 0.0;
            return false;
        }
        value = atof(_arg[name].value.c_str());
        return true;
    } 
    bool get_value(std::string name, size_t & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value = 0;
            return false;
        }
        value = atol(_arg[name].value.c_str());
        return true;
    } 
    bool get_value(std::string name, unsigned & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value = 0;
            return false;
        }
        value = atoi(_arg[name].value.c_str());
        return true;
    } 
    bool get_value(std::string name, int & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value = 0;
            return false;
        }
        value = atoi(_arg[name].value.c_str());
        return true;
    }
    bool get_value(std::string name, bool & value)
    {
        if (_arg.find(name)==_arg.end())
        {
            value = false;
            return false;
        }
        int num = atoi(_arg[name].value.c_str());
        value = (num==0)? false : true;
        return true;
    } 
private:
    std::map<std::string, struct arg_t> _arg;
};
//================================================================//
class graphBIG
{
public:
    static void print(void)
    {
        std::cout<<"=================================================================="<<std::endl;
        std::cout<<"   ________                    .__   __________.___  ________ \n";
        std::cout<<"  /  _____/___________  ______ |  |__\\______   \\   |/  _____/ \n";
        std::cout<<" /   \\  __\\_  __ \\__  \\ \\____ \\|  |  \\|    |  _/   /   \\  ___ \n";
        std::cout<<" \\    \\_\\  \\  | \\// __ \\|  |_> >   Y  \\    |   \\   \\    \\_\\  \\\n";
        std::cout<<"  \\______  /__|  (____  /   __/|___|  /______  /___|\\______  /\n";
        std::cout<<"         \\/           \\/|__|        \\/       \\/            \\/ \n";
        std::cout<<"                                                                 "<<std::endl;
        std::cout<<"=================================================================="<<std::endl;
    }
};
//================================================================//
class timer
{
public:
    static double get_usec()
    {
        timeval tim;
        gettimeofday(&tim, NULL);
        return tim.tv_sec+(tim.tv_usec/1000000.0);
    } 
};
//================================================================//
// Performance Counter
#define DEFAULT_EVENT "HW_INSTRUCTIONS:CACHE_L1D_READ_ACCESS:CACHE_L1D_READ_MISS:CACHE_LL_READ_ACCESS:CACHE_LL_READ_MISS"

//================================================================//

class thread_utility
{
public:
    static void get_cpuinfo(size_t & smt_cnt, size_t & core_cnt)
    {
        std::ifstream ifs;

        ifs.open("/proc/cpuinfo");
        if (!ifs.is_open()) 
        {
            std::cerr << "can not open /proc/cpuinfo" << std::endl;
            return;
        }

        size_t proc_cnt = 0;
        core_cnt = 0;
        while (ifs.good()) 
        {
            std::string line;
            getline(ifs,line);
            if (line.empty()) continue;

            if (line.size() > 10 && line.substr(0,9)=="processor") proc_cnt++;
            if (core_cnt==0 && line.size() > 9 && line.substr(0,8)=="siblings") 
            {
                core_cnt = atoi(line.substr(10).c_str());
            }
        }

        smt_cnt = proc_cnt / core_cnt;

        ifs.close();
    }
    // bind caller thread to a core
    // smt_cnt: # of smt threads per core
    // core_cnt: # of cores
    static void thread_bind(size_t thread_id, size_t smt_cnt, size_t core_cnt)
    {
        size_t tid;
        cpu_set_t mask;
        CPU_ZERO(&mask);
#ifndef ENABLE_SMT
        tid = ((thread_id%core_cnt)*smt_cnt) + ((thread_id/core_cnt)%smt_cnt);
#else
        tid = thread_id;
#endif
        CPU_SET(tid, &mask);
        if (sched_setaffinity (0, sizeof(cpu_set_t), &mask) != 0)
        {
            std::cerr<<"ERROR while binding thread-"<<thread_id<<"\n";
            exit(-1);
        }
        std::cout<<"binding thread-"<<thread_id<<" with core-"<<tid<<std::endl<<std::flush;
    }
};


#endif
