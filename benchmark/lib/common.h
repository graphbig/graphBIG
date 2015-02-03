#ifndef _UBENCH_COMMON_H
#define _UBENCH_COMMON_H

#include "def.h"
#include "perf.h"
#include <sys/time.h>

#include <stdio.h>
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
#include <unordered_map>
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
public:
    static void initialize(int argc, char* argv[], std::vector<std::string>& arguments)
    {
        arguments.clear();
        for (int i=0;i<argc;i++) 
            arguments.push_back(std::string(argv[i]));
    }
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
