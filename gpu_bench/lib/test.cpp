#include "common.h"
#include "def.h"
#include "algo.h"
#include "perf.h"
using namespace std;


int main(int argc, char * argv[])
{
    vector<string> inputarg;
    argument_parser::initialize(argc,argv,inputarg);
    gBenchPerf_event perf(inputarg);

    //perf.open(false,false,false);
    perf.start();
    priority_heap<UINT64> test;

    for (size_t i=0;i<20;i++) 
    {
        test.insert(i, 100);
    }
    for (size_t i=0;i<20;i++) 
    {
        test.insert(i, i);
        size_t ret = test.erase_min();
        cout<<ret<<" ";
        test.print_all();
    }

    perf.stop();
    perf.print();
    cout<<"======================"<<endl;
    perf.start();

    size_t k;
    for (int i=0;i<10000;i++)
    {
        k = i*i - i;
    }

    perf.stop();
    perf.print();

    perf.start();

    for (int i=0;i<30000;i++)
    {
        k = i*i - i;
    }

    perf.stop();
    perf.print();

	
    return 0;
}
