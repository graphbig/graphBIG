#include "SIM.h"


unsigned __attribute__ ((noinline)) SIM_BEGIN(bool i)
{
    std::cout<<"sim begin\n";
    if (i)
        return 1;
    else
        0;
}
unsigned __attribute__ ((noinline)) SIM_END(bool i)
{
    std::cout<<"sim end\n";
    if (i) 
        return 1;
    else
        return 0;
} 


