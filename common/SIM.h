#ifndef _SIM_H
#define _SIM_H
#include <stdint.h>
#include <iostream>

extern "C" unsigned __attribute__ ((noinline)) SIM_BEGIN(bool i);
extern "C" unsigned __attribute__ ((noinline)) SIM_END(bool i); 

#endif
