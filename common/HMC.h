#ifndef _HMC_H
#define _HMC_H
#include <stdint.h>
#include <iostream>

extern "C" uint16_t  __attribute__ ((noinline)) HMC_CAS_greater_16B(uint16_t * ptr1, uint16_t newdata);
extern "C" uint16_t  __attribute__ ((noinline)) HMC_CAS_less_16B(uint16_t * ptr1, uint16_t newdata);
extern "C" uint16_t  __attribute__ ((noinline)) HMC_CAS_equal_16B(uint16_t * ptr1, uint16_t olddata, uint16_t newdata);
extern "C" uint16_t  __attribute__ ((noinline)) HMC_CAS_zero_16B(uint16_t * ptr1, uint16_t newdata);

extern "C" int16_t  __attribute__ ((noinline)) HMC_ADD_16B(int16_t *ptr1, int16_t newdata);

extern "C" float  __attribute__ ((noinline)) HMC_FP_ADD(float *ptr1, float newdata);


#endif
