#include "HMC.h"

uint16_t global_lock=0;

uint16_t  __attribute__ ((noinline)) HMC_CAS_greater_16B(uint16_t * ptr1,  uint16_t newdata)
{
    while(__sync_lock_test_and_set(&global_lock, 1));

    uint16_t olddata = *ptr1;
    if (olddata > newdata) 
        *ptr1 = newdata;

    __sync_lock_release(&global_lock);
    return olddata;
}


uint16_t  __attribute__ ((noinline)) HMC_CAS_less_16B(uint16_t * ptr1,  uint16_t newdata)
{
    while(__sync_lock_test_and_set(&global_lock, 1));

    uint16_t olddata = *ptr1;
    if (olddata < newdata) 
        *ptr1 = newdata;

    __sync_lock_release(&global_lock);
    return olddata;
}

uint16_t  __attribute__ ((noinline)) HMC_CAS_equal_16B(uint16_t * ptr1,  uint16_t olddata, uint16_t newdata)
{
    uint16_t data = *ptr1;
    return __sync_val_compare_and_swap(ptr1, olddata, newdata);
}

uint16_t  __attribute__ ((noinline)) HMC_CAS_zero_16B(uint16_t * ptr1,  uint16_t newdata)
{
    uint16_t data = *ptr1;
    return __sync_val_compare_and_swap(ptr1, 0, newdata);
}

int16_t  __attribute__ ((noinline)) HMC_ADD_16B(int16_t *ptr1, int16_t newdata)
{
    int16_t data = *ptr1;
    return __sync_fetch_and_add(ptr1, newdata);
}

float  __attribute__ ((noinline)) HMC_FP_ADD(float *ptr1, float newdata)
{
    while(__sync_lock_test_and_set(&global_lock, 1));

    float olddata = *ptr1;
    *ptr1 = newdata;

    __sync_lock_release(&global_lock);
    return olddata;
}

