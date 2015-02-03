#include <iostream>
#include <tr1/memory>

#include <vector>
#include <list>
#include "openG.h"

using namespace std;

/*
int main(int argc, char * argv[])                                       
{                                                                       
    std::tr1::shared_ptr<int> * A;                                      
    std::tr1::shared_ptr<int> * B;                                      
                                                                        
    A = new std::tr1::shared_ptr<int>;                                  
    B = new std::tr1::shared_ptr<int>;                                  
                                                                        
    int * p = new int;                                                  
    *p = 5;                                                             
    A->reset(p);                                                        
    B->reset(p);                                                        
                                                                        
    cout<<A->use_count()<<"  "<<B->use_count()<<" "<<*(A->get()) <<endl;
    delete B;                                                           
    cout<<A->use_count()<<"  "<<B->use_count()<<endl;                   
}                                                                       
*/



class data_t                                                      
{                                                                 
public:                                                           
    data_t(size_t a, double b):_id(a),val(b){}                    
                                                                  
    size_t id(){return _id;}                                      
                                                                  
    double val;                                                   
    size_t _id;                                                   
};                                                                
                                                                  
                                                                  
int main(int argc, char * argv[])                                 
{                                                                 
    openG::storage::indexed_vector_storage<data_t> A;             
                                                                  
    openG::storage::indexed_vector_storage<data_t>::iterator iter;
                                                                  
    for (size_t i=0;i<100;i++)                                    
    {                                                             
        data_t tmp(i, i*i/2.1);                                   
        A.push_back(tmp);                                         
    }                                                             
                                                                  
    for (size_t i=1;i<10;i++)                                     
    {                                                             
        A.erase(i*5);                                             
    }                                                             
                                                                  
    for (size_t i=100;i>0;i--)                                    
    {                                                             
        iter = A.find(i);                                         
        if (iter==A.end())                                        
            cout<<"cannot find: "<<i<<endl;                       
        else                                                      
            cout<<iter->id()<<"\t"<<(*iter).val<<endl;              
    }                                                             
                                                                  
    return 0;                                                     
}                                                                 


