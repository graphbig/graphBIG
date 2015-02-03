#include <iostream>
#include <random>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cout<<"Wrong arguments\n";
        cout<<"Usage: ./randgen <max> <num>"<<endl;
        return -1;
    }

    uint64_t min = 0;
    uint64_t max = atoll(argv[1]);
    uint64_t num = atoll(argv[2]);

    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(min,max-1);

    cout<<"## NUM-"<<num<<" MAX-"<<max<<endl;

    for (uint64_t i=0;i<num;i++)
    {
        uint64_t outcome = distribution(generator);
        cout<<outcome<<"\n";
    }
    return 0;
}


