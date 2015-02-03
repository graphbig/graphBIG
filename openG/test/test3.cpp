#include <iostream>
#include <tr1/memory>

#include <vector>
#include <list>
#include <set>
#include "openG.h"

using namespace std;


class prop
{
public:
    prop():A(0),B(0){}
    prop(size_t a, size_t b):A(a),B(b){}

    size_t A;
    size_t B;
};

typedef typename openG::extGraph<prop,prop>     graph_t;
typedef typename graph_t::vertex_iterator       vertex_iterator;
typedef typename graph_t::edge_iterator         edge_iterator;

#define VFILE "../../dataset/small/person.csv"
#define EFILE "../../dataset/small/person_knows_person.csv"

int main(int argc, char * argv[])
{
    

    graph_t g(openG::DIRECTED);

    cout<<"=============================================="<<endl;
    cout<<"Loading csv files: \n";
    cout<<"\t"<<VFILE<<endl;
    g.load_csv_vertices(VFILE, true, "|", 0);
    cout<<"\t"<<EFILE<<endl;
    g.load_csv_edges(EFILE, true, "|", 0, 1);
    cout<<"=============================================="<<endl;
    cout<<g.vertex_num()<<" vertices"<<endl;
    cout<<g.edge_num()<<" edges"<<endl;
    
    cout<<"=============================================="<<endl;
}


