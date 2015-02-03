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

typedef typename openG::Graph<prop,prop>    graph_t;
typedef typename graph_t::vertex_iterator   vertex_iterator;
typedef typename graph_t::edge_iterator     edge_iterator;

size_t NV=20;
size_t NE=200;

vector<multiset<uint64_t> > out_edges;
vector<multiset<uint64_t> > in_edges;

vector<pair<uint64_t,uint64_t> > edges;

void gen_directed_edges(void)
{
    out_edges.resize(NV);
    in_edges.resize(NV);

    for (size_t i=0;i<NE;i++)
    {
        uint64_t src = rand()%NV;
        uint64_t dest = rand()%NV;

        edges.push_back(make_pair(src,dest));
        out_edges[src].insert(dest);
        in_edges[dest].insert(src);
    }
}

void gen_undirected_edges(void)
{
    out_edges.clear();
    in_edges.clear();
    edges.clear();

    out_edges.resize(NV);
    in_edges.resize(NV);

    for (size_t i=0;i<NE;i++)
    {
        uint64_t src = rand()%NV;
        uint64_t dest = rand()%NV;

        edges.push_back(make_pair(src,dest));
        out_edges[src].insert(dest);
        out_edges[dest].insert(src);
    }
}

void ref_delete_vertex(uint64_t vid)
{
    for (size_t i=0;i<out_edges.size();i++)
    {
        if (i==vid) 
        {
            out_edges[vid].clear();
            in_edges[vid].clear();
            continue;
        }

        out_edges[i].erase(vid);
        in_edges[i].erase(vid);
    }
}

int main(int argc, char * argv[])
{
    

    graph_t g(openG::DIRECTED);

    cout<<"=============================================="<<endl;
    for (size_t i=0;i<NV;i++)
    {
        vertex_iterator vit = g.add_vertex();
        vit->set_property(prop(1,vit->id()));
    }

    gen_directed_edges();
    
    for (size_t i=0;i<edges.size();i++)
    {
        uint64_t src = edges[i].first;
        uint64_t dest = edges[i].second;
        edge_iterator eit;
        bool ret = g.add_edge(src,dest,eit);
        
        if (ret) 
            eit->set_property(prop(src,dest));
        else 
            cout<<"Add Edge Fail: "<<src<<"-"<<dest<<endl;
    }
    cout<<"Graph Construction: "<<NV<<" vertices\t"<<edges.size()<<" edges"<<endl;
    cout<<"=============================================="<<endl;
    for (int i=0;i<10;i++) 
    {
        uint64_t vid = rand()%NV;
        ref_delete_vertex(vid);
        g.delete_vertex(vid);
    }
    cout<<"Delete Vertex: 10"<<endl;
    cout<<"=============================================="<<endl;
    vertex_iterator vit;
    for (vit=g.vertices_begin();vit!=g.vertices_end();vit++)
    {
        // check vertex property
        if (vit->property().A!=1 || vit->property().B!=vit->id()) 
        {
            cout<<"vertex property mismatch\tvertex: "<<vit->id()<<endl;
            break;
        }
        // check in_edge/out_edge number
        if (vit->in_edges_size() != in_edges[vit->id()].size())
        {
            cout<<"in_edges_size mismatch\tvertex: "<<vit->id()<<endl;
            break;
        }
        if (vit->out_edges_size() != out_edges[vit->id()].size())
        {
            cout<<"out_edges_size mismatch\tvertex: "<<vit->id()<<endl;
            break;
        }
        
        bool propvalid=true;
        // check edges
        multiset<uint64_t> tmp;
        for (edge_iterator eit=vit->out_edges_begin();eit!=vit->out_edges_end();eit++)
        {
            tmp.insert(eit->target());
            // check edge property
            pair<uint64_t, uint64_t> prop = make_pair(eit->property().A,eit->property().B);
            pair<uint64_t, uint64_t> e1 = make_pair(vit->id(),eit->target());
            pair<uint64_t, uint64_t> e2 = make_pair(eit->target(),vit->id());
            if (prop!=e1 && prop!=e2) 
            {
                cout<<"edge property mismatch\tsrc: "<<vit->id()<<" dest: "<<eit->target()<<endl;
                propvalid=false;
                break;
            }
        }
        if (!propvalid) break;
        if (tmp != out_edges[vit->id()])
        {
            cout<<"out_edges mismatch\tvertex: "<<vit->id()<<endl;
            break;
        }
        tmp.clear();
        for (edge_iterator eit=vit->in_edges_begin();eit!=vit->in_edges_end();eit++)
        {
            tmp.insert(eit->target());
            // check edge property
            pair<uint64_t, uint64_t> prop = make_pair(eit->property().A,eit->property().B);
            pair<uint64_t, uint64_t> e1 = make_pair(vit->id(),eit->target());
            pair<uint64_t, uint64_t> e2 = make_pair(eit->target(),vit->id());
            if (prop!=e1 && prop!=e2) 
            {
                cout<<"edge property mismatch\tsrc: "<<vit->id()<<" dest: "<<eit->target()<<endl;
                propvalid=false;
                break;
            }
        }
        if (!propvalid) break;
        if (tmp != in_edges[vit->id()])
        {
            cout<<"in_edges mismatch\tvertex: "<<vit->id()<<endl;
            break;
        }
    }
    if (vit == g.vertices_end())
        cout<<"Verification PASS"<<endl;
    else
        cout<<"Verification FAIL"<<endl;
    cout<<"=============================================="<<endl;


    

}


