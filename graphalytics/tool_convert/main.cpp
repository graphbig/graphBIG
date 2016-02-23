//====== Graph Benchmark Suites ======//
// Convertor tool
// => this tool converts the ldbc-X data 
//     in graphalytics to the CSR data files
//     used in openG CSR graph APIs. 

#include <vector>
#include <string>
#include <fstream>
#include "common.h"
#include "def.h"
#include "openG.h"
#include <unordered_map>

using namespace std;

class vertex_property
{
public:
    vertex_property():value(0){}
    vertex_property(uint64_t x):value(x){}

    uint64_t value;
};
class edge_property
{
public:
    edge_property():value(0){}
    edge_property(uint64_t x):value(x){}

    uint64_t value;
};

typedef openG::extGraph<vertex_property, edge_property> graph_t;
typedef graph_t::vertex_iterator    vertex_iterator;
typedef graph_t::edge_iterator      edge_iterator;


unordered_map<uint64_t, uint64_t> idmap; // mapping from external id to internal id

//==============================================================//
void arg_init(argument_parser & arg)
{
    arg.add_arg("outpath","./","path for generated graph");
}
//==============================================================//

struct sort_pred 
{
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) 
    {
        return left.second < right.second;
    }
};

//==============================================================//
bool convert_vertices(string vfile, string outpath)
{
    ifstream ifs;
    ifs.open(vfile.c_str());

    if (!ifs.is_open())
        return false;


    uint64_t vid = 0;
    while(ifs.good())
    {
        uint64_t num;
        ifs>>num;
        if (ifs.eof()) break;
        idmap[num]=vid;
        vid++;
    }

    ifs.close();

    vector<uint64_t> vmap(idmap.size());
    for (unordered_map<uint64_t, uint64_t>::iterator iter=idmap.begin();
            iter!=idmap.end();iter++)
    {
        vmap[iter->second] = iter->first;
    }

    string vfn = outpath + "/vmap.csr";
    ofstream ofs;
    ofs.open(vfn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(vmap[0]), sizeof(uint64_t)*vmap.size());
    }
    ofs.close();

    cout<<"== "<<vmap.size()<<" vertex"<<endl;
    return true;
}
bool convert_edges(string efile, string outpath)
{
    vector<pair<uint64_t,uint64_t> > raw_edges;

    vector<uint64_t> verts,edges;
    vector<double> weights;
        
    // Load edges into in-mem buffer
    ifstream ifs;

    uint64_t filesize;
    ifs.open(efile.c_str(), std::ifstream::binary);
    if (!ifs.is_open()) return false;
    ifs.seekg(0, ifs.end);
    filesize = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.close();

    raw_edges.reserve(filesize / 10);
    weights.reserve(filesize / 10);

    ifs.open(efile.c_str());
    if (!ifs.is_open())
        return false;

    while(ifs.good())
    {
        string line;
        getline(ifs, line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        uint64_t src, dest;
        double weight=0.0;

        char * ptr = const_cast<char *>(line.c_str());

        src = strtoll(ptr, &ptr, 10);
        dest = strtoll(ptr, &ptr, 10);
        weight = strtod(ptr, &ptr);

        weights.push_back(weight);
        raw_edges.push_back(make_pair(idmap[src],idmap[dest]));
    }

    ifs.close();

    // prepare out-going edge file
    string vfn = outpath + "/verts_out.csr";
    string efn = outpath + "/edges_out.csr";
    string wfn = outpath + "/eweights.csr";

    verts.resize(idmap.size()+1, 0);
    edges.resize(raw_edges.size());

    verts[0] = 0;
    edges[0] = raw_edges[0].second;
    for (uint64_t i=1;i<raw_edges.size();i++)
    {
        edges[i]=raw_edges[i].second;
        if (raw_edges[i].first != raw_edges[i-1].first)
            verts[raw_edges[i].first] = i;
    }
    verts[idmap.size()] = edges.size();

    for (uint64_t i=1;i<verts.size();i++)
    {
        if (verts[i] == 0)
            verts[i] = verts[i-1];
    }

    ofstream ofs;
    ofs.open(vfn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(verts[0]), sizeof(uint64_t)*verts.size());
    }
    ofs.close();
    ofs.open(efn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(edges[0]), sizeof(uint64_t)*edges.size());
    }
    ofs.close();
    ofs.open(wfn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(weights[0]), sizeof(double)*weights.size());
    }
    ofs.close();


    // prepare incoming edge file
    vfn = outpath + "/verts_in.csr";
    efn = outpath + "/edges_in.csr";

    std::sort(raw_edges.begin(),raw_edges.end(),sort_pred());
    verts.clear();
    verts.resize(idmap.size()+1, 0);

    verts[0] = 0;
    edges[0] = raw_edges[0].first;
    for (uint64_t i=1;i<raw_edges.size();i++)
    {
        edges[i]=raw_edges[i].first;
        if (raw_edges[i].second != raw_edges[i-1].second)
            verts[raw_edges[i].second] = i;
    }
    verts[idmap.size()] = edges.size();
    
    for (uint64_t i=1;i<verts.size();i++)
    {
        if (verts[i] == 0)
            verts[i] = verts[i-1];
    }

    ofs.open(vfn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(verts[0]), sizeof(uint64_t)*verts.size());
    }
    ofs.close();
    ofs.open(efn.c_str());
    if (ofs.is_open())
    {
        ofs.write((char*)&(edges[0]), sizeof(uint64_t)*edges.size());
    }
    ofs.close();

    cout<<"== "<<edges.size()<<" edge"<<endl;
    return true;
}
//==============================================================//
int main(int argc, char * argv[])
{
    graphBIG::print();
    cout<<"Tool: CSR data generation\n";

    argument_parser arg;
    arg_init(arg);
#ifndef NO_PERF    
    gBenchPerf_event perf;
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
#else
    if (arg.parse(argc,argv,NULL,false)==false)
    {
        arg.help();
        return -1;
    }
#endif

    string path, separator, outpath;
    arg.get_value("dataset",path);
    arg.get_value("outpath",outpath);

    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";


    convert_vertices(vfile, outpath);
    convert_edges(efile, outpath);

    t2 = timer::get_usec();

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

