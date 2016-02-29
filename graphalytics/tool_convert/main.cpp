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
    arg.add_arg("undirected","0","input raw dataset is undirected graph");
    arg.add_arg("weight","0","edge file has edge weights");
}
//==============================================================//
typedef struct ebuf
{
    uint64_t src = 0;
    uint64_t dest = 0;
    double weight = 0.0;
    ebuf(uint64_t s, uint64_t d, double w)
    {
        src = s;
        dest = d;
        weight = w;
    }
    ebuf(const ebuf& other)
    {
        src = other.src;
        dest = other.dest;
        weight = other.weight;
    }
    ebuf& operator=(const ebuf& other)
    {
        src = other.src;
        dest = other.dest;
        weight = other.weight;
        return *this;
    }
}ebuf_t;

struct sort_src 
{
    bool operator()(const ebuf_t &left, const ebuf_t &right) 
    {
        if (left.src < right.src)
            return true;
        else if (left.src > right.src)
            return false;
        else if (left.dest < right.dest)
            return true;
        else
            return false;
    }
};
struct sort_dest
{
    bool operator()(const ebuf_t &left, const ebuf_t &right) 
    {
        if (left.dest < right.dest)
            return true;
        else if (left.dest > right.dest)
            return false;
        else if (left.src < right.src)
            return true;
        else
            return false;
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

    
bool convert_edges(string efile, string outpath, bool undirected=false,
        bool enable_edge_weight=true)
{
    vector<ebuf_t> raw_edges;

    vector<uint64_t> verts,edges;
    vector<double> weights;
        
    ifstream ifs;

    // estimate edge number and reserve mem space
    uint64_t filesize;
    ifs.open(efile.c_str(), std::ifstream::binary);
    if (!ifs.is_open()) return false;
    ifs.seekg(0, ifs.end);
    filesize = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    ifs.close();

    if (undirected)
        raw_edges.reserve(filesize / 5);
    else
        raw_edges.reserve(filesize / 10);


    // loading edges into in-mem buffer
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
        double w=0.0;

        char * ptr = const_cast<char *>(line.c_str());

        src = strtoll(ptr, &ptr, 10);
        dest = strtoll(ptr, &ptr, 10);
        if (enable_edge_weight)
            w = strtod(ptr, &ptr);
        
        raw_edges.push_back(ebuf_t(idmap[src],idmap[dest], w));
        if (undirected) raw_edges.push_back(ebuf_t(idmap[dest],idmap[src],w));
    }

    ifs.close();

    // prepare out-going edge file
    string vfn = outpath + "/verts_out.csr";
    string efn = outpath + "/edges_out.csr";
    string wfn = outpath + "/eweights.csr";


    std::sort(raw_edges.begin(), raw_edges.end(), sort_src());

    verts.clear();
    verts.resize(idmap.size()+1, 0);
    edges.resize(raw_edges.size());
    if (enable_edge_weight)
        weights.resize(raw_edges.size());

    verts[0] = 0;
    edges[0] = raw_edges[0].dest;
    if (enable_edge_weight) weights[0] = raw_edges[0].weight;
    for (uint64_t i=1;i<raw_edges.size();i++)
    {
        edges[i]=raw_edges[i].dest;
        if (enable_edge_weight) 
            weights[i] = raw_edges[i].weight;
        if (raw_edges[i].src != raw_edges[i-1].src)
            verts[raw_edges[i].src] = i;
    }
    verts[idmap.size()] = edges.size();

    for (uint64_t i=1;i<verts.size();i++)
    {
        if (verts[i] == 0)
            verts[i] = verts[i+1];
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
    if (enable_edge_weight)
    {
        ofs.open(wfn.c_str());
        if (ofs.is_open())
        {
            ofs.write((char*)&(weights[0]), sizeof(double)*weights.size());
        }
        ofs.close();
    }

    // prepare incoming edge file
    vfn = outpath + "/verts_in.csr";
    efn = outpath + "/edges_in.csr";

    std::sort(raw_edges.begin(),raw_edges.end(),sort_dest());

    verts.clear();
    verts.resize(idmap.size()+1, 0);

    verts[0] = 0;
    edges[0] = raw_edges[0].src;
    for (uint64_t i=1;i<raw_edges.size();i++)
    {
        edges[i]=raw_edges[i].src;
        if (raw_edges[i].dest != raw_edges[i-1].dest)
            verts[raw_edges[i].dest] = i;
    }
    verts[idmap.size()] = edges.size();
    
    for (uint64_t i=1;i<verts.size();i++)
    {
        if (verts[i] == 0)
            verts[i] = verts[i+1];
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

    bool undirected, weight;
    arg.get_value("undirected", undirected);
    arg.get_value("weight", weight);

    graph_t g;
    double t1, t2;
    
    cout<<"loading data... \n";

    t1 = timer::get_usec();
    string vfile = path + "/vertex.csv";
    string efile = path + "/edge.csv";


    convert_vertices(vfile, outpath);
    convert_edges(efile, outpath, undirected, weight);

    t2 = timer::get_usec();

#ifndef ENABLE_VERIFY
    cout<<"== time: "<<t2-t1<<" sec\n";
#endif

    cout<<"==================================================================\n";
    return 0;
}  // end main

