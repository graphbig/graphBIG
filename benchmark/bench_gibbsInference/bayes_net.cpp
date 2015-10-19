/**************************************************************
 This code shows an example of using Gibbs sampling for
 Bayesian network approximate inference. The interfaces
 are similar to the Dlib inference tool, but built on
 System G native store. 

 verfied according to CSci 5512 UMN
 Gibbs Sampling for Approximate Inference in Bayesian Networks
***************************************************************/

#include <time.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include "common.h"
#include "def.h"
#include "openG.h"
#ifdef SIM
#include "SIM.h"
#endif
#define _ERROR_ID 99999
#define SEED 123

using namespace std;

class vertex_property
{
public:
    vertex_property() {}
    void set_node_num_values(unsigned num)
    {
        num_values = num;
        prob.resize(num, 0.0);
        value = rand()%num;
    }
    void set_parents_num_values(vector<unsigned> & num_vec)
    {
        pred_num_value = num_vec;
        unsigned ofs = num_values;
        for (unsigned i=0; i<pred_num_value.size(); i++)
        {
            ofs *= pred_num_value[i];
        }
        cpt.resize(ofs, 0.0);
    }
    void set_node_probability(unsigned node_value, vector<unsigned> & parents_value, double prob)
    {
        cpt[index_gen(node_value, parents_value)] = prob;
    }
    double get_node_probability(unsigned node_value, vector<unsigned> & parents_value)
    {
        return cpt[index_gen(node_value, parents_value)];
    }
    void set_node_value(unsigned ivalue)
    {
        if (ivalue >= num_values) return;
        value = ivalue;
    }
    double probability(unsigned ivalue)
    {
        if (ivalue >= num_values) return 0;
        return prob[ivalue];
    }
    //==== property values ====//
    unsigned num_values;
    vector<unsigned> pred_num_value;

    unsigned value;
    vector<double> cpt;
    vector<double> prob;
private:
    unsigned index_gen(unsigned node_value, vector<unsigned> & parents_value)
    {
        unsigned idx = node_value;
        unsigned ofs = num_values;

        if (parents_value.size()!=pred_num_value.size()) return 0;

        for(unsigned i=0; i<parents_value.size(); i++)
        {
            idx += parents_value[i]*ofs;
            ofs *= pred_num_value[i];
        }
        return idx;
    }
};

class edge_property
{
public:
    edge_property():value(0) {}

    char value;
};

typedef openG::extGraph<vertex_property,edge_property> graph_t;

typedef graph_t::vertex_iterator   vit_t;
typedef graph_t::edge_iterator     eit_t;
typedef graph_t::edge_iterator    peit_t;
typedef uint64_t                   vid_t;

//==============================================================//
void arg_init(argument_parser & arg)
{
    arg.add_arg("root","0","root/starting vertex");
    arg.add_arg("val","1","estimate value");
    arg.add_arg("iter","1000","iteration number");
}
//==============================================================//

void gibbs_sampler(graph_t * g, set<vid_t> & evidence_nodes)
{
    // update the probability for each vertex
    for (vit_t vit=g->vertices_begin(); vit!=g->vertices_end(); vit++)
    {
        vid_t vid = vit->id();
        if (evidence_nodes.find(vid) == evidence_nodes.end())    // not evidence
        {
            // Calculate p(x_i|pa(x_i))\prod_{y\in ch(x_i)}p(y|pa(y))
            for (size_t i=0; i<vit->property().num_values; i++)
            {
                vector<unsigned> pa;
                for (peit_t peit=vit->preds_begin(); peit!=vit->preds_end(); peit++)
                {
                    vid_t  svid  = peit->target();
                    vit_t  svit = g->find_vertex(svid);
                    pa.push_back(svit->property().value);
                }
                vit->property().prob[i] = vit->property().get_node_probability(i, pa);

                for (eit_t eit=vit->edges_begin(); eit!=vit->edges_end(); eit++)
                {
                    vid_t  tvid  = eit->target();
                    vit_t  tvit = g->find_vertex(tvid);
                    pa.clear();

                    for (peit_t tpeit=tvit->preds_begin(); tpeit!=tvit->preds_end(); tpeit++)
                    {
                        vid_t  stvid  = tpeit->target();
                        vit_t stvit = g->find_vertex(stvid);
                        if (stvid == vid)
                            pa.push_back(i);
                        else
                            pa.push_back(stvit->property().value);
                    }
                    vit->property().prob[i] *= tvit->property().get_node_probability(tvit->property().value, pa);
                }
            }

            // normalize the result
            double norm_factor = 0.0;
            for (size_t i=0; i<vit->property().num_values; i++)
                norm_factor += vit->property().prob[i];
            for (size_t i=0; i<vit->property().num_values; i++)
                vit->property().prob[i] /= norm_factor;
        }
    }

    // update the values (states) for each non-evidence node
    for (vit_t vit=g->vertices_begin(); vit!=g->vertices_end(); vit++)
    {
        vid_t vid = vit->id();
        if (evidence_nodes.find(vid) != evidence_nodes.end()) continue;

        double  p = ((double)rand() / (double)(RAND_MAX));
        double acc= 0;
        for (size_t i=0; i<vit->property().num_values; i++)
        {
            acc += vit->property().prob[i];
            if (p < acc)
            {
                vit->property().value = i;
                break;
            }
        }
    }
}

double gibbs_estimate(graph_t *g, set<vid_t> & evidence_nodes, uint64_t node, unsigned val, size_t m)
{
    size_t cnt = 0;
    for (size_t i=0; i<m; i++)
    {
        gibbs_sampler(g, evidence_nodes);
        vit_t vit = g->find_vertex(node);
        if (vit->property().value == val)
            cnt++;
    }

    return (double)cnt/(double)m;
}

void dsc_parser(
    string fname,
    map<string, unsigned>&  node2vid,
    map<unsigned, unsigned>& vid2numvalues,
    map<unsigned, vector<double> >& vid2cpt,
    vector<pair<unsigned, unsigned> >& edges
)

{
    ifstream ifs;
    ifs.open(fname.c_str());

    if (ifs.is_open()==false)
    {
        cout<<"[ERROR] cannot open dataset file: \""<<fname<<"\""<<endl;
        exit(-1);
    }

    unsigned vid_gen = 0;
    while(ifs.good())
    {
        string line;
        getline(ifs, line);

        if (line.empty()) continue;

        size_t pos;
        pos = line.find("node");
        if (pos != string::npos)
        {
            size_t head, tail;
            string element;
            unsigned vid;

            // get the node name
            head = line.find_first_of(" ", pos);
            head = line.find_first_not_of(" ", head);
            tail = line.find_first_of(" ", head);
            element = line.substr(head, tail-head);
            vid = vid_gen++;
            node2vid[element]=vid;

            // get the # of node values
            getline(ifs, line);
            head = line.find("[",0);
            head = line.find_first_not_of(" ", head+1);
            tail = line.find_first_of(" ", head);
            element = line.substr(head, tail-head);
            vid2numvalues[vid] = atol(element.c_str());

            do
            {
                getline(ifs, line);
            }
            while(line.empty()||line[0]!='}');
            continue;
        }

        pos = line.find("probability");
        if (pos != string::npos)
        {
            size_t head, tail, mid;
            string element;

            head = line.find("(",pos);
            head = line.find_first_not_of(" ", head+1);
            tail = line.find_first_of(")", head);
            element = line.substr(head, tail-head);

            mid = element.find("|", 0);
            // no conditional variables
            if (mid==string::npos)
            {
                string name;
                tail = element.find_first_of(" ");
                name = element.substr(0, tail);

                unsigned vid = node2vid[name];
                unsigned num = vid2numvalues[vid];
                getline(ifs, line);

                head = line.find_first_not_of(" ");
                for(unsigned i=0; i<num; i++)
                {
                    tail = line.find_first_of(",;", head);
                    string number = line.substr(head,tail-head);
                    double a = atof(number.c_str());
                    vid2cpt[vid].push_back(a);

                    head = line.find_first_not_of(" ", tail+1);
                }
                do
                {
                    getline(ifs, line);
                }
                while(line.empty()||line[0]!='}');

            }
            else
            {
                string dest_name, tmpstr;
                vector<string> src_names;
                unsigned dest_vid;

                tail = element.find_first_of(" |");
                dest_name = element.substr(0, tail);
                dest_vid = node2vid[dest_name];

                head = element.find_first_not_of(" ", mid+1);
                tail = element.find_last_not_of(" ");
                tmpstr = element.substr(head, tail-head+1);

                head = 0;
                // get conditional variables
                while(head!=string::npos)
                {
                    tail = tmpstr.find_first_of(" ,", head);
                    src_names.push_back(tmpstr.substr(head, tail-head));
                    head = tmpstr.find_first_not_of(" ,", tail);
                }

                unsigned linenum = 1;
                for (unsigned i=0; i<src_names.size(); i++)
                {
                    unsigned vid = node2vid[src_names[i]];
                    linenum *= vid2numvalues[vid];
                    edges.push_back(make_pair(vid, dest_vid));
                }

                // get probability values
                for (unsigned i=0; i<linenum; i++)
                {
                    getline(ifs, line);

                    head = line.find_first_of(":");
                    head = line.find_first_not_of(" ", head+1);

                    while(head!=string::npos)
                    {
                        tail = line.find_first_of(" ,;", head);
                        string str = line.substr(head, tail-head);
                        double num = atof(str.c_str());
                        vid2cpt[dest_vid].push_back(num);
                        head = line.find_first_not_of(" ,;", tail);
                    }
                }

                do
                {
                    getline(ifs, line);
                }
                while(line.empty()||line[0]!='}');

            }

            continue;
        }
    }


    ifs.close();
}
void load_graph(graph_t & g, string dataset, vector<string> & vid2node)
{
    map<string, unsigned>  node2vid;
    map<unsigned, unsigned> vid2numvalues;
    map<unsigned, vector<double> > vid2cpt;
    vector<pair<unsigned, unsigned> > edges;

    // parsing dataset file
    dsc_parser(dataset, node2vid, vid2numvalues, vid2cpt, edges);

    vid2node.resize(node2vid.size());
    for(map<string,unsigned>::iterator iter=node2vid.begin();
            iter!=node2vid.end(); iter++)
    {
        vid2node[iter->second] = iter->first;
    }

    // add vertices
    unsigned vertex_num = node2vid.size();
    for(unsigned i=0; i<vertex_num; i++)
        g.add_vertex();

    // add edges
    for (unsigned i=0; i<edges.size(); i++)
    {
        eit_t eit;
        g.add_edge(edges[i].first, edges[i].second, eit);
    }

    // vertex prop: num_values
    for (unsigned i=0; i<vertex_num; i++)
    {
        vit_t vit = g.find_vertex(i);
        vit->property().set_node_num_values(vid2numvalues[i]);
    }

    // vertex prop: parents_num_values
    for (unsigned i=0; i<vertex_num; i++)
    {
        vit_t vit = g.find_vertex(i);
        vector<unsigned> pred_num;
        for (eit_t pit = vit->preds_begin(); pit!=vit->preds_end(); pit++)
        {
            vit_t iter = g.find_vertex(pit->target());
            pred_num.push_back(iter->property().num_values);
        }
        vit->property().set_parents_num_values(pred_num);
    }

    // vertex prop: cpt
    for (unsigned i=0; i<vertex_num; i++)
    {
        vit_t vit = g.find_vertex(i);
        vit->property().cpt = vid2cpt[i];
    }

    // TODO
    // set node initial state value
    // [now] it is generated randomly
}

int main(int argc, char* argv[])
{
    graphBIG::print();
    cout<<"Benchmark: Gibbs Inference\n";

    argument_parser arg;
    gBenchPerf_event perf;
    arg_init(arg);
    if (arg.parse(argc,argv,perf,false)==false)
    {
        arg.help();
        return -1;
    }
    string path;
    arg.get_value("dataset",path);

    size_t root,value,iteration;
    arg.get_value("root",root);
    arg.get_value("val",value);
    arg.get_value("iter",iteration);

    set<vid_t> evidence_nodes;
    vector<string> vid2node;
    double result;

    unsigned run_num = ceil(perf.get_event_cnt() /(double) DEFAULT_PERF_GRP_SZ);
    if (run_num==0) run_num = 1;

    for (unsigned i=0; i<run_num; i++)
    {
        cout<<"\nRun #"<<i<<endl;

        graph_t g;
        srand(SEED); // use the same seed for each run

        double t1, t2;
        cout<<"loading data... \n";
        t1 = timer::get_usec();

        vid2node.clear();
        load_graph(g, path, vid2node);

        size_t vertex_num = g.vertex_num();
        size_t edge_num = g.edge_num();
        t2 = timer::get_usec();
        cout<<"== "<<vertex_num<<" vertices  "<<edge_num<<" edges\n";
#ifndef ENABLE_VERIFY
        cout<<"== time: "<<t2-t1<<" sec\n";
#endif


        // set evidence nodes randomly
        evidence_nodes.clear();
        unsigned evidnum = ceil(vertex_num/(double)4);
        for (unsigned i=0; i<evidnum; i++)
            evidence_nodes.insert(rand()%vertex_num);

        t1 = timer::get_usec();
        perf.open(i);
        perf.start(i);
#ifdef SIM
    SIM_BEGIN(true);
#endif
        result = gibbs_estimate(&g, evidence_nodes, root, value, iteration);
#ifdef SIM
    SIM_END(true);
#endif
        perf.stop(i);
        t2 = timer::get_usec();

        cout<<"== Gibbs Inference Finish"<<endl;

#ifndef ENABLE_VERIFY
        cout<<"== time: "<<t2-t1<<" sec\n";
#endif
    }
    // perform Gibbs sampling for 2000 steps
    cout << "\nEstimate by "<< iteration <<" steps Gibbs sampling:\n";
    cout << "p("<<vid2node[root]<<"="<<value<<"|";
    for(set<vid_t>::iterator iter=evidence_nodes.begin(); iter!=evidence_nodes.end(); iter++)
    {
        if (iter!=evidence_nodes.begin())
            cout<<", ";
        cout<<vid2node[*iter];
    }
    cout <<") = "<< result << endl;
#ifndef ENABLE_VERIFY
    perf.print();
#endif
    return 0;
}

