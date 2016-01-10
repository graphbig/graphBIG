#ifndef OPENG_GRAPH_H
#define OPENG_GRAPH_H

#include <tr1/unordered_map>

#include <iterator>
#include <list>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "openG_storage.h"
//#include "openG_property.h"
#ifdef SIM
#include "SIM.h"
#endif
namespace openG
{

#ifdef SIM
#define LOCKSZ 2000000
#endif

enum Directness_t
{
    DIRECTED=0,
    UNDIRECTED=1
};


template<class EPROP>
class edge
{
public:
    typedef EPROP eproperty_t;
    typedef typename std::tr1::shared_ptr<EPROP>  shared_eproperty_t;

    edge(uint64_t eid, uint64_t vid):_eid(eid),_vid(vid){}
    edge(uint64_t eid, uint64_t vid, const eproperty_t& prop):_eid(eid),_vid(vid)
    {
        eproperty_t * ptr = new eproperty_t;
        *ptr = prop;
        _eproperty.reset(ptr);
    }
    edge(uint64_t eid, uint64_t vid, shared_eproperty_t& prop):_eid(eid),_vid(vid){_eproperty=prop;}

    uint64_t id(void)
    {
        return _eid;
    }
    uint64_t target(void)
    {
        return _vid;
    }
    eproperty_t & property(void)
    {
        assert(_eproperty.use_count()>0);
        return *(_eproperty.get());
    }

    void set_property(const eproperty_t & prop)
    {
        if (_eproperty.use_count()==0) 
        {
            eproperty_t * ptr = new eproperty_t;
            _eproperty.reset(ptr);
        }
        eproperty_t * ptr = _eproperty.get();
        *ptr = prop;
    }
    void set_property(shared_eproperty_t& prop)
    {
        _eproperty = prop;
    }

    bool has_property(void)
    {
        return (_eproperty.use_count()>0);
    }
    shared_eproperty_t& shared_property(void)
    {
        return _eproperty;
    }
protected:
    uint64_t _eid; // edge id
    uint64_t _vid; // id of source/target vertex
    shared_eproperty_t _eproperty; // edge property
};

template<class VPROP, class EDGE, class edgelist_t>
class vertex
{
public:
    typedef VPROP vproperty_t;
    typedef typename EDGE::eproperty_t eproperty_t; 
    typedef typename EDGE::shared_eproperty_t shared_eproperty_t;
    typedef typename edgelist_t::iterator edge_iterator;

    vertex(uint64_t vid):_id(vid){}
    vertex(uint64_t vid, const vproperty_t & vp):_id(vid),_vproperty(vp){}

    uint64_t id()
    {
        return _id;
    }

    vproperty_t & property(void)
    {
        return _vproperty;
    }

    void set_property(const vproperty_t& vprop)
    {
        _vproperty = vprop;
    }

    //================= Traverse Edge =================//
    edge_iterator in_edges_begin(void){return in_edges.begin();}
    edge_iterator in_edges_end(void){return in_edges.end();}

    edge_iterator out_edges_begin(void){return out_edges.begin();}
    edge_iterator out_edges_end(void){return out_edges.end();}

    uint64_t in_edges_size(void){return in_edges.size();}
    uint64_t out_edges_size(void){return out_edges.size();}

    // for compatibility with IBM SystemG internal version
    edge_iterator edges_begin(void){return out_edges_begin();}
    edge_iterator edges_end(void){return out_edges_end();}
    uint64_t edges_size(void){return out_edges_size();}

    edge_iterator preds_begin(void){return in_edges_begin();}
    edge_iterator preds_end(void){return in_edges_end();}
    uint64_t preds_size(void){return in_edges_size();}
    //================= Add Edge =================//
    edge_iterator add_in_edge(uint64_t eid, uint64_t vid)
    {
        in_edges.push_back(EDGE(eid,vid));
        edge_iterator iter = in_edges.end();
        iter--;
        return iter;
    }
    edge_iterator add_out_edge(uint64_t eid, uint64_t vid)
    {
        out_edges.push_back(EDGE(eid,vid));
        edge_iterator iter = out_edges.end();
        iter--;
        return iter;
    }
    edge_iterator add_in_edge(uint64_t eid, uint64_t vid, const eproperty_t& ep)
    {
        in_edges.push_back(EDGE(eid,vid));
        edge_iterator iter = in_edges.end();
        iter--;
        iter->set_property(ep);
        return iter;
    }
    edge_iterator add_out_edge(uint64_t eid, uint64_t vid, const eproperty_t& ep)
    {
        out_edges.push_back(EDGE(eid,vid));
        edge_iterator iter = out_edges.end();
        iter--;
        iter->set_property(ep);
        return iter;
    }
    edge_iterator add_in_edge(uint64_t eid, uint64_t vid, shared_eproperty_t& eprop)
    {
        in_edges.push_back(EDGE(eid,vid,eprop));
        edge_iterator iter = in_edges.end();
        iter--;
        return iter;
    }
    edge_iterator add_out_edge(uint64_t eid, uint64_t vid, shared_eproperty_t& eprop)
    {
        out_edges.push_back(EDGE(eid,vid,eprop));
        edge_iterator iter = out_edges.end();
        iter--;
        return iter;
    }

    //================= Delete Edge =================//

    // delete edge with given eid
    edge_iterator delete_in_edge(uint64_t eid)
    {
        return in_edges.erase(eid);
    }
    edge_iterator delete_out_edge(uint64_t eid)
    {
        return out_edges.erase(eid);
    }

    // delete edge with given iterator
    edge_iterator delete_in_edge(edge_iterator eit)
    {
        return in_edges.erase(eit);
    }
    edge_iterator delete_out_edge(edge_iterator eit)
    {
        return out_edges.erase(eit);
    }

    // delete all edges with given source/destination vertex id
    uint64_t delete_in_edge_v(uint64_t vid)
    {
        uint64_t ret=0;
        edge_iterator iter = in_edges.begin();
        while (iter != in_edges.end()) 
        {
            if (iter->target()!=vid) 
            {
                iter++;
                continue;
            }

            iter = in_edges.erase(iter);
            ret++;
        }
        return ret;
    }
    uint64_t delete_out_edge_v(uint64_t vid)
    {
        uint64_t ret=0;
        edge_iterator iter = out_edges.begin();
        while (iter != out_edges.end()) 
        {
            if (iter->target()!=vid) 
            {
                iter++;
                continue;
            }

            iter = out_edges.erase(iter);
            ret++;
        }
        return ret;
    }

    //================= Find Edge =================//
    edge_iterator find_in_edge(uint64_t eid)
    {
        return in_edges.find(eid);
    }
    edge_iterator find_out_edge(uint64_t eid)
    {
        return out_edges.find(eid);
    }

    edge_iterator find_in_edge(uint64_t vid, edge_iterator eit)
    {
        for (;eit!=in_edges.begin();eit++) 
        {
            if (eit->target()==vid) 
                return eit;
        }
    }
    edge_iterator find_out_edge(uint64_t vid, edge_iterator eit)
    {
        for (;eit!=out_edges.begin();eit++) 
        {
            if (eit->target()==vid) 
                return eit;
        }
    }

protected:
    uint64_t _id;
    vproperty_t _vproperty;

    edgelist_t in_edges;
    edgelist_t out_edges;
};


template<class VERTEX, class EDGE, class vertexlist_t>
class adjacency_list
{
public:
    typedef typename VERTEX::edge_iterator edge_iterator;
    typedef typename vertexlist_t::iterator vertex_iterator;
    typedef typename VERTEX::vproperty_t vproperty_t;
    typedef typename EDGE::eproperty_t eproperty_t;

    adjacency_list(Directness_t d=DIRECTED):_directness(d),
        _vid_gen(0),_eid_gen(0),_vertex_num(0),_edge_num(0)
    {
#ifdef SIM
        memset(locks,0,sizeof(bool)*LOCKSZ);
#endif
    }

    Directness_t get_directness(void){return _directness;}
    uint64_t vertex_num(void){return _vertex_num;}
    uint64_t edge_num(void){return _edge_num;}

    vertex_iterator vertices_begin(void){return _vertices.begin();}
    vertex_iterator vertices_end(void){return _vertices.end();}

    // for compatibility with IBM SystemG internal version
    uint64_t num_vertices(void){return _vertex_num;}
    uint64_t num_edges(void){return _edge_num;}
    //================= Add Vertex =================//
    vertex_iterator add_vertex(void)
    {
        _vertices.push_back(VERTEX(gen_vid()));
        _vertex_num++;
        vertex_iterator iter = _vertices.end();
        iter--;
        return iter;
    }
    vertex_iterator add_vertex(const vproperty_t & vprop)
    {
        _vertices.push_back(VERTEX(gen_vid()));
        _vertex_num++;
        vertex_iterator iter = _vertices.end();
        iter--;
        iter->set_property(vprop);
        return iter;
    }

    //================= Find Vertex =================//
    vertex_iterator find_vertex(uint64_t vid)
    {
        return _vertices.find(vid);
    }

    //================= Delete Vertex =================//
    vertex_iterator delete_vertex(uint64_t vid)
    {
        vertex_iterator viter = this->find_vertex(vid);
        if (viter == _vertices.end()) return viter;

        _vertex_num--;
        if (_directness == UNDIRECTED) 
        {
            // undirected graph doesn't have in_edges
            // only need to remove target verices' corresponding out_edges
            for (edge_iterator eit=viter->out_edges_begin();
                  eit!=viter->out_edges_end();eit++) 
            {
                vertex_iterator vit2 = this->find_vertex(eit->target());
                assert(vit2!=_vertices.end());

                _edge_num -= vit2->delete_out_edge_v(vid);
            }

            return _vertices.erase(vid);
        }
        else if (_directness == DIRECTED) 
        {
            // remove out_edges of source vertices
            for (edge_iterator eit=viter->in_edges_begin();
                  eit!=viter->in_edges_end();eit++) 
            {
                uint64_t targ = eit->target();
                vertex_iterator vit2 = this->find_vertex(targ);
                assert(vit2!=_vertices.end());
#ifdef SIM
                SIM_LOCK(&(locks[targ]));
#endif
                _edge_num -= vit2->delete_out_edge_v(vid);
#ifdef SIM
                SIM_UNLOCK(&(locks[targ]));
#endif
            }
            // remove in_edges of destination vertices
            for (edge_iterator eit=viter->out_edges_begin();
                  eit!=viter->out_edges_end();eit++) 
            {
                uint64_t targ = eit->target();
                vertex_iterator vit2 = this->find_vertex(targ);
                assert(vit2!=_vertices.end());
#ifdef SIM
                SIM_LOCK(&(locks[targ]));
#endif
                vit2->delete_in_edge_v(vid);
#ifdef SIM
                SIM_UNLOCK(&(locks[targ]));
#endif
            }
#ifdef SIM
                SIM_LOCK(&(locks[vid]));
#endif
            vertex_iterator ret = _vertices.erase(vid);
#ifdef SIM
                SIM_UNLOCK(&(locks[vid]));
#endif
            return ret;
        }

        // should not reach here
        assert(0);
    }

    //================= Add Edge =================//
    bool add_edge(uint64_t src, uint64_t dest, edge_iterator& eiter)
    {
        vertex_iterator src_iter = this->find_vertex(src);
        vertex_iterator dest_iter = this->find_vertex(dest);
        if (src_iter == _vertices.end() ||
            dest_iter == _vertices.end()) return false;

        if (_directness == UNDIRECTED) 
        {
            uint64_t eid;
            eproperty_t eprop;
            eid = gen_eid();
#ifdef SIM
            SIM_LOCK(&(locks[src]));
#endif      
            eiter = src_iter->add_out_edge(eid, dest, eprop);
#ifdef SIM
            SIM_UNLOCK(&(locks[src]));
#endif
#ifdef SIM
            __sync_fetch_and_add(&_edge_num,1);
#else
            _edge_num++;
#endif

            eid = gen_eid();
#ifdef SIM
            SIM_LOCK(&(locks[dest]));
#endif
            dest_iter->add_out_edge(eid, src, eiter->shared_property());
#ifdef SIM
            SIM_UNLOCK(&(locks[dest]));
#endif
#ifdef SIM
            __sync_fetch_and_add(&_edge_num,1);
#else
            _edge_num++;
#endif
            return true;
        }
        else if (_directness == DIRECTED) 
        {
            uint64_t eid;
            eproperty_t eprop;
            eid = gen_eid();
#ifdef SIM
            SIM_LOCK(&(locks[src]));
#endif
            eiter = src_iter->add_out_edge(eid, dest, eprop);
#ifdef SIM
            SIM_UNLOCK(&(locks[src]));
#endif
#ifdef SIM
            SIM_LOCK(&(locks[dest]));
#endif
            dest_iter->add_in_edge(eid, src, eiter->shared_property());
#ifdef SIM
            SIM_UNLOCK(&(locks[dest]));
#endif
#ifdef SIM
            __sync_fetch_and_add(&_edge_num,1);
#else
            _edge_num++;
#endif
            return true;
        }
        
        return false;
    }


    //================= Find Edge =================//
    bool find_in_edge(uint64_t src, uint64_t eid, edge_iterator& eiter)
    {
        vertex_iterator src_iter = this->find_vertex(src);
        if (src_iter == _vertices.end()) return false;

        eiter = src_iter->find_in_edge(eid);
        return eiter!=src_iter->in_edges_end();
    }
    bool find_out_edge(uint64_t src, uint64_t eid, edge_iterator& eiter)
    {
        vertex_iterator src_iter = this->find_vertex(src);
        if (src_iter == _vertices.end()) return false;

        eiter = src_iter->find_out_edge(eid);
        return eiter!=src_iter->out_edges_end();
    }

    //================= Delete Edge =================//
    bool delete_edge(uint64_t src, uint64_t eid, edge_iterator& next_eiter)
    {
        vertex_iterator src_iter, dest_iter;
        edge_iterator eiter;

        src_iter = this->find_vertex(src);
        if (src_iter == _vertices.end()) return false;
        eiter = src_iter->find_out_edge(eid);
        if (eiter == src_iter->out_edges_end()) return false;
        dest_iter = this->find_vertex(eiter->target());
        if (dest_iter == _vertices.end()) return false;

        if (_directness == UNDIRECTED) 
        {
            next_eiter = src_iter->delete_out_edge(eid);
            dest_iter->delete_out_edge(eid);
            _edge_num--;
            return true;
        }
        else if (_directness == DIRECTED) 
        {
            next_eiter = src_iter->delete_out_edge(eid);
            dest_iter->delete_in_edge(eid);
            _edge_num--;
            return true;
        }
        return false;
    }
    bool delete_edge_v(uint64_t src, uint64_t dest)
    {
        vertex_iterator src_iter, dest_iter;

        src_iter = this->find_vertex(src);
        if (src_iter == _vertices.end()) return false;
        dest_iter = this->find_vertex(dest);
        if (dest_iter == _vertices.end()) return false;

        if (_directness == UNDIRECTED) 
        {
            _edge_num -= src_iter->delete_out_edge_v(dest);
            _edge_num -= dest_iter->delete_out_edge_v(src);

            return true;
        }
        else if (_directness == DIRECTED) 
        {
            _edge_num -= src_iter->delete_out_edge(dest);
            _edge_num -= dest_iter->delete_in_edge(src);

            return true;
        }

        return false;
    }
protected:
    // local functions
    uint64_t gen_vid(void)
    {
#ifdef SIM
        return __sync_fetch_and_add(&_vid_gen, 1);
#else
        uint64_t ret = _vid_gen;
        _vid_gen++;
        return ret;
#endif
    }
    uint64_t gen_eid(void)
    {
#ifdef SIM
        return __sync_fetch_and_add(&_eid_gen, 1);
#else
        uint64_t ret = _eid_gen;
        _eid_gen++;
        return ret;
#endif
    }
protected:
    // local variables
    vertexlist_t _vertices;
    
    Directness_t _directness; 
    uint64_t _vid_gen;
    uint64_t _eid_gen;

    uint64_t _vertex_num;
    uint64_t _edge_num;
#ifdef SIM
    bool locks[LOCKSZ];
#endif
};



}//end of namespace openG


#endif
