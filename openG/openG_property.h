#ifndef OPENG_PROPERTY_H
#define OPENG_PROPERTY_H

#include <tr1/unordered_map>

#include <iterator>
#include <list>
#include <vector>
#include <stdint.h>
#include <assert.h>

#include "openG_storage.h"
#include "openG_graph.h"

namespace openG
{

enum GLayout
{
    ILV_ILE=0,  //  vertexlist->indexed_list    edgelist->indexed_list
    LV_LE=1,    //  vertexlist->list            edgelist->list
    IVV_IVE=2,  //  vertexlist->indexed_vector  edgelist->indexed_vector
    VV_VE=3     //  vertexlist->vector          edgelist->vector
};

template<class VPROP, class EPROP, GLayout L=ILV_ILE>
class openG_configure;

template<class VPROP, class EPROP>
class openG_configure<VPROP,EPROP,ILV_ILE>
{
public:
    typedef VPROP       vproperty_t;
    typedef EPROP       eproperty_t;

    typedef typename openG::edge<eproperty_t>                       edge_t;
    typedef typename openG::storage::indexed_list_storage<edge_t>   edgelist_t;
    typedef typename openG::vertex<vproperty_t, edge_t, edgelist_t> vertex_t;
    typedef typename openG::storage::indexed_list_storage<vertex_t> vertexlist_t;
};

}

#endif
