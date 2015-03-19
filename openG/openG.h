#ifndef OPENG_H
#define OPENG_H

#include <tr1/unordered_map>
#include <iostream>
#include <tr1/memory>
#include <iterator>
#include <list>
#include <vector>
#include <stdint.h>
#include <assert.h>
#include <string>
#include <fstream>

#include "openG_storage.h"
#include "openG_property.h"
#include "openG_graph.h"

namespace openG
{


template<class VPROP, class EPROP, class CONFIG=openG_configure<VPROP,EPROP> >
class Graph: public adjacency_list<typename CONFIG::vertex_t, typename CONFIG::edge_t, typename CONFIG::vertexlist_t>
{
    typedef adjacency_list<typename CONFIG::vertex_t, typename CONFIG::edge_t, typename CONFIG::vertexlist_t> base_t;
public:
    Graph(Directness_t d=DIRECTED):base_t(d){}

    typedef typename base_t::edge_iterator      edge_iterator;
    typedef typename base_t::vertex_iterator    vertex_iterator;
    typedef typename base_t::vproperty_t        vproperty_t;
    typedef typename base_t::eproperty_t        eproperty_t;
};

template<class VPROP, class EPROP, class CONFIG=openG_configure<VPROP,EPROP> >
class extGraph: public Graph<VPROP, EPROP, CONFIG>
{
    typedef Graph<VPROP, EPROP, CONFIG> base_t;
public:
    extGraph(Directness_t d=DIRECTED):base_t(d){}

    typedef typename base_t::edge_iterator      edge_iterator;
    typedef typename base_t::vertex_iterator    vertex_iterator;
    typedef typename base_t::vproperty_t        vproperty_t;
    typedef typename base_t::eproperty_t        eproperty_t;

    void add_vertex_key(uint64_t vid, std::string key)
    {
        _key2id[key] = vid;
        _id2key[vid] = key;
    }

    void delete_vertex_key(uint64_t vid)
    {
        std::tr1::unordered_map<uint64_t, std::string>::iterator it;
        it = _id2key.find(vid);
        if (it == _id2key.end()) return;

        _key2id.erase(it->second);
        _id2key.erase(it);
    }


    /**
    *   @brief load vertices from a csv file into graph. If the
    *          vertex already exists in graph, it'll be skipped
    *   @param filename     csv file name
    *   @param has_header   if csv file has header
    *   @param separators   separators used in the csv file
    *   @param keypos       column # of external vertex id (starting
    *                       from 0)
    *   @param loop_ctrl    pointer to a bool controlling variable.
    *                       if not NULL, setting the controlling
    *                       variable to false can break the loading
    *                       process
    *   @return long int: if sucess, return number of processed
    *                       vertices. Otherwise, return -1.
    */
    //===================================================================//
    long int load_csv_vertices(std::string filename, bool has_header, std::string separators,
                               size_t keypos, bool * loop_ctrl=NULL)
    {
        std::ifstream file(filename.c_str());
        if (!file.is_open())
        {
            std::cerr<<"cannot open csv file\n";
            return -1;
        }
        if (!file.good())
        {
            std::cerr<<"csv file empty\n";
            return -1;
        }

        std::string line;
        std::vector<std::string> csv_header;
        size_t next_pos=0;

        // get csv file header
        getline(file, line);
        if (!has_header)
        {
            while (next_pos != std::string::npos)
            {
                std::string cellstr;
                next_pos = csv_nextCell(line,separators,cellstr,next_pos);
                csv_header.push_back(cellstr);
            }
            file.seekg(0, std::ios::beg);
        }
        else
        {
            while (next_pos != std::string::npos)
            {
                std::string cellstr;
                next_pos = csv_nextCell(line,separators,cellstr,next_pos);
                csv_header.push_back(cellstr);
            }
        }

        size_t line_num=0;

        // processing data lines
        size_t vertex_num=0;
        std::vector<std::string> _pos_buffer(csv_header.size());
        while (file.good())
        {
            line_num++;

            if (loop_ctrl!=NULL && (*loop_ctrl)==false) break;

            // fetch one line from csv file
            line.clear();
            getline(file, line);

            if (line.empty()) continue; // skip empty lines

            next_pos = 0;

            // get cells from the line
            for (size_t i=0; i<_pos_buffer.size(); i++)
            {
                next_pos = csv_nextCell(line,separators,_pos_buffer[i],next_pos);

                if (next_pos == std::string::npos) break;
            }

            // sanity check
            if (next_pos != std::string::npos)
            {
                std::cerr<<line_num<<" wrong data line in csv file\n";
                continue;
            }
            if (keypos >= _pos_buffer.size())
            {
                std::cerr<<line_num<<" wrong key position or wrong data line in csv file\n";
                continue;
            }
          
            std::string& key_str = _pos_buffer[keypos];

            // add the vertex if not already present
            if (_key2id.find(key_str) == _key2id.end())
            {
                vertex_iterator vit = this->add_vertex();
                _key2id[key_str] = vit->id();
                _id2key[vit->id()] = key_str;
            }

            vertex_num++;
        }

        return vertex_num;
    }
    /**
    *   @brief load edges from a csv file into graph. 
    *   @param filename     csv file name
    *   @param has_header   if csv file has header
    *   @param separators   separators used in the csv file
    *   @param srcpos       column # of src vertex key (starting
    *                       from 0)
    *   @param destpos      column # of dest vertex key (starting 
    *                       from 0)
    *   @param dag_check    if true, checking new edges for DAG
    *                       requirements                
    *   @param loop_ctrl    pointer to a bool controlling variable.
    *                       if not NULL, setting the controlling
    *                       variable to false can break the loading
    *                       process
    *   @return long int: if sucess, return number of processed
    *                       vertices. Otherwise, return -1.
    */
    //===================================================================//
    long int load_csv_edges(std::string filename, bool has_header, std::string separators,
                               size_t srcpos, size_t destpos, bool dag_check=false, bool * loop_ctrl=NULL)
    {
        std::ifstream file(filename.c_str());
        if (!file.is_open())
        {
            std::cerr<<"cannot open csv file\n";
            return -1;
        }
        if (!file.good())
        {
            std::cerr<<"csv file empty\n";
            return -1;
        }

        std::string line;
        std::vector<std::string> csv_header;
        size_t next_pos=0;

        // get csv file header
        getline(file, line);
        if (!has_header)
        {
            while (next_pos != std::string::npos)
            {
                std::string cellstr;
                next_pos = csv_nextCell(line,separators,cellstr,next_pos);
                csv_header.push_back(cellstr);
            }
            file.seekg(0, std::ios::beg);
        }
        else
        {
            while (next_pos != std::string::npos)
            {
                std::string cellstr;
                next_pos = csv_nextCell(line,separators,cellstr,next_pos);
                csv_header.push_back(cellstr);
            }
        }

        size_t line_num=0;

        // processing data lines
        size_t edge_num=0;
        std::vector<std::string> _pos_buffer(csv_header.size());
        while (file.good())
        {
            line_num++;

            if (loop_ctrl!=NULL && (*loop_ctrl)==false) break;

            // fetch one line from csv file
            line.clear();
            getline(file, line);

            if (line.empty()) continue; // skip empty lines

            next_pos = 0;

            // get cells from the line
            for (size_t i=0; i<_pos_buffer.size(); i++)
            {
                next_pos = csv_nextCell(line,separators,_pos_buffer[i],next_pos);

                if (next_pos == std::string::npos) break;
            }

            // sanity check
            if (next_pos != std::string::npos)
            {
                std::cerr<<line_num<<" wrong data line in csv file\n";
                continue;
            }
            if (srcpos >= _pos_buffer.size())
            {
                std::cerr<<line_num<<" wrong src position or wrong data line in csv file\n";
                continue;
            }
            if (destpos >= _pos_buffer.size())
            {
                std::cerr<<line_num<<" wrong destpos position or wrong data line in csv file\n";
                continue;
            }

            std::string& src_str = _pos_buffer[srcpos];
            std::string& dest_str = _pos_buffer[destpos];

            std::tr1::unordered_map<std::string, uint64_t>::iterator srciter,destiter;
            srciter  = _key2id.find(src_str);
            destiter = _key2id.find(dest_str);

            // add the vertex if not already present
            if (srciter == _key2id.end())
            {
                vertex_iterator vit = this->add_vertex();
                _key2id[src_str] = vit->id();
                _id2key[vit->id()] = src_str;
                srciter  = _key2id.find(src_str);
            }
            if (destiter == _key2id.end())
            {
                vertex_iterator vit = this->add_vertex();
                _key2id[dest_str] = vit->id();
                _id2key[vit->id()] = dest_str;
                destiter = _key2id.find(dest_str);
            }

            if (dag_check && srciter->second>=destiter->second)
                continue; // ensure DAG property

            edge_iterator eit;
            bool eret = this->add_edge(srciter->second,destiter->second,eit);
            if (eret == false)
            {
                std::cerr<<line_num<<" error when adding edge\n";
                continue;
            }

            edge_num++;
        }

        return edge_num;
    }

    // convert property graph structure to a CSR graph
    void to_CSR_Graph(std::vector<uint64_t> & vertexlist,
            std::vector<uint64_t> & degreelist, 
            std::vector<uint64_t> & edgelist)
    {
        vertexlist.clear();
        degreelist.clear();
        edgelist.clear();

        // initialize vertex/edge list
        vertexlist.resize(this->num_vertices(), 0); 
        degreelist.resize(this->num_vertices(), 0);
        edgelist.resize(this->num_edges(), 0);


        size_t vidx=0;
        size_t eidx=0;
        for (vertex_iterator vit=this->vertices_begin();vit!=this->vertices_end();vit++)
        {
            vidx = vit->id();
            // double check in case vector overflow happens
            if (vidx >= vertexlist.size())
            {  
                vertexlist.resize(vidx+1);
                degreelist.resize(vidx+1);
            }
            if (eidx >= edgelist.size())    edgelist.resize(eidx+1);

            degreelist[vidx]=vit->edges_size();
            if (degreelist[vidx]==0) continue; // skip no-edge vertices
            
            vertexlist[vidx]=eidx;

            for (edge_iterator eit=vit->edges_begin();eit!=vit->edges_end();eit++)
            {
                edgelist[eidx]=eit->target();
                eidx++;
            }

        }
    }
protected:
    std::tr1::unordered_map<std::string, uint64_t> _key2id;
    std::tr1::unordered_map<uint64_t, std::string> _id2key;
private:
    size_t csv_nextCell(std::string& line, std::string sepr, std::string& ret, size_t pos=0)
    {
        sepr.append("\r\n");

        ret.clear();

        size_t head, tail;
        bool in_quotation = false;

        head = line.find_first_not_of(sepr, pos);
        if (head == std::string::npos)
        {
            ret.clear();
            return std::string::npos;
        }

        if (line[head]=='\"')
        {
            head++;
            in_quotation = true;
        }
        if (in_quotation)
        {
            size_t prev;
            tail = line.find_first_of('\"', head);
            ret = line.substr(head, tail-head);

            if (tail == (line.size()-1) ) // reach line end
                return std::string::npos;

            while (line[tail+1]=='\"') // double quote means a quote mark in fileds
            {
                ret.append("\"");
                prev = tail+2;
                tail = line.find_first_of('\"', prev);
                if (tail == std::string::npos)
                {
                    ret.append(line.substr(prev));
                    return std::string::npos;
                }
                ret.append(line.substr(prev, tail-prev));
                if (tail == (line.size()-1))
                    return std::string::npos;
            }
            return line.find_first_not_of(sepr, tail+1);
        }
        else
        {
            tail = line.find_first_of(sepr, head);
            if (tail != std::string::npos)
            {
                ret = line.substr(head, tail - head);
                return line.find_first_not_of(sepr, tail);
            }
            else
            {
                ret = line.substr(head);
                return std::string::npos;
            }

        }

        return std::string::npos; // should not reach here
    }
};

}

#endif
