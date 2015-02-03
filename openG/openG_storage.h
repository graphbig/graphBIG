#ifndef OPENG_STORAGE_H
#define OPENG_STORAGE_H

#include <tr1/unordered_map>

#include <iterator>
#include <list>
#include <vector>

namespace openG
{

namespace storage
{
// class T must have a member function id()
template <class T>
class vector_storage : public std::vector<T>
{
    typedef std::vector<T> base_t;

public:
    typedef typename base_t::iterator iterator;
    typedef typename base_t::const_iterator const_iterator;
    typedef typename base_t::value_type value_type;

    vector_storage() : base_t() {}
    vector_storage(size_t i) : base_t(i) {}

    iterator find(const size_t& id)
    {
        // guess first
        iterator iter;
        if (id >= this->size())
            iter = this->end()-1;
        else
            iter = this->begin()+id;

        if (iter->id() == id) return iter;


        // if not found search one by one
        for (iter=this->begin(); iter!=this->end(); ++iter)
        {
            if (iter->id() == id) return iter;
        }
        return this->end();
    }

    void push_back(const value_type& val)
    {
        base_t::push_back(val);
    }

    iterator erase(iterator iter)
    {
        if (iter == this->end()) return this->end();
        return base_t::erase(iter);
    }

    iterator erase(const size_t& id)
    {
        iterator iter = this->find(id);
        if (iter == this->end()) return this->end();
        return base_t::erase(iter);
    }
    
};  // end of vector_storage class

template <class T>
class indexed_vector_storage : public std::vector<T>
{
    typedef std::vector<T> base_t;
    typedef std::vector<bool> flag_t;
    typedef typename base_t::iterator base_iterator;
    typedef typename flag_t::iterator flag_iterator;

    //================ iterator ===============//
    class new_iterator
    {
    public:
        new_iterator():_data_ptr(NULL),_flag_ptr(NULL),_cur_pos(0){}
        new_iterator(base_t* a, flag_t * b, size_t c)
        :_data_ptr(a),_flag_ptr(b),_cur_pos(c){}

        new_iterator& operator++(int x)
        {
            if (_data_ptr==NULL || _flag_ptr==NULL) return *this;
            do
            {
                _cur_pos++;
            }while(_cur_pos<_data_ptr->size() && (*_flag_ptr)[_cur_pos]==false);

            return *this;
        }

        new_iterator& operator++()
        {
            if (_data_ptr==NULL || _flag_ptr==NULL) return *this;
            do
            {
                _cur_pos++;
            }while(_cur_pos<_data_ptr->size() && (*_flag_ptr)[_cur_pos]==false);

            return *this;
        }

        new_iterator& operator--(int x)
        {
            if (_data_ptr==NULL || _flag_ptr==NULL) return *this;
            do
            {
                _cur_pos--;
            }while(_cur_pos>0 && (*_flag_ptr)[_cur_pos]==false);

            return *this;
        }

        new_iterator& operator--()
        {
            if (_data_ptr==NULL || _flag_ptr==NULL) return *this;
            do
            {
                _cur_pos--;
            }while(_cur_pos>0 && (*_flag_ptr)[_cur_pos]==false);

            return *this;
        }

        bool operator==(const new_iterator& rhs)
        {

            return (this->_cur_pos==rhs._cur_pos && this->_data_ptr==rhs._data_ptr
                    && this->_flag_ptr==rhs._flag_ptr);
        }

        bool operator!=(const new_iterator& rhs)
        {
            return (this->_cur_pos!=rhs._cur_pos || this->_data_ptr!=rhs._data_ptr
                    || this->_flag_ptr!=rhs._flag_ptr);
        }

        void operator=(const new_iterator& rhs)
        {
            this->_cur_pos = rhs._cur_pos;
            this->_data_ptr = rhs._data_ptr;
            this->_flag_ptr = rhs._flag_ptr;
        }

        T* operator->()
        {
            return &((*_data_ptr)[_cur_pos]);
        }

        T& operator*()
        {
            return (*_data_ptr)[_cur_pos];
        }

        size_t curpos()
        {
            return _cur_pos;
        }
    protected:
        base_t * _data_ptr;
        flag_t * _flag_ptr;
        size_t _cur_pos;
    };

public:
    typedef new_iterator iterator;
    typedef new_iterator const_iterator;
    typedef typename base_t::value_type value_type;

    indexed_vector_storage() : base_t() {}

    iterator begin(void)
    {
        size_t i=0;
        for (;i<_flags.size();i++) 
        {
            if (_flags[i]) break;
        }
        return new_iterator(this,&_flags,i);
    }

    iterator end(void)
    {
        return new_iterator(this,&_flags,_flags.size());
    }

    iterator find(const size_t& id)
    {
        typename std::tr1::unordered_map<size_t, size_t>::iterator idx_iter;
        idx_iter = _index.find(id);
        if (idx_iter == _index.end()) 
            return this->end();
        else
            return new_iterator(this,&_flags,idx_iter->second);

        return this->end();
    }

    void clear(void)
    {
        base_t::clear();
        _flags.clear();
        _index.clear();
    }

    void push_back(const value_type& val)
    {
        base_t::push_back(val);
        _flags.push_back(true);

        _index[(const_cast<value_type&>(val)).id()] = this->size() - 1;
    }

    iterator erase(iterator iter)
    {
        if (iter == this->end()) return this->end();
        _index.erase(iter->id());
        invalidate(iter);
        iter++;
        return iter;
    }

    iterator erase(const size_t& id)
    {
        iterator iter = this->find(id);
        if (iter == this->end()) return this->end();
        _index.erase(id);
        invalidate(iter);
        iter++;
        return iter;
    }
protected:
    void invalidate(iterator iter)
    {
        size_t idx = iter.curpos();
        if (_flags.size()<=idx) return;

        _flags[idx] = false;
    }
    std::vector<bool> _flags;

    std::tr1::unordered_map<size_t, size_t> _index;
};  // end of indexed_vector_storage class

template <class T>
class list_storage : public std::list<T>
{
    typedef std::list<T> base_t;

public:
    typedef typename base_t::iterator iterator;
    typedef typename base_t::const_iterator const_iterator;
    typedef typename base_t::value_type value_type;

    list_storage() : base_t() {}

    iterator find(const size_t& id)
    {
        iterator iter;
        
        for (iter=this->begin(); iter!=this->end(); ++iter)
        {
            if (iter->id() == id) return iter;
        }
        return this->end();
    }

    void push_back(const value_type& val)
    {
        base_t::push_back(val);
    }

    iterator erase(iterator iter)
    {
        if (iter == this->end()) return this->end();
        return base_t::erase(iter);
    }

    iterator erase(const size_t& id)
    {
        iterator iter = this->find(id);
        if (iter == this->end()) return this->end();
        return base_t::erase(iter);
    }
};  // end of list_storage class


template<class T>
class indexed_list_storage : public list_storage<T>
{
    typedef list_storage<T> base_t;
public:
    typedef typename base_t::iterator iterator;
    typedef typename base_t::const_iterator const_iterator;
    typedef typename base_t::value_type value_type;

    indexed_list_storage() : base_t() {}

    iterator find(const size_t& id)
    {
        typename std::tr1::unordered_map<size_t, iterator>::iterator idx_iter;
        idx_iter = _index.find(id);
        if (idx_iter == _index.end()) 
            return this->end();
        else
            return idx_iter->second;
    }

    void clear(void)
    {
        base_t::clear();
        _index.clear();
    }

    void push_back(const value_type& val)
    {
        base_t::push_back(val);
        iterator iter = this->end();
        iter--;
        _index[iter->id()] = iter;
    }

    iterator erase(iterator iter)
    {
        if (iter == this->end()) return this->end();
        _index.erase(iter->id());
        return base_t::erase(iter);
    }

    iterator erase(const size_t& id)
    {
        iterator iter = this->find(id);
        if (iter == this->end()) return this->end();

        _index.erase(iter->id());
        return base_t::erase(iter);
    }

protected:
    std::tr1::unordered_map<size_t, iterator> _index;
};//end of indexed_storage class



}//end of namespace storage
}//end of namespace openG



#endif

