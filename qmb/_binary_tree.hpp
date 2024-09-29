#ifndef QMB_BINARY_TREE
#define QMB_BINARY_TREE

#include <memory>

// Binary tree, with each leaf associated with a value,
// and we can set and get leaf value by an iterator.
template<typename T, T miss>
class Tree {
    std::unique_ptr<Tree<T, miss>> _left;
    std::unique_ptr<Tree<T, miss>> _right;
    std::unique_ptr<T> _value;

    T& value() {
        if (!_value) {
            _value = std::make_unique<T>(miss);
        }
        return *_value;
    }


  public:
    template<typename It>
    void set(It begin, It end, T v) {
        if (begin == end) {
            value() = v;
        } else {
            if (*(begin++)) {
                if (!_right) {
                    _right = std::make_unique<Tree<T, miss>>();
                }
                _right->set(begin, end, v);
            } else {
                if (!_left) {
                    _left = std::make_unique<Tree<T, miss>>();
                }
                _left->set(begin, end, v);
            }
        }
    }

    template<typename It>
    T get(It begin, It end) {
        if (begin == end) {
            return value();
        } else {
            if (*(begin++)) {
                if (_right) {
                    return _right->get(begin, end);
                } else {
                    return miss;
                }
            } else {
                if (_left) {
                    return _left->get(begin, end);
                } else {
                    return miss;
                }
            }
        }
    }
};

#endif
