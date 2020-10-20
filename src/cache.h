#ifndef __CACHE__
#define __CACHE__

#include <unordered_map>
#include "my_types.h"

/*
data to be cached
*/
#define INFO_SIZE (1 << 13)

struct INFO {
    uint8_t  block[INFO_SIZE];
    uint64_t key;
};
/*
cache class
*/
#define CACHE_HIT   1
#define CACHE_MISS  0

struct CACHE {
    INFO   info;
    CACHE*  prev;
    CACHE*  next;
    CACHE() {
        prev = 0;
        next = 0;
    }
};

class LRU_CACHE {
public:
    CACHE* head;
    CACHE* tail;
    LOCK lock;
    static CACHE* cache;
    static uint32_t size;
    static uint32_t used;
    static std::unordered_map<uint64_t,CACHE*> cacheMap;
public:
    LRU_CACHE();
    void add(uint64_t,INFO* info);
    int  get(uint64_t,uint32_t probe_index,uint8_t& value);
    static void alloc(uint32_t);
    static void free();
};

#endif
