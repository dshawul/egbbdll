#ifndef __CACHE__
#define __CACHE__

#define PARALLEL
#include "my_types.h"
#include <unordered_map>

/*
data to be cached
*/
#define INFO_SIZE (1 << 13)

struct INFO {
    UBMP8  block[INFO_SIZE];
    UBMP64 key;
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
    static UBMP32 size;
    static UBMP32 used;
    static std::unordered_map<UBMP64,CACHE*> cacheMap;
public:
    LRU_CACHE();
    void add(UBMP64,INFO* info);
    int  get(UBMP64,UBMP32 probe_index,UBMP8& value);
    static void alloc(UBMP32);
    static void free();
};

#endif
