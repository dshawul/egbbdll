#include <cstring>
#include "cache.h"

CACHE* LRU_CACHE::cache;
uint32_t LRU_CACHE::size;
uint32_t LRU_CACHE::used;
std::unordered_map<uint64_t,CACHE*> 
    LRU_CACHE::cacheMap;

LRU_CACHE::LRU_CACHE() {
    tail = 0;
    head = 0;
    l_create(lock);
}
void LRU_CACHE::alloc(uint32_t tsize) {
    size  = tsize / sizeof(CACHE);
    cache = new CACHE[size];
    cacheMap.reserve(size);
    used  = 0;
}
void LRU_CACHE::free() {
    delete[] cache;
    cacheMap.clear();
}

/*Add new block to LRU cache*/
void LRU_CACHE::add(uint64_t key,INFO* info) {
    CACHE* freec = 0;
    l_lock(lock);
    if(used < size) {
        freec = cache + used;
        used++;
    } else { 
        cacheMap.erase(tail->info.key);
        freec = tail;
        tail = freec->prev;
        tail->next = 0;
        freec->prev = 0;
    }
    l_unlock(lock);

    /*copy info*/
    memcpy(&freec->info,info,sizeof(INFO));

    /*make head*/
    l_lock(lock);
    CACHE* temph = head;
    head = freec;
    head->next = temph;
    if(temph)
        temph->prev = head;
    else
        tail = head;
    cacheMap[key] = head;
    l_unlock(lock);
}

/*check lru cache for value*/
int LRU_CACHE::get(uint64_t key,uint32_t probe_index,uint8_t& value) {
    l_lock(lock);
    if(cacheMap.count(key)) {
        CACHE* curr = cacheMap[key];
        if(curr != head) {
            if(curr == tail)
                tail = tail->prev;

            if(curr->prev)
                curr->prev->next = curr->next;
            if(curr->next)
                curr->next->prev = curr->prev;
            curr->next = 0;
            curr->prev = 0;

            CACHE *temp = head;
            head = curr;
            head->next = temp;
            temp->prev = head;
        }
        value = head->info.block[probe_index];
        l_unlock(lock);
        return CACHE_HIT;
    }
    l_unlock(lock);
    return CACHE_MISS;
}
