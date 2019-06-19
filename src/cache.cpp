#include "cache.h"
#include <cstring>

CACHE* LRU_CACHE::cache;
UBMP32 LRU_CACHE::size;
UBMP32 LRU_CACHE::used;
std::unordered_map<UBMP64,CACHE*> 
    LRU_CACHE::cacheMap;

LRU_CACHE::LRU_CACHE() {
    tail = 0;
    head = 0;
    l_create(lock);
}
void LRU_CACHE::alloc(UBMP32 tsize) {
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
void LRU_CACHE::add(UBMP64 key,INFO* info) {
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
int LRU_CACHE::get(UBMP64 key,UBMP32 probe_index,UBMP8& value) {
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
