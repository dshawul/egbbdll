#ifndef __CACHE__
#define __CACHE__

#define PARALLEL
#include "my_types.h"

/*
data to be cached
*/
#define INFO_SIZE (1 << 13)

struct INFO {
	UBMP8  block[INFO_SIZE];
	UBMP32 start_index;
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
	LRU_CACHE* lru_prev;
	LRU_CACHE* lru_next;
	LOCK lock;
	static CACHE* cache;
	static LRU_CACHE* lru_head;
	static LRU_CACHE* lru_tail;
	static LOCK lock_lru;
	static UBMP32 size;
	static UBMP32 used;

public:
	LRU_CACHE();
	void add(INFO* info);
	int  get(UBMP32 start_index,UBMP32 probe_index,UBMP8& value);
	void insert_head(CACHE*);
	void bring_to_front();
	static void alloc(UBMP32);
};

#endif
