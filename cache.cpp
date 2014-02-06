#include "cache.h"
#include <cstring>

CACHE* LRU_CACHE::cache;
LRU_CACHE* LRU_CACHE::lru_head;
LRU_CACHE* LRU_CACHE::lru_tail;
LOCK LRU_CACHE::lock_lru;
UBMP32 LRU_CACHE::size;
UBMP32 LRU_CACHE::used;

LRU_CACHE::LRU_CACHE() {
	tail = 0;
	head = 0;
	lru_next = 0;
	lru_prev = 0;
	l_create(lock);
}
void LRU_CACHE::alloc(UBMP32 tsize) {
	size  = tsize / sizeof(CACHE);
	cache = new CACHE[size];
	used  = 0;
	lru_head = 0;
	lru_tail = 0;
	l_create(lock_lru);
}
void LRU_CACHE::free() {
	delete[] cache;
}

/*
Bring recently accessed data (after add/get) to the front of the lru-list.
Assumes bucket lock is not acquired
*/
void LRU_CACHE::bring_to_front() {
    l_lock(lock_lru);

	if(lru_head) {
		if(lru_head != this) {
			if(lru_tail == this)
				lru_tail = lru_tail->lru_prev;
			
			if(lru_prev)
				lru_prev->lru_next = lru_next;
			if(lru_next)
				lru_next->lru_prev = lru_prev;
			lru_next = 0;
			lru_prev = 0;
			
			LRU_CACHE *lru_temp = lru_head;
			lru_head = this;
			lru_head->lru_next = lru_temp;
			lru_temp->lru_prev = lru_head;
		}
	} else {
		lru_tail = this;
		lru_head = this;
	}

	l_unlock(lock_lru);
}

/*Replace head by new one*/
void LRU_CACHE::insert_head(CACHE* freec) {
	l_lock(lock);

	CACHE* temph = head;
	head = freec;
	head->next = temph;
	if(temph)
		temph->prev = head;
	else
		tail = head;

	l_unlock(lock);
}

/*Add new block to LRU cache*/
void LRU_CACHE::add(INFO* info) {
	CACHE* freec = 0;

	l_lock(lock_lru);
    if(used < size) {
		/*we have an unused block*/
		freec = cache + used;
		used++;
		l_unlock(lock_lru);

	} else { 
		/*find a tail to replace and detach it*/
		LRU_CACHE* lru_target = lru_tail;
		while(lru_target) {
			l_lock(lru_target->lock);
			if(lru_target->head != lru_target->tail) {
				freec = lru_target->tail;
				lru_target->tail = freec->prev;
				lru_target->tail->next = 0;
				freec->prev = 0;
				l_unlock(lru_target->lock);
				break;
			}
            l_unlock(lru_target->lock);
			lru_target = lru_target->lru_prev;
		}
		l_unlock(lock_lru);
	}

	/*insert info at the free space*/
	memcpy(&freec->info,info,sizeof(INFO));
	insert_head(freec);
	bring_to_front();
}

/*check lru cache for value*/
int LRU_CACHE::get(UBMP32 start_index,UBMP32 probe_index,UBMP8& value) {
	register CACHE* curr = head;

	l_lock(lock);
	while(curr) {
		if(curr->info.start_index == start_index) {
			/*update cache list and copy value*/
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
			/*put the lru list at the front & return a hit*/
			bring_to_front();
			return CACHE_HIT;
		}
		curr = curr->next;
	}

	l_unlock(lock);
	return CACHE_MISS;
}
