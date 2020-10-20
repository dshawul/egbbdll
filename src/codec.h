#ifndef __CODEC__
#define __CODEC__

#ifdef _MSC_VER
#    define _CRT_SECURE_NO_DEPRECATE
#    define _SCL_SECURE_NO_DEPRECATE
#    pragma warning (disable: 4127)
#endif

#include <stdio.h>

#include "my_types.h"

/*
const bytes
*/
#define _byte_1   UINT64(0x00000000000000ff)
#define _byte_all UINT64(0xffffffffffffffff)
#define _byte_32  0xffffffff

/*
block size
*/
#define BLOCK_SIZE                         (1 << 13)
#define MAX_LEN                            32
#define INVALID                            -1

/*
Lempel Ziv consts
*/
#define DISTANCE_BITS                      12
#define LENGTH_BITS                        8
#define PAIR_BITS                          (DISTANCE_BITS + LENGTH_BITS)
#define LITERAL_BITS                       8
#define F_PAIR_BITS                        (PAIR_BITS + 1)
#define F_LITERAL_BITS                     (LITERAL_BITS + 1)

#define MIN_MATCH_LENGTH                   ((PAIR_BITS >> 3) + 1 + 1)
#define MAX_MATCH_LENGTH                   ((1 << LENGTH_BITS) - 1 + MIN_MATCH_LENGTH)

#define WINDOW_SIZE                        (1 << DISTANCE_BITS)

#define LENGTH_CODES                       30
#define DISTANCE_CODES                     31

#define EOB_MARKER                         (1 << LITERAL_BITS)
#define LENGTH_MARKER                      (EOB_MARKER + 1)
#define LITERAL_CODES                      ((1 << LITERAL_BITS) + LENGTH_CODES + 1)

/*
Mapping of distances.This greatly reduces space requirement 
for huffman trees.Distrances from 0-4096  are mapped to 0-23
*/
extern const int extra_dbits[];
extern const int base_dist[];
extern const int extra_lbits[];
extern const int base_length[];
/*
match length/position pair
*/
struct PAIR {
    int pos;
    int len;
    PAIR() {
        pos = 0;
        len = 0;
    }
};

/*bitset*/
class BITSET {
    uint64_t code;
    uint16_t length;
    const uint8_t*& in;
    uint8_t*& out;
public:
    BITSET(const uint8_t*& input,uint8_t*& output) : 
        in(input),out(output) {
        code = 0;
        length = 0;
    }
    void addbits(int x) {
        while(x > length) {                                
            code = (code << 8) | *in++;                        
            length += 8;                                       
        }   
    }
    uint32_t getbits(int x) {
        addbits(x);                                          
        length -= x;                                     
        return uint32_t(code >> length) & (_byte_32 >> (32 - x)); 
    }
    void writebits() {
        while(length >= 8) {
            length -= 8;
            *out++ = uint8_t((code >> length) & _byte_1);
            code &= (_byte_all >> (64 - length));
        }
    }
    void flushbits() {
        if(length) {
            length -= 8;
            *out++ = uint8_t((code  << (-length)) & _byte_1);
        }
    }
    void write(int value, int len) {
        code = ((code << len) | (value));
        length += len;
    }
    uint32_t read(int len) {
        return uint32_t(code >> (length - len));
    }
    void trim(int len) {
        length -= len;
    }
    void writeliteral(int value) {
        code = ((code << F_LITERAL_BITS) |
            (value));
        length += F_LITERAL_BITS;
    }
    void writepair(const PAIR& pairv) {
        code = ((code << F_PAIR_BITS) |
            (1 << PAIR_BITS) |
            ((pairv.len - MIN_MATCH_LENGTH) << DISTANCE_BITS) |
            (pairv.pos));
        length += F_PAIR_BITS;
    }
};

/*
huffman tree
*/
struct NODE {
    int   symbol;
    uint32_t freq;
    uint8_t  skip;
    NODE*  left;
    NODE*  right;
    NODE() {
        clear();
    }
    void clear() {
        symbol = INVALID;
        freq   = 0;
        skip   = 1;
        left   = 0;
        right  = 0;
    }
};

struct CANN {
    int   symbol;
    uint32_t code;
    uint32_t mask;
    uint8_t length;
    CANN() {
        symbol = INVALID;
        code = 0;
        length = 0;
    }
};

struct HUFFMAN {
    NODE*   nodes;
    CANN*   cann;
    CANN*   pstart[MAX_LEN];
    uint8_t   min_length,
            max_length;
    uint32_t  MAX_LEAFS;
    uint32_t  MAX_NODES;
    void clear_nodes();
    void build_cann_from_nodes();
    void build_cann_from_length();
    void print_all_node();
    void print_all_cann();
};
/*
file info
*/
class COMP_INFO {

public:
    FILE*   pf;
    char    output_fname[256];
    uint32_t* index_table;
    uint32_t  size,
            orgsize,
            cmpsize,
            n_blocks,
            block_size,
            read_start;
    HUFFMAN huffman;
    HUFFMAN huffman_pos;
    COMP_INFO();
    ~COMP_INFO();
public:
    bool open(FILE*,int);
    void compress(FILE*,FILE*,int=0);
    void decompress(FILE*,FILE*,int=0);
    int encode_huff(const uint8_t*,uint8_t*,const uint32_t);
    int encode_lz(const uint8_t*,uint8_t*,const uint32_t);
    template<bool> 
    int _decode(const uint8_t*,uint8_t*,const uint32_t);
    int decode(const uint8_t*,uint8_t*,const uint32_t);
    int decode_huff(const uint8_t*,uint8_t*,const uint32_t);
    int decode_lz(const uint8_t*,uint8_t*,const uint32_t);
    void write_crc(bool);
    void collect_frequency(const uint8_t*,const uint32_t);
};

unsigned long Crc32_ComputeBuf(unsigned long inCrc32, const void *buf,unsigned int bufLen);

#endif
