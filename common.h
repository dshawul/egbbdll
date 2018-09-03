#ifndef __COMMON__
#define __COMMON__

#ifdef _MSC_VER
#    define _CRT_SECURE_NO_DEPRECATE
#    define _SCL_SECURE_NO_DEPRECATE
#    pragma warning (disable: 4127)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#define PARALLEL
#include "my_types.h"
#include "cache.h"
#include "codec.h"

using namespace std;

/*types*/
enum COLORS {
    white,black,neutral
};
enum CHESSMEN {
    king = 1,queen,rook,bishop,knight,pawn
};
enum OCCUPANCY {
    blank,wking,wqueen,wrook,wbishop,wknight,wpawn,
          bking,bqueen,brook,bbishop,bknight,bpawn,elephant
};
enum RANKS {
    RANK1,RANK2,RANK3,RANK4,RANK5,RANK6,RANK7,RANK8
};
enum FILES {
    FILEA,FILEB,FILEC,FILED,FILEE,FILEF,FILEG,FILEH
};
enum SQUARES {
    A1 = 0,B1,C1,D1,E1,F1,G1,H1,
        A2 = 16,B2,C2,D2,E2,F2,G2,H2,
        A3 = 32,B3,C3,D3,E3,F3,G3,H3,
        A4 = 48,B4,C4,D4,E4,F4,G4,H4,
        A5 = 64,B5,C5,D5,E5,F5,G5,H5,
        A6 = 80,B6,C6,D6,E6,F6,G6,H6,
        A7 = 96,B7,C7,D7,E7,F7,G7,H7,
        A8 = 112,B8,C8,D8,E8,F8,G8,H8
};
enum RESULTS{
    DONT_KNOW = -3,ILLEGAL = -2,LOSS = -1,DRAW = 0,WIN = 1
};

#define RR    0x01
#define LL   -0x01
#define RU    0x11
#define LD   -0x11
#define UU    0x10
#define DD   -0x10
#define LU    0x0f
#define RD   -0x0f

#define RRU   0x12
#define LLD  -0x12
#define LLU   0x0e
#define RRD  -0x0e
#define RUU   0x21
#define LDD  -0x21
#define LUU   0x1f
#define RDD  -0x1f

#define UUU   0x20
#define DDD  -0x20
#define RRR   0x02
#define LLL  -0x02


#define KM       1
#define QM       2
#define RM       4
#define BM       8
#define NM      16
#define WPM     32
#define BPM     64
#define QRBM    14
#define KNM     17    

#define MAX_STR            256
#define MAX_MOVES          256
#define MAX_PLY              8
#define MAX_CPUS           256

/*square*/
#define file(x)          ((x) &  7)
#define rank(x)          ((x) >> 4)
#define file64(x)        ((x) &  7)
#define rank64(x)        ((x) >> 3)
#define SQ(x,y)          (((x) << 4) | (y))
#define SQ64(x,y)        (((x) << 3) | (y))
#define SQ8864(x)        SQ64(rank(x),file(x))
#define SQ6488(x)        SQ(rank64(x),file(x))
#define SQ6448(x)        ((x) - 8)
#define SQ4864(x)        ((x) + 8) 
#define MIRRORF(sq)      ((sq) ^ 0x07)
#define MIRRORR(sq)      ((sq) ^ 0x70)
#define MIRRORD(sq)      SQ(file(sq),rank(sq))
#define MIRRORF64(sq)    ((sq) ^ 0x07)
#define MIRRORR64(sq)    ((sq) ^ 0x38)
#define MIRRORD64(sq)    SQ64(file64(sq),rank64(sq))

/*distance*/
#undef MAX
#undef MIN
#define MAX(a, b)        (((a) > (b)) ? (a) : (b))
#define MIN(a, b)        (((a) < (b)) ? (a) : (b))
#define f_distance(x,y)  abs(file(x)-file(y))
#define r_distance(x,y)  abs(rank(x)-rank(y))
#define distance(x,y)    MAX(f_distance(x,y),r_distance(x,y))
#define is_light(x)      ((file(x)+rank(x)) & 1)
#define is_light64(x)    ((file64(x)+rank64(x)) & 1)

#define COLOR(x)         (col_tab[x])
#define PIECE(x)         (pic_tab[x]) 
#define DECOMB(c,x)      ((x) - ((c) ? 6 : 0)) 
#define COMBINE(c,x)     ((x) + ((c) ? 6 : 0)) 
#define is_white(x)      ((x) <= 6)
#define is_black(x)      ((x) > 6)
#define invert(x)        (!(x))
#define invert_color(x)  (((x) > 6) ? ((x) - 6) : ((x) + 6))

/*move*/
#define FROM_FLAG        0x000000ff
#define TO_FLAG          0x0000ff00
#define PIECE_FLAG       0x000f0000
#define CAPTURE_FLAG     0x00f00000
#define PROMOTION_FLAG   0x0f000000
#define CAP_PROM         0x0ff00000
#define FROM_TO_PROM     0x0f00ffff
#define EP_FLAG          0x10000000
#define CASTLE_FLAG      0x20000000
#define m_from(x)        ((x) & FROM_FLAG)
#define m_to(x)          (((x) & TO_FLAG) >> 8)
#define m_piece(x)       (((x) & PIECE_FLAG) >> 16)
#define m_capture(x)     (((x) & CAPTURE_FLAG) >> 20)
#define m_promote(x)     (((x) & PROMOTION_FLAG) >> 24)
#define is_cap_prom(x)   ((x) & CAP_PROM)
#define is_ep(x)         ((x) & EP_FLAG)
#define is_castle(x)     ((x) & CASTLE_FLAG)

#define WSC_FLAG       1
#define WLC_FLAG       2
#define BSC_FLAG       4
#define BLC_FLAG       8
#define WSLC_FLAG      3
#define BSLC_FLAG     12 
#define WBC_FLAG      15

#define MYINT     UBMP64

/*
Type definitions
*/
typedef struct LIST{
    int   sq;
    LIST* prev;
    LIST* next;
}*PLIST;

typedef struct STACK{
    int move_st[MAX_MOVES];
    int count;
    int legal_moves;
    int fifty;
    int epsquare;
    int castle;
}*PSTACK;

/*
searcher
*/
typedef struct SEARCHER{
    int player;
    int opponent;
    int castle;
    int epsquare;
    int fifty;
    int temp_board[224];
    int* const board;
    PLIST list[128];
    PLIST plist[15];
    int ply;
    PSTACK pstack;
    STACK stack[MAX_PLY];

    int used;
    INFO info;
    UBMP8 temp_block[BLOCK_SIZE];
    
    SEARCHER();
    ~SEARCHER();
    int   blocked(int,int) const;
    int   attacks(int,int) const;
    void  pcAdd(int,int);
    void  pcRemove(int,int);
    void  pcSwap(int,int);
    void  do_move(const int&);
    void  undo_move(const int&);
    void  gen_all();
    void  set_pos(
        int side, int* piece,int* square);
    void  clear_pos(int* piece,int* square);
    int get_score(int alpha,int beta,
        int side, int* piece,int* square);
    int get_children_score(int alpha,int beta,
        int side, int* piece,int* square, bool onlyEp);
    void get_index(MYINT& pos_index,UBMP32& tab_index,
        int side, int* piece,int* square);
} *PSEARCHER;

/*
inline piece list functions
*/
FORCEINLINE void SEARCHER::pcAdd(int pic,int sq) {
    PLIST* pHead;
    PLIST pPc;
    pHead = &plist[pic];
    pPc = list[sq];
    if(!(*pHead)) {
        (*pHead) = pPc;
        (*pHead)->next = 0;
        (*pHead)->prev = 0;
    } else {
        pPc->next = (*pHead)->next;
        if((*pHead)->next) (*pHead)->next->prev = pPc;
        (*pHead)->next = pPc;
        pPc->prev = (*pHead);
    }
};
FORCEINLINE void SEARCHER::pcRemove(int pic,int sq) {
    PLIST* pHead;
    PLIST pPc;
    pHead = &plist[pic];
    pPc = list[sq];
    if(pPc->next) pPc->next->prev = pPc->prev;
    if(pPc->prev) pPc->prev->next = pPc->next;
    if((*pHead) == pPc) (*pHead) = (*pHead)->next;
};
FORCEINLINE void SEARCHER::pcSwap(int from,int to) {
    PLIST pPc;
    PLIST& pTo = list[to];
    PLIST& pFrom = list[from];
    pPc = pTo;
    pTo = pFrom;
    pFrom = pPc;
    pTo->sq = to;
    pFrom->sq = from;
}

/*
globals
*/
extern const int col_tab[15];
extern const int pic_tab[15];
extern const int pawn_dir[2];

extern void init_sqatt();
extern void init_indices();

const char piece_name[] = "_KQRBNPkqrbnp_";
const char rank_name[] = "12345678";
const char file_name[] = "abcdefgh";
const char col_name[] = "WwBb";
const char cas_name[] = "KQkq";

extern const UBMP8* const _sqatt_pieces;
extern const BMP8* const _sqatt_step;
#define sqatt_pieces(sq)        _sqatt_pieces[sq]
#define sqatt_step(sq)          _sqatt_step[sq]

/*
Some defs
*/
#define MAX_PIECES  9
#define rotF  1
#define rotR  2
#define rotD  4

/*
Enumerator
*/
struct ENUMERATOR {
    int piece[MAX_PIECES];
    int square[MAX_PIECES];
    int res1[MAX_PIECES];
    int res2[MAX_PIECES];
    MYINT index[MAX_PIECES];
    MYINT divisor[MAX_PIECES];
    int n_piece;
    int n_pawn;
    int player;
    int king_loc;
    int pawn_loc;
    MYINT size;
    char name[16];

    ENUMERATOR() {
        n_piece = 0;
        n_pawn = 0;
        size = 1;
        player = white;
    }
    void add(int pc) {
        piece[n_piece++] = pc;
        if(PIECE(pc) == pawn)
            n_pawn++;
    }
    void add(int pc,int sq) {
        piece[n_piece] = pc;
        square[n_piece] = sq;
        n_piece++;
        if(PIECE(pc) == pawn)
            n_pawn++;
    }
    void add(int side,int* piece) {
        player = side;
        for(int i = 0;i < MAX_PIECES && piece[i];i++)
            add(piece[i]);
    }
    void clear() {
        n_piece = 0;
        n_pawn = 0;
        size = 1;
        player = white;
    }
    void init();
    void sort(int type);
    void check_flip();
    bool get_index(MYINT&,bool = false);
};
/*
EGBB
*/
class EGBB : public COMP_INFO {
public:
    UBMP32 id;
    char name[256];
    UBMP8*  table;
    int  state;
    bool use_search;
    bool is_loaded;
    LOCK lock;
    ENUMERATOR enumerator;
    EGBB() {
        is_loaded = false;
        use_search = false;
        table = 0;
        l_create(lock);
    }
    ~EGBB();
    static char path[256];
    static std::unordered_map<int,EGBB*> egbbs;
    static LRU_CACHE LRUcache;

    void open(int egbb_state);
    int get_score(MYINT,PSEARCHER);
    static int GetIndex(ENUMERATOR* penum);
};

/*Searchers*/
extern SEARCHER searchers[MAX_CPUS];
extern LOCK searcher_lock;
/*
End
*/
#endif
