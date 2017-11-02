#include "common.h"

/*
Initialize index tables and pointers
*/

static const UBMP8 K_TR[64] = {
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 4, 5, 6, 6, 5, 4, 1,
    2, 5, 7, 8, 8, 7, 5, 2,
    3, 6, 8, 9, 9, 8, 6, 3,
    3, 6, 8, 9, 9, 8, 6, 3,
    2, 5, 7, 8, 8, 7, 5, 2,
    1, 4, 5, 6, 6, 5, 4, 1,
    0, 1, 2, 3, 3, 2, 1, 0
};
static const UBMP8 K1_TR[64] = {
    0, 1, 2, 3, 4, 5, 6, 7,
    1, 8, 9,10,11,12,13,14,
    2, 9,15,16,17,18,19,20,
    3,10,16,21,22,23,24,25,
    4,11,17,22,26,27,28,29,
    5,12,18,23,27,30,31,32,
    6,13,19,24,28,31,33,34,
    7,14,20,25,29,32,34,35
};
static const UBMP8 K2_TR[64] = {
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 9,10,11,11,10, 9, 8,
   12,13,14,15,15,14,13,12,
   16,17,18,19,19,18,17,16,
   20,21,22,23,23,22,21,20,
   24,25,26,27,27,26,25,24,
   28,29,30,31,31,30,29,28
};
static const UBMP8 _mirror64SL[] = {
     0, 0, 0, 0, 0, 0, 0, 0,
    48,40,32,24,16, 8,55,47,
    39,31,23,15,49,41,33,25,
    17, 9,54,46,38,30,22,14,
    50,42,34,26,18,10,53,45,
    37,29,21,13,51,43,35,27,
    19,11,52,44,36,28,20,12,
     0, 0, 0, 0, 0, 0, 0, 0
};
static const UBMP8 _mirrorSL64[] = {
     0, 0, 0, 0, 0, 0, 0, 0,
    13,25,37,49,55,43,31,19,
    12,24,36,48,54,42,30,18,
    11,23,35,47,53,41,29,17,
    10,22,34,46,52,40,28,16,
     9,21,33,45,51,39,27,15,
     8,20,32,44,50,38,26,14,
     0, 0, 0, 0, 0, 0, 0, 0
};

BMP16 KK_index[4096];
BMP16 KK_WP_index[4096];
BMP16 KK_rotation[4096];
BMP16 KK_WP_rotation[4096];
BMP16 KK_square[462];
BMP16 KK_WP_square[1806];

#define SQ64SL(x)   _mirror64SL[x]
#define SQSL64(x)   _mirrorSL64[x]
/*
Initialize index tables and pointers
*/
void init_indices() {
    int temp[2048];
    int index;
    int i,j;
    int u1,u2;
    int rot,val;

    for( i = 0; i < 4096; i++) {
        KK_index[i] = ILLEGAL;
        KK_WP_index[i] = ILLEGAL;
    }

    /*without pawn*/
    for( i = 0; i < 2048; i++) {
        temp[i] = ILLEGAL;
    }

    index = 0;
    for(i = 0;i < 64;i++) {
        for(j = 0;j < 64;j++) {
            if(distance(SQ6488(i),SQ6488(j)) <= 1)
                continue;
            /*rotations*/
            u1 = i;
            u2 = j;
            rot = 0;

            if(file64(u1) > FILED) {
                u1 = MIRRORF64(u1);
                u2 = MIRRORF64(u2);
                rot ^= rotF;
            }
            if(rank64(u1) > RANK4) {
                u1 = MIRRORR64(u1);
                u2 = MIRRORR64(u2);
                rot ^= rotR;
            }
            if(rank64(u1) > file64(u1)) {
                u1 = MIRRORD64(u1);
                u2 = MIRRORD64(u2);
                rot ^= rotD;
            }
            if(file64(u1) == rank64(u1)) {
                if(rank64(u2) > file64(u2)) {
                    u1 = MIRRORD64(u1);
                    u2 = MIRRORD64(u2);
                    rot ^= rotD;
                }
            }

            val = (u1 << 6) | u2;

            /*actual index*/
            if(file64(u1) == rank64(u1)) {
                u1 = K_TR[u1];
                u2 = K1_TR[u2];
            } else {
                u1 = K_TR[u1];
                u2 = u2;
            }

            if(temp[u1 * 64 + u2] == ILLEGAL) {
                temp[u1 * 64 + u2] = index;
                KK_index[i * 64 + j] = index;
                KK_rotation[i * 64 + j] = rot;
                KK_square[index] = val;
                index++;
            } else {
                KK_index[i * 64 + j] = temp[u1 * 64 + u2];
                KK_rotation[i * 64 + j] = rot;
            }
        }
    }
    /*with pawn*/
    for( i = 0; i < 2048; i++) {
        temp[i] = ILLEGAL;
    }

    index = 0;
    for( i = 0;i < 64;i++) {
        for( j = 0; j < 64;j++) {
            if(distance(SQ6488(i),SQ6488(j)) <= 1)
                continue;

            /*reflection*/
            u1 = i;
            u2 = j;
            rot = 0;
            if(file64(u1) > FILED) {
                u1 = MIRRORF64(u1);
                u2 = MIRRORF64(u2);
                rot ^= rotF;
            }

            val = (u1 << 6) | u2;
            /*actual index*/
            u1 = K2_TR[u1];
            u2 = u2;

            if(temp[u1 * 64 + u2] == ILLEGAL) {
                temp[u1 * 64 + u2] = index;
                KK_WP_index[i * 64 + j] = index;
                KK_WP_rotation[i * 64 + j] = rot;
                KK_WP_square[index] = val;
                index++;
            } else {
                KK_WP_index[i * 64 + j] = temp[u1 * 64 + u2];
                KK_WP_rotation[i * 64 + j] = rot;
            }
            /*end*/
        }
    }
}

/*
** Indexing and de-indexing functions for k similar pieces which
** have k! unique placements on a chess board. This method of binomial
** indexing is courtesy of syzygy of CCRL forum.
*/

/*
Lookup table for combination 
*/
static int combination[64][5] = {
      0,       0,       0,       0,       0,
      1,       0,       0,       0,       0,
      2,       1,       0,       0,       0,
      3,       3,       1,       0,       0,
      4,       6,       4,       1,       0,
      5,      10,      10,       5,       1,
      6,      15,      20,      15,       6,
      7,      21,      35,      35,      21,
      8,      28,      56,      70,      56,
      9,      36,      84,     126,     126,
     10,      45,     120,     210,     252,
     11,      55,     165,     330,     462,
     12,      66,     220,     495,     792,
     13,      78,     286,     715,    1287,
     14,      91,     364,    1001,    2002,
     15,     105,     455,    1365,    3003,
     16,     120,     560,    1820,    4368,
     17,     136,     680,    2380,    6188,
     18,     153,     816,    3060,    8568,
     19,     171,     969,    3876,   11628,
     20,     190,    1140,    4845,   15504,
     21,     210,    1330,    5985,   20349,
     22,     231,    1540,    7315,   26334,
     23,     253,    1771,    8855,   33649,
     24,     276,    2024,   10626,   42504,
     25,     300,    2300,   12650,   53130,
     26,     325,    2600,   14950,   65780,
     27,     351,    2925,   17550,   80730,
     28,     378,    3276,   20475,   98280,
     29,     406,    3654,   23751,  118755,
     30,     435,    4060,   27405,  142506,
     31,     465,    4495,   31465,  169911,
     32,     496,    4960,   35960,  201376,
     33,     528,    5456,   40920,  237336,
     34,     561,    5984,   46376,  278256,
     35,     595,    6545,   52360,  324632,
     36,     630,    7140,   58905,  376992,
     37,     666,    7770,   66045,  435897,
     38,     703,    8436,   73815,  501942,
     39,     741,    9139,   82251,  575757,
     40,     780,    9880,   91390,  658008,
     41,     820,   10660,  101270,  749398,
     42,     861,   11480,  111930,  850668,
     43,     903,   12341,  123410,  962598,
     44,     946,   13244,  135751, 1086008,
     45,     990,   14190,  148995, 1221759,
     46,    1035,   15180,  163185, 1370754,
     47,    1081,   16215,  178365, 1533939,
     48,    1128,   17296,  194580, 1712304,
     49,    1176,   18424,  211876, 1906884,
     50,    1225,   19600,  230300, 2118760,
     51,    1275,   20825,  249900, 2349060,
     52,    1326,   22100,  270725, 2598960,
     53,    1378,   23426,  292825, 2869685,
     54,    1431,   24804,  316251, 3162510,
     55,    1485,   26235,  341055, 3478761,
     56,    1540,   27720,  367290, 3819816,
     57,    1596,   29260,  395010, 4187106,
     58,    1653,   30856,  424270, 4582116,
     59,    1711,   32509,  455126, 5006386,
     60,    1770,   34220,  487635, 5461512,
     61,    1830,   35990,  521855, 5949147,
     62,    1891,   37820,  557845, 6471002,
     63,    1953,   39711,  595665, 7028847
};
/*
Accepts:
     sq[] = set of squares (unsorted)
       N = number of squares
Returns:
     Unique index calculated using binomial coefficients.
     Sorted squares in ascending order.
*/
int get_index_like(int* square, const int N) {
   register int index = 0,i,j,temp;
   for(i = 0;i < N;i++) {
       for(j = i + 1;j < N;j++) {
           if(square[j] < square[i]) {
               temp = square[i];
               square[i] = square[j];
               square[j] = temp;
           }
       }
   }
   for(i = 0;i < N;i++)
      index += combination[square[i]][i];
   return index;
}
/*
Accepts:
    index = unique index
Returns:
    sq[]  = set of N squares corresponding to the index
           in ascending order
*/
void get_squares_like(int* sq,const int N,const int index) {
   int comb;
   sq[0] = index;
   for(int i = N - 1;i >= 1;i--) {
      sq[i] = i;
      comb = 1;
      while(sq[0] >= comb) {
         sq[0] -= comb;
         comb = combination[++sq[i]][i - 1];
      }
   }
}

/*
 * globals
 */
static const int piece_v[15] = {
    0,0,975,500,326,325,100,0,975,500,326,325,100,0
};

/*
 * Streaming order for pieces for any of the following cases.
 *  - Optimized for better compression
 *  - Slicing egbbs
 *  - Highest mobility piece placed last for better cache use
 */
static const int piece_order[2][12] = {
    {bpawn,wpawn,bking,wking,bknight,wknight,bbishop,wbishop,brook,wrook,bqueen,wqueen}, 
    {wpawn,bpawn,wking,bking,wknight,bknight,wbishop,bbishop,wrook,brook,wqueen,bqueen}
};
/*
 * Original piece order
 */
static const int original_order[12] = {
    wking,wqueen,wrook,wbishop,wknight,wpawn,bking,bqueen,brook,bbishop,bknight,bpawn
};

/*
 * order pieces based on the streaming order used
 * by the generator.
 */     
void ENUMERATOR::sort(int type) {
    int i,j,pic,order,stronger;
    int vcount[2] = {0,0};

    /*check whether to switch sides*/
    for(i = 0;i < n_piece; i++)
        vcount[COLOR(piece[i])] += piece_v[piece[i]];

    /*who is stronger?*/
    if(vcount[white] > vcount[black]) stronger = white;
    else if(vcount[black] > vcount[white]) stronger = black;
    else stronger = white;

    /*ordered list*/
    for(i = 0;i < n_piece;i++) {
        res2[i] = piece[i];
        res1[i] = square[i];
    }

    order = 0;
    for(i = 0;i < 12;i++) {
        if(type == 0) pic = piece_order[stronger][i];
        else pic = original_order[i];
        for(j = 0;j < n_piece; j++) {
            if(res2[j] == pic) {
                piece[order] = res2[j];
                square[order] = res1[j];
                order++;
            }
        }
    }
}

/*
 * Initialize enumerator
 */
//#define SIMPLE

void ENUMERATOR::init() {
    int n_pawns = 0,n_pieces = 0,order = 0,
        i,j,pic;

    /*name*/
    for(i = 0;i < n_piece; i++) {
        name[i] = piece_name[piece[i]];
    }
    name[i++] = '.';
    name[i++] = (player == white) ? 'w':'b';
    name[i++] = 0;

    /*determine streaming order*/
    int vcount[2] = {0,0}, stronger;
    for(i = 0;i < n_piece; i++)
        vcount[COLOR(piece[i])] += piece_v[piece[i]];

    if(vcount[white] > vcount[black]) stronger = white;
    else if(vcount[black] > vcount[white]) stronger = black;
    else stronger = white;

    /*ordered list*/
    for(i = 0;i < n_piece;i++) 
        res2[i] = piece[i];

    for(i = 0;i < 12;i++) {
        pic = piece_order[stronger][i];
        for(j = 0;j < n_piece; j++) {
            if(res2[j] == pic) {
                if(PIECE(pic) == king) {
                    index[order] = 1;
                    n_pieces++;
                } else if (PIECE(pic) == pawn) {
                    if(!n_pawns) pawn_loc = order;
#ifndef SIMPLE
                    index[order] = 48 - n_pawns;
#else
                    index[order] = 48;
#endif
                    n_pawns++;
                } else {
#ifndef SIMPLE
                    index[order] = 64 - n_pieces - n_pawns;
#else
                    index[order] = 64;
#endif
                    n_pieces++;
                }
                piece[order] = pic;
                order++;
            }
        }
    }
    /*kings*/
    for(i = 0;i < n_piece;i++) {
        if(PIECE(piece[i]) == king) {
            king_loc = i;
            if(n_pawns)
                index[i + 1] = 1806;
            else
                index[i + 1] = 462;
            break;
        }
    }

#ifndef SIMPLE
    /*same pieces*/
    for(i = 1;i < n_piece; i++) {
        for(j = i + 1;j < n_piece;j++) {
            if(piece[i] != piece[j]) break;
        }
        j--;

        /*for now limit to five similar pieces*/
        if(j - i > 5) 
            j = i + 5;

        /*calculate number of unique positions*/
        if(i != j) {
            for(int k = i;k < j;k++)  {
                index[j] *= index[k];
                index[k] = 1;
            }
            switch(j - i) {
            case 1: index[j] /= (2 * 1); break;
            case 2: index[j] /= (3 * 2 * 1); break;
            case 3: index[j] /= (4 * 3 * 2 * 1); break;
            case 4: index[j] /= (5 * 4 * 3 * 2 * 1); break;
            case 5: index[j] /= (6 * 5 * 4 * 3 * 2 * 1); break;
            }
        }
        i = j;
    }
#endif
    /*divisor*/
    divisor[n_piece - 1] = 1;
    for(i = 0;i < n_piece; i++) {
        j = n_piece - 1 - i;
        size *= index[j];
        if(j >= 1) 
            divisor[j - 1] = size;
    }
}
/*
 * Get index of position
 */
bool ENUMERATOR::get_index(MYINT& pindex,bool special) {
    MYINT temp;
    int i,k,rot,sq,ispawn,N;

#ifdef SIMPLE
    for(i = 0;i < n_piece;i++) {
        for(k = 0;k < i;k++) {
            if(square[i] == square[k])
                return false;
        }
    }
#else
    /*illegal pawn placement on kings' square*/
    if(n_pawn) {
        for(i = pawn_loc;i < pawn_loc + n_pawn; i++) {
            if(square[i] == square[king_loc] || square[i] == square[king_loc + 1]) {
                return false;
            }
        }
    }
#endif

    /*save*/
    memcpy(res2,square,n_piece * sizeof(int));

    /*rotate*/
    if(n_pawn) 
        rot = KK_WP_rotation[square[king_loc] * 64 + square[king_loc + 1]];
    else 
        rot = KK_rotation[square[king_loc] * 64 + square[king_loc + 1]];
    if(rot || special) {
        for(i = 0;i < n_piece;i++) {
            sq = square[i];
            if(rot & rotF) sq = MIRRORF64(sq);
            if(rot & rotR) sq = MIRRORR64(sq);
            if(rot & rotD) sq = MIRRORD64(sq);
            if(special) sq = MIRRORD64(sq);
            square[i] = sq;
        }
    }

    /*legal placement based on other piece location*/
    for(i = n_piece - 1;i >= 0; --i) {

        /*skip kings*/
        if(i == king_loc + 1) {
            --i;
            continue;
        }

        /*adjust pawn squares here*/
        if(i == pawn_loc + n_pawn - 1) {
            for(k = pawn_loc;k < pawn_loc + n_pawn; k++) {
                square[k] = SQSL64(square[k]);
            }
        }

#ifndef SIMPLE
        /*start and finish*/
        int k,l,temp,start,finish;
        ispawn = (PIECE(piece[i]) == pawn);
        start = (ispawn ? pawn_loc : 0);
        finish = i;
        if(i > 1 && index[i - 1] == 1) {
            finish = i - 1;
            if(i > 2 && index[i - 2] == 1) {
                finish = i - 2;
                if(i > 3 && index[i - 3] == 1) {
                    finish = i - 3;
                    if(i > 4 && index[i - 4] == 1) {
                        finish = i - 4;
                    }
                }
            }
        }

        /*legal placement of pieces/pawns*/
        for(k = start; k < finish;k++) 
            res1[k] = square[k];
        for(k = start; k < finish;k++) {
            for(l = k + 1; l < finish;l++) {
                if(res1[k] > res1[l]) {
                    temp = res1[k];
                    res1[k] = res1[l];
                    res1[l] = temp;
                }
            }
        }
        for(k = finish - 1; k >= start ; k--) {
            if(square[i] >= res1[k]) 
                square[i]--;
        }
#endif
        /*end*/
    }

    /*primary locations*/
    pindex = 0;
    for(i = n_piece - 1;i >= 0; --i) {
        /*king squares*/
        if(i == king_loc + 1) {
            if(n_pawn) 
                temp = KK_WP_index[square[i - 1] * 64 + square[i]];
            else 
                temp = KK_index[square[i - 1] * 64 + square[i]];
            pindex += temp * divisor[i];
            --i;
            continue;
        }

        /*others*/
        ispawn = (PIECE(piece[i]) == pawn);
        N = 0;
#ifndef SIMPLE
        for(;i >= 2;i--) {
            if(index[i - 1] == 1) N++;
            else break;
        }
        if(N) {
            if(ispawn) {
                for(k = 0;k <= N;k++) 
                    square[i + k] = SQ6448(square[i + k]);
            }
            temp = get_index_like(&square[i],N + 1);
        } else 
#endif
        {
            if(ispawn) 
                temp = SQ6448(square[i]);
            else 
                temp = square[i];   
        }
        pindex += temp * divisor[i + N];
    }
    
    /*restore*/
    memcpy(square,res2,n_piece * sizeof(int));

    return true;
}
/*
flip sides ?
*/
void ENUMERATOR::check_flip() {
    int i,pic;
    int count[2] = {0,0},vcount[2] = {0,0};

    /*check whether to switch sides*/
    for(i = 0;i < n_piece; i++) {
        pic = piece[i];
        count[COLOR(pic)]++;
        vcount[COLOR(pic)] += piece_v[pic];
    }

    if(count[white] > count[black]);
    else if(count[white] == count[black] 
        && vcount[white] >= vcount[black]);
    else  {
        player = invert(player);
        for(i = 0;i < n_piece;i++) {
            piece[i] = invert_color(piece[i]);
            square[i] = MIRRORR64(square[i]);
        }
    }

}

