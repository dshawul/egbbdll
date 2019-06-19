#include "common.h"

/*external variables*/
const int pawn_dir[2] = {UU,DD};
const int col_tab[15] = {neutral,white,white,white,white,white,white,
black,black,black,black,black,black,neutral};
const int pic_tab[15] = {blank,king,queen,rook,bishop,knight,pawn,
king,queen,rook,bishop,knight,pawn,elephant};

/*
MOVES
*/
void SEARCHER::do_move(const int& move) {

    int from = m_from(move),to = m_to(move),sq;

    pstack->epsquare = epsquare;
    pstack->castle = castle;
    pstack->fifty = fifty;

    /*remove captured piece*/
    if(m_capture(move)) {
        if(is_ep(move)) {
            sq = to - pawn_dir[player];
        } else {
            sq = to;
        }
        pcRemove(m_capture(move),sq);
        board[sq] = blank;
    }

    /*move piece*/
    if(m_promote(move)) {
        board[to] = m_promote(move);
        board[from] = blank;
        pcAdd(m_promote(move),to);
        pcRemove(COMBINE(player,pawn),from);
    } else {
        board[to] = board[from];
        board[from] = blank;
        pcSwap(from,to);
    }

    /*move castle*/
    if(is_castle(move)) {
        int fromc,toc;
        if(to > from) {
           fromc = to + RR;
           toc = to + LL;
        } else {
           fromc = to + 2*LL;
           toc = to + RR;
        }
        board[toc] = board[fromc];
        board[fromc] = blank;
        pcSwap(fromc,toc);
    } 

    /*update current state*/
    epsquare = 0;
    fifty++;
    if(PIECE(m_piece(move)) == pawn) {
        fifty = 0;
        if(to - from == (2 * pawn_dir[player])) {
            epsquare = ((to + from) >> 1);
        }
    }else if(m_capture(move)) {
        fifty = 0;
    }
    int p_castle = castle;
    if(from == E1 || to == A1 || from == A1) castle &= ~WLC_FLAG;
    if(from == E1 || to == H1 || from == H1) castle &= ~WSC_FLAG;
    if(from == E8 || to == A8 || from == A8) castle &= ~BLC_FLAG;
    if(from == E8 || to == H8 || from == H8) castle &= ~BSC_FLAG;

    player = invert(player);
    opponent = invert(opponent);

    ply++;
    pstack++;
}

void SEARCHER::undo_move(const int& move) {
    int to,from,sq;
    pstack--;
    ply--;

    player = invert(player);
    opponent = invert(opponent);

    epsquare = pstack->epsquare;
    castle = pstack->castle;
    fifty = pstack->fifty;

    to = m_to(move);
    from = m_from(move);

    /*unmove castle*/
    if(is_castle(move)) {
        int fromc,toc;
        if(to > from) {
           fromc = to + LL;
           toc = to + RR;
        } else {
           fromc = to + RR;
           toc = to + 2*LL;
        }
        board[toc] = board[fromc];
        board[fromc] = blank;
        pcSwap(fromc,toc);
    } 

    /*unmove piece*/
    if(m_promote(move)) {
        board[from] = COMBINE(player,pawn);
        board[to] = blank;
        pcAdd(COMBINE(player,pawn),from);
        pcRemove(m_promote(move),to);

    } else {
        board[from] = board[to];
        board[to] = blank;
        pcSwap(to,from);
    }

    /*insert captured piece*/
    if(m_capture(move)) {
        if(is_ep(move)) {
            sq = to - pawn_dir[player];
        } else {
            sq = to;
        }
        board[sq] = m_capture(move);
        pcAdd(m_capture(move),sq);
    }
}
/*
generate all
*/
#define NK_MOVES(dir) {                                                 \
        to = from + dir;                                                \
        if(board[to] == blank)                                          \
            *pmove++ = tmove | (to<<8);                                 \
        else if(COLOR(board[to]) == opponent)                           \
            *pmove++ = tmove | (to<<8) | (board[to]<<20);               \
};
#define BRQ_MOVES(dir) {                                                \
        to = from + dir;                                                \
        while(board[to] == blank) {                                     \
            *pmove++ = tmove | (to<<8);                                 \
            to += dir;                                                  \
        }                                                               \
        if(COLOR(board[to]) == opponent)                                \
            *pmove++ = tmove | (to<<8) | (board[to]<<20);               \
};

void SEARCHER::gen_all() {
    int* pmove = &pstack->move_st[pstack->count],*spmove = pmove,tmove;
    int  from,to;
    PLIST current;
    
    if(player == white) {
        /*enpassant*/
        if(epsquare) {
            from = epsquare + LD;
            if(board[from] == wpawn)
                *pmove++ = from | (epsquare<<8) | (wpawn<<16) | (bpawn<<20) | EP_FLAG;
            
            from = epsquare + RD;
            if(board[from] == wpawn)
                *pmove++ = from | (epsquare<<8) | (wpawn<<16) | (bpawn<<20) | EP_FLAG;
        }
        /*castling*/
        if((castle & WSLC_FLAG) && !attacks(black,E1)) {
            if(castle & WSC_FLAG &&
                board[F1] == blank &&
                board[G1] == blank &&
                !attacks(black,F1) &&
                !attacks(black,G1))
                *pmove++ = E1 | (G1<<8) | (wking<<16) | CASTLE_FLAG;
            if(castle & WLC_FLAG &&
                board[B1] == blank &&
                board[C1] == blank &&
                board[D1] == blank &&
                !attacks(black,C1) &&
                !attacks(black,D1)) {
                *pmove++ = E1 | (C1<<8) | (wking<<16) | CASTLE_FLAG;
            }
        }
        /*knight*/
        current = plist[wknight];
        while(current) {
            from = current->sq;
            tmove = from | (wknight<<16);
            NK_MOVES(RRU);
            NK_MOVES(LLD);
            NK_MOVES(RUU);
            NK_MOVES(LDD);
            NK_MOVES(LLU);
            NK_MOVES(RRD);
            NK_MOVES(RDD);
            NK_MOVES(LUU);
            current = current->next;
        }
        /*bishop*/
        current = plist[wbishop];
        while(current) {
            from = current->sq;
            tmove = from | (wbishop<<16);
            BRQ_MOVES(RU);
            BRQ_MOVES(LD);
            BRQ_MOVES(LU);
            BRQ_MOVES(RD);
            current = current->next;
        }
        /*rook*/
        current = plist[wrook];
        while(current) {
            from = current->sq;
            tmove = from | (wrook<<16);
            BRQ_MOVES(UU);
            BRQ_MOVES(DD);
            BRQ_MOVES(RR);
            BRQ_MOVES(LL);
            current = current->next;
        }
        /*queen*/
        current = plist[wqueen];
        while(current) {
            from = current->sq;
            tmove = from | (wqueen<<16);
            BRQ_MOVES(RU);
            BRQ_MOVES(LD);
            BRQ_MOVES(LU);
            BRQ_MOVES(RD);
            BRQ_MOVES(UU);
            BRQ_MOVES(DD);
            BRQ_MOVES(RR);
            BRQ_MOVES(LL);
            current = current->next;
        }
        /*king*/
        from = plist[wking]->sq;
        tmove = from | (wking<<16);
        NK_MOVES(RU);
        NK_MOVES(LD);
        NK_MOVES(LU);
        NK_MOVES(RD);
        NK_MOVES(UU);
        NK_MOVES(DD);
        NK_MOVES(RR);
        NK_MOVES(LL);
        
        /*pawn*/
        current = plist[wpawn];
        while(current) {
            from = current->sq;
            //caps
            to = from + RU;
            if(COLOR(board[to]) == black) {
                if(rank(to) == RANK8) {
                    tmove = from | (to<<8) | (wpawn<<16) | (board[to]<<20);
                    *pmove++ = tmove | (wqueen<<24);
                    *pmove++ = tmove | (wknight<<24);
                    *pmove++ = tmove | (wrook<<24);
                    *pmove++ = tmove | (wbishop<<24);
                } else {
                    *pmove++ = from | (to<<8) | (wpawn<<16) | (board[to]<<20);
                }
            }
            to = from + LU;
            if(COLOR(board[to]) == black) {
                if(rank(to) == RANK8) {
                    tmove = from | (to<<8) | (wpawn<<16) | (board[to]<<20);
                    *pmove++ = tmove | (wqueen<<24);
                    *pmove++ = tmove | (wknight<<24);
                    *pmove++ = tmove | (wrook<<24);
                    *pmove++ = tmove | (wbishop<<24);
                } else {
                    *pmove++ = from | (to<<8) | (wpawn<<16) | (board[to]<<20);
                }
            }
            //noncaps
            to = from + UU;
            if(board[to] == blank) {
                if(rank(to) == RANK8) {
                    if(board[to] == blank) {
                        tmove = from | (to<<8) | (wpawn<<16);
                        *pmove++ = tmove | (wqueen<<24);
                        *pmove++ = tmove | (wknight<<24);
                        *pmove++ = tmove | (wrook<<24);
                        *pmove++ = tmove | (wbishop<<24);
                    }
                } else {
                    *pmove++ = from | (to<<8) | (wpawn<<16);
                    if(rank(from) == RANK2) {
                        to += UU;
                        if(board[to] == blank)
                            *pmove++ = from | (to<<8) | (wpawn<<16);
                    }
                }
            }   
            current = current->next;
        }
    } else {
        /*enpassant*/
        if(epsquare) {
            from = epsquare + RU;
            if(board[from] == bpawn)
                *pmove++ = from | (epsquare<<8) | (bpawn<<16) | (wpawn<<20) | EP_FLAG;
            
            from = epsquare + LU;
            if(board[from] == bpawn)
                *pmove++ = from | (epsquare<<8) | (bpawn<<16) | (wpawn<<20) | EP_FLAG;
        }
        /*end*/
        /*castling*/
        if((castle & BSLC_FLAG) && !attacks(white,E8)) {
            if(castle & BSC_FLAG &&
                board[F8] == blank &&
                board[G8] == blank &&
                !attacks(white,F8) &&
                !attacks(white,G8))
                *pmove++ = E8 | (G8<<8) | (bking<<16) | CASTLE_FLAG;
            if(castle & BLC_FLAG &&
                board[B8] == blank &&
                board[C8] == blank &&
                board[D8] == blank &&
                !attacks(white,C8) &&
                !attacks(white,D8)) {
                *pmove++ = E8 | (C8<<8) | (bking<<16) | CASTLE_FLAG;
            }
        }

        /*knight*/
        current = plist[bknight];
        while(current) {
            from = current->sq;
            tmove = from | (bknight<<16);
            NK_MOVES(RRU);
            NK_MOVES(LLD);
            NK_MOVES(RUU);
            NK_MOVES(LDD);
            NK_MOVES(LLU);
            NK_MOVES(RRD);
            NK_MOVES(RDD);
            NK_MOVES(LUU);
            current = current->next;
        }
        /*bishop*/
        current = plist[bbishop];
        while(current) {
            from = current->sq;
            tmove = from | (bbishop<<16);
            BRQ_MOVES(RU);
            BRQ_MOVES(LD);
            BRQ_MOVES(LU);
            BRQ_MOVES(RD);
            current = current->next;
        }
        /*rook*/
        current = plist[brook];
        while(current) {
            from = current->sq;
            tmove = from | (brook<<16);
            BRQ_MOVES(UU);
            BRQ_MOVES(DD);
            BRQ_MOVES(RR);
            BRQ_MOVES(LL);
            current = current->next;
        }
        /*queen*/
        current = plist[bqueen];
        while(current) {
            from = current->sq;
            tmove = from | (bqueen<<16);
            BRQ_MOVES(RU);
            BRQ_MOVES(LD);
            BRQ_MOVES(LU);
            BRQ_MOVES(RD);
            BRQ_MOVES(UU);
            BRQ_MOVES(DD);
            BRQ_MOVES(RR);
            BRQ_MOVES(LL);
            current = current->next;
        }
        
        /*king*/
        from = plist[bking]->sq;
        tmove = from | (bking<<16);
        NK_MOVES(RU);
        NK_MOVES(LD);
        NK_MOVES(LU);
        NK_MOVES(RD);
        NK_MOVES(UU);
        NK_MOVES(DD);
        NK_MOVES(RR);
        NK_MOVES(LL);
        
        /*pawn*/
        current = plist[bpawn];
        while(current) {
            from = current->sq;
            //caps
            to = from + LD;
            if(COLOR(board[to]) == white) {
                if(rank(to) == RANK1) {
                    tmove = from | (to<<8) | (bpawn<<16) | (board[to]<<20);
                    *pmove++ = tmove | (bqueen<<24);
                    *pmove++ = tmove | (bknight<<24);
                    *pmove++ = tmove | (brook<<24);
                    *pmove++ = tmove | (bbishop<<24);
                } else {
                    *pmove++ = from | (to<<8) | (bpawn<<16) | (board[to]<<20);
                }
            }
            to = from + RD;
            if(COLOR(board[to]) == white) {
                if(rank(to) == RANK1) {
                    tmove = from | (to<<8) | (bpawn<<16) | (board[to]<<20);
                    *pmove++ = tmove | (bqueen<<24);
                    *pmove++ = tmove | (bknight<<24);
                    *pmove++ = tmove | (brook<<24);
                    *pmove++ = tmove | (bbishop<<24);
                } else {
                    *pmove++ = from | (to<<8) | (bpawn<<16) | (board[to]<<20);
                }
            }
            //noncaps
            to = from + DD;
            if(board[to] == blank) {
                if(rank(to) == RANK1) {
                    tmove = from | (to<<8) | (bpawn<<16);
                    *pmove++ = tmove | (bqueen<<24);
                    *pmove++ = tmove | (bknight<<24);
                    *pmove++ = tmove | (brook<<24);
                    *pmove++ = tmove | (bbishop<<24);
                } else {
                    *pmove++ = from | (to<<8) | (bpawn<<16);
                    if(rank(from) == RANK7) {
                        to += DD;
                        if(board[to] == blank)
                            *pmove++ = from | (to<<8) | (bpawn<<16);
                    }
                }
            }   
            current = current->next;
        }

    }
    /*count*/
    pstack->count += int(pmove - spmove);
}
/*constructor*/
SEARCHER::SEARCHER() : board(&temp_board[48])
{
    int sq;
    for(sq = 0;sq < 128; sq++)
        list[sq] = new LIST;
    for(sq = 0;sq < 48;sq++)
        temp_board[sq] = elephant;
    for(sq = 176;sq < 224;sq++)
        temp_board[sq] = elephant;
    for(sq = A1;sq < A1 + 128;sq++) {
        if(sq & 0x88)
           board[sq] = elephant;
        else
           board[sq] = blank;
    }

    used = 0;

    ply = 0;
    pstack = stack + 0;
    for(int i = wking;i < elephant;i++) {
       plist[i] = 0;
    }
    for(sq = A1;sq <= H8;sq++) {
        if(!(sq & 0x88)) { 
            list[sq]->sq = sq;
            list[sq]->prev = 0;
            list[sq]->next = 0;
        }
    }
}
/*Distructor*/
SEARCHER::~SEARCHER() {
    for(int sq = 0;sq < 128; sq++)
        delete list[sq];
}
/*clear squares*/
void SEARCHER::clear_pos(int* piece,int* square) {
    register int i;
    for(i = 0;i < MAX_PIECES && piece[i];i++) {
        int sq = SQ6488(square[i]);
        pcRemove(piece[i],sq);
        board[sq] = blank;
    }
}
/*set pos*/
void SEARCHER::set_pos(int side,int* piece,int* square) {
    register int i;
    for(i = 0;i < MAX_PIECES && piece[i];i++) {
        int sq = SQ6488(square[i]);
        board[sq] = piece[i];
        pcAdd(piece[i],sq);
    }
    epsquare = SQ6488(square[i]);
    player = side;
    opponent = invert(side);
    fifty = 0;
    castle = 0;
    ply = 0;
    pstack = stack + 0;
}
/*attacks*/
static const UBMP8 t_sqatt_pieces[] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0,
  6,  0,  0,  0,  0,  0,  0, 10,  0,  0, 10,  0,  0,  0,  0,  0,
  6,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0, 10,  0,  0,  0,  0,
  6,  0,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,
  6,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0,  0,
  6,  0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 16,
  6, 16, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 75,
  7, 75, 16,  0,  0,  0,  0,  0,  0,  6,  6,  6,  6,  6,  6,  7,
  0,  7,  6,  6,  6,  6,  6,  6,  0,  0,  0,  0,  0,  0, 16, 43,
  7, 43, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 16,
  6, 16, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0,  0,
  6,  0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,
  6,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0,
  6,  0,  0,  0,  0, 10,  0,  0,  0,  0, 10,  0,  0,  0,  0,  0,
  6,  0,  0,  0,  0,  0, 10,  0,  0, 10,  0,  0,  0,  0,  0,  0,
  6,  0,  0,  0,  0,  0,  0, 10,  0,  0,  0,  0,  0,  0,  0,  0
};

static const BMP8 t_sqatt_step[] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,-17,  0,  0,  0,  0,  0,  0,
-16,  0,  0,  0,  0,  0,  0,-15,  0,  0,-17,  0,  0,  0,  0,  0,
-16,  0,  0,  0,  0,  0,-15,  0,  0,  0,  0,-17,  0,  0,  0,  0,
-16,  0,  0,  0,  0,-15,  0,  0,  0,  0,  0,  0,-17,  0,  0,  0,
-16,  0,  0,  0,-15,  0,  0,  0,  0,  0,  0,  0,  0,-17,  0,  0,
-16,  0,  0,-15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,-17,  0,
-16,  0,-15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,-17,
-16,-15,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1,
  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0, 15,
 16, 17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,
 16,  0, 17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,
 16,  0,  0, 17,  0,  0,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0,
 16,  0,  0,  0, 17,  0,  0,  0,  0,  0,  0, 15,  0,  0,  0,  0,
 16,  0,  0,  0,  0, 17,  0,  0,  0,  0, 15,  0,  0,  0,  0,  0,
 16,  0,  0,  0,  0,  0, 17,  0,  0, 15,  0,  0,  0,  0,  0,  0,
 16,  0,  0,  0,  0,  0,  0, 17,  0,  0,  0,  0,  0,  0,  0,  0
};

const UBMP8* const _sqatt_pieces = t_sqatt_pieces + 0x80;
const BMP8* const _sqatt_step = t_sqatt_step + 0x80;

/*any blocking piece in between?*/
int SEARCHER::blocked(int from, int to) const {
    register int step,sq;
    if(step = sqatt_step(to - from)) {
        sq = from + step;
        while(board[sq] == blank && (sq != to)) sq += step;
        return (sq != to);
    }
    return true;
};

/*is square attacked by color?*/
int SEARCHER::attacks(int col,int sq) const {
    register PLIST current;
    
    if(col == white) {
        /*pawn*/
        if(board[sq + LD] == wpawn) return true;
        if(board[sq + RD] == wpawn) return true;
        /*knight*/
        current = plist[wknight];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & NM)
                return true;
            current = current->next;
        }
        /*bishop*/
        current = plist[wbishop];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & BM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*rook*/
        current = plist[wrook];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & RM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*queen*/
        current = plist[wqueen];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & QM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*king*/
        if(sqatt_pieces(sq - plist[wking]->sq) & KM)
            return true;
    } else if(col == black) {
        /*pawn*/
        if(board[sq + RU] == bpawn) return true;
        if(board[sq + LU] == bpawn) return true;
        /*knight*/
        current = plist[bknight];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & NM)
                return true;
            current = current->next;
        }
        /*bishop*/
        current = plist[bbishop];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & BM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*rook*/
        current = plist[brook];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & RM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*queen*/
        current = plist[bqueen];
        while(current) {
            if(sqatt_pieces(sq - current->sq) & QM)
                if(!blocked(current->sq,sq))
                    return true;
            current = current->next;
        }
        /*king*/
        if(sqatt_pieces(sq - plist[bking]->sq) & KM)
            return true;
    }
    return false;
}
