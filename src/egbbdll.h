#ifndef __EGBBPROBE__
#define __EGBBPROBE__

enum {_WHITE,_BLACK};
enum {_EMPTY,_WKING,_WQUEEN,_WROOK,_WBISHOP,_WKNIGHT,_WPAWN,
             _BKING,_BQUEEN,_BROOK,_BBISHOP,_BKNIGHT,_BPAWN};
enum {LOAD_NONE,LOAD_4MEN,SMART_LOAD,LOAD_5MEN,LOAD_5MEN_LZ};

#define _NOTFOUND 99999

#if defined (_WIN32)
#   define _CDECL __cdecl
#ifdef DLL_EXPORT
#   define DLLExport extern "C" __declspec(dllexport)
#else
#   define DLLExport extern "C" __declspec(dllimport)
#endif
#else
#   define _CDECL
#   define DLLExport extern "C"
#endif

/*4 men*/
DLLExport int  _CDECL probe_egbb(int player, int w_king, int b_king,
               int piece1 = _EMPTY, int square1 = 0,
               int piece2 = _EMPTY, int square2 = 0);
DLLExport void _CDECL load_egbb(char* path);
/*5 men*/
DLLExport int  _CDECL probe_egbb_5men(int player, int w_king, int b_king,
               int piece1 = _EMPTY, int square1 = 0,
               int piece2 = _EMPTY, int square2 = 0,
               int piece3 = _EMPTY, int square3 = 0
               );
DLLExport void _CDECL load_egbb_5men(char* path,int cache_size = 4194304, int load_options = LOAD_4MEN);
/*X men*/
DLLExport int  _CDECL probe_egbb_fen(char* fen);
DLLExport int  _CDECL probe_egbb_xmen(int player, int* piece, int* square);
DLLExport void _CDECL load_egbb_xmen(char* path,int cache_size = 4194304, int load_options = LOAD_4MEN);
/*private*/
DLLExport void _CDECL load_egbb_into_ram(int side,int* piece);
DLLExport void _CDECL unload_egbb_from_ram(int side,int* piece);
DLLExport void _CDECL open_egbb(int* piece);

#undef _CDECL
#endif

