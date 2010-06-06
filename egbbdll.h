#ifndef __EGBBPROBE__
#define __EGBBPROBE__

enum {_WHITE,_BLACK};
enum {_EMPTY,_WKING,_WQUEEN,_WROOK,_WBISHOP,_WKNIGHT,_WPAWN,
             _BKING,_BQUEEN,_BROOK,_BBISHOP,_BKNIGHT,_BPAWN};
enum {LOAD_NONE,LOAD_4MEN,SMART_LOAD,LOAD_5MEN};

#define _NOTFOUND 99999

#if defined (_WIN32) || defined(_WIN64)
     #define DLLExport extern "C" __declspec(dllexport)
#else
     #define DLLExport extern "C"
#endif

#ifdef _MSC_VER
#   define CDECL __cdecl
#else
#   define CDECL
#endif
/*4 men*/
DLLExport int  CDECL probe_egbb(int player, int w_king, int b_king,
			   int piece1 = _EMPTY, int square1 = 0,
			   int piece2 = _EMPTY, int square2 = 0);
DLLExport void CDECL load_egbb(char* path);
/*5 men*/
DLLExport int  CDECL probe_egbb_5men(int player, int w_king, int b_king,
			   int piece1 = _EMPTY, int square1 = 0,
			   int piece2 = _EMPTY, int square2 = 0,
			   int piece3 = _EMPTY, int square3 = 0
			   );
DLLExport void CDECL load_egbb_5men(char* path,int cache_size = 4194304,int load_options = LOAD_4MEN);
/*X men*/
DLLExport int  CDECL probe_egbb_xmen(int player, int* piece,int* square);
DLLExport void CDECL load_egbb_xmen(char* path,int cache_size = 4194304,int load_options = LOAD_4MEN);
/*private*/
DLLExport void CDECL load_egbb_into_ram(int side,int* piece);
DLLExport void CDECL unload_egbb_from_ram(int side,int* piece);
DLLExport void CDECL open_egbb(int* piece);

#endif
