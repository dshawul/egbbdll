#ifndef __EGBBPROBE__
#define __EGBBPROBE__

enum {_WHITE,_BLACK};
enum {_EMPTY,_WKING,_WQUEEN,_WROOK,_WBISHOP,_WKNIGHT,_WPAWN,
             _BKING,_BQUEEN,_BROOK,_BBISHOP,_BKNIGHT,_BPAWN};
enum {LOAD_NONE,LOAD_4MEN,SMART_LOAD,LOAD_5MEN,LOAD_5MEN_LZ};
enum {CPU, GPU};

#define _NOTFOUND 99999

#if defined (_WIN32)
#   define DLLExport extern "C" __declspec(dllexport)
#else
#   define DLLExport extern "C"
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
DLLExport int  CDECL probe_egbb_fen(char* fen);
DLLExport int  CDECL probe_egbb_xmen(int player, int* piece, int* square);
DLLExport void CDECL load_egbb_xmen(char* path,int cache_size = 4194304,int load_options = LOAD_4MEN);
/*NN eval*/
DLLExport void CDECL set_num_active_searchers(int n_searchers);
DLLExport int  CDECL add_to_batch(int player, int* piece, int* square, int batch_id = 0);
DLLExport int  CDECL probe_neural_network(int player, int* piece, int* square);
DLLExport void CDECL probe_neural_network_batch(int* scores, int batch_id = 0);
DLLExport void CDECL load_neural_network(char* path, int n_threads = 1, int n_devices = 1, int dev_type = CPU, int delay = 0, int float_type = 1);
/*private*/
DLLExport void CDECL load_egbb_into_ram(int side,int* piece);
DLLExport void CDECL unload_egbb_from_ram(int side,int* piece);
DLLExport void CDECL open_egbb(int* piece);

#endif

