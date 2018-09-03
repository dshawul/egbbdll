#ifdef TENSORFLOW
#include <vector>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "common.h"
#include "egbbdll.h"

using namespace tensorflow;

static Session* session;
static const string main_input_layer = "main_input";
static const string aux_input_layer = "aux_input";
static const string output_layer = "value_0";
static const int CHANNELS = 12;
static const int PARAMS = 5;

static int BATCH_SIZE;
static VOLATILE int n_active_searchers;
static int n_searchers;
static int N_DEVICES = 1;
static VOLATILE int chosen_device = 0;

struct InputData {
    Tensor* main_input;
    Tensor* aux_input;
    int* scores;
    VOLATILE int n_batch;
    VOLATILE int save_n_batch;
    VOLATILE int n_finished;
    
    InputData() {
        main_input = 0;
        aux_input = 0;
        scores = 0;
        n_batch = 0;
        save_n_batch = 0;
        n_finished = 0;
    }
    void alloc() {
        static const TensorShape main_input_shape({BATCH_SIZE, 8, 8, CHANNELS});
        static const TensorShape aux_input_shape({BATCH_SIZE, PARAMS});
        main_input = new Tensor(DT_FLOAT, main_input_shape);
        aux_input = new Tensor(DT_FLOAT, aux_input_shape);
        scores = new int[BATCH_SIZE];
    }
};

static std::unordered_map<int,InputData> input_map;

/*
   Load NN
*/
static Status LoadGraph(const string& graph_file_name,
        Session** session) {

    GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return errors::NotFound("Failed to load compute graph at '",
                graph_file_name, "'");
    }

    SessionOptions options;
    options.config.set_intra_op_parallelism_threads(1);
    options.config.set_inter_op_parallelism_threads(1);

    Status status = NewSession(options, session);
    Status session_create_status = (*session)->Create(graph_def);

    if (!session_create_status.ok()) {
        return session_create_status;
    }

    return Status::OK();
}

/*
   Initialize tensorflow
*/
DLLExport void CDECL load_neural_network(char* path, int n_searchers_l) {

    /*Message*/
    printf("Loading neural network....\n");
    fflush(stdout);

    /*setenv variables*/
#ifdef _WIN32
    SetEnvironmentVariable((LPCWSTR)"TF_CPP_MIN_LOG_LEVEL",(LPCWSTR)"3");
#else
    setenv("TF_CPP_MIN_LOG_LEVEL","3",1);
#endif

    /*Load NN*/
    TF_CHECK_OK( LoadGraph(path, &session) );

    /*Initialize tensors*/
    n_searchers = n_searchers_l;
    n_active_searchers = n_searchers;
    BATCH_SIZE = n_searchers / N_DEVICES;
    for(int i = 0;i < N_DEVICES;i++) {
        InputData& inp = input_map[i];
        inp.alloc();
    }

    /*warm up nn*/
    int piece = _EMPTY, square = _EMPTY;
    InputData& inp = input_map[0];
    for(int i = 0;i < BATCH_SIZE;i++)
        add_to_batch(0, &piece, &square);
    probe_neural_network_batch(inp.scores);
    inp.n_finished = BATCH_SIZE;

    /*Message*/
    printf("Neural network loaded !      \n");
    fflush(stdout);
}

/*
   Add position to batch
*/
static void fill_input_planes(int player, int* piece,int* square, float* data, float* adata);

DLLExport int CDECL add_to_batch(int player, int* piece, int* square, int batch_id) {
    
    //get identifier
    InputData& inp = input_map[batch_id];

    //input
    float* minput = (float*)(inp.main_input->tensor_data().data());
    float* ainput = (float*)(inp.aux_input->tensor_data().data());

    int offset;

    l_lock(searcher_lock);
    offset = inp.n_batch;
    inp.n_batch++;
    l_unlock(searcher_lock);

    //offsets
    minput += offset * (8 * 8 * CHANNELS);
    ainput += offset * (PARAMS);

    //fill planes
    fill_input_planes(player, piece, square, minput, ainput);

    return offset;
}

/*
   Do batch inference
*/

static inline int logit(double p);

DLLExport void CDECL probe_neural_network_batch(int* scores, int batch_id) {

    //get identifier
    InputData& inp = input_map[batch_id];

    //outputs
    std::vector<Tensor> outputs;

    //run session
    std::vector<std::pair<string, Tensor> > inputs = {
        {main_input_layer, *inp.main_input},
        {aux_input_layer,  *inp.aux_input}
    };

    TF_CHECK_OK( session->Run(inputs, {output_layer}, {}, &outputs) );

    //extract and return score in centi-pawns
    for(int i = 0;i < inp.n_batch; i++) {
        float p = outputs[0].flat<float>()(i);
        scores[i] = logit(p);
    }

    l_lock(searcher_lock);
    inp.n_finished = 0;
    inp.save_n_batch = inp.n_batch;
    inp.n_batch = 0;
    l_unlock(searcher_lock);
}

/*
   Evaluate position using NN
*/

DLLExport int CDECL probe_neural_network(int player, int* piece, int* square) {

    //choose batch id
    int batch_id;
    do {
        l_lock(searcher_lock);
        batch_id = chosen_device++;
        if(chosen_device == N_DEVICES)
            chosen_device = 0;
        l_unlock(searcher_lock);
    } while(input_map[batch_id].n_batch == BATCH_SIZE);
    
    //get identifier
    InputData& inp = input_map[batch_id];

    //add to batch
    int offset = add_to_batch(player,piece,square,batch_id);

    //pause threads till eval completes
    if(offset + 1 < BATCH_SIZE) {

        while(inp.n_batch) {
            t_sleep(0);

            if(offset + 1 == inp.n_batch
               && n_active_searchers < n_searchers 
               && inp.n_batch >= n_active_searchers
               ) {
#if 0
                    printf("\n# batchsize %d / %d workers %d / %d\n",
                        inp.n_batch,BATCH_SIZE,
                        n_active_searchers, n_searchers);
                    fflush(stdout);
#endif
                probe_neural_network_batch(inp.scores,batch_id);
                break;
            }
        }

    } else {
        probe_neural_network_batch(inp.scores,batch_id);
    }

    //mark finished
    l_lock(searcher_lock);
    inp.n_finished++;
    l_unlock(searcher_lock);

    //wait for previous eval to finish
    while(inp.n_finished < inp.save_n_batch) {
        t_sleep(0);
    }

    return inp.scores[offset];
}

/*
   Set number of active workers
*/
DLLExport void CDECL set_active_searchers(int n_searchers) {
    l_lock(searcher_lock);
    n_active_searchers = n_searchers;
    l_unlock(searcher_lock);
}

/*
   Convert winning percentage to centi-pawns
*/
static const double Kfactor = -log(10.0) / 400.0;

static inline int logit(double p) {
    if(p < 1e-15) p = 1e-15;
    else if(p > 1 - 1e-15) p = 1 - 1e-15;
    return int(log((1 - p) / p) / Kfactor);
}

/*
   Fill input planes
*/
static void fill_input_planes(int player, int* piece,int* square, float* data, float* adata) {
    int pc, col, sq, to;
    int board[128];

    /*zero board*/
    memset(board, 0, sizeof(int) * 128);
    memset(data,  0, sizeof(float) * 8 * 8 * CHANNELS);
    memset(adata, 0, sizeof(float) * PARAMS);

    /*fill board*/
    int ksq = SQ6488(square[player]);
    bool fliph = (file(ksq) <= FILED);

    for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
        sq = SQ6488(square[i]);
        if(player == _BLACK) {
            sq = MIRRORR(sq);
            pc = invert_color(pc);
        }
        if(fliph) 
            sq = MIRRORF(sq);

        board[sq] = pc;
    }

    /* 
       Add the attack map planes 
    */
#define D(sq,C)     data[rank(sq) * 8 * CHANNELS + file(sq) * CHANNELS + C]

#define NK_MOVES(dir, off) {                    \
        to = sq + dir;                          \
        if(!(to & 0x88)) D(to, off) = 1.0f;     \
}

#define BRQ_MOVES(dir, off) {                   \
        to = sq + dir;                          \
        while(!(to & 0x88)) {                   \
            D(to, off) = 1.0f;                  \
            if(board[to] != 0) break;           \
                to += dir;                      \
        }                                       \
}

    for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {

        sq = SQ6488(square[i]);
        if(player == _BLACK) {
            sq = MIRRORR(sq);
            pc = invert_color(pc);
        }
        if(fliph) 
            sq = MIRRORF(sq);

        switch(pc) {
            case wking:
                NK_MOVES(RU,0);
                NK_MOVES(LD,0);
                NK_MOVES(LU,0);
                NK_MOVES(RD,0);
                NK_MOVES(UU,0);
                NK_MOVES(DD,0);
                NK_MOVES(RR,0);
                NK_MOVES(LL,0);
                break;
            case wqueen:
                BRQ_MOVES(RU,1);
                BRQ_MOVES(LD,1);
                BRQ_MOVES(LU,1);
                BRQ_MOVES(RD,1);
                BRQ_MOVES(UU,1);
                BRQ_MOVES(DD,1);
                BRQ_MOVES(RR,1);
                BRQ_MOVES(LL,1);
                break;
            case wrook:
                BRQ_MOVES(UU,2);
                BRQ_MOVES(DD,2);
                BRQ_MOVES(RR,2);
                BRQ_MOVES(LL,2);
                break;
            case wbishop:
                BRQ_MOVES(RU,3);
                BRQ_MOVES(LD,3);
                BRQ_MOVES(LU,3);
                BRQ_MOVES(RD,3);
                break;
            case wknight:
                NK_MOVES(RRU,4);
                NK_MOVES(LLD,4);
                NK_MOVES(RUU,4);
                NK_MOVES(LDD,4);
                NK_MOVES(LLU,4);
                NK_MOVES(RRD,4);
                NK_MOVES(RDD,4);
                NK_MOVES(LUU,4);
                break;
            case wpawn:
                NK_MOVES(RU,5);
                NK_MOVES(LU,5);
                break;
            case bking:
                NK_MOVES(RU,6);
                NK_MOVES(LD,6);
                NK_MOVES(LU,6);
                NK_MOVES(RD,6);
                NK_MOVES(UU,6);
                NK_MOVES(DD,6);
                NK_MOVES(RR,6);
                NK_MOVES(LL,6);
                break;
            case bqueen:
                BRQ_MOVES(RU,7);
                BRQ_MOVES(LD,7);
                BRQ_MOVES(LU,7);
                BRQ_MOVES(RD,7);
                BRQ_MOVES(UU,7);
                BRQ_MOVES(DD,7);
                BRQ_MOVES(RR,7);
                BRQ_MOVES(LL,7);
                break;
            case brook:
                BRQ_MOVES(UU,8);
                BRQ_MOVES(DD,8);
                BRQ_MOVES(RR,8);
                BRQ_MOVES(LL,8);
                break;
            case bbishop:
                BRQ_MOVES(RU,9);
                BRQ_MOVES(LD,9);
                BRQ_MOVES(LU,9);
                BRQ_MOVES(RD,9);
                break;
            case bknight:
                NK_MOVES(RRU,10);
                NK_MOVES(LLD,10);
                NK_MOVES(RUU,10);
                NK_MOVES(LDD,10);
                NK_MOVES(LLU,10);
                NK_MOVES(RRD,10);
                NK_MOVES(RDD,10);
                NK_MOVES(LUU,10);
                break;
            case bpawn:
                NK_MOVES(RD,11);
                NK_MOVES(LD,11);
                break;
        }

        col = COLOR(pc);
        pc = PIECE(pc);

        if(pc != king) {
            if(col == white)
                adata[pc - queen]++;
            else
                adata[pc - queen]--;
        }
    }

#undef NK_MOVES
#undef BRQ_MOVES
#undef D

}
#endif
