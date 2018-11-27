#include <vector>
#include <math.h>
#include <iostream>

#ifdef TENSORFLOW
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#endif

#ifdef TRT
#include <fstream>
#include <string>
#include "include/cuda_runtime_api.h"
#include "include/NvInfer.h"
#include "include/NvUffParser.h"
#endif

#include "common.h"
#include "egbbdll.h"

static const int CHANNELS = 12;
static const int NPARAMS = 5;

static int N_DEVICES;
static int BATCH_SIZE;
static int n_searchers;
static VOLATILE int n_active_searchers;
static VOLATILE int chosen_device = 0;
static int delayms = 0;
static int floatPrecision = 1;
static bool is_trt = false;

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
  Network model
*/
class Model {
public:
    int* scores;
    VOLATILE int wait;
    VOLATILE int n_batch;
    VOLATILE int n_batch_i;
    VOLATILE int n_finished_threads;
    int id;
    Model() {
        scores = new int[BATCH_SIZE];
        n_batch = 0;
        n_batch_i = 0;
        n_finished_threads = 0;
        id = 0;
        wait = 1;
    }
    ~Model() {
        delete[] scores;
    }
    virtual float* get_main_input() = 0;
    virtual float* get_aux_input() = 0;
    virtual void predict() = 0;
    virtual void LoadGraph(const string&, int, int) = 0;
    static char path[256];
    static int dev_type;
};
char Model::path[256];
int Model::dev_type;

static Model** netModel;
static const string main_input_layer = "main_input";
static const string aux_input_layer = "aux_input";
static const string output_layer = "value/Softmax";

/*
  TensorFlow model
*/
#ifdef TENSORFLOW

using namespace tensorflow;

class TfModel : public Model {
    Tensor* main_input;
    Tensor* aux_input;
    Session* session;
public:
    TfModel();
    ~TfModel();
    void LoadGraph(const string& graph_file_name, int dev_id, int dev_type);
    void predict();
    float* get_main_input() {
        return (float*)(main_input->tensor_data().data());
    }
    float* get_aux_input() {
        return (float*)(aux_input->tensor_data().data());
    }
};

TfModel::TfModel() : Model() {
    static const TensorShape main_input_shape({BATCH_SIZE, 8, 8, CHANNELS});
    static const TensorShape aux_input_shape({BATCH_SIZE, NPARAMS});
    main_input = new Tensor(DT_FLOAT, main_input_shape);
    aux_input = new Tensor(DT_FLOAT, aux_input_shape);
}
TfModel::~TfModel() {
    delete main_input;
    delete aux_input;
}
void TfModel::LoadGraph(const string& graph_file_name, int dev_id, int dev_type) {
    Model::id = dev_id;

    GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(Env::Default(), graph_file_name, &graph_def);

    std::string dev_name = ((dev_type == GPU) ? "/gpu:" : "/cpu:") + std::to_string(dev_id);
    graph::SetDefaultDevice(dev_name, &graph_def);
    printf("Loading graph on %s\n",dev_name.c_str());
    fflush(stdout);

    SessionOptions options;
    Status status = NewSession(options, &session);
    session->Create(graph_def);    
}

void TfModel::predict() {
    std::vector<Tensor> outputs;

    std::vector<std::pair<string, Tensor> > inputs = {
        {main_input_layer, *main_input},
        {aux_input_layer, *aux_input}
    };

    TF_CHECK_OK( session->Run(inputs, {output_layer}, {}, &outputs) );

    auto pp = outputs[0].matrix<float>();
    for(int i = 0;i < BATCH_SIZE; i++) {
        float p = pp(i,0) * 1.0 + pp(i,1) * 0.5;
        scores[i] = logit(p);
    }
}
#endif

/*
  TensorRT model
*/
#ifdef TRT

using namespace nvuffparser;
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class TrtModel : public Model {
    ICudaEngine* engine;
    IExecutionContext* context;
    IUffParser* parser;
    Logger logger;
    int numBindings;
    nvinfer1::DataType floatMode;
    std::vector<void*> buffers;
    float* main_input;
    float* aux_input;
    float* output;
public:
    TrtModel();
    ~TrtModel();
    void LoadGraph(const string& uff_file_name, int dev_id, int dev_type);
    void predict();
    float* get_main_input() {
        return main_input;
    }
    float* get_aux_input() {
        return aux_input;
    }
};

TrtModel::TrtModel() : Model() {
    parser = createUffParser();
    parser->registerInput(main_input_layer.c_str(), nvinfer1::DimsCHW(CHANNELS, 8, 8), UffInputOrder::kNCHW);
    parser->registerInput(aux_input_layer.c_str(), nvinfer1::DimsCHW(NPARAMS, 1, 1), UffInputOrder::kNC);
    parser->registerOutput(output_layer.c_str());
    context = 0;
    engine = 0;
    numBindings = 0;
    if(floatPrecision == 0)
        floatMode = nvinfer1::DataType::kFLOAT;
    else if(floatPrecision == 1)
        floatMode = nvinfer1::DataType::kHALF;
    else
        floatMode = nvinfer1::DataType::kINT8;
    main_input = new float[BATCH_SIZE * 8 * 8 * CHANNELS];
    aux_input = new float[BATCH_SIZE * NPARAMS];
    output = new float[BATCH_SIZE * 3];
}

TrtModel::~TrtModel() {
    context->destroy();
    engine->destroy();
    delete[] main_input;
    delete[] aux_input;
    delete[] output;
}

void TrtModel::LoadGraph(const string& uff_file_name, int dev_id, int dev_type) {
    std::string dev_name = ((dev_type == GPU) ? "/gpu:" : "/cpu:") + std::to_string(dev_id);
    printf("Loading graph on %s\n",dev_name.c_str());
    fflush(stdout);

    Model::id = dev_id;
    cudaSetDevice(Model::id);

    IBuilder* builder = createInferBuilder(logger);
    INetworkDefinition* network = builder->createNetwork();
    if(!parser->parse(uff_file_name.c_str(), *network, floatMode)) {
        std::cout << "Fail to parse network " << uff_file_name << std::endl;;
        return;
    }     
    if (floatMode == nvinfer1::DataType::kHALF) {
        builder->setHalf2Mode(true);
    } else if (floatMode == nvinfer1::DataType::kINT8) {
        builder->setInt8Mode(true);
    }
    builder->setMaxBatchSize(BATCH_SIZE);
    builder->setMaxWorkspaceSize((1 << 30));
    engine = builder->buildCudaEngine(*network);
    if (!engine) {
        std::cout << "Unable to create engine" << std::endl;
        return;
    }
    network->destroy();
    builder->destroy();
    parser->destroy();

    context = engine->createExecutionContext();
    numBindings = engine->getNbBindings();
    
    /*prepare buffer*/
    for(int i = 0; i < numBindings; i++) {
        Dims d = engine->getBindingDimensions(i);
        size_t size = 1;
        for(size_t i = 0; i < d.nbDims; i++) 
            size*= d.d[i];

        void* buf;
        cudaMalloc(&buf, BATCH_SIZE * size * sizeof(float));
        buffers.push_back(buf);
    }
}
void TrtModel::predict() {
    cudaSetDevice(Model::id);

    cudaMemcpy(buffers[0], main_input, BATCH_SIZE * sizeof(float) * 8 * 8 * CHANNELS, cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[1], aux_input, BATCH_SIZE * sizeof(float) * NPARAMS, cudaMemcpyHostToDevice);

    context->execute(BATCH_SIZE, buffers.data());

    cudaMemcpy(output, buffers[2], BATCH_SIZE * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0;i < BATCH_SIZE;i++) {
        float p = output[3*i+0] * 1.0 + output[3*i+1] * 0.5;
        scores[i] = logit(p);
    }
}

#endif

/*
  Thread procedure for loading NN
*/
static VOLATILE int nn_loaded = 0;
static void CDECL nn_thread_proc(void* id) {
    int dev_id = *((int*)id);
    netModel[dev_id]->LoadGraph(Model::path, dev_id, Model::dev_type);
    l_add(nn_loaded,1);
}

/*
   Initialize tensorflow
*/
DLLExport void CDECL load_neural_network(char* path, int n_threads, int n_devices, int dev_type, int delay, int float_type) {

    /*Message*/
    printf("Loading neural network : %s\n",path);
    fflush(stdout);

#ifdef _WIN32
#   define setenv(n,v,o) _putenv_s(n,v)
#endif

    /*setenv variables*/
#ifdef TENSORFLOW
    setenv("TF_CPP_MIN_LOG_LEVEL","3",1);
#endif

    delayms = delay;
    n_searchers = n_threads;
    N_DEVICES = n_devices;
    n_active_searchers = n_searchers;
    BATCH_SIZE = n_searchers / N_DEVICES;
    floatPrecision = float_type;
    
    /*Load tensorflow or tensorrt graphs on GPU*/
    netModel = new Model*[N_DEVICES];
    is_trt = false;
#if defined(TENSORFLOW) && defined(TRT)
    if(strstr(path, ".pb") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[i] = new TfModel;
    } else if(strstr(path, ".uff") != NULL) {
        for(int i = 0; i < N_DEVICES; i++)
            netModel[i] = new TrtModel;
        is_trt = true;
    }
#elif defined(TENSORFLOW)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[i] = new TfModel;
#elif defined(TRT)
    for(int i = 0; i < N_DEVICES; i++)
        netModel[i] = new TrtModel;
    is_trt = true;
#endif

    /*Load NN with multiple threads*/
    strcpy(Model::path, path);
    Model::dev_type = dev_type;
    int* tid = new int[N_DEVICES];
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {
        tid[dev_id] = dev_id;
        t_create(nn_thread_proc,&tid[dev_id]);
    }
    while(nn_loaded < N_DEVICES)
        t_sleep(1);
	delete[] tid;

    /*warm up nn*/
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {
        int piece = _EMPTY, square = _EMPTY;
        Model* net = netModel[dev_id];
        for(int i = 0;i < BATCH_SIZE;i++)
            add_to_batch(0, &piece, &square, dev_id);
        net->predict();
        net->n_batch = 0;
    }

    /*Message*/
    printf("Neural network loaded !\t\n");
    fflush(stdout);
}

/*
   Add position to batch
*/
static void fill_input_planes(int player, int* piece,int* square, float* data, float* adata);

DLLExport int CDECL add_to_batch(int player, int* piece, int* square, int batch_id) {

    //get identifier
    Model* net = netModel[batch_id];

    //input
    float* minput = net->get_main_input();
    float* ainput = net->get_aux_input();

    //offsets
    int offset = l_add(net->n_batch,1);
    minput += offset * (8 * 8 * CHANNELS);
    ainput += offset * (NPARAMS);

    //fill planes
    fill_input_planes(player, piece, square, minput, ainput);

    return offset;
}

DLLExport void CDECL probe_neural_network_batch(int* scores, int batch_id) {
    //get identifier
    Model* net = netModel[batch_id];

    //prediction
    net->predict();

    //copy result
    memcpy(scores, net->scores, BATCH_SIZE * sizeof(int));
}
/*
   Evaluate position using NN
*/

#define SLEEP() {     \
    t_yield();        \
    t_sleep(delayms); \
}

DLLExport int CDECL probe_neural_network(int player, int* piece, int* square) {

    //choose batch id
    int batch_id;
    l_lock(searcher_lock);
    do {
        for(batch_id = chosen_device;batch_id < N_DEVICES; batch_id++) {
            Model* net = netModel[batch_id];
            if(net->n_batch_i < BATCH_SIZE) {
                net->n_batch_i++;
                break;
            }
        }
        chosen_device = 0;
    } while(batch_id + 1 > N_DEVICES);

    chosen_device = batch_id + 1;
    if(chosen_device == N_DEVICES)
        chosen_device = 0;
    l_unlock(searcher_lock);

    //get identifier
    Model* net = netModel[batch_id];

    //add to batch
    int offset = add_to_batch(player,piece,square,batch_id);

    //pause threads till eval completes
    if(offset + 1 < BATCH_SIZE) {

        while(net->wait) {
            SLEEP();

            if(offset + 1 == net->n_batch
               && n_active_searchers < n_searchers 
               && net->n_batch >= BATCH_SIZE - (n_searchers - n_active_searchers)
               ) {
#if 0
                    printf("\n# batchsize %d / %d workers %d / %d\n",
                        net->n_batch,BATCH_SIZE,
                        n_active_searchers, n_searchers);
                    fflush(stdout);
#endif
                net->predict();
                net->wait = 0;
                break;
            }
        }

    } else {
        net->predict();
        net->wait = 0;
    }

    //Wait until all eval calls are finished
    l_lock(searcher_lock);
    net->n_finished_threads++;
    if(net->n_finished_threads == net->n_batch) {
        net->wait = 1;
        net->n_finished_threads = 0;
        net->n_batch = 0;
        net->n_batch_i = 0;
    } 
    l_unlock(searcher_lock);

    while (net->n_finished_threads > 0 
        && net->n_finished_threads < net->n_batch
        ) {
        SLEEP();
    }

    return net->scores[offset];
}

#undef SLEEP

/*
   Set number of active workers
*/
DLLExport void CDECL set_num_active_searchers(int n_searchers) {
    l_set(n_active_searchers,n_searchers);
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
    memset(adata, 0, sizeof(float) * NPARAMS);

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
#define DHWC(sq,C)     data[rank(sq) * 8 * CHANNELS + file(sq) * CHANNELS + C]
#define DCHW(sq,C)     data[C * 8 * 8 + rank(sq) * 8 + file(sq)]
#define D(sq,C)        ( is_trt ? DCHW(sq,C) : DHWC(sq,C) )

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
