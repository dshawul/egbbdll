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
#include <iterator>
#include "include/cuda_runtime_api.h"
#include "include/NvInfer.h"
#include "include/NvUffParser.h"
#endif

#include "common.h"
#include "egbbdll.h"

#define AZPOLICY 1

enum NNTYPE {
    DEFAULT=0, LCZERO, SIMPLE
};

static int CHANNELS;
static int NPOLICY;
static const int NPARAMS = 5;
static const string main_input_layer = "main_input";
static const string aux_input_layer = "aux_input";
static string policy_layer;
static string value_layer;

static int N_DEVICES;
static int BATCH_SIZE;
static int n_searchers;
static VOLATILE int n_active_searchers;
static VOLATILE int chosen_device = 0;
static int delayms = 0;
static int nn_type = DEFAULT;
static int floatPrecision = 1;
static bool is_trt = false;

/*
   Convert winning percentage to centi-pawns
*/
static const double Kfactor = -log(10.0) / 400.0;

double logistic(double score) {
    return 1 / (1 + exp(Kfactor * score));
}

static inline int logit(double p) {
    if(p < 1e-15) p = 1e-15;
    else if(p > 1 - 1e-15) p = 1 - 1e-15;
    return int(log((1 - p) / p) / Kfactor);
}

/*
Move policy format
*/

/* 1. AlphaZero format: 56=queen moves, 8=knight moves, 9 pawn promotions */
static const UBMP8 t_move_map[] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0, 35,  0,  0,  0,  0,  0,  0,
 27,  0,  0,  0,  0,  0,  0, 55,  0,  0, 36,  0,  0,  0,  0,  0,
 26,  0,  0,  0,  0,  0, 54,  0,  0,  0,  0, 37,  0,  0,  0,  0,
 25,  0,  0,  0,  0, 53,  0,  0,  0,  0,  0,  0, 38,  0,  0,  0,
 24,  0,  0,  0, 52,  0,  0,  0,  0,  0,  0,  0,  0, 39,  0,  0,
 23,  0,  0, 51,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 40, 60,
 22, 56, 50,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 61, 41,
 21, 49, 57,  0,  0,  0,  0,  0,  0,  7,  8,  9, 10, 11, 12, 13,
  0,  0,  1,  2,  3,  4,  5,  6,  0,  0,  0,  0,  0,  0, 63, 48,
 14, 28, 59,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 47, 62,
 15, 58, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 46,  0,  0,
 16,  0,  0, 30,  0,  0,  0,  0,  0,  0,  0,  0, 45,  0,  0,  0,
 17,  0,  0,  0, 31,  0,  0,  0,  0,  0,  0, 44,  0,  0,  0,  0,
 18,  0,  0,  0,  0, 32,  0,  0,  0,  0, 43,  0,  0,  0,  0,  0,
 19,  0,  0,  0,  0,  0, 33,  0,  0, 42,  0,  0,  0,  0,  0,  0,
 20,  0,  0,  0,  0,  0,  0, 34,  0,  0,  0,  0,  0,  0,  0,  0
};
static const UBMP8* const move_map = t_move_map + 0x80;

/* 2. LcZero format: flat move representation */
static const int MOVE_TAB_SIZE = 64*64+8*3*3;

static unsigned short move_index_table[MOVE_TAB_SIZE];

static void init_index_table() {

    memset(move_index_table, 0, MOVE_TAB_SIZE * sizeof(short));

    int cnt = 0;

    for(int from = 0; from < 64; from++) {

        int from88 = SQ6488(from);
        for(int to = 0; to < 64; to++) {
            int to88 = SQ6488(to);
            if(from != to) {
                if(sqatt_pieces(to88 - from88))
                    move_index_table[from * 64 + to] = cnt++;
            }
        }

    }

    for(int from = 48; from < 56; from++) {
        int idx = 4096 + file64(from) * 9;

        if(from > 48) {
            move_index_table[idx+0] = cnt++;
            move_index_table[idx+1] = cnt++;
            move_index_table[idx+2] = cnt++;
        }

        move_index_table[idx+3] = cnt++;
        move_index_table[idx+4] = cnt++;
        move_index_table[idx+5] = cnt++;

        if(from < 55) {
            move_index_table[idx+6] = cnt++;
            move_index_table[idx+7] = cnt++;
            move_index_table[idx+8] = cnt++;
        }
    }
}

/*
Fill input planes
*/
static void fill_input_planes(
    int player, int cast, int fifty, int hist, int* draw, int* piece, int* square, float* data, float* adata);

/*
  Network model
*/
class Model {
public:
    int* scores;
    float* policy_scores;
    int* policy_index;
    int* policy_size;
    VOLATILE int wait;
    VOLATILE int n_batch;
    VOLATILE int n_batch_i;
    VOLATILE int n_finished_threads;
    int id;
    Model() {
        scores = new int[BATCH_SIZE];
        policy_scores = new float[BATCH_SIZE * MAX_MOVES];
        policy_index = new int[BATCH_SIZE * MAX_MOVES];
        policy_size = new int[BATCH_SIZE];
        memset(policy_size,0,BATCH_SIZE * sizeof(int));
        n_batch = 0;
        n_batch_i = 0;
        n_finished_threads = 0;
        id = 0;
        wait = 1;
    }
    ~Model() {
        delete[] scores;
        delete[] policy_scores;
        delete[] policy_index;
        delete[] policy_size;
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
    if(nn_type == DEFAULT)
        main_input = new Tensor(DT_FLOAT, {BATCH_SIZE, 8, 8, CHANNELS});
    else
        main_input = new Tensor(DT_FLOAT, {BATCH_SIZE, CHANNELS, 8, 8});
    aux_input = new Tensor(DT_FLOAT, {BATCH_SIZE, NPARAMS});
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

#if 0
    std::cout << "=============================" << std::endl;
    for (auto &node: *graph_def.mutable_node())
        std::cout << node.name() << std::endl;
    std::cout << "=============================" << std::endl;
#endif

    SessionOptions options;
    Status status = NewSession(options, &session);
    session->Create(graph_def);    
}

void TfModel::predict() {
    std::vector<Tensor> outputs;

    if(nn_type == DEFAULT || nn_type == SIMPLE) {

        std::vector<std::pair<string, Tensor> > inputs;
        if(nn_type == DEFAULT) {
            inputs = {
                {main_input_layer, *main_input},
                {aux_input_layer, *aux_input}
            };
        } else {
            inputs = {
                {main_input_layer, *main_input}
            };
        }

        TF_CHECK_OK( session->Run(inputs, {value_layer,policy_layer}, {}, &outputs) );

        auto pp0 = outputs[0].matrix<float>();
        auto pp1 = outputs[1].matrix<float>();
        for(int i = 0;i < n_batch; i++) {
            float p = pp0(i,0) * 1.0 + pp0(i,1) * 0.5;
            scores[i] = logit(p);

            for(int j = 0;j < policy_size[i];j++) {
                int idx = policy_index[i * MAX_MOVES + j];
                float p = pp1(i,idx);
                policy_scores[i * MAX_MOVES + j] = p;
            }
        }
    } else {
        std::vector<std::pair<string, Tensor> > inputs = {
            {main_input_layer, *main_input}
        };

        TF_CHECK_OK( session->Run(inputs, {value_layer,policy_layer}, {}, &outputs) );

        auto pp0 = outputs[0].matrix<float>();
        auto pp1 = outputs[1].matrix<float>();
        for(int i = 0;i < n_batch; i++) {
            float p = (pp0(i,0) + 1.0) * 0.5;
            scores[i] = logit(p);

            for(int j = 0;j < policy_size[i];j++) {
                int idx = policy_index[i * MAX_MOVES + j];
                float p = pp1(i,idx);
                policy_scores[i * MAX_MOVES + j] = p;
            }
        }
    }

}
#endif

/*
  TensorRT model
*/
#ifdef TRT

using namespace nvuffparser;
using namespace nvinfer1;

class Int8CacheCalibrator : public IInt8EntropyCalibrator {
public:

  Int8CacheCalibrator() {

    void* buf;
    cudaMalloc(&buf, CAL_BATCH_SIZE * sizeof(float) * 8 * 8 * CHANNELS);
    buffers.push_back(buf);
    if(nn_type == DEFAULT) {
        cudaMalloc(&buf, CAL_BATCH_SIZE * sizeof(float) * NPARAMS);
        buffers.push_back(buf);
    }
    counter = 0;
    main_input = new float[CAL_BATCH_SIZE * 8 * 8 * CHANNELS];
    aux_input = new float[CAL_BATCH_SIZE * NPARAMS];

    if (floatPrecision == 2) {
        epd_file = fopen(calib_file_name.c_str(),"r");
        if(!epd_file) {
            printf("Epd file needed for calibration not found!\n");
            fflush(stdout);
            exit(0);
        }
    }
  }

  ~Int8CacheCalibrator() override {
    cudaFree(buffers[0]);
    if(nn_type == DEFAULT)
        cudaFree(buffers[1]);
    delete[] main_input;
    delete[] aux_input;
    if(epd_file)
        fclose(epd_file);
  }
  
  int getBatchSize() const override {
    return CAL_BATCH_SIZE;
  }
  
  bool getBatch(void* bindings[], const char* names[], int nbBindings) override {
    if (counter >= NUM_CAL_BATCH)
        return false;

    std::cout << "Calibrating on Batch " << counter + 1 << " of " << NUM_CAL_BATCH << "\r";

    int piece[33],square[33],isdraw[1],hist=1,player,castle,fifty;
    char fen[MAX_STR];
    for(int i = 0; i < CAL_BATCH_SIZE; i++) {
        if(!fgets(fen,MAX_STR,epd_file))
            return false;
        decode_fen(fen,player,castle,fifty,piece,square);

        float* minput = main_input + i * (8 * 8 * CHANNELS);
        float* ainput = aux_input + i * (NPARAMS);
        fill_input_planes(player,castle,fifty,hist,isdraw,piece,square,minput,ainput);
    }

    cudaMemcpy(buffers[0], main_input, 
        CAL_BATCH_SIZE * sizeof(float) * 8 * 8 * CHANNELS, cudaMemcpyHostToDevice);
    bindings[0] = buffers[0];
    if(nn_type == DEFAULT) {
        cudaMemcpy(buffers[1], aux_input, 
            CAL_BATCH_SIZE * sizeof(float) * NPARAMS, cudaMemcpyHostToDevice);
        bindings[1] = buffers[1];
    }

    counter++;
    return true;
  }
  
  const void* readCalibrationCache(size_t& length) override {
    return nullptr;
  }

  void writeCalibrationCache(const void* cache, size_t length) override {
  }

private:
  std::vector<void*> buffers;
  float* main_input;
  float* aux_input;
  int counter;
  FILE* epd_file;
  static const int CAL_BATCH_SIZE = 256;
  static const int NUM_CAL_BATCH = 10;
  static const std::string calib_file_name;
};

const std::string Int8CacheCalibrator::calib_file_name = "calibrate.epd";

class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

class TrtModel : public Model {
    ICudaEngine* engine;
    IExecutionContext* context;
    Logger logger;
    int numBindings;
    nvinfer1::DataType floatMode;
    std::vector<void*> buffers;
    std::vector<float*> buffers_h;
    int maini, auxi, policyi, valuei;
public:
    TrtModel();
    ~TrtModel();
    void LoadGraph(const string& uff_file_name, int dev_id, int dev_type);
    void predict();
    float* get_main_input() {
        return buffers_h[maini];
    }
    float* get_aux_input() {
        return buffers_h[auxi];
    }
};

TrtModel::TrtModel() : Model() {
    context = 0;
    engine = 0;
    numBindings = 0;
    if(floatPrecision == 0)
        floatMode = nvinfer1::DataType::kFLOAT;
    else if(floatPrecision == 1)
        floatMode = nvinfer1::DataType::kHALF;
    else
        floatMode = nvinfer1::DataType::kINT8;
}

TrtModel::~TrtModel() {
    context->destroy();
    engine->destroy();
    cudaFreeHost(buffers_h[0]);
}

void TrtModel::LoadGraph(const string& uff_file_name, int dev_id, int dev_type) {
    std::string dev_name = ((dev_type == GPU) ? "/gpu:" : "/cpu:") + std::to_string(dev_id);
    printf("Loading graph on %s\n",dev_name.c_str());
    fflush(stdout);

    Model::id = dev_id;
    cudaSetDevice(Model::id);

    std::string trtName = uff_file_name + "." + 
                          std::to_string(BATCH_SIZE)+ "_" + 
                          std::to_string(floatPrecision) +
                          ".trt";
    std::ifstream ifs(trtName.c_str(), std::ios::in | std::ios::binary);

    if (!ifs.is_open()) {

        IUffParser* parser;
        parser = createUffParser();
        parser->registerInput(main_input_layer.c_str(), 
            nvinfer1::DimsCHW(CHANNELS, 8, 8), UffInputOrder::kNCHW);
        if(nn_type == DEFAULT)
            parser->registerInput(aux_input_layer.c_str(), 
                nvinfer1::DimsCHW(NPARAMS, 1, 1), UffInputOrder::kNC);
        parser->registerOutput(value_layer.c_str());
        parser->registerOutput(policy_layer.c_str());

        IBuilder* builder = createInferBuilder(logger);
        if ((floatMode == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) 
         || (floatMode == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
            std::cout << "Device does not support this low precision mode." << std::endl;
            return;
        }

        INetworkDefinition* network = builder->createNetwork();
        nvinfer1::DataType loadMode = (floatMode == nvinfer1::DataType::kINT8) ?
                            nvinfer1::DataType::kFLOAT : floatMode;
        if(!parser->parse(uff_file_name.c_str(), *network, loadMode)) {
            std::cout << "Fail to parse network " << uff_file_name << std::endl;
            return;
        }

        Int8CacheCalibrator calibrator;

        if (floatMode == nvinfer1::DataType::kHALF) {
            builder->setFp16Mode(true);
        } else if (floatMode == nvinfer1::DataType::kINT8) {
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator);
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

        if(Model::id == 0) {
            IHostMemory *trtModelStream = engine->serialize();
            std::ofstream ofs(trtName.c_str(), std::ios::out | std::ios::binary);
            ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
            ofs.close();
            trtModelStream->destroy();
        }
    } else {
        char* trtModelStream{nullptr};
        size_t size{0};

        ifs.seekg(0, ifs.end);
        size = ifs.tellg();
        ifs.seekg(0, ifs.beg);
        trtModelStream = new char[size];
        ifs.read(trtModelStream, size);
        ifs.close();


        IRuntime* infer = createInferRuntime(logger);
        engine = infer->deserializeCudaEngine(trtModelStream, size, nullptr);
        if (trtModelStream) delete[] trtModelStream;
    }

    context = engine->createExecutionContext();
    numBindings = engine->getNbBindings();
    
    /*Pinned memory*/
    float* pDevice, *pHost;
    cudaHostAlloc((void**)&pHost, 
        BATCH_SIZE * (8 * 8 * CHANNELS + NPARAMS + 3 + NPOLICY) * sizeof(float), 
        cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**)&pDevice,(void*)pHost,0);

    for(int i = 0; i < numBindings; i++) {

        Dims d = engine->getBindingDimensions(i);
        size_t size = 1;
        for(size_t j = 0; j < d.nbDims; j++) 
            size*= d.d[j];

#if 0
        printf("%d. %s %d =",i,engine->getBindingName(i),size);
        for(size_t j = 0; j < d.nbDims; j++) 
            printf(" %d",d.d[j]);
        printf("\n");
        fflush(stdout);
#endif

        buffers.push_back(pDevice);
        buffers_h.push_back(pHost);
        pDevice += BATCH_SIZE * size;
        pHost += BATCH_SIZE * size;
    }

    maini = engine->getBindingIndex(main_input_layer.c_str());
    auxi = engine->getBindingIndex(aux_input_layer.c_str());
    policyi = engine->getBindingIndex(policy_layer.c_str());
    valuei = engine->getBindingIndex(value_layer.c_str());
}
void TrtModel::predict() {

    cudaSetDevice(Model::id);

    context->execute(BATCH_SIZE, buffers.data());

    if(nn_type == DEFAULT || nn_type == SIMPLE) {
        for(int i = 0;i < n_batch;i++) {
            float p = buffers_h[valuei][3*i+0] * 1.0 + buffers_h[valuei][3*i+1] * 0.5;
            scores[i] = logit(p);

            for(int j = 0;j < policy_size[i];j++) {
                int idx = policy_index[i * MAX_MOVES + j];
                float p = buffers_h[policyi][i * NPOLICY + idx];
                policy_scores[i * MAX_MOVES + j] = p;
            }
        }
    } else {
        for(int i = 0;i < n_batch;i++) {
            float p = (buffers_h[valuei][i] + 1.0) * 0.5;
            scores[i] = logit(p);

            for(int j = 0;j < policy_size[i];j++) {
                int idx = policy_index[i * MAX_MOVES + j];
                float p = buffers_h[policyi][i * NPOLICY + idx];
                policy_scores[i * MAX_MOVES + j] = p;
            }
        }
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
static int add_to_batch(
    int player, int cast, int fifty, int hist, int* draw, int* piece, int* square, int batch_id);

DLLExport void CDECL load_neural_network(
    char* path, int n_threads, int n_devices, int dev_type, int delay, int float_type, int lnn_type) {

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

    nn_type = lnn_type;
    delayms = delay;
    n_searchers = n_threads;
    N_DEVICES = n_devices;
    n_active_searchers = n_searchers;
    BATCH_SIZE = n_searchers / N_DEVICES;
    floatPrecision = float_type;

    init_index_table();

    /*constants based on network type*/
    if(nn_type == DEFAULT || nn_type == SIMPLE) {
        if(nn_type == DEFAULT)
            CHANNELS = 24;
        else
            CHANNELS = 12;
        value_layer = "value/Softmax";
#if AZPOLICY
        policy_layer = "policy/Reshape";
        NPOLICY = 4672;
#else
        policy_layer = "policy/BiasAdd";
        NPOLICY = 1858;
#endif
    } else {
        CHANNELS = 112;
        NPOLICY = 1858;
        value_layer = "value_head";
        policy_layer = "policy_head";
    }
    
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
#if 1
    /*warm up nn*/
    for(int dev_id = 0; dev_id < N_DEVICES; dev_id++) {
        int piece = _EMPTY, square = _EMPTY;
        Model* net = netModel[dev_id];
        for(int i = 0;i < BATCH_SIZE;i++)
            add_to_batch(0, 0, 1, 1, 0, &piece, &square, dev_id);
        net->predict();
        net->n_batch = 0;
    }
#endif
    /*Message*/
    printf("Neural network loaded !\t\n");
    fflush(stdout);
}

/*
   Add position to batch
*/

static int add_to_batch(
    int player, int cast, int fifty, int hist, int* draw, int* piece, int* square, int batch_id) {

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
    fill_input_planes(player, cast, fifty, hist, draw, piece, square, minput, ainput);

    return offset;
}

/*
   Evaluate position using NN
*/

#define SLEEP() {     \
    t_yield();        \
    t_sleep(delayms); \
}

DLLExport int CDECL probe_neural_network(
    int player, int cast, int fifty, int hist, int* draw, int* piece, int* square, int* moves, int* probs) {

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
    int offset = add_to_batch(player,cast,fifty,hist,draw,piece,square,batch_id);

    //policy
    if(moves) {
        int* s = moves, cnt = 0;
        while(*s >= 0) {
            int index, from, to, prom;
            from = *s++;
            to = *s++;
            prom = *s++;
            if(player == _BLACK) {
                from = MIRRORR64(from);
                to = MIRRORR64(to);
            }

#if AZPOLICY
            if(nn_type == DEFAULT || nn_type == SIMPLE) {
                index = from * 73;
                if(prom) {
                    prom = PIECE(prom);
                    if(prom != queen)
                        index += 64 + (to - from - 7) * 3  + (prom - queen);
                    else
                        index += move_map[SQ6488(to) - SQ6488(from)];
                } else {
                    index += move_map[SQ6488(to) - SQ6488(from)];
                }
            } else 
#endif
            {
                int compi = from * 64 + to;
                if(prom) {
                    prom = PIECE(prom);
                    if(prom != knight) {
                        compi = 4096 +  file64(from) * 9 + 
                                (to - from - 7) * 3 + (prom - queen);
                    }
                }

                index = move_index_table[compi];
            }

            net->policy_index[offset * MAX_MOVES + cnt] = index;
            cnt++;
        }
        net->policy_size[offset] = cnt;
    }

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

    //policy
    if(moves) {
        for(int j = 0;j < net->policy_size[offset];j++) {
            float p = net->policy_scores[offset * MAX_MOVES + j];
            probs[j] = p * 10000;
        }
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
static void fill_input_planes(
    int player, int cast, int fifty, int hist, int* draw, int* piece, int* square, float* data, float* adata) {
    
    int pc, col, sq, to;

    /* 
       Add the attack map planes 
    */
#define DHWC(sq,C)     data[rank(sq) * 8 * CHANNELS + file(sq) * CHANNELS + C]
#define DCHW(sq,C)     data[C * 8 * 8 + rank(sq) * 8 + file(sq)]
#define D(sq,C)        ( (is_trt || (nn_type > DEFAULT) ) ? DCHW(sq,C) : DHWC(sq,C) )

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

    memset(data,  0, sizeof(float) * 8 * 8 * CHANNELS);

    if(nn_type == DEFAULT) {

        int board[128];
        int ksq = SQ6488(square[player]);

        memset(board, 0, sizeof(int) * 128);
        memset(adata, 0, sizeof(float) * NPARAMS);

        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = SQ6488(square[i]);
            if(player == _BLACK) {
                sq = MIRRORR(sq);
                pc = invert_color(pc);
            }

            board[sq] = pc;
        }

        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = SQ6488(square[i]);
            if(player == _BLACK) {
                sq = MIRRORR(sq);
                pc = invert_color(pc);
            }
            D(sq,(pc+11)) = 1.0f;
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
    } else if (nn_type == SIMPLE) {
        for(int i = 0; (pc = piece[i]) != _EMPTY; i++) {
            sq = SQ6488(square[i]);
            if(player == _BLACK) {
                sq = MIRRORR(sq);
                pc = invert_color(pc);
            }
            D(sq,(pc-1)) = 1.0f;
        }
    } else {

        static const int piece_map[2][12] = {
            {
                wpawn,wknight,wbishop,wrook,wqueen,wking,
                bpawn,bknight,bbishop,brook,bqueen,bking
            },
            {
                bpawn,bknight,bbishop,brook,bqueen,bking,
                wpawn,wknight,wbishop,wrook,wqueen,wking
            }
        };

        for(int h = 0, i = 0; h < hist; h++) {
            for(; (pc = piece[i]) != _EMPTY; i++) {
                sq = SQ6488(square[i]);
                if(player == _BLACK) 
                    sq = MIRRORR(sq);
                int off = piece_map[player][pc - wking]
                         - wking + 13 * h;
                D(sq,off) = 1.0f;
            }
            if(draw && draw[h]) {
                int off = 13 * h + 12;
                for(int sq = 0; sq < 64; sq++) {
                    sq = SQ6488(sq);
                    D(sq,off) = 1.0;
                }
            }
            i++;
        }

        for(int i = 0; i < 64; i++) {
            sq = SQ6488(i);
            if(player == _BLACK) {
                if(cast & 8) D(sq,(CHANNELS - 8)) = 1.0;
                if(cast & 4) D(sq,(CHANNELS - 7)) = 1.0;
                if(cast & 2) D(sq,(CHANNELS - 6)) = 1.0;
                if(cast & 1) D(sq,(CHANNELS - 5)) = 1.0;
                D(sq,(CHANNELS - 4)) = 1.0;
            } else {
                if(cast & 2) D(sq,(CHANNELS - 8)) = 1.0;
                if(cast & 1) D(sq,(CHANNELS - 7)) = 1.0;
                if(cast & 8) D(sq,(CHANNELS - 6)) = 1.0;
                if(cast & 4) D(sq,(CHANNELS - 5)) = 1.0;
                D(sq,(CHANNELS - 4)) = 0.0;
            }
            D(sq,(CHANNELS - 3)) = fifty / 100.0;
            D(sq,(CHANNELS - 1)) = 1.0;
        }
    }

#if 0
        for(int c = 0; c < CHANNELS;c++) {
            printf("Channel %d\n",c);
            for(int i = 0; i < 8; i++) {
                for(int j = 0; j < 8; j++) {
                    int sq = SQ(i,j);
                    printf("%d ",int(D(sq,c)));
                }
                printf("\n");
            }
            printf("\n");
        }
        fflush(stdout);
#endif

#undef NK_MOVES
#undef BRQ_MOVES
#undef D
}
