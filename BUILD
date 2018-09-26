load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object")

tf_cc_shared_object(
    name = "egbbdll64.dll",
    srcs = [
	"cache.cpp",
	"codec.cpp",
	"decompress.cpp",
	"egbbdll.cpp",
	"eval_nn.cpp",
	"index.cpp",
	"moves.cpp",
	"codec.h",
	"my_types.h",
	"common.h",
	"cache.h",
	"egbbdll.h"
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow"
    ],
    defines = [ "TENSORFLOW" ]
)
