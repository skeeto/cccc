### What?

❇️ cccc.h is a single header pure C implementation of primitive autodiff w/ no dependencies, kernel fusion and automatic compute kernel generation

❇️ cccc.h is ~1000 lines of code, so hopefully it can be useful as a learning opportunity

❇️ cccc.h follows the design philosophy of libraries like tinygrad and luminal, where all operations are defined in terms of a small set of primitive operations

### How?

❇️ the following example defines a computation `sin(ln(x)) = y` and generates a kernel that calculates the value of `y` as well as `dy/dx` gradient accumulation

```c
#include "cccc.h"

int main() {
    // define 2d tensor x with fp32 data type, with gradient tracking
    cccc_tensor * x = cccc_new_tensor_2d(CCCC_TYPE_FP32, 2, 3, true);
    cccc_tensor * y = cccc_sin(cccc_log(x));

    cccc_graph * graph = cccc_new_graph(y);
    const char * ir = cccc_parser_cuda(graph);

    printf("%s\n", ir);
    cccc_graph_free(graph);
}
```

this program outputs the following cuda kernel which includes both the forward and the backward pass all fused into a single kernel

```cuda
__global__ void cccc_kernel(float * data_0, float * data_2, float * data_11) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < 6) return;

    float data_1 = log(data_0[idx]);
    data_2[idx] = sin(data_1);
    float data_3 = 1.000000;
    float data_4 = 1.570796;
    float data_5 = data_1 + data_4;
    float data_6 = sin(data_5);
    float data_7 = data_3 * data_6;
    float data_8 = data_8 + data_7;
    float data_9 = 1/(data_0[idx]);
    float data_10 = data_8 * data_9;
    data_11[idx] = data_11[idx] + data_10;
}

```

