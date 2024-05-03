### What?

❇️ cccc.h is a single header pure C implementation of primitive autodiff w/ no dependencies, kernel fusion and automatic compute kernel generation (only CUDA for now)

❇️ cccc.h is ~1000 lines of code, so hopefully it can be useful as a learning opportunity (cccc.h is experimental and probably contains unsafe code and errors)

❇️ cccc.h follows the design philosophy of libraries like tinygrad and luminal, where all operations are defined in terms of a small set of primitive operations

### How?

❇️ the following example defines a computation `ln(x) = y`, constructs a computational graph that evaluates `y` as well as `d(y)/d(x)`, keep in mind that  this is a simple example, and supported operations can be arbitrarily stacked on top of each other

```c
#include "cccc.h"

int main(){
    // set up a 2d 32-bit float tensor w/ gradien tracking
    cccc_tensor * x = cccc_new_tensor_2d(CCCC_TYPE_FP32, 2, 3, true);
    cccc_tensor * y = cccc_log(x);

    // constructing a computational graph
    cccc_graph * graph = cccc_new_graph(y);

    // parsing the graph w/ a cuda parser
    const char * kernel_string = cccc_parser_cuda(graph);

    // printing the resulting kernel
    printf("%s\n", kernel_string);

    // free graph and nodes
    cccc_graph_free(graph);
}
```

this program outputs the following cuda kernel which includes both the forward and the backward pass

```cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void cccc_kernel(float * data_0, float * data_5) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < 2 * 3 * 1 * 1) return;

    float data_1 = log(data_0[idx]);
    float data_2 = 1.000000;
    float data_3 = 1/(data_0[idx]);
    float data_4 = data_2 * data_3;
    data_5[idx] = data_5[idx] + data_4;
}

```

in particular, `data_1` calculates the forward pass, `data_3` calculates the partial `dy/dx`, and `data_4` multiplies it by the pre-existing `x` gradient, and `data_5` adds the result to previous `x` gradient

