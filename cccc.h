#if !defined(CCCC_IMPL)
#define CCCC_IMPL

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CCCC_ASSERT(x, ...) do { if (!(x)) {                                               \
    fflush(stderr);                                                                        \
    fprintf(stderr, "CCCC_ASSERT %s:%d: %s ", __FILE__, __LINE__, #x);                     \
    __VA_OPT__(fprintf(stderr, __VA_ARGS__);)                                              \
    fprintf(stderr, "\n");                                                                 \
    exit(EXIT_FAILURE);                                                                    \
} } while (0);

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define CCCC_SRCS_MAX 2
#define CCCC_TYPE_MAX 3
#define CCCC_DIMS_MAX 4
#define CCCC_CHAR_MAX 100
#define CCCC_NODE_MAX 256

// clang-format off

//
//  ████████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
//  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
//     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
//     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
//     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
//     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝
//

// clang-format on

enum cccc_type {
    CCCC_TYPE_FP16,
    CCCC_TYPE_FP32,
    CCCC_TYPE_FP64
};

const static int cccc_type_sizes[CCCC_TYPE_MAX] = {
    [CCCC_TYPE_FP16] = sizeof(_Float16),
    [CCCC_TYPE_FP32] = sizeof(float),
    [CCCC_TYPE_FP64] = sizeof(double)
};

enum cccc_buff {
    CCCC_BUFF_NONE,
    CCCC_BUFF_INTR,
    CCCC_BUFF_LOAD,
    CCCC_BUFF_SAVE
};

enum cccc_oper {
    CCCC_OPER_NONE,
    CCCC_OPER_CNST,

    CCCC_OPER_LOG,
    CCCC_OPER_EXP,
    CCCC_OPER_SIN,
    CCCC_OPER_REC,
    CCCC_OPER_SQRT,

    CCCC_OPER_ADD,
    CCCC_OPER_MUL,

    CCCC_OPER_RESHAPE,
    CCCC_OPER_PERMUTE,

    CCCC_OPER_SUM_REDUCE,
    CCCC_OPER_MAX_REDUCE
};

struct cccc_tensor {
    enum cccc_type type;
    enum cccc_oper oper;

    enum cccc_buff buff;

    int shape[CCCC_DIMS_MAX];
    int stride[CCCC_DIMS_MAX];

    struct cccc_tensor * src[CCCC_SRCS_MAX];
    struct cccc_tensor * grad;

    bool has_grad;

    void * data;
    int index;
};

// clang-format off

//
//  ██╗███╗   ██╗██╗████████╗
//  ██║████╗  ██║██║╚══██╔══╝
//  ██║██╔██╗ ██║██║   ██║
//  ██║██║╚██╗██║██║   ██║
//  ██║██║ ╚████║██║   ██║
//  ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

static struct cccc_tensor * cccc_new_tensor_impl(enum cccc_type type, int shape[CCCC_DIMS_MAX]) {
    struct cccc_tensor * result = malloc(sizeof(struct cccc_tensor));

    *result = (struct cccc_tensor){
       /*.type       =*/ type,
       /*.oper       =*/ CCCC_OPER_NONE,
       /*.buff       =*/ CCCC_BUFF_NONE,
       /*.shape      =*/ {shape[0], shape[1], shape[2], shape[3]},
       /*.stride     =*/ {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1},
       /*.src        =*/ {NULL},
       /*.grad       =*/ NULL,
       /*.has_grad   =*/ false,
       /*.data       =*/ NULL,
       /*.index      =*/ -1,
    };

    return result;
}

// clang-format on

struct cccc_tensor * cccc_new_tensor(enum cccc_type type, int ne0, int ne1, int ne2, int ne3,
                                     bool has_grad) {
    int shape[CCCC_DIMS_MAX] = {ne0, ne1, ne2, ne3};

    struct cccc_tensor * result = cccc_new_tensor_impl(type, shape);
    result->buff = CCCC_BUFF_LOAD;
    result->has_grad = has_grad;

    return result;
}

int cccc_tensor_size(struct cccc_tensor * tensor) {
    return tensor->shape[0] * tensor->shape[1] * tensor->shape[2] * tensor->shape[3];
}

struct cccc_tensor * cccc_const(enum cccc_type type, int shape[CCCC_DIMS_MAX], float value) {
    struct cccc_tensor * result = cccc_new_tensor_impl(type, shape);

    result->buff = CCCC_BUFF_INTR;
    result->oper = CCCC_OPER_CNST;
    result->data = malloc(shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
    for (int i = 0; i < shape[0] * shape[1] * shape[2] * shape[3]; i++) {
        *((float *)result->data + i) = value;
    }

    return result;
}

// clang-format off

//
//  ███╗   ███╗██╗███████╗ ██████╗
//  ████╗ ████║██║██╔════╝██╔════╝
//  ██╔████╔██║██║███████╗██║
//  ██║╚██╔╝██║██║╚════██║██║
//  ██║ ╚═╝ ██║██║███████║╚██████╗
//  ╚═╝     ╚═╝╚═╝╚══════╝ ╚═════╝
//
//  ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
//  ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

// clang-format on

static bool cccc_can_broadcast(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    if (rhs == NULL || lhs == NULL)
        return true;
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        if (lhs->shape[i] != rhs->shape[i] && lhs->shape[i] != 1 && rhs->shape[i] != 1) {
            return false;
        }
    }

    return true;
}

static bool cccc_broadcasted(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    return lhs->shape[0] != rhs->shape[0] || lhs->shape[1] != rhs->shape[1] ||
           lhs->shape[2] != rhs->shape[2] || lhs->shape[3] != rhs->shape[3];
}

static bool cccc_has_buffer(struct cccc_tensor * tensor) {
    switch (tensor->buff) {
        case CCCC_BUFF_NONE: return false;
        default: return true;
    }
}

static bool cccc_owns_buffer(struct cccc_tensor * tensor) {
    switch (tensor->buff) {
        case CCCC_BUFF_NONE:
        case CCCC_BUFF_INTR: return false;
        default: return true;
    }
}

// clang-format off

static int cccc_tensor_n_dim(struct cccc_tensor * tensor) {
    int last_dim = 0;
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        if(tensor->shape[i] != 1) last_dim = i;
    }
    return last_dim == 0 ? 1 : last_dim + 1;
}

//
//   ██████╗ ██████╗ ██████╗ ███████╗
//  ██╔════╝██╔═══██╗██╔══██╗██╔════╝
//  ██║     ██║   ██║██████╔╝█████╗
//  ██║     ██║   ██║██╔══██╗██╔══╝
//  ╚██████╗╚██████╔╝██║  ██║███████╗
//   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

// clang-format on

#define CCCC_UNARY_OPERATION(function, operation)                                                  \
    struct cccc_tensor * function(struct cccc_tensor * tensor) {                                   \
        struct cccc_tensor * result = cccc_new_tensor_impl(tensor->type, tensor->shape);           \
                                                                                                   \
        for (int i = 0; i < CCCC_DIMS_MAX; i++) {                                                  \
            result->stride[i] = tensor->stride[i];                                                 \
        }                                                                                          \
                                                                                                   \
        result->oper = operation;                                                                  \
        result->src[0] = tensor;                                                                   \
        result->has_grad = tensor->has_grad;                                                       \
                                                                                                   \
        return result;                                                                             \
    }

CCCC_UNARY_OPERATION(cccc_log, CCCC_OPER_LOG);
CCCC_UNARY_OPERATION(cccc_exp, CCCC_OPER_EXP);
CCCC_UNARY_OPERATION(cccc_sin, CCCC_OPER_SIN);
CCCC_UNARY_OPERATION(cccc_rec, CCCC_OPER_REC);
CCCC_UNARY_OPERATION(cccc_sqrt, CCCC_OPER_SQRT);

struct cccc_tensor * cccc_add(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    CCCC_ASSERT(cccc_can_broadcast(lhs, rhs));

    bool null_input = lhs == NULL || rhs == NULL;
    struct cccc_tensor * non_null = lhs != NULL ? lhs : rhs;

    int shape[CCCC_DIMS_MAX] = {0};
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        shape[i] = null_input ? non_null->shape[i] :
            (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    struct cccc_tensor * result = cccc_new_tensor_impl(lhs->type, shape);

    result->oper = CCCC_OPER_ADD;
    result->src[0] = null_input ? result : lhs;
    result->src[1] = null_input ? non_null : rhs;
    result->has_grad = null_input ? non_null->has_grad : lhs->has_grad || rhs->has_grad;

    return result;
}

struct cccc_tensor * cccc_mul(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    CCCC_ASSERT(cccc_can_broadcast(lhs, rhs));

    bool null_input = lhs == NULL || rhs == NULL;
    struct cccc_tensor * non_null = lhs != NULL ? lhs : rhs;

    int shape[CCCC_DIMS_MAX] = {0};
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        shape[i] = null_input ? non_null->shape[i] :
            (lhs->shape[i] + rhs->shape[i] + abs(lhs->shape[i] - rhs->shape[i])) / 2;
    }

    struct cccc_tensor * result = cccc_new_tensor_impl(lhs->type, shape);

    result->oper = CCCC_OPER_MUL;
    result->src[0] = null_input ? result : lhs;
    result->src[1] = null_input ? non_null : rhs;
    result->has_grad = null_input ? non_null->has_grad : lhs->has_grad || rhs->has_grad;

    return result;
}

struct cccc_tensor * cccc_reshape(struct cccc_tensor * tensor, int shape[CCCC_DIMS_MAX]) {
    int size = cccc_tensor_size(tensor);
    int new_size = shape[0] * shape[1] * shape[2] * shape[3];
    CCCC_ASSERT(size == new_size, "reshaped and source tensor must have the same size");

    struct cccc_tensor * result = cccc_new_tensor_impl(tensor->type, shape);

    result->oper = CCCC_OPER_RESHAPE;
    result->buff = CCCC_BUFF_INTR;
    result->src[0] = tensor;
    result->has_grad = tensor->has_grad;

    return result;
}

struct cccc_tensor * cccc_permute(struct cccc_tensor * tensor, int perm[CCCC_DIMS_MAX]) {
    struct cccc_tensor * result = cccc_new_tensor_impl(tensor->type, tensor->shape);
    int n_dim = cccc_tensor_n_dim(tensor);
    for (int i = 0; i < n_dim; i++) {
        result->shape[i] = tensor->shape[perm[i]];
        result->stride[i] = tensor->stride[perm[i]];
    }

    result->oper = CCCC_OPER_PERMUTE;
    result->buff = CCCC_BUFF_INTR;
    result->src[0] = tensor;
    result->has_grad = tensor->has_grad;

    return result;
}

struct cccc_tensor * cccc_sum(struct cccc_tensor * tensor, int n_axes, int axes[CCCC_DIMS_MAX]) {
    CCCC_ASSERT(n_axes > 0 && n_axes < CCCC_DIMS_MAX);

    int shape[CCCC_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        shape[i] = tensor->shape[i];
    }

    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    struct cccc_tensor * result = cccc_new_tensor_impl(tensor->type, shape);

    result->oper = CCCC_OPER_SUM_REDUCE;
    result->buff = CCCC_BUFF_INTR;
    result->src[0] = tensor;
    result->has_grad = tensor->has_grad;

    return result;
}

struct cccc_tensor * cccc_max(struct cccc_tensor * tensor, int n_axes, int axes[CCCC_DIMS_MAX]) {
    CCCC_ASSERT(n_axes > 0 && n_axes < CCCC_DIMS_MAX);

    int shape[CCCC_DIMS_MAX] = {1, 1, 1, 1};
    for (int i = 0; i < CCCC_DIMS_MAX; i++) {
        shape[i] = tensor->shape[i];
    }

    for (int i = 0; i < n_axes; i++) {
        shape[axes[i]] = 1;
    }

    struct cccc_tensor * result = cccc_new_tensor_impl(tensor->type, shape);

    result->oper = CCCC_OPER_MAX_REDUCE;
    result->buff = CCCC_BUFF_INTR;
    result->src[0] = tensor;
    result->has_grad = tensor->has_grad;

    return result;
}

// clang-format off

//
//  ███╗   ██╗ ██████╗ ███╗   ██╗       ██████╗ ██████╗ ██████╗ ███████╗
//  ████╗  ██║██╔═══██╗████╗  ██║      ██╔════╝██╔═══██╗██╔══██╗██╔════╝
//  ██╔██╗ ██║██║   ██║██╔██╗ ██║█████╗██║     ██║   ██║██████╔╝█████╗
//  ██║╚██╗██║██║   ██║██║╚██╗██║╚════╝██║     ██║   ██║██╔══██╗██╔══╝
//  ██║ ╚████║╚██████╔╝██║ ╚████║      ╚██████╗╚██████╔╝██║  ██║███████╗
//  ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═══╝       ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
//
//   ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗██╗ ██████╗ ███╗   ██╗███████╗
//  ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
//  ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║██║   ██║██╔██╗ ██║███████╗
//  ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║██║   ██║██║╚██╗██║╚════██║
//  ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║███████║
//   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝
//

// clang-format on

struct cccc_tensor * cccc_neg(struct cccc_tensor * tensor) {
    return cccc_log(cccc_rec(cccc_exp(tensor)));
}

struct cccc_tensor * cccc_square(struct cccc_tensor * tensor) {
    return cccc_mul(tensor, tensor);
}

struct cccc_tensor * cccc_sub(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    return cccc_add(lhs, cccc_neg(rhs));
}

struct cccc_tensor * cccc_div(struct cccc_tensor * lhs, struct cccc_tensor * rhs) {
    return cccc_mul(lhs, cccc_rec(rhs));
}

struct cccc_tensor * cccc_cos(struct cccc_tensor * tensor) {
    return cccc_sin(cccc_add(tensor, cccc_const(tensor->type, tensor->shape, M_PI_2)));
}

struct cccc_tensor * cccc_tanh(struct cccc_tensor * tensor) {
    return cccc_div(cccc_sub(cccc_exp(tensor), cccc_exp(cccc_neg(tensor))),
                    cccc_add(cccc_exp(tensor), cccc_exp(cccc_neg(tensor))));
}

// clang-format off

//
//  ██╗  ██╗ █████╗ ███████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗
//  ██║  ██║██╔══██╗██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗
//  ███████║███████║███████╗███████║██╔████╔██║███████║██████╔╝
//  ██╔══██║██╔══██║╚════██║██╔══██║██║╚██╔╝██║██╔══██║██╔═══╝
//  ██║  ██║██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║  ██║██║
//  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝
//

// clang-format on

#define CCCC_FNV_PRIME 1099511628211LU
#define CCCC_FNV_OFFSET 14695981039346656037LU

struct cccc_hashmap_entry {
    uintptr_t key;
    int value;
};

struct cccc_hashmap {
    int used;
    struct cccc_hashmap_entry * entries;
    int capacity;
};

static uint64_t cccc_hash_key(void * key) {
    uint64_t hash = CCCC_FNV_OFFSET;
    hash ^= (uint64_t)(uintptr_t)key;
    hash *= CCCC_FNV_PRIME;
    return hash;
}

static struct cccc_hashmap * cccc_new_hashmap() {
    int capacity = CCCC_NODE_MAX;

    struct cccc_hashmap * map = malloc(sizeof(struct cccc_hashmap));
    *map = (struct cccc_hashmap){
        .used = 0,
        .entries = malloc(sizeof(struct cccc_hashmap_entry) * capacity),
        .capacity = capacity,
    };

    for (int i = 0; i < capacity; i++) {
        map->entries[i].key = 0;
        map->entries[i].value = -1;
    }

    return map;
}

static int cccc_hashmap_get(struct cccc_hashmap * map, void * key) {
    if (key == NULL) {
        return -1;
    }

    uint64_t hash = cccc_hash_key(key);
    int index = (int)(hash & (uint64_t)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            return map->entries[index].value;
        }

        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    return -1;
};

static void cccc_hashmap_set(struct cccc_hashmap * map, void * key, int value) {
    if (map->used >= map->capacity) {
        CCCC_ASSERT(false, "hashmap size overflow");
    }

    uint64_t hash = cccc_hash_key(key);
    int index = (int)(hash & (int)(map->capacity - 1));

    while (map->entries[index].key != 0) {
        if ((uintptr_t)key == map->entries[index].key) {
            // Found key (it already exists), update value.
            map->entries[index].value = value;
            return;
        }
        // Key wasn't in this slot, move to next (linear
        // probing).
        index++;
        if (index >= map->capacity) {
            index = 0;
        }
    }

    map->entries[index].key = (uintptr_t)key;
    map->entries[index].value = value;
}

//
//  ██╗   ██████╗
//  ██║   ██╔══██╗
//  ██║   ██████╔╝
//  ██║   ██╔══██╗
//  ██║██╗██║  ██║██╗
//  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
//

// intermediary representation

static const char * cccc_oper_to_string(enum cccc_oper oper) {
    switch (oper) {
        case CCCC_OPER_NONE: return "";
        case CCCC_OPER_CNST: return "";
        case CCCC_OPER_LOG: return "log";
        case CCCC_OPER_EXP: return "exp";
        case CCCC_OPER_SIN: return "sin";
        case CCCC_OPER_REC: return "1/";
        case CCCC_OPER_SQRT: return "sqrt";
        case CCCC_OPER_ADD: return "+";
        case CCCC_OPER_MUL: return "*";
        case CCCC_OPER_RESHAPE: return "";
        case CCCC_OPER_PERMUTE: return "";
        default: CCCC_ASSERT(false, "invalid conversion of type to string");
    }
}

static const char * cccc_type_to_string(enum cccc_type type) {
    switch (type) {
        case CCCC_TYPE_FP16: return "fp16";
        case CCCC_TYPE_FP32: return "fp32";
        case CCCC_TYPE_FP64: return "fp64";
        default: CCCC_ASSERT(false, "unknown variant of cccc_type");
    }
}

// super unsafe i think?
static void cccc_find_and_replace(char * string, const char * needle, const char * replacement) {
    char * haystack = string;
    char * result = haystack;

    while ((result = strstr(result, needle)) != NULL) {
        size_t position = result - haystack;
        size_t length = strlen(needle);

        memmove(haystack + position + strlen(replacement), haystack + position + length,
                strlen(haystack + position + length) + 1);
        memcpy(haystack + position, replacement, strlen(replacement));

        result += strlen(replacement);
    }
}

// both reduction and broadcasting index functions look terrible, there has to be a cleaner
// and concise way to express this

static const char * cccc_reduction_index(struct cccc_tensor * parent, struct cccc_tensor * child) {
    char * result = malloc(CCCC_CHAR_MAX * CCCC_DIMS_MAX * sizeof(char));
    int size = CCCC_CHAR_MAX * CCCC_DIMS_MAX;
    *result = '\0';

    for (int i = 0; i < cccc_tensor_n_dim(parent); i++) {
        snprintf(result + strlen(result), size, "%s(idx/%d)%%%d*%d",
                 i != 0 && i != cccc_tensor_n_dim(parent) ? "+" : "", child->stride[i],
                 parent->shape[i], parent->stride[i]);
    }

    return result;
}

static const char * cccc_broadcasting_index(struct cccc_tensor * parent, struct cccc_tensor * child, bool broadcasted) {
    if (broadcasted == false) {
        return "idx";
    }

    char * result = malloc(CCCC_CHAR_MAX * CCCC_DIMS_MAX * sizeof(char));
    int size = CCCC_CHAR_MAX * CCCC_DIMS_MAX;
    *result = '\0';

    for (int i = 0; i < cccc_tensor_n_dim(parent); i++) {
        // disgusting 🤢
        snprintf(result + strlen(result), size, "%s(idx/%d)%%%d*%d",
                 i != 0 && i != cccc_tensor_n_dim(parent) ? "+" : "", parent->stride[i],
                 child->shape[i] == 1 && i < cccc_tensor_n_dim(child) ? 1 : child->shape[i],
                 child->stride[i]);
    }

    return result;
}

// clang-format off

//
//   ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗
//  ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║
//  ██║  ███╗██████╔╝███████║██████╔╝███████║
//  ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║
//  ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║
//   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝
//

// clang-format on

struct cccc_graph {
    int n_nodes;
    struct cccc_tensor * nodes[CCCC_NODE_MAX];
    struct cccc_hashmap * map;
    char ir[CCCC_NODE_MAX * CCCC_CHAR_MAX];
};

static void cccc_graph_forward(struct cccc_graph * graph, struct cccc_tensor * tensor,
                               int * node_counter) {
    if (tensor == NULL) {
        return;
    }

    // also checking if tensor has itself as a child to prevent (infinite)
    // cycles

    if (tensor != tensor->src[0] && cccc_hashmap_get(graph->map, tensor->src[0]) == -1) {
        cccc_graph_forward(graph, tensor->src[0], node_counter);
    }

    if (tensor != tensor->src[1] && cccc_hashmap_get(graph->map, tensor->src[1]) == -1) {
        cccc_graph_forward(graph, tensor->src[1], node_counter);
    }

    if (cccc_hashmap_get(graph->map, tensor) == -1) {
        tensor->index = *node_counter;
        graph->nodes[*node_counter] = tensor;
        cccc_hashmap_set(graph->map, tensor, (*node_counter)++);
    }
}

static void cccc_graph_backward(struct cccc_graph * graph, struct cccc_tensor * root) {
    if (root->has_grad == false) {
        return;
    } else {
        // setting root's gradient as 1, because d(root)/d(root) = 1
        int shape[CCCC_DIMS_MAX] = {1, 1, 1, 1};
        root->grad = cccc_const(root->type, shape, 1);
    }

    // in this loop create gradient tensors corresponding to each tensor
    // that requires gradient tracking, and set their buffers to the correct
    // option respectively (because, for example, a intermediary tensor w/o
    // a buffer also needs an intermediary gradient tensor w/o a buffer)

    struct cccc_tensor * queue[CCCC_NODE_MAX] = {NULL};
    int queue_start = 0;
    int queue_end = 0;
    queue[queue_end++] = root;

    while (queue_end != queue_start) {
        struct cccc_tensor * tensor = queue[queue_start++];
        if (tensor->has_grad == true) {
            // processing node here

            // declaring partials d(tensor)/d(tensor->src[0]) and
            // d(tensor)/d(tensor->src[1])

            struct cccc_tensor * partial_0 = NULL;
            struct cccc_tensor * partial_1 = NULL;

            int shape[CCCC_DIMS_MAX] = {1, 1, 1, 1};

            // calculating partials

            switch (tensor->oper) {
                case CCCC_OPER_NONE:
                case CCCC_OPER_CNST: break;
                case CCCC_OPER_LOG: partial_0 = cccc_rec(tensor->src[0]); break;
                case CCCC_OPER_EXP: partial_0 = cccc_exp(tensor->src[0]); break;
                case CCCC_OPER_SIN: partial_0 = cccc_cos(tensor->src[0]); break;
                case CCCC_OPER_REC:
                    partial_0 = cccc_neg(cccc_rec(cccc_square(tensor->src[0])));
                    break;
                case CCCC_OPER_SQRT:
                    partial_0 = cccc_rec(
                        cccc_mul(cccc_const(tensor->type, shape, 2), cccc_sqrt(tensor->src[0])));
                    break;
                case CCCC_OPER_ADD:
                    partial_0 = cccc_const(tensor->type, shape, 1);
                    partial_1 = cccc_const(tensor->type, shape, 1);
                    break;
                case CCCC_OPER_MUL:
                    partial_0 = tensor->src[1];
                    partial_1 = tensor->src[0];
                    break;
                case CCCC_OPER_RESHAPE:
                case CCCC_OPER_PERMUTE: partial_0 = cccc_const(tensor->type, shape, 1); break;
                case CCCC_OPER_SUM_REDUCE:
                case CCCC_OPER_MAX_REDUCE: partial_0 = cccc_const(tensor->type, shape, 1); break;
                default: CCCC_ASSERT(false, "unknown variant of cccc_oper");
            }

            // multiplying tensor->grad by partials and adding them to
            // the gradients of the tensor's children (we have to do a
            // mini DFS traversal w/ cccc_graph_forward() since the gradient
            // calculation forms a mini sub-graph that needs to be tra-
            // versed separately)

            tensor->src[0]->grad = cccc_add(cccc_mul(tensor->grad, partial_0), NULL);
            cccc_graph_forward(graph, tensor->src[0]->grad, &graph->n_nodes);

            if (tensor->src[1] != NULL) {
                tensor->src[1]->grad = cccc_add(cccc_mul(tensor->grad, partial_1), NULL);
                cccc_graph_forward(graph, tensor->src[1]->grad, &graph->n_nodes);
            }

            // finished processing node and adding children

            if (tensor->src[0] != NULL) {
                queue[queue_end++] = tensor->src[0];
            }
            if (tensor->src[1] != NULL) {
                queue[queue_end++] = tensor->src[1];
            }
        }
    }
}

static void cccc_graph_generate_ir(struct cccc_graph * graph) {
    int size = CCCC_NODE_MAX * CCCC_CHAR_MAX;
    for (int i = 0; i < graph->n_nodes; i++) {
        struct cccc_tensor * tensor = graph->nodes[i];

        switch (tensor->oper) {
            case CCCC_OPER_NONE: break;
            case CCCC_OPER_CNST:
                if (cccc_tensor_size(tensor) == 1) {
                    snprintf(graph->ir + strlen(graph->ir), size, "\t%s data_%d = %f;\n",
                             cccc_type_to_string(tensor->type), i, *(float *)tensor->data);
                }
                break;
            case CCCC_OPER_LOG:
            case CCCC_OPER_EXP:
            case CCCC_OPER_SIN:
            case CCCC_OPER_REC:
            case CCCC_OPER_SQRT:
                if (cccc_has_buffer(tensor)) {
                    snprintf(graph->ir + strlen(graph->ir), size, "\tdata_%d[idx] = ", i);
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size,
                             "\t%s data_%d = ", cccc_type_to_string(tensor->type), i);
                }

                if (cccc_has_buffer(tensor->src[0])) {
                    snprintf(graph->ir + strlen(graph->ir), size, "%s(data_%d[idx]);\n",
                             cccc_oper_to_string(tensor->oper), tensor->src[0]->index);
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size, "%s(data_%d);\n",
                             cccc_oper_to_string(tensor->oper), tensor->src[0]->index);
                }

                break;
            case CCCC_OPER_ADD:
            case CCCC_OPER_MUL:
                if (cccc_has_buffer(tensor)) {
                    snprintf(graph->ir + strlen(graph->ir), size, "\tdata_%d[idx] = ", i);
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size,
                             "\t%s data_%d = ", cccc_type_to_string(tensor->type), i);
                }

                bool broadcasted = cccc_broadcasted(tensor->src[0], tensor->src[1]);

                if (cccc_has_buffer(tensor->src[0])) {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d[%s] %s ",
                             tensor->src[0]->index,
                             cccc_broadcasting_index(tensor, tensor->src[0], broadcasted),
                             cccc_oper_to_string(tensor->oper));
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d %s ",
                             tensor->src[0]->index, cccc_oper_to_string(tensor->oper));
                }

                if (cccc_has_buffer(tensor->src[1])) {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d[%s];\n",
                             tensor->src[1]->index,
                             cccc_broadcasting_index(tensor, tensor->src[1], broadcasted));
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d;\n",
                             tensor->src[1]->index);
                }

                break;
            case CCCC_OPER_RESHAPE:
            case CCCC_OPER_PERMUTE: break;
            case CCCC_OPER_SUM_REDUCE:
                snprintf(graph->ir + strlen(graph->ir), size, "\tdata_%d[%s] += ", tensor->index,
                         cccc_reduction_index(tensor, tensor->src[0]));

                if (cccc_has_buffer(tensor->src[0])) {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d[idx];\n",
                             tensor->src[0]->index);
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d;\n",
                             tensor->src[0]->index);
                }

                break;
            case CCCC_OPER_MAX_REDUCE:
                snprintf(graph->ir + strlen(graph->ir), size, "\tdata_%d[%s] = max(data_%d[idx], ",
                         tensor->index, cccc_reduction_index(tensor, tensor->src[0]),
                         tensor->index);

                if (cccc_has_buffer(tensor->src[0])) {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d[idx]);\n",
                             tensor->src[0]->index);
                } else {
                    snprintf(graph->ir + strlen(graph->ir), size, "data_%d);\n",
                             tensor->src[0]->index);
                }
                break;
            default: CCCC_ASSERT(false, "unknown variant of cccc_oper");
        }
    }
}

static void cccc_graph_node_buffers(struct cccc_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct cccc_tensor * tensor = graph->nodes[i];
        int size = cccc_tensor_size(tensor);

        if (cccc_has_buffer(tensor)) {
            tensor->data = malloc(size * cccc_type_sizes[tensor->type]);
            if (tensor->grad != NULL) {
                tensor->grad->buff = CCCC_BUFF_SAVE;
                tensor->grad->data = malloc(size * cccc_type_sizes[tensor->type]);
            }
        }
    }
}

// clang-format off

struct cccc_graph * cccc_new_graph(struct cccc_tensor * root) {
    root->data = malloc(cccc_tensor_size(root) * sizeof(cccc_type_sizes[root->type]));
    root->buff = CCCC_BUFF_SAVE;
    struct cccc_graph * graph = malloc(sizeof(struct cccc_graph));

    *graph = (struct cccc_graph){
        /*.n_nodes =*/ 0,
        /*.nodes   =*/ {NULL},
        /*.map     =*/ cccc_new_hashmap(),
        /*.ir      =*/ {'\0'}
    };

    cccc_graph_forward(graph, root, &graph->n_nodes);
    cccc_graph_backward(graph, root);

    cccc_graph_node_buffers(graph);
    cccc_graph_generate_ir(graph);

    return graph;
};

// clang-format on

static void cccc_graph_free(struct cccc_graph * graph) {
    for (int i = 0; i < graph->n_nodes; i++) {
        struct cccc_tensor * tensor = graph->nodes[i];

        // only freeing when tensor isn't of type reshape/permute, because those tensors
        // just use their children's data pointer, so we avoid a double free this way :)
        if (tensor->oper != CCCC_OPER_RESHAPE && tensor->oper != CCCC_OPER_PERMUTE) {
            free(tensor->data);
        }

        free(tensor);
    }

    free(graph);
}

// clang-format off

//
//  ██████╗  █████╗  ██████╗██╗  ██╗███████╗███╗   ██╗██████╗
//  ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝████╗  ██║██╔══██╗
//  ██████╔╝███████║██║     █████╔╝ █████╗  ██╔██╗ ██║██║  ██║
//  ██╔══██╗██╔══██║██║     ██╔═██╗ ██╔══╝  ██║╚██╗██║██║  ██║
//  ██████╔╝██║  ██║╚██████╗██║  ██╗███████╗██║ ╚████║██████╔╝
//  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝
//
//   ██████╗██╗   ██╗██████╗  █████╗
//  ██╔════╝██║   ██║██╔══██╗██╔══██╗
//  ██║     ██║   ██║██║  ██║███████║
//  ██║     ██║   ██║██║  ██║██╔══██║
//  ╚██████╗╚██████╔╝██████╔╝██║  ██║
//   ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝
//

// clang-format on

const char * cccc_parser_cuda(struct cccc_graph * graph) {
    int size = CCCC_NODE_MAX * CCCC_CHAR_MAX;
    char * kernel_string = malloc(size * sizeof(char));
    *kernel_string = '\0';

    // adding includes and kernel function signature to the
    // kernel string

    int offset = snprintf(kernel_string + strlen(kernel_string), size,
                          "#include <cuda_runtime.h>\n"
                          "#include <cuda_fp16.h>\n\n"
                          "__global__ void cccc_kernel(");

    // adding kernel input parameters to the kernel string

    int n_kernel_parameters = 0;
    int largest_tensor = 1;

    for (int i = 0; i < graph->n_nodes; i++) {
        struct cccc_tensor * tensor = graph->nodes[i];
        int tensor_size = cccc_tensor_size(tensor);
        if (tensor_size > largest_tensor)
            largest_tensor = tensor_size;

        if (cccc_owns_buffer(tensor) && cccc_tensor_size(tensor) != 1 &&
            tensor->oper != CCCC_OPER_RESHAPE && tensor->oper != CCCC_OPER_PERMUTE) {
            if (n_kernel_parameters == 0) {
                snprintf(kernel_string + strlen(kernel_string), size, "%s * data_%d",
                         cccc_type_to_string(tensor->type), i);
                n_kernel_parameters++;
            } else {
                snprintf(kernel_string + strlen(kernel_string), size, ", %s * data_%d",
                         cccc_type_to_string(tensor->type), i);
                n_kernel_parameters++;
            }
        }
    }

    snprintf(kernel_string + strlen(kernel_string), size,
             ") {\n\tint idx = blockDim.x * blockIdx.x + threadIdx.x;\n"
             "\tif (idx < %d) return;\n\n",
             largest_tensor);

    // prepending kernel_string to graph->ir
    memmove(kernel_string + strlen(kernel_string), graph->ir, strlen(graph->ir) + 1);

    // cuda specific type substitution (i.e. fp16 to __half and fp32 to float)
    // adding offset so that fp16 from the include header name doesn't get
    // replaced
    cccc_find_and_replace(kernel_string + offset, "fp16", "__half");
    cccc_find_and_replace(kernel_string + offset, "fp32", "float");
    cccc_find_and_replace(kernel_string + offset, "fp64", "double");

    // adding the closing braces/brackets :)

    snprintf(kernel_string + strlen(kernel_string), size, "}");
    return kernel_string;
}

#endif
