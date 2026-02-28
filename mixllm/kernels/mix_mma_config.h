// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#define MAX_M_FOR_SEARCH 163840
#define M_BIN_SIZE 8
#define N_PARTIAL_BIN_SIZE 1000

#define CALL_GEMM(stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n) \
    output = this->gemm<stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n>(\
    matrix_C_computed, matrix_A, \
    matrix_scale_act, matrix_zero, \
    matrix_scale_int8, matrix_scale_int4, \
    matrix_indices_int8, matrix_indices_int4, \
    matrix_B_int8, matrix_B_interleaved)

#define CALL_GEMM_CASE(case_int, stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n) \
    case case_int: \
    assert(config_map[vectorToString({blocksize_m, blocksize_n, tilesize_m, tilesize_n})] == case_int); \
    CALL_GEMM(stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n); \
    break;

#define CALL_GEMM_SWITCH_5(stage, config_id) \
    switch (config_id) { \
        CALL_GEMM_CASE(1, stage, 16, 16, 16, 16); \
        CALL_GEMM_CASE(2, stage, 16, 32, 16, 16); \
        CALL_GEMM_CASE(3, stage, 16, 32, 16, 32); \
        CALL_GEMM_CASE(4, stage, 16, 64, 16, 32); \
        CALL_GEMM_CASE(5, stage, 16, 64, 16, 64); \
        CALL_GEMM_CASE(6, stage, 16, 128, 16, 64); \
        CALL_GEMM_CASE(7, stage, 32, 16, 32, 16); \
        CALL_GEMM_CASE(8, stage, 32, 32, 16, 32); \
        CALL_GEMM_CASE(9, stage, 32, 32, 32, 16); \
        CALL_GEMM_CASE(10, stage, 32, 32, 32, 32); \
        CALL_GEMM_CASE(11, stage, 32, 64, 16, 32); \
        CALL_GEMM_CASE(12, stage, 32, 64, 16, 64); \
        CALL_GEMM_CASE(13, stage, 32, 64, 32, 16); \
        CALL_GEMM_CASE(14, stage, 32, 64, 32, 32); \
        CALL_GEMM_CASE(15, stage, 32, 64, 32, 64); \
        CALL_GEMM_CASE(16, stage, 32, 128, 16, 64); \
        CALL_GEMM_CASE(17, stage, 32, 128, 32, 32); \
        CALL_GEMM_CASE(18, stage, 32, 128, 32, 64); \
        CALL_GEMM_CASE(19, stage, 64, 16, 64, 16); \
        CALL_GEMM_CASE(20, stage, 64, 32, 32, 32); \
        CALL_GEMM_CASE(21, stage, 64, 32, 64, 16); \
        CALL_GEMM_CASE(22, stage, 64, 32, 64, 32); \
        CALL_GEMM_CASE(23, stage, 64, 64, 16, 64); \
        CALL_GEMM_CASE(24, stage, 64, 64, 32, 32); \
        CALL_GEMM_CASE(25, stage, 64, 64, 32, 64); \
        CALL_GEMM_CASE(26, stage, 64, 64, 64, 16); \
        CALL_GEMM_CASE(27, stage, 64, 64, 64, 32); \
        CALL_GEMM_CASE(28, stage, 64, 64, 64, 64); \
        CALL_GEMM_CASE(29, stage, 64, 128, 16, 64); \
        CALL_GEMM_CASE(30, stage, 64, 128, 32, 32); \
        CALL_GEMM_CASE(31, stage, 64, 128, 32, 64); \
        CALL_GEMM_CASE(32, stage, 64, 128, 64, 16); \
        CALL_GEMM_CASE(33, stage, 64, 128, 64, 32); \
        CALL_GEMM_CASE(34, stage, 64, 128, 64, 64); \
        CALL_GEMM_CASE(35, stage, 128, 32, 64, 32); \
        CALL_GEMM_CASE(36, stage, 128, 64, 32, 64); \
        CALL_GEMM_CASE(37, stage, 128, 64, 64, 32); \
        CALL_GEMM_CASE(38, stage, 128, 64, 64, 64); \
        CALL_GEMM_CASE(39, stage, 128, 128, 32, 64); \
        CALL_GEMM_CASE(40, stage, 128, 128, 64, 32); \
        CALL_GEMM_CASE(41, stage, 128, 128, 64, 64); \
        default: assert(false); \
    }

#define CALL_GEMM_SWITCH_11(stage, config_id) \
    switch (config_id) { \
        CALL_GEMM_CASE(1, stage, 16, 16, 16, 16); \
        CALL_GEMM_CASE(2, stage, 16, 32, 16, 16); \
        CALL_GEMM_CASE(3, stage, 16, 32, 16, 32); \
        CALL_GEMM_CASE(4, stage, 16, 64, 16, 32); \
        CALL_GEMM_CASE(5, stage, 16, 64, 16, 64); \
        CALL_GEMM_CASE(6, stage, 16, 128, 16, 64); \
        CALL_GEMM_CASE(7, stage, 32, 16, 32, 16); \
        CALL_GEMM_CASE(8, stage, 32, 32, 16, 32); \
        CALL_GEMM_CASE(9, stage, 32, 32, 32, 16); \
        CALL_GEMM_CASE(10, stage, 32, 32, 32, 32); \
        CALL_GEMM_CASE(11, stage, 32, 64, 16, 32); \
        CALL_GEMM_CASE(12, stage, 32, 64, 16, 64); \
        CALL_GEMM_CASE(13, stage, 32, 64, 32, 16); \
        CALL_GEMM_CASE(14, stage, 32, 64, 32, 32); \
        CALL_GEMM_CASE(15, stage, 32, 64, 32, 64); \
        CALL_GEMM_CASE(16, stage, 32, 128, 16, 64); \
        CALL_GEMM_CASE(17, stage, 32, 128, 32, 32); \
        CALL_GEMM_CASE(18, stage, 32, 128, 32, 64); \
        CALL_GEMM_CASE(19, stage, 64, 16, 64, 16); \
        CALL_GEMM_CASE(20, stage, 64, 32, 32, 32); \
        CALL_GEMM_CASE(21, stage, 64, 32, 64, 16); \
        CALL_GEMM_CASE(22, stage, 64, 32, 64, 32); \
        CALL_GEMM_CASE(23, stage, 64, 64, 16, 64); \
        CALL_GEMM_CASE(24, stage, 64, 64, 32, 32); \
        CALL_GEMM_CASE(25, stage, 64, 64, 32, 64); \
        CALL_GEMM_CASE(26, stage, 64, 64, 64, 16); \
        CALL_GEMM_CASE(27, stage, 64, 64, 64, 32); \
        CALL_GEMM_CASE(28, stage, 64, 64, 64, 64); \
        CALL_GEMM_CASE(29, stage, 64, 128, 16, 64); \
        CALL_GEMM_CASE(30, stage, 64, 128, 32, 32); \
        CALL_GEMM_CASE(31, stage, 64, 128, 32, 64); \
        CALL_GEMM_CASE(32, stage, 64, 128, 64, 16); \
        CALL_GEMM_CASE(33, stage, 64, 128, 64, 32); \
        CALL_GEMM_CASE(34, stage, 64, 128, 64, 64); \
        default: CALL_GEMM(3, 16, 16, 16, 16); \
    }



#define CALL_GEMM_SWITCH(stage, config_id) \
    switch (stage) { \
        case 5: {CALL_GEMM_SWITCH_5(5, config_id) break;} \
        case 11: {CALL_GEMM_SWITCH_11(11, config_id) break;} \
        default: assert(false); \
    }

#define CALL_GEMM_RM(stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n) \
    output = this->gemm_rm<stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n>(\
    matrix_C_computed, matrix_A, \
    matrix_scale_act, matrix_zero, \
    matrix_scale_int8, matrix_scale_int4, \
    matrix_indices_int8, matrix_indices_int4, \
    matrix_B_int8, matrix_B_interleaved)

#define CALL_GEMM_CASE_RM(case_int, stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n) \
    case case_int: \
    assert(config_map_rm[vectorToString({blocksize_m, blocksize_n, tilesize_m, tilesize_n})] == case_int); \
    CALL_GEMM_RM(stage, blocksize_m, blocksize_n, tilesize_m, tilesize_n); \
    break;

#define CALL_GEMM_SWITCH_RM_5(stage, config_id) \
    switch (config_id) { \
        CALL_GEMM_CASE_RM(1, stage, 16,32,16,32); \
        CALL_GEMM_CASE_RM(2, stage, 16,64,16,32); \
        CALL_GEMM_CASE_RM(3, stage, 16,64,16,64); \
        CALL_GEMM_CASE_RM(4, stage, 16,128,16,64); \
        CALL_GEMM_CASE_RM(5, stage, 32,32,16,32); \
        CALL_GEMM_CASE_RM(6, stage, 32,32,32,32); \
        CALL_GEMM_CASE_RM(7, stage, 32,64,16,32); \
        CALL_GEMM_CASE_RM(8, stage, 32,64,16,64); \
        CALL_GEMM_CASE_RM(9, stage, 32,64,32,32); \
        CALL_GEMM_CASE_RM(10, stage, 32,64,32,64); \
        CALL_GEMM_CASE_RM(11, stage, 32,128,16,64); \
        CALL_GEMM_CASE_RM(12, stage, 32,128,32,32); \
        CALL_GEMM_CASE_RM(13, stage, 32,128,32,64); \
        CALL_GEMM_CASE_RM(14, stage, 64,32,32,32); \
        CALL_GEMM_CASE_RM(15, stage, 64,32,64,32); \
        CALL_GEMM_CASE_RM(16, stage, 64,64,16,64); \
        CALL_GEMM_CASE_RM(17, stage, 64,64,32,32); \
        CALL_GEMM_CASE_RM(18, stage, 64,64,32,64); \
        CALL_GEMM_CASE_RM(19, stage, 64,64,64,32); \
        CALL_GEMM_CASE_RM(20, stage, 64,64,64,64); \
        CALL_GEMM_CASE_RM(21, stage, 64,128,16,64); \
        CALL_GEMM_CASE_RM(22, stage, 64,128,32,32); \
        CALL_GEMM_CASE_RM(23, stage, 64,128,32,64); \
        CALL_GEMM_CASE_RM(24, stage, 64,128,64,32); \
        CALL_GEMM_CASE_RM(25, stage, 64,128,64,64); \
        CALL_GEMM_CASE_RM(26, stage, 128,32,64,32); \
        CALL_GEMM_CASE_RM(27, stage, 128,64,32,64); \
        CALL_GEMM_CASE_RM(28, stage, 128,64,64,32); \
        CALL_GEMM_CASE_RM(29, stage, 128,64,64,64); \
        CALL_GEMM_CASE_RM(30, stage, 128,64,128,32); \
        CALL_GEMM_CASE_RM(31, stage, 128,128,32,64); \
        CALL_GEMM_CASE_RM(32, stage, 128,128,64,32); \
        CALL_GEMM_CASE_RM(33, stage, 128,128,64,64); \
        CALL_GEMM_CASE_RM(34, stage, 128,128,128,32); \
        CALL_GEMM_CASE_RM(35, stage, 128,128,128,64); \
        CALL_GEMM_CASE_RM(36, stage, 32,256,32,64); \
        CALL_GEMM_CASE_RM(37, stage, 64,256,32,64); \
        CALL_GEMM_CASE_RM(38, stage, 64,256,64,32); \
        CALL_GEMM_CASE_RM(39, stage, 64,256,64,64); \
        CALL_GEMM_CASE_RM(40, stage, 128,256,64,64); \
        CALL_GEMM_CASE_RM(41, stage, 128,256,128,32); \
        CALL_GEMM_CASE_RM(42, stage, 128,256,128,64); \
        CALL_GEMM_CASE_RM(43, stage, 256,64,64,64); \
        CALL_GEMM_CASE_RM(44, stage, 256,64,128,32); \
        CALL_GEMM_CASE_RM(45, stage, 256,128,64,64); \
        CALL_GEMM_CASE_RM(46, stage, 256,128,128,32); \
        CALL_GEMM_CASE_RM(47, stage, 256,128,128,64); \
        default: assert(false); \
    }

#define CALL_GEMM_SWITCH_RM_11(stage, config_id) \
    switch (config_id) { \
        CALL_GEMM_CASE_RM(1, stage, 16,32,16,32); \
        CALL_GEMM_CASE_RM(2, stage, 16,64,16,32); \
        CALL_GEMM_CASE_RM(3, stage, 16,64,16,64); \
        CALL_GEMM_CASE_RM(4, stage, 16,128,16,64); \
        CALL_GEMM_CASE_RM(5, stage, 32,32,16,32); \
        CALL_GEMM_CASE_RM(6, stage, 32,32,32,32); \
        CALL_GEMM_CASE_RM(7, stage, 32,64,16,32); \
        CALL_GEMM_CASE_RM(8, stage, 32,64,16,64); \
        CALL_GEMM_CASE_RM(9, stage, 32,64,32,32); \
        CALL_GEMM_CASE_RM(10, stage, 32,64,32,64); \
        CALL_GEMM_CASE_RM(11, stage, 32,128,16,64); \
        CALL_GEMM_CASE_RM(12, stage, 32,128,32,32); \
        CALL_GEMM_CASE_RM(13, stage, 32,128,32,64); \
        CALL_GEMM_CASE_RM(14, stage, 64,32,32,32); \
        CALL_GEMM_CASE_RM(15, stage, 64,32,64,32); \
        CALL_GEMM_CASE_RM(16, stage, 64,64,16,64); \
        CALL_GEMM_CASE_RM(17, stage, 64,64,32,32); \
        CALL_GEMM_CASE_RM(18, stage, 64,64,32,64); \
        CALL_GEMM_CASE_RM(19, stage, 64,64,64,32); \
        CALL_GEMM_CASE_RM(20, stage, 64,64,64,64); \
        CALL_GEMM_CASE_RM(21, stage, 64,128,16,64); \
        CALL_GEMM_CASE_RM(22, stage, 64,128,32,32); \
        CALL_GEMM_CASE_RM(23, stage, 64,128,32,64); \
        CALL_GEMM_CASE_RM(24, stage, 64,128,64,32); \
        CALL_GEMM_CASE_RM(25, stage, 64,128,64,64); \
        CALL_GEMM_CASE_RM(26, stage, 128,32,64,32); \
        CALL_GEMM_CASE_RM(27, stage, 128,64,32,64); \
        CALL_GEMM_CASE_RM(28, stage, 128,64,64,32); \
        CALL_GEMM_CASE_RM(29, stage, 128,64,64,64); \
        default: CALL_GEMM_RM(5, 16,32,16,32); \
    }



#define CALL_GEMM_SWITCH_RM(stage, config_id) \
    switch (stage) { \
        case 5: {CALL_GEMM_SWITCH_RM_5(5, config_id) break;} \
        case 11: {CALL_GEMM_SWITCH_RM_11(11, config_id) break;} \
        default: assert(false); \
    }


namespace mixllm{
std::vector<std::vector<int>> gemm_configs{
    {16,16,16,16},
    {16,32,16,16},
    {16,32,16,32},
    {16,64,16,32},
    {16,64,16,64},
    {16,128,16,64},
    {32,16,32,16},
    {32,32,16,32},
    {32,32,32,16},
    {32,32,32,32},
    {32,64,16,32},
    {32,64,16,64},
    {32,64,32,16},
    {32,64,32,32},
    {32,64,32,64},
    {32,128,16,64},
    {32,128,32,32},
    {32,128,32,64},
    {64,16,64,16},
    {64,32,32,32},
    {64,32,64,16},
    {64,32,64,32},
    {64,64,16,64},
    {64,64,32,32},
    {64,64,32,64},
    {64,64,64,16},
    {64,64,64,32},
    {64,64,64,64},
    {64,128,16,64},
    {64,128,32,32},
    {64,128,32,64},
    {64,128,64,16},
    {64,128,64,32},
    {64,128,64,64},
    {128,32,64,32},
    {128,64,32,64},
    {128,64,64,32},
    {128,64,64,64},
    {128,128,32,64},
    {128,128,64,32},
    {128,128,64,64}};

std::vector<std::vector<int>> gemm_configs_rm{
    {16,32,16,32},
    {16,64,16,32},
    {16,64,16,64},
    {16,128,16,64},
    {32,32,16,32},
    {32,32,32,32},
    {32,64,16,32},
    {32,64,16,64},
    {32,64,32,32},
    {32,64,32,64},
    {32,128,16,64},
    {32,128,32,32},
    {32,128,32,64},
    {64,32,32,32},
    {64,32,64,32},
    {64,64,16,64},
    {64,64,32,32},
    {64,64,32,64},
    {64,64,64,32},
    {64,64,64,64},
    {64,128,16,64},
    {64,128,32,32},
    {64,128,32,64},
    {64,128,64,32},
    {64,128,64,64},
    {128,32,64,32},
    {128,64,32,64},
    {128,64,64,32},
    {128,64,64,64},
    {128,64,128,32},
    {128,128,32,64},
    {128,128,64,32},
    {128,128,64,64},
    {128,128,128,32},
    {128,128,128,64},
    {32,256,32,64},
    {64,256,32,64},
    {64,256,64,32},
    {64,256,64,64},
    {128,256,64,64},
    {128,256,128,32},
    {128,256,128,64},
    {256,64,64,64},
    {256,64,128,32},
    {256,128,64,64},
    {256,128,128,32},
    {256,128,128,64}
};
}

//end-snippet