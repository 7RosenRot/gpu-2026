#ifndef SHKENEV_ILYA_GELU_OCL_H
#define SHKENEV_ILYA_GELU_OCL_H

#include <vector>

std::vector<float> GeluOCL(const std::vector<float>& input, int platform);

#endif
