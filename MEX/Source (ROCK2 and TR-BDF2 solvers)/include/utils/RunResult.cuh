#pragma once

#include <cublas_v2.h>
#include "cuda_runtime.h"

#include <iostream>
#include <string>



struct RunResult
{
	std::string message;
	int code;

	RunResult(int _code, std::string _message)
		: code(_code), message(_message) {};
	RunResult()
		: code(0), message() {};

	operator int() const { return this->code; };
};

std::ostream& operator << (std::ostream& os, const RunResult& res)
{
	os << "code: " << res.code << ", message: " << res.message;
	return os;
}

#define CUDA_LOG(op){auto res = op;  if (res != 0){std::cerr << "FILE " << __FILE__ << " LINE " << __LINE__ << " CUDA ERROR: " << res << std::endl;}}
#define CUDA_CALL(op,message) {cudaError_t res = op;  if (res != cudaSuccess) { return RunResult(static_cast<int>(res), message);}}
#define CUBLAS_CALL(op,message) {cublasStatus_t res = op;  if (res != CUBLAS_STATUS_SUCCESS) { return RunResult(static_cast<int>(res), message);}}
