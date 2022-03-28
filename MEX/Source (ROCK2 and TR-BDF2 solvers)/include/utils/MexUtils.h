#pragma once

#include "mex.h"
#include <sstream>
#include <string>
#include <vector>

void checkmxScalar(const mxArray* val, double& result, const std::string& message)
{
	if (!(mxIsScalar(val)) || (mxGetNumberOfDimensions(val) != 2) || (mxGetNumberOfElements(val) != 1))
	{
		mexErrMsgTxt(&message[0]);
		return;
	}
	result = mxGetScalar(val);
}

void checkmxStruct(const mxArray* val, const std::vector<std::string>& fields, std::vector<double>& values, const std::string& message)
{
	if (!mxIsStruct(val))
	{
		mexErrMsgTxt(&message[0]);
		return;
	}
	for (unsigned int k = 0; k < fields.size(); ++k)
	{
		mxArray *value = mxGetField(val, 0, &fields[k][0]);
		if (value == NULL)
		{
			std::ostringstream os;
			os << message << " Field " << fields[k] << "was not found!" << std::endl;
			mexErrMsgTxt(os.str().data());
			return;
		}
		values[k] = mxGetScalar(value);
	}
}

template<typename T> void checkmxArray(const mxArray* val, T*& data, unsigned int Nel, const std::string& message)
{
	if (typeid(T) == typeid(float) && mxIsSingle(val) && mxGetNumberOfElements(val) == Nel)
	{
		data = (T*)mxGetSingles(val);
	}
	else if (typeid(T) == typeid(double) && mxIsDouble(val) && mxGetNumberOfElements(val) == Nel)
	{
		data = (T*)mxGetDoubles(val);
	}
	else
	{
		mexErrMsgTxt(&message[0]);
		return;
	}
}

template<typename T> void checkmxArray(const mxArray* val, T*& data, const std::string& message)
{
	if (typeid(T) == typeid(float) && mxIsSingle(val))
	{
		data = (T*)mxGetSingles(val);
	}
	else if (typeid(T) == typeid(double) && mxIsDouble(val))
	{
		data = (T*)mxGetDoubles(val);
	}
	else if (typeid(T) == typeid(unsigned int) && mxIsUint32(val))
	{
		data = (T*)mxGetUint32s(val);
	}
	else
	{
		mexErrMsgTxt(&message[0]);
		return;
	}
}
