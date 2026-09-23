#pragma once
#include "runtime.h"
#include <string>

// Upload the entire continuous byte stream once; training samples on the GPU.
void load_stream(Engine *engine, const std::string &path, int64_t size);
