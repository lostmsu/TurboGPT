#pragma once
#include <cstddef>
#include <string>

extern const unsigned char qwen_revision[];
extern const size_t qwen_revision_size;
inline std::string build_revision() {
    return std::string(reinterpret_cast<const char *>(qwen_revision), qwen_revision_size);
}
