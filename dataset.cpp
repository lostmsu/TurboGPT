#include "dataset.h"
#include <filesystem>
#include <fstream>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

void load_stream(Engine *engine, const std::string &path, int64_t size) {
#ifdef _WIN32
    // Map the existing file instead of zero-filling and then copying a second
    // 1GB CPU buffer. CUDA still receives an owned, fully resident byte array.
    HANDLE file = CreateFileW(std::filesystem::path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                              nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Cannot open dataset " + path);
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    const void *view = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
    int status = -1;
    if (view)
        status = tg_set_data(engine, static_cast<const uint8_t *>(view), size);
    if (view)
        UnmapViewOfFile(view);
    if (mapping)
        CloseHandle(mapping);
    CloseHandle(file);
    if (!view)
        throw std::runtime_error("Cannot map dataset " + path);
    check_status(status);
#else
    std::vector<uint8_t> bytes(size);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!file)
        throw std::runtime_error("Cannot read dataset " + path);
    check_status(tg_set_data(engine, bytes.data(), size));
#endif
}
