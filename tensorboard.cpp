#include "tensorboard.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
void varint(std::vector<uint8_t> &out, uint64_t value) {
    while (value >= 128) {
        out.push_back(uint8_t(value) | 128);
        value >>= 7;
    }
    out.push_back(uint8_t(value));
}

void key(std::vector<uint8_t> &out, int field, int wire) {
    varint(out, uint64_t(field << 3) | wire);
}

void bytes(std::vector<uint8_t> &out, int field, const void *data, size_t size) {
    key(out, field, 2);
    varint(out, size);
    const auto *first = static_cast<const uint8_t *>(data);
    out.insert(out.end(), first, first + size);
}

void message(std::vector<uint8_t> &out, int field, const std::vector<uint8_t> &value) {
    bytes(out, field, value.data(), value.size());
}

void string_field(std::vector<uint8_t> &out, int field, const std::string &value) {
    bytes(out, field, value.data(), value.size());
}

void integer(std::vector<uint8_t> &out, int field, uint64_t value) {
    key(out, field, 0);
    varint(out, value);
}

void floating(std::vector<uint8_t> &out, int field, float value) {
    key(out, field, 5);
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t(bits >> (8 * i)));
}

void wide(std::vector<uint8_t> &out, int field, double value) {
    key(out, field, 1);
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 8; ++i)
        out.push_back(uint8_t(bits >> (8 * i)));
}

void little32(std::vector<uint8_t> &out, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

void little64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

constexpr std::array<std::array<uint32_t, 256>, 8> crc32c_tables() {
    std::array<std::array<uint32_t, 256>, 8> tables{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit)
            crc = crc & 1 ? (crc >> 1) ^ 0x82f63b78 : crc >> 1;
        tables[0][i] = crc;
    }
    for (int k = 1; k < 8; ++k)
        for (int i = 0; i < 256; ++i)
            tables[k][i] = (tables[k - 1][i] >> 8) ^ tables[0][tables[k - 1][i] & 255];
    return tables;
}
constexpr auto Crc32cTables = crc32c_tables();

uint32_t load32(const uint8_t *data) {
    return uint32_t(data[0]) | uint32_t(data[1]) << 8 | uint32_t(data[2]) << 16 |
           uint32_t(data[3]) << 24;
}

uint32_t crc32c(const uint8_t *data, size_t size) {
    const auto &t = Crc32cTables;
    uint32_t crc = 0xffffffff;
    for (; size >= 8; data += 8, size -= 8) {
        const uint32_t low = crc ^ load32(data), high = load32(data + 4);
        crc = t[7][low & 255] ^ t[6][(low >> 8) & 255] ^ t[5][(low >> 16) & 255] ^ t[4][low >> 24] ^
              t[3][high & 255] ^ t[2][(high >> 8) & 255] ^ t[1][(high >> 16) & 255] ^ t[0][high >> 24];
    }
    for (; size; ++data, --size)
        crc = (crc >> 8) ^ t[0][(crc ^ *data) & 255];
    return crc ^ 0xffffffff;
}

uint32_t masked_crc(uint32_t crc) {
    return ((crc >> 15) | (crc << 17)) + 0xa282ead8;
}

double wall_time() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string host_name() {
    const char *windows = std::getenv("COMPUTERNAME");
    return windows ? windows : "localhost";
}

int process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

std::vector<uint8_t> scalar_tensor(float value) {
    std::vector<uint8_t> tensor, shape;
    integer(tensor, 1, 1);
    message(tensor, 2, shape);
    floating(tensor, 5, value);
    return tensor;
}

std::vector<uint8_t> text_tensor(const std::string &value) {
    std::vector<uint8_t> tensor, shape, dimension;
    integer(tensor, 1, 7);
    integer(dimension, 1, 1);
    message(shape, 2, dimension);
    message(tensor, 2, shape);
    string_field(tensor, 8, value);
    return tensor;
}

std::vector<uint8_t> metadata(const std::string &plugin, const std::string &display_name,
                              int data_class) {
    std::vector<uint8_t> plugin_data, result;
    string_field(plugin_data, 1, plugin);
    message(result, 1, plugin_data);
    string_field(result, 2, display_name);
    integer(result, 4, data_class);
    return result;
}

std::vector<uint8_t> summary_value(const std::string &tag, const std::vector<uint8_t> &tensor,
                                   const std::string &plugin, const std::string &display_name,
                                   int data_class) {
    std::vector<uint8_t> value;
    string_field(value, 1, tag);
    message(value, 8, tensor);
    message(value, 9, metadata(plugin, display_name, data_class));
    return value;
}

constexpr size_t BufferBytes = size_t(1) << 20;
} // namespace

TensorBoardLogger::TensorBoardLogger(const std::filesystem::path &directory) {
    std::filesystem::create_directories(directory);
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    const std::string name = "events.out.tfevents." + std::to_string(seconds) + "." + host_name() +
                             "." + std::to_string(process_id()) + ".0";
    file.open(directory / name, std::ios::binary | std::ios::trunc);
    if (!file)
        throw std::runtime_error("Cannot create TensorBoard event file");
    buffer.reserve(BufferBytes);
    std::vector<uint8_t> source;
    wide(event_bytes, 1, wall_time());
    string_field(event_bytes, 3, "brain.Event:2");
    string_field(source, 1, "turbogpt-native");
    message(event_bytes, 10, source);
    record();
}

TensorBoardLogger::~TensorBoardLogger() {
    try {
        write_buffer();
    } catch (...) {
        // Destructors cannot report; flush() reports write failures.
    }
}

void TensorBoardLogger::record() {
    const size_t start = buffer.size();
    little64(buffer, uint64_t(event_bytes.size()));
    little32(buffer, masked_crc(crc32c(buffer.data() + start, 8)));
    buffer.insert(buffer.end(), event_bytes.begin(), event_bytes.end());
    little32(buffer, masked_crc(crc32c(event_bytes.data(), event_bytes.size())));
    if (buffer.size() >= BufferBytes)
        write_buffer();
}

void TensorBoardLogger::write_buffer() {
    file.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(buffer.size()));
    buffer.clear();
    if (!file)
        throw std::runtime_error("Cannot write TensorBoard events");
}

void TensorBoardLogger::event(const std::vector<uint8_t> &summary, int64_t step) {
    event_bytes.clear();
    wide(event_bytes, 1, wall_time());
    integer(event_bytes, 2, uint64_t(step));
    message(event_bytes, 5, summary);
    record();
}

void TensorBoardLogger::scalar(const std::string &tag, double value, int64_t step) {
    auto found = scalars.find(tag);
    if (found == scalars.end()) {
        std::vector<uint8_t> value_bytes;
        string_field(value_bytes, 1, tag);
        // The float is the tensor's last field, so its 4 bytes end the tensor message.
        message(value_bytes, 8, scalar_tensor(0.f));
        const size_t value_end = value_bytes.size();
        message(value_bytes, 9, metadata("scalars", tag, 1));
        ScalarSummary summary;
        message(summary.bytes, 1, value_bytes);
        summary.value_offset = summary.bytes.size() - value_bytes.size() + value_end - 4;
        found = scalars.emplace(tag, std::move(summary)).first;
    }
    const float value32 = float(value);
    std::memcpy(found->second.bytes.data() + found->second.value_offset, &value32, 4);
    event(found->second.bytes, step);
}

void TensorBoardLogger::text(const std::string &tag, const std::string &value, int64_t step) {
    const std::string name = tag + "/text_summary";
    std::vector<uint8_t> summary;
    message(summary, 1, summary_value(name, text_tensor(value), "text", name, 2));
    event(summary, step);
}

void TensorBoardLogger::flush() {
    write_buffer();
    file.flush();
    if (!file)
        throw std::runtime_error("Cannot flush TensorBoard event file");
}
