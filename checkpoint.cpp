#include "checkpoint.h"
#include "model.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace {
constexpr uint16_t ZipDataDescriptorFlag = 0x0808;
constexpr const char *SerializationId = "3407000000000000000000000000000000000000";
constexpr uint32_t ZipLocalSignature = 0x04034b50, ZipDataDescriptorSignature = 0x08074b50,
                   ZipCentralSignature = 0x02014b50, ZipEndSignature = 0x06054b50,
                   Zip64EndSignature = 0x06064b50, Zip64LocatorSignature = 0x07064b50;

void atomic_replace(const std::string &path,
                    const std::function<void(const std::string &)> &write) {
    const std::string temporary = path + ".tmp";
    const std::string backup = path + ".bak";
    write(temporary);
    std::error_code ignored;
    std::filesystem::remove(backup, ignored);
    if (std::filesystem::exists(path))
        std::filesystem::rename(path, backup);
    std::filesystem::rename(temporary, path);
    std::filesystem::remove(backup, ignored);
}

void append16(std::vector<uint8_t> &out, uint16_t value) {
    for (int i = 0; i < 2; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

void append32(std::vector<uint8_t> &out, uint32_t value) {
    for (int i = 0; i < 4; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

void append64(std::vector<uint8_t> &out, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        out.push_back(uint8_t(value >> (8 * i)));
}

void append_bytes(std::vector<uint8_t> &out, const void *data, size_t size) {
    const auto *first = static_cast<const uint8_t *>(data);
    out.insert(out.end(), first, first + size);
}

void append_text(std::vector<uint8_t> &out, const std::string &value) {
    append_bytes(out, value.data(), value.size());
}

void append_int64(std::vector<uint8_t> &out, int64_t value) {
    append64(out, uint64_t(value));
}

void append_float64(std::vector<uint8_t> &out, double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    append64(out, bits);
}

void append_float32s(std::vector<uint8_t> &out, const std::vector<float> &values) {
    for (float value : values) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        append32(out, bits);
    }
}

void append_float64s(std::vector<uint8_t> &out, const std::vector<double> &values) {
    for (double value : values) {
        uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        append64(out, bits);
    }
}

uint32_t zip_crc32(const uint8_t *data, size_t size) {
    uint32_t crc = 0xffffffff;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320 & (0 - uint32_t(crc & 1)));
    }
    return crc ^ 0xffffffff;
}

uint16_t read16(const std::vector<uint8_t> &data, size_t offset) {
    return uint16_t(data[offset]) | (uint16_t(data[offset + 1]) << 8);
}

uint32_t read32(const std::vector<uint8_t> &data, size_t offset) {
    return uint32_t(data[offset]) | (uint32_t(data[offset + 1]) << 8) |
           (uint32_t(data[offset + 2]) << 16) | (uint32_t(data[offset + 3]) << 24);
}

uint64_t read64(const std::vector<uint8_t> &data, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= uint64_t(data[offset + i]) << (8 * i);
    return value;
}

void pickle_text(std::vector<uint8_t> &out, const std::string &value) {
    out.push_back('X');
    append32(out, uint32_t(value.size()));
    append_text(out, value);
}

void pickle_global(std::vector<uint8_t> &out, const std::string &module, const std::string &name) {
    out.push_back('c');
    append_text(out, module);
    out.push_back('\n');
    append_text(out, name);
    out.push_back('\n');
}

void pickle_integer(std::vector<uint8_t> &out, uint32_t value) {
    out.push_back('J');
    append32(out, value);
}

void pickle_tensor(std::vector<uint8_t> &out, const std::string &key, const char *storage_type,
                   const std::string &storage_key, uint32_t size) {
    pickle_text(out, key);
    pickle_global(out, "torch._utils", "_rebuild_tensor_v2");
    out.push_back('(');
    out.push_back('(');
    pickle_text(out, "storage");
    pickle_global(out, "torch", storage_type);
    pickle_text(out, storage_key);
    pickle_text(out, "cpu");
    pickle_integer(out, size);
    out.push_back('t');
    out.push_back('Q');
    out.push_back('K');
    out.push_back(0);
    pickle_integer(out, size);
    out.push_back(0x85);
    pickle_integer(out, 1);
    out.push_back(0x85);
    out.push_back(0x89);
    pickle_global(out, "collections", "OrderedDict");
    out.push_back(')');
    out.push_back('R');
    out.push_back('t');
    out.push_back('R');
}

std::vector<uint8_t> checkpoint_pickle(uint32_t parameter_count, uint32_t final_loss_positions) {
    std::vector<uint8_t> result;
    result.push_back(0x80);
    result.push_back(2);
    result.push_back('}');
    result.push_back('(');
    pickle_text(result, "model");
    result.push_back('}');
    result.push_back('(');
    pickle_tensor(result, "depth", "LongStorage", "0", 1);
    pickle_tensor(result, "context", "LongStorage", "1", 1);
    pickle_tensor(result, "weights", "FloatStorage", "2", parameter_count);
    result.push_back('u');
    pickle_text(result, "opt");
    result.push_back('}');
    result.push_back('(');
    pickle_tensor(result, "momentum", "FloatStorage", "3", parameter_count);
    pickle_tensor(result, "variance", "FloatStorage", "4", parameter_count);
    pickle_tensor(result, "step", "LongStorage", "5", 1);
    result.push_back('u');
    pickle_text(result, "lr_scheduler");
    result.push_back('}');
    result.push_back('(');
    pickle_tensor(result, "total_steps", "LongStorage", "6", 1);
    pickle_tensor(result, "step", "LongStorage", "5", 1);
    pickle_tensor(result, "warmup_fraction", "DoubleStorage", "7", 1);
    pickle_tensor(result, "initial_lr_fraction", "DoubleStorage", "8", 1);
    pickle_tensor(result, "final_lr_fraction", "DoubleStorage", "9", 1);
    pickle_tensor(result, "low_momentum", "DoubleStorage", "10", 1);
    pickle_tensor(result, "high_momentum", "DoubleStorage", "11", 1);
    result.push_back('u');
    pickle_text(result, "trainer");
    result.push_back('}');
    result.push_back('(');
    pickle_tensor(result, "train_seconds", "DoubleStorage", "12", 1);
    pickle_tensor(result, "final_loss_sums", "DoubleStorage", "13", final_loss_positions);
    pickle_tensor(result, "final_loss_batches", "LongStorage", "14", 1);
    result.push_back('u');
    result.push_back('u');
    result.push_back('.');
    return result;
}

struct ArchiveEntry {
    std::string name;
    std::vector<uint8_t> data;
    uint32_t crc = 0;
    uint64_t local_offset = 0;
};

ArchiveEntry archive_entry(const std::string &name, const std::vector<uint8_t> &data) {
    ArchiveEntry entry;
    entry.name = name;
    entry.data = data;
    entry.crc = zip_crc32(entry.data.data(), entry.data.size());
    return entry;
}

ArchiveEntry text_entry(const std::string &name, const std::string &text) {
    std::vector<uint8_t> data(text.begin(), text.end());
    return archive_entry(name, data);
}

void append_local_entry(std::vector<uint8_t> &archive, ArchiveEntry &entry) {
    entry.local_offset = archive.size();
    uint16_t extra_size = uint16_t(64 - ((entry.local_offset + 30 + entry.name.size()) % 64));
    if (extra_size < 4)
        extra_size = uint16_t(extra_size + 64);
    append32(archive, ZipLocalSignature);
    append16(archive, 20);
    append16(archive, ZipDataDescriptorFlag);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, 0);
    append32(archive, 0);
    append32(archive, 0);
    append16(archive, uint16_t(entry.name.size()));
    append16(archive, uint16_t(extra_size));
    append_text(archive, entry.name);
    append16(archive, 0x4246);
    append16(archive, uint16_t(extra_size - 4));
    archive.insert(archive.end(), size_t(extra_size - 4), 'Z');
    append_bytes(archive, entry.data.data(), entry.data.size());
    append32(archive, ZipDataDescriptorSignature);
    append32(archive, entry.crc);
    append32(archive, uint32_t(entry.data.size()));
    append32(archive, uint32_t(entry.data.size()));
}

void append_central_entry(std::vector<uint8_t> &archive, const ArchiveEntry &entry) {
    append32(archive, ZipCentralSignature);
    append16(archive, 20);
    append16(archive, 20);
    append16(archive, ZipDataDescriptorFlag);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, entry.crc);
    append32(archive, uint32_t(entry.data.size()));
    append32(archive, uint32_t(entry.data.size()));
    append16(archive, uint16_t(entry.name.size()));
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, 0);
    append32(archive, uint32_t(entry.local_offset));
    append_text(archive, entry.name);
}

std::vector<uint8_t> make_checkpoint(const Config &config, const TrainingState &state,
                                     uint32_t parameter_count) {
    std::vector<ArchiveEntry> entries;
    entries.push_back(
        archive_entry("archive/data.pkl",
                      checkpoint_pickle(parameter_count, uint32_t(state.final_loss_sums.size()))));
    entries.push_back(text_entry("archive/.format_version", "2"));
    entries.push_back(text_entry("archive/.storage_alignment", "64"));
    entries.push_back(text_entry("archive/byteorder", "little"));

    std::vector<uint8_t> scalar;
    append_int64(scalar, config.depth);
    entries.push_back(archive_entry("archive/data/0", scalar));
    scalar.clear();
    append_int64(scalar, config.context);
    entries.push_back(archive_entry("archive/data/1", scalar));
    scalar.clear();
    append_float32s(scalar, state.weights);
    entries.push_back(archive_entry("archive/data/2", scalar));
    scalar.clear();
    append_float32s(scalar, state.momentum);
    entries.push_back(archive_entry("archive/data/3", scalar));
    scalar.clear();
    append_float32s(scalar, state.variance);
    entries.push_back(archive_entry("archive/data/4", scalar));
    scalar.clear();
    append_int64(scalar, int64_t(state.step));
    entries.push_back(archive_entry("archive/data/5", scalar));
    scalar.clear();
    append_int64(scalar, state.scheduler.total_steps);
    entries.push_back(archive_entry("archive/data/6", scalar));
    scalar.clear();
    append_float64(scalar, state.scheduler.warmup_fraction);
    entries.push_back(archive_entry("archive/data/7", scalar));
    scalar.clear();
    append_float64(scalar, state.scheduler.initial_lr_fraction);
    entries.push_back(archive_entry("archive/data/8", scalar));
    scalar.clear();
    append_float64(scalar, state.scheduler.final_lr_fraction);
    entries.push_back(archive_entry("archive/data/9", scalar));
    scalar.clear();
    append_float64(scalar, state.scheduler.low_momentum);
    entries.push_back(archive_entry("archive/data/10", scalar));
    scalar.clear();
    append_float64(scalar, state.scheduler.high_momentum);
    entries.push_back(archive_entry("archive/data/11", scalar));
    scalar.clear();
    append_float64(scalar, state.train_seconds);
    entries.push_back(archive_entry("archive/data/12", scalar));
    scalar.clear();
    append_float64s(scalar, state.final_loss_sums);
    entries.push_back(archive_entry("archive/data/13", scalar));
    scalar.clear();
    append_int64(scalar, int64_t(state.final_loss_batches));
    entries.push_back(archive_entry("archive/data/14", scalar));

    entries.push_back(text_entry("archive/version", "3\n"));
    entries.push_back(text_entry("archive/.data/serialization_id", SerializationId));
    std::vector<uint8_t> archive;
    for (ArchiveEntry &entry : entries) {
        append_local_entry(archive, entry);
    }
    const uint64_t central_offset = archive.size();
    for (const ArchiveEntry &entry : entries)
        append_central_entry(archive, entry);
    const uint64_t central_size = archive.size() - central_offset;

    const uint64_t zip64_end_offset = archive.size();
    append32(archive, Zip64EndSignature);
    append64(archive, 44);
    append16(archive, 20);
    append16(archive, 20);
    append32(archive, 0);
    append32(archive, 0);
    append64(archive, entries.size());
    append64(archive, entries.size());
    append64(archive, central_size);
    append64(archive, central_offset);
    append32(archive, Zip64LocatorSignature);
    append32(archive, 0);
    append64(archive, zip64_end_offset);
    append32(archive, 1);
    append32(archive, ZipEndSignature);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, uint16_t(entries.size()));
    append16(archive, uint16_t(entries.size()));
    append32(archive, uint32_t(central_size));
    append32(archive, uint32_t(central_offset));
    append16(archive, 0);
    return archive;
}

struct ZipRecord {
    std::string name;
    uint64_t data_offset = 0;
    uint32_t size = 0;
    uint32_t crc = 0;
};

std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open PyTorch checkpoint");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
}

std::unordered_map<std::string, ZipRecord> read_zip_records(const std::vector<uint8_t> &archive) {
    if (archive.size() < 22)
        throw std::runtime_error("Truncated PyTorch checkpoint");
    size_t end_offset = archive.size() - 22;
    bool found = false;
    for (size_t offset = end_offset; offset != std::string::npos; --offset) {
        if (read32(archive, offset) == ZipEndSignature) {
            end_offset = offset;
            found = true;
            break;
        }
        if (offset == 0)
            break;
    }
    if (!found)
        throw std::runtime_error("Invalid PyTorch checkpoint ZIP directory");
    const uint16_t entries = read16(archive, end_offset + 10);
    const uint64_t central_offset = read32(archive, end_offset + 16);
    if (central_offset + uint64_t(entries) * 46 > archive.size())
        throw std::runtime_error("Invalid PyTorch checkpoint central directory");

    std::unordered_map<std::string, ZipRecord> records;
    size_t offset = size_t(central_offset);
    for (uint16_t index = 0; index < entries; ++index) {
        if (offset + 46 > archive.size() || read32(archive, offset) != ZipCentralSignature)
            throw std::runtime_error("Invalid PyTorch checkpoint central entry");
        const uint16_t method = read16(archive, offset + 10);
        const uint32_t compressed_size = read32(archive, offset + 20);
        const uint32_t size = read32(archive, offset + 24);
        const uint16_t name_size = read16(archive, offset + 28);
        const uint16_t extra_size = read16(archive, offset + 30);
        const uint16_t comment_size = read16(archive, offset + 32);
        if (method != 0 || compressed_size != size ||
            offset + 46 + uint64_t(name_size) + extra_size + comment_size > archive.size())
            throw std::runtime_error("Unsupported PyTorch checkpoint compression");
        const std::string name(reinterpret_cast<const char *>(archive.data() + offset + 46),
                               name_size);
        const uint64_t local_offset = read32(archive, offset + 42);
        if (local_offset + 30 + name_size > archive.size() ||
            read32(archive, size_t(local_offset)) != ZipLocalSignature)
            throw std::runtime_error("Invalid PyTorch checkpoint local entry");
        const uint16_t local_name_size = read16(archive, size_t(local_offset) + 26);
        const uint16_t local_extra_size = read16(archive, size_t(local_offset) + 28);
        ZipRecord record;
        record.name = name;
        record.size = size;
        record.crc = read32(archive, offset + 16);
        record.data_offset = local_offset + 30 + local_name_size + local_extra_size;
        if (record.data_offset + size > archive.size())
            throw std::runtime_error("Truncated PyTorch checkpoint record");
        if (zip_crc32(archive.data() + record.data_offset, size) != record.crc)
            throw std::runtime_error("PyTorch checkpoint record CRC mismatch");
        if (!records.emplace(name, record).second)
            throw std::runtime_error("Duplicate PyTorch checkpoint record");
        offset += 46 + size_t(name_size) + extra_size + comment_size;
    }
    return records;
}

ZipRecord require_record(const std::unordered_map<std::string, ZipRecord> &records,
                         const std::string &name) {
    const auto found = records.find(name);
    if (found == records.end())
        throw std::runtime_error("Missing PyTorch checkpoint record " + name);
    return found->second;
}

std::string archive_prefix(const std::unordered_map<std::string, ZipRecord> &records) {
    for (const auto &record : records) {
        const std::string suffix = "/data.pkl";
        if (record.first.size() > suffix.size() &&
            record.first.compare(record.first.size() - suffix.size(), suffix.size(), suffix) == 0)
            return record.first.substr(0, record.first.size() - suffix.size());
    }
    throw std::runtime_error("Missing PyTorch checkpoint data.pkl");
}

uint64_t final_loss_window_batches(uint64_t completed_steps, int64_t total_steps) {
    const int64_t first_step =
        std::max(int64_t(0), total_steps - std::max(int64_t(1), (total_steps + 99) / 100));
    return uint64_t(std::max(int64_t(0), int64_t(completed_steps) - first_step));
}

void require_text_record(const std::unordered_map<std::string, ZipRecord> &records,
                         const std::vector<uint8_t> &archive, const std::string &name,
                         const std::string &expected) {
    const ZipRecord record = require_record(records, name);
    if (record.size != expected.size() ||
        !std::equal(expected.begin(), expected.end(), archive.data() + record.data_offset)) {
        throw std::runtime_error("Unsupported PyTorch checkpoint metadata");
    }
}

int64_t read_scalar(const std::vector<uint8_t> &archive, const ZipRecord &record) {
    if (record.size != 8)
        throw std::runtime_error("Invalid PyTorch checkpoint scalar size");
    return int64_t(read64(archive, record.data_offset));
}

double read_double(const std::vector<uint8_t> &archive, const ZipRecord &record) {
    if (record.size != 8)
        throw std::runtime_error("Invalid PyTorch checkpoint scalar size");
    const uint64_t bits = read64(archive, record.data_offset);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> read_float32s(const std::vector<uint8_t> &archive, const ZipRecord &record,
                                 uint32_t count) {
    if (record.size != uint64_t(count) * 4)
        throw std::runtime_error("Invalid PyTorch checkpoint tensor size");
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t bits = read32(archive, record.data_offset + uint64_t(i) * 4);
        std::memcpy(&values[i], &bits, sizeof(values[i]));
    }
    return values;
}

std::vector<double> read_float64s(const std::vector<uint8_t> &archive, const ZipRecord &record,
                                  uint32_t count) {
    if (record.size != uint64_t(count) * 8)
        throw std::runtime_error("Invalid PyTorch checkpoint tensor size");
    std::vector<double> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t bits = read64(archive, record.data_offset + uint64_t(i) * 8);
        std::memcpy(&values[i], &bits, sizeof(values[i]));
    }
    return values;
}
} // namespace

TrainingState load_training_state(const std::string &path, const Config &config) {
    TrainingState state;

    const std::vector<uint8_t> archive = read_file(path);
    const auto records = read_zip_records(archive);
    const std::string prefix = archive_prefix(records);
    const auto record = [&](const std::string &name) {
        return require_record(records, prefix + "/" + name);
    };
    require_text_record(records, archive, prefix + "/.format_version", "2");
    require_text_record(records, archive, prefix + "/.storage_alignment", "64");
    require_text_record(records, archive, prefix + "/byteorder", "little");
    require_text_record(records, archive, prefix + "/version", "3\n");
    const ZipRecord serialization_id = record(".data/serialization_id");
    if (serialization_id.size != std::strlen(SerializationId) ||
        !std::equal(archive.data() + serialization_id.data_offset,
                    archive.data() + serialization_id.data_offset + serialization_id.size,
                    SerializationId))
        throw std::runtime_error("Invalid PyTorch checkpoint serialization id");

    const int64_t depth = read_scalar(archive, record("data/0"));
    const int64_t context = read_scalar(archive, record("data/1"));
    const uint32_t count = uint32_t(parameter_layout(config.depth, config.context).count);
    if (depth != config.depth || context != config.context)
        throw std::runtime_error("Checkpoint architecture mismatch");
    state.weights = read_float32s(archive, record("data/2"), count);
    state.momentum = read_float32s(archive, record("data/3"), count);
    state.variance = read_float32s(archive, record("data/4"), count);
    state.step = uint64_t(read_scalar(archive, record("data/5")));
    const int64_t total_steps = read_scalar(archive, record("data/6"));
    if (total_steps < 1 || total_steps > INT_MAX || state.step > uint64_t(total_steps))
        throw std::runtime_error("Invalid checkpoint scheduler state");
    state.scheduler.total_steps = int(total_steps);
    state.scheduler.warmup_fraction = read_double(archive, record("data/7"));
    state.scheduler.initial_lr_fraction = read_double(archive, record("data/8"));
    state.scheduler.final_lr_fraction = read_double(archive, record("data/9"));
    state.scheduler.low_momentum = read_double(archive, record("data/10"));
    state.scheduler.high_momentum = read_double(archive, record("data/11"));
    state.train_seconds = read_double(archive, record("data/12"));
    state.final_loss_sums = read_float64s(archive, record("data/13"), uint32_t(config.context));
    state.final_loss_batches = uint64_t(read_scalar(archive, record("data/14")));
    if (state.final_loss_batches >
        final_loss_window_batches(state.step, state.scheduler.total_steps))
        throw std::runtime_error("Invalid checkpoint final loss state");
    onecycle_point(state.scheduler, 0);
    return state;
}

void save_training_state(const std::string &path, const Config &config,
                         const TrainingState &state) {
    const int count = parameter_layout(config.depth, config.context).count;
    if (int(state.weights.size()) != count || int(state.momentum.size()) != count ||
        int(state.variance.size()) != count)
        throw std::runtime_error("Checkpoint state does not match the model");
    if (state.final_loss_sums.size() != size_t(config.context))
        throw std::runtime_error("Checkpoint final loss state does not match the model");
    if (state.final_loss_batches >
        final_loss_window_batches(state.step, state.scheduler.total_steps))
        throw std::runtime_error("Invalid checkpoint final loss state");
    if (state.scheduler.total_steps < 1 || state.step > uint64_t(state.scheduler.total_steps))
        throw std::runtime_error("Invalid checkpoint scheduler state");
    onecycle_point(state.scheduler, 0);
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    atomic_replace(path, [&](const std::string &destination) {
        const std::vector<uint8_t> archive = make_checkpoint(config, state, uint32_t(count));
        std::ofstream file(destination, std::ios::binary);
        file.write(reinterpret_cast<const char *>(archive.data()), std::streamsize(archive.size()));
        file.close();
        if (!file)
            throw std::runtime_error("Cannot write PyTorch checkpoint");
    });
}
