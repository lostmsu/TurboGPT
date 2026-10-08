// Resumable PyTorch checkpoints: an uncompressed torch.save ZIP with a protocol-2 pickle.
use crate::config::Config;
use crate::error::{Error, Result, fail};
use crate::model::parameter_layout;
use crate::optim::{OneCycle, onecycle_point};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, Default)]
pub struct TrainingState {
    pub weights: Vec<f32>,
    pub momentum: Vec<f32>,
    pub variance: Vec<f32>,
    pub scheduler: OneCycle,
    pub step: u64,
    pub train_seconds: f64,
    pub final_loss_sums: Vec<f64>,
    pub final_loss_batches: u64,
}

const ZIP_DATA_DESCRIPTOR_FLAG: u16 = 0x0808;
const SERIALIZATION_ID: &str = "3407000000000000000000000000000000000000";
const ZIP_LOCAL_SIGNATURE: u32 = 0x04034b50;
const ZIP_DATA_DESCRIPTOR_SIGNATURE: u32 = 0x08074b50;
const ZIP_CENTRAL_SIGNATURE: u32 = 0x02014b50;
const ZIP_END_SIGNATURE: u32 = 0x06054b50;
const ZIP64_END_SIGNATURE: u32 = 0x06064b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x07064b50;

fn atomic_replace(path: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let temporary =
        path.with_file_name(format!("{}.tmp", path.file_name().unwrap().to_string_lossy()));
    let backup =
        path.with_file_name(format!("{}.bak", path.file_name().unwrap().to_string_lossy()));
    write(&temporary)?;
    let _ = std::fs::remove_file(&backup);
    if path.exists() {
        std::fs::rename(path, &backup)?;
    }
    std::fs::rename(&temporary, path)?;
    let _ = std::fs::remove_file(&backup);
    Ok(())
}

fn append16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn append32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn append64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn zip_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & 0u32.wrapping_sub(crc & 1));
        }
    }
    crc ^ 0xffff_ffff
}

fn read16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}
fn read32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}
fn read64(data: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
}

fn pickle_text(out: &mut Vec<u8>, value: &str) {
    out.push(b'X');
    append32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

fn pickle_global(out: &mut Vec<u8>, module: &str, name: &str) {
    out.push(b'c');
    out.extend_from_slice(module.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(name.as_bytes());
    out.push(b'\n');
}

fn pickle_integer(out: &mut Vec<u8>, value: u32) {
    out.push(b'J');
    append32(out, value);
}

fn pickle_tensor(out: &mut Vec<u8>, key: &str, storage_type: &str, storage_key: &str, size: u32) {
    pickle_text(out, key);
    pickle_global(out, "torch._utils", "_rebuild_tensor_v2");
    out.extend_from_slice(b"((");
    pickle_text(out, "storage");
    pickle_global(out, "torch", storage_type);
    pickle_text(out, storage_key);
    pickle_text(out, "cpu");
    pickle_integer(out, size);
    out.extend_from_slice(b"tQK\0");
    pickle_integer(out, size);
    out.push(0x85);
    pickle_integer(out, 1);
    out.push(0x85);
    out.push(0x89);
    pickle_global(out, "collections", "OrderedDict");
    out.extend_from_slice(b")RtR");
}

fn checkpoint_pickle(parameter_count: u32, final_loss_positions: u32) -> Vec<u8> {
    let mut r = vec![0x80, 2, b'}', b'('];
    pickle_text(&mut r, "model");
    r.extend_from_slice(b"}(");
    pickle_tensor(&mut r, "depth", "LongStorage", "0", 1);
    pickle_tensor(&mut r, "context", "LongStorage", "1", 1);
    pickle_tensor(&mut r, "weights", "FloatStorage", "2", parameter_count);
    r.push(b'u');
    pickle_text(&mut r, "opt");
    r.extend_from_slice(b"}(");
    pickle_tensor(&mut r, "momentum", "FloatStorage", "3", parameter_count);
    pickle_tensor(&mut r, "variance", "FloatStorage", "4", parameter_count);
    pickle_tensor(&mut r, "step", "LongStorage", "5", 1);
    r.push(b'u');
    pickle_text(&mut r, "lr_scheduler");
    r.extend_from_slice(b"}(");
    pickle_tensor(&mut r, "total_steps", "LongStorage", "6", 1);
    pickle_tensor(&mut r, "step", "LongStorage", "5", 1);
    pickle_tensor(&mut r, "warmup_fraction", "DoubleStorage", "7", 1);
    pickle_tensor(&mut r, "initial_lr_fraction", "DoubleStorage", "8", 1);
    pickle_tensor(&mut r, "final_lr_fraction", "DoubleStorage", "9", 1);
    pickle_tensor(&mut r, "low_momentum", "DoubleStorage", "10", 1);
    pickle_tensor(&mut r, "high_momentum", "DoubleStorage", "11", 1);
    r.push(b'u');
    pickle_text(&mut r, "trainer");
    r.extend_from_slice(b"}(");
    pickle_tensor(&mut r, "train_seconds", "DoubleStorage", "12", 1);
    pickle_tensor(&mut r, "final_loss_sums", "DoubleStorage", "13", final_loss_positions);
    pickle_tensor(&mut r, "final_loss_batches", "LongStorage", "14", 1);
    r.extend_from_slice(b"uu.");
    r
}

struct ArchiveEntry {
    name: String,
    data: Vec<u8>,
    crc: u32,
    local_offset: u64,
}

fn archive_entry(name: &str, data: Vec<u8>) -> ArchiveEntry {
    let crc = zip_crc32(&data);
    ArchiveEntry { name: name.to_string(), data, crc, local_offset: 0 }
}

fn append_local_entry(archive: &mut Vec<u8>, entry: &mut ArchiveEntry) {
    entry.local_offset = archive.len() as u64;
    let mut extra_size = (64 - ((entry.local_offset as usize + 30 + entry.name.len()) % 64)) as u16;
    if extra_size < 4 {
        extra_size += 64;
    }
    append32(archive, ZIP_LOCAL_SIGNATURE);
    append16(archive, 20);
    append16(archive, ZIP_DATA_DESCRIPTOR_FLAG);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, 0);
    append32(archive, 0);
    append32(archive, 0);
    append16(archive, entry.name.len() as u16);
    append16(archive, extra_size);
    archive.extend_from_slice(entry.name.as_bytes());
    append16(archive, 0x4246);
    append16(archive, extra_size - 4);
    archive.resize(archive.len() + extra_size as usize - 4, b'Z');
    archive.extend_from_slice(&entry.data);
    append32(archive, ZIP_DATA_DESCRIPTOR_SIGNATURE);
    append32(archive, entry.crc);
    append32(archive, entry.data.len() as u32);
    append32(archive, entry.data.len() as u32);
}

fn append_central_entry(archive: &mut Vec<u8>, entry: &ArchiveEntry) {
    append32(archive, ZIP_CENTRAL_SIGNATURE);
    append16(archive, 20);
    append16(archive, 20);
    append16(archive, ZIP_DATA_DESCRIPTOR_FLAG);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, entry.crc);
    append32(archive, entry.data.len() as u32);
    append32(archive, entry.data.len() as u32);
    append16(archive, entry.name.len() as u16);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append16(archive, 0);
    append32(archive, 0);
    append32(archive, entry.local_offset as u32);
    archive.extend_from_slice(entry.name.as_bytes());
}

fn make_checkpoint(config: &Config, state: &TrainingState, parameter_count: u32) -> Vec<u8> {
    let i64s = |value: i64| value.to_le_bytes().to_vec();
    let f64s = |values: &[f64]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let f32s = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let s = &state.scheduler;
    let mut entries = vec![
        archive_entry(
            "archive/data.pkl",
            checkpoint_pickle(parameter_count, state.final_loss_sums.len() as u32),
        ),
        archive_entry("archive/.format_version", b"2".to_vec()),
        archive_entry("archive/.storage_alignment", b"64".to_vec()),
        archive_entry("archive/byteorder", b"little".to_vec()),
        archive_entry("archive/data/0", i64s(config.depth as i64)),
        archive_entry("archive/data/1", i64s(config.context as i64)),
        archive_entry("archive/data/2", f32s(&state.weights)),
        archive_entry("archive/data/3", f32s(&state.momentum)),
        archive_entry("archive/data/4", f32s(&state.variance)),
        archive_entry("archive/data/5", i64s(state.step as i64)),
        archive_entry("archive/data/6", i64s(s.total_steps as i64)),
        archive_entry("archive/data/7", f64s(&[s.warmup_fraction])),
        archive_entry("archive/data/8", f64s(&[s.initial_lr_fraction])),
        archive_entry("archive/data/9", f64s(&[s.final_lr_fraction])),
        archive_entry("archive/data/10", f64s(&[s.low_momentum])),
        archive_entry("archive/data/11", f64s(&[s.high_momentum])),
        archive_entry("archive/data/12", f64s(&[state.train_seconds])),
        archive_entry("archive/data/13", f64s(&state.final_loss_sums)),
        archive_entry("archive/data/14", i64s(state.final_loss_batches as i64)),
        archive_entry("archive/version", b"3\n".to_vec()),
        archive_entry("archive/.data/serialization_id", SERIALIZATION_ID.as_bytes().to_vec()),
    ];
    let mut archive = Vec::new();
    for entry in &mut entries {
        append_local_entry(&mut archive, entry);
    }
    let central_offset = archive.len() as u64;
    for entry in &entries {
        append_central_entry(&mut archive, entry);
    }
    let central_size = archive.len() as u64 - central_offset;
    let zip64_end_offset = archive.len() as u64;
    append32(&mut archive, ZIP64_END_SIGNATURE);
    append64(&mut archive, 44);
    append16(&mut archive, 20);
    append16(&mut archive, 20);
    append32(&mut archive, 0);
    append32(&mut archive, 0);
    append64(&mut archive, entries.len() as u64);
    append64(&mut archive, entries.len() as u64);
    append64(&mut archive, central_size);
    append64(&mut archive, central_offset);
    append32(&mut archive, ZIP64_LOCATOR_SIGNATURE);
    append32(&mut archive, 0);
    append64(&mut archive, zip64_end_offset);
    append32(&mut archive, 1);
    append32(&mut archive, ZIP_END_SIGNATURE);
    append16(&mut archive, 0);
    append16(&mut archive, 0);
    append16(&mut archive, entries.len() as u16);
    append16(&mut archive, entries.len() as u16);
    append32(&mut archive, central_size as u32);
    append32(&mut archive, central_offset as u32);
    append16(&mut archive, 0);
    archive
}

#[derive(Clone, Copy)]
struct ZipRecord {
    data_offset: usize,
    size: usize,
}

fn read_zip_records(archive: &[u8]) -> Result<HashMap<String, ZipRecord>> {
    if archive.len() < 22 {
        return fail("Truncated PyTorch checkpoint");
    }
    let end_offset = (0..=archive.len() - 22)
        .rev()
        .find(|&offset| read32(archive, offset) == ZIP_END_SIGNATURE)
        .ok_or_else(|| Error::new("Invalid PyTorch checkpoint ZIP directory"))?;
    let entries = read16(archive, end_offset + 10) as usize;
    let central_offset = read32(archive, end_offset + 16) as usize;
    if central_offset + entries * 46 > archive.len() {
        return fail("Invalid PyTorch checkpoint central directory");
    }
    let mut records = HashMap::new();
    let mut offset = central_offset;
    for _ in 0..entries {
        if offset + 46 > archive.len() || read32(archive, offset) != ZIP_CENTRAL_SIGNATURE {
            return fail("Invalid PyTorch checkpoint central entry");
        }
        let method = read16(archive, offset + 10);
        let compressed_size = read32(archive, offset + 20) as usize;
        let size = read32(archive, offset + 24) as usize;
        let name_size = read16(archive, offset + 28) as usize;
        let extra_size = read16(archive, offset + 30) as usize;
        let comment_size = read16(archive, offset + 32) as usize;
        if method != 0
            || compressed_size != size
            || offset + 46 + name_size + extra_size + comment_size > archive.len()
        {
            return fail("Unsupported PyTorch checkpoint compression");
        }
        let name =
            String::from_utf8_lossy(&archive[offset + 46..offset + 46 + name_size]).into_owned();
        let local_offset = read32(archive, offset + 42) as usize;
        if local_offset + 30 + name_size > archive.len()
            || read32(archive, local_offset) != ZIP_LOCAL_SIGNATURE
        {
            return fail("Invalid PyTorch checkpoint local entry");
        }
        let local_name_size = read16(archive, local_offset + 26) as usize;
        let local_extra_size = read16(archive, local_offset + 28) as usize;
        let crc = read32(archive, offset + 16);
        let data_offset = local_offset + 30 + local_name_size + local_extra_size;
        if data_offset + size > archive.len() {
            return fail("Truncated PyTorch checkpoint record");
        }
        if zip_crc32(&archive[data_offset..data_offset + size]) != crc {
            return fail("PyTorch checkpoint record CRC mismatch");
        }
        if records.insert(name, ZipRecord { data_offset, size }).is_some() {
            return fail("Duplicate PyTorch checkpoint record");
        }
        offset += 46 + name_size + extra_size + comment_size;
    }
    Ok(records)
}

fn final_loss_window_batches(completed_steps: u64, total_steps: i64) -> u64 {
    let first_step = 0.max(total_steps - 1.max((total_steps + 99) / 100));
    0.max(completed_steps as i64 - first_step) as u64
}

pub fn load_training_state(path: &str, config: &Config) -> Result<TrainingState> {
    let archive = std::fs::read(path).map_err(|_| Error::new("Cannot open PyTorch checkpoint"))?;
    let records = read_zip_records(&archive)?;
    let prefix = records
        .keys()
        .find_map(|name| name.strip_suffix("/data.pkl").filter(|p| !p.is_empty()))
        .ok_or_else(|| Error::new("Missing PyTorch checkpoint data.pkl"))?
        .to_string();
    let record = |name: &str| -> Result<ZipRecord> {
        let full = format!("{prefix}/{name}");
        records
            .get(&full)
            .copied()
            .ok_or_else(|| Error::new(format!("Missing PyTorch checkpoint record {full}")))
    };
    let bytes = |r: ZipRecord| &archive[r.data_offset..r.data_offset + r.size];
    let require_text = |name: &str, expected: &str| -> Result<()> {
        if bytes(record(name)?) != expected.as_bytes() {
            return fail("Unsupported PyTorch checkpoint metadata");
        }
        Ok(())
    };
    require_text(".format_version", "2")?;
    require_text(".storage_alignment", "64")?;
    require_text("byteorder", "little")?;
    require_text("version", "3\n")?;
    if bytes(record(".data/serialization_id")?) != SERIALIZATION_ID.as_bytes() {
        return fail("Invalid PyTorch checkpoint serialization id");
    }
    let scalar = |name: &str| -> Result<i64> {
        let r = record(name)?;
        if r.size != 8 {
            return fail("Invalid PyTorch checkpoint scalar size");
        }
        Ok(read64(&archive, r.data_offset) as i64)
    };
    let double = |name: &str| -> Result<f64> { Ok(f64::from_bits(scalar(name)? as u64)) };
    let float32s = |name: &str, count: usize| -> Result<Vec<f32>> {
        let r = record(name)?;
        if r.size != count * 4 {
            return fail("Invalid PyTorch checkpoint tensor size");
        }
        Ok(bytes(r).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    };
    let float64s = |name: &str, count: usize| -> Result<Vec<f64>> {
        let r = record(name)?;
        if r.size != count * 8 {
            return fail("Invalid PyTorch checkpoint tensor size");
        }
        Ok(bytes(r).chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    };

    let depth = scalar("data/0")?;
    let context = scalar("data/1")?;
    let count = parameter_layout(config.depth).count as usize;
    if depth != config.depth as i64 || context != config.context as i64 {
        return fail("Checkpoint architecture mismatch");
    }
    let mut state = TrainingState {
        weights: float32s("data/2", count)?,
        momentum: float32s("data/3", count)?,
        variance: float32s("data/4", count)?,
        step: scalar("data/5")? as u64,
        ..Default::default()
    };
    let total_steps = scalar("data/6")?;
    if total_steps < 1 || total_steps > i32::MAX as i64 || state.step > total_steps as u64 {
        return fail("Invalid checkpoint scheduler state");
    }
    state.scheduler = OneCycle {
        total_steps: total_steps as i32,
        warmup_fraction: double("data/7")?,
        initial_lr_fraction: double("data/8")?,
        final_lr_fraction: double("data/9")?,
        low_momentum: double("data/10")?,
        high_momentum: double("data/11")?,
    };
    state.train_seconds = double("data/12")?;
    state.final_loss_sums = float64s("data/13", config.context as usize)?;
    state.final_loss_batches = scalar("data/14")? as u64;
    if state.final_loss_batches > final_loss_window_batches(state.step, total_steps) {
        return fail("Invalid checkpoint final loss state");
    }
    onecycle_point(&state.scheduler, 0)?;
    Ok(state)
}

pub fn save_training_state(path: &str, config: &Config, state: &TrainingState) -> Result<()> {
    let count = parameter_layout(config.depth).count as usize;
    if state.weights.len() != count
        || state.momentum.len() != count
        || state.variance.len() != count
    {
        return fail("Checkpoint state does not match the model");
    }
    if state.final_loss_sums.len() != config.context as usize {
        return fail("Checkpoint final loss state does not match the model");
    }
    if state.final_loss_batches
        > final_loss_window_batches(state.step, state.scheduler.total_steps as i64)
    {
        return fail("Invalid checkpoint final loss state");
    }
    if state.scheduler.total_steps < 1 || state.step > state.scheduler.total_steps as u64 {
        return fail("Invalid checkpoint scheduler state");
    }
    onecycle_point(&state.scheduler, 0)?;
    let path = Path::new(path);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    atomic_replace(path, |destination| {
        let archive = make_checkpoint(config, state, count as u32);
        std::fs::write(destination, archive)
            .map_err(|_| Error::new("Cannot write PyTorch checkpoint"))
    })
}
