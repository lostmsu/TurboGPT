// TensorBoard event files. Events accumulate in memory and reach the file when the
// buffer fills or on flush().
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn key(out: &mut Vec<u8>, field: u32, wire: u32) {
    varint(out, (field << 3 | wire) as u64);
}
fn bytes(out: &mut Vec<u8>, field: u32, data: &[u8]) {
    key(out, field, 2);
    varint(out, data.len() as u64);
    out.extend_from_slice(data);
}
fn integer(out: &mut Vec<u8>, field: u32, value: u64) {
    key(out, field, 0);
    varint(out, value);
}
fn floating(out: &mut Vec<u8>, field: u32, value: f32) {
    key(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}
fn wide(out: &mut Vec<u8>, field: u32, value: f64) {
    key(out, field, 1);
    out.extend_from_slice(&value.to_le_bytes());
}

const fn crc32c_tables() -> [[u32; 256]; 8] {
    let mut tables = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0x82f63b78 } else { crc >> 1 };
            bit += 1;
        }
        tables[0][i] = crc;
        i += 1;
    }
    let mut k = 1;
    while k < 8 {
        let mut i = 0;
        while i < 256 {
            tables[k][i] = (tables[k - 1][i] >> 8) ^ tables[0][(tables[k - 1][i] & 255) as usize];
            i += 1;
        }
        k += 1;
    }
    tables
}
static CRC32C_TABLES: [[u32; 256]; 8] = crc32c_tables();

fn crc32c(mut data: &[u8]) -> u32 {
    let t = &CRC32C_TABLES;
    let mut crc = 0xffff_ffffu32;
    while data.len() >= 8 {
        let low = crc ^ u32::from_le_bytes(data[..4].try_into().unwrap());
        let high = u32::from_le_bytes(data[4..8].try_into().unwrap());
        crc = t[7][(low & 255) as usize]
            ^ t[6][((low >> 8) & 255) as usize]
            ^ t[5][((low >> 16) & 255) as usize]
            ^ t[4][(low >> 24) as usize]
            ^ t[3][(high & 255) as usize]
            ^ t[2][((high >> 8) & 255) as usize]
            ^ t[1][((high >> 16) & 255) as usize]
            ^ t[0][(high >> 24) as usize];
        data = &data[8..];
    }
    for &byte in data {
        crc = (crc >> 8) ^ t[0][((crc ^ byte as u32) & 255) as usize];
    }
    crc ^ 0xffff_ffff
}

fn masked_crc(crc: u32) -> u32 {
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

fn wall_time() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

pub fn host_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok().map(|h| h.trim().to_string()))
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "localhost".into())
}

fn scalar_tensor(value: f32) -> Vec<u8> {
    let mut tensor = Vec::new();
    integer(&mut tensor, 1, 1);
    bytes(&mut tensor, 2, &[]);
    floating(&mut tensor, 5, value);
    tensor
}

fn text_tensor(value: &[u8]) -> Vec<u8> {
    let (mut tensor, mut shape, mut dimension) = (Vec::new(), Vec::new(), Vec::new());
    integer(&mut tensor, 1, 7);
    integer(&mut dimension, 1, 1);
    bytes(&mut shape, 2, &dimension);
    bytes(&mut tensor, 2, &shape);
    bytes(&mut tensor, 8, value);
    tensor
}

fn metadata(plugin: &str, display_name: &str, data_class: u64) -> Vec<u8> {
    let (mut plugin_data, mut result) = (Vec::new(), Vec::new());
    bytes(&mut plugin_data, 1, plugin.as_bytes());
    bytes(&mut result, 1, &plugin_data);
    bytes(&mut result, 2, display_name.as_bytes());
    integer(&mut result, 4, data_class);
    result
}

fn summary_value(
    tag: &str,
    tensor: &[u8],
    plugin: &str,
    display_name: &str,
    data_class: u64,
) -> Vec<u8> {
    let mut value = Vec::new();
    bytes(&mut value, 1, tag.as_bytes());
    bytes(&mut value, 8, tensor);
    bytes(&mut value, 9, &metadata(plugin, display_name, data_class));
    value
}

const BUFFER_BYTES: usize = 1 << 20;

// A scalar summary is constant per tag except for its 4 value bytes.
struct ScalarSummary {
    bytes: Vec<u8>,
    value_offset: usize,
}

pub struct TensorBoardLogger {
    file: File,
    scalars: HashMap<String, ScalarSummary>,
    event_bytes: Vec<u8>,
    buffer: Vec<u8>,
}

impl TensorBoardLogger {
    pub fn new(directory: &Path) -> Result<Self> {
        std::fs::create_dir_all(directory)?;
        let seconds =
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let name =
            format!("events.out.tfevents.{seconds}.{}.{}.0", host_name(), std::process::id());
        let file = File::create(directory.join(name))
            .map_err(|_| Error::new("Cannot create TensorBoard event file"))?;
        let mut logger = TensorBoardLogger {
            file,
            scalars: HashMap::new(),
            event_bytes: Vec::new(),
            buffer: Vec::with_capacity(BUFFER_BYTES),
        };
        let mut source = Vec::new();
        wide(&mut logger.event_bytes, 1, wall_time());
        bytes(&mut logger.event_bytes, 3, b"brain.Event:2");
        bytes(&mut source, 1, b"turbogpt-native");
        bytes(&mut logger.event_bytes, 10, &source);
        logger.record()?;
        Ok(logger)
    }

    fn record(&mut self) -> Result<()> {
        let start = self.buffer.len();
        self.buffer.extend_from_slice(&(self.event_bytes.len() as u64).to_le_bytes());
        let header_crc = masked_crc(crc32c(&self.buffer[start..start + 8]));
        self.buffer.extend_from_slice(&header_crc.to_le_bytes());
        self.buffer.extend_from_slice(&self.event_bytes);
        self.buffer.extend_from_slice(&masked_crc(crc32c(&self.event_bytes)).to_le_bytes());
        if self.buffer.len() >= BUFFER_BYTES {
            self.write_buffer()?;
        }
        Ok(())
    }

    fn write_buffer(&mut self) -> Result<()> {
        self.file
            .write_all(&self.buffer)
            .map_err(|_| Error::new("Cannot write TensorBoard events"))?;
        self.buffer.clear();
        Ok(())
    }

    fn event(&mut self, summary: &[u8], step: i64) -> Result<()> {
        self.event_bytes.clear();
        wide(&mut self.event_bytes, 1, wall_time());
        integer(&mut self.event_bytes, 2, step as u64);
        bytes(&mut self.event_bytes, 5, summary);
        self.record()
    }

    pub fn scalar(&mut self, tag: &str, value: f64, step: i64) -> Result<()> {
        if !self.scalars.contains_key(tag) {
            let mut value_bytes = Vec::new();
            bytes(&mut value_bytes, 1, tag.as_bytes());
            // The float is the tensor's last field, so its 4 bytes end the tensor message.
            bytes(&mut value_bytes, 8, &scalar_tensor(0.0));
            let value_end = value_bytes.len();
            bytes(&mut value_bytes, 9, &metadata("scalars", tag, 1));
            let mut summary = Vec::new();
            bytes(&mut summary, 1, &value_bytes);
            let value_offset = summary.len() - value_bytes.len() + value_end - 4;
            self.scalars.insert(tag.to_string(), ScalarSummary { bytes: summary, value_offset });
        }
        let summary = self.scalars.get_mut(tag).unwrap();
        summary.bytes[summary.value_offset..][..4].copy_from_slice(&(value as f32).to_le_bytes());
        let bytes = std::mem::take(&mut summary.bytes);
        let result = self.event(&bytes, step);
        self.scalars.get_mut(tag).unwrap().bytes = bytes;
        result
    }

    pub fn text(&mut self, tag: &str, value: impl AsRef<[u8]>, step: i64) -> Result<()> {
        let name = format!("{tag}/text_summary");
        let mut summary = Vec::new();
        bytes(
            &mut summary,
            1,
            &summary_value(&name, &text_tensor(value.as_ref()), "text", &name, 2),
        );
        self.event(&summary, step)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.write_buffer()?;
        self.file.flush().map_err(|_| Error::new("Cannot flush TensorBoard event file"))?;
        Ok(())
    }
}

impl Drop for TensorBoardLogger {
    fn drop(&mut self) {
        // Destructors cannot report; flush() reports write failures.
        let _ = self.write_buffer();
    }
}
