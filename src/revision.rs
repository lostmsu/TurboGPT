// build.rs embeds the HEAD commit hash, then the working-tree diff (tracked and
// untracked files) after a blank line.
static REVISION: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/revision.bin"));

pub fn build_revision() -> &'static [u8] {
    REVISION
}
