#pragma once
#include <cstddef>
#include <string>

// build.ps1 generates the definition in the build directory: the HEAD commit
// hash, then the working-tree diff (tracked and untracked files) after a blank line.
extern const unsigned char turbogpt_revision[];
extern const size_t turbogpt_revision_size;

inline std::string build_revision() {
    return std::string(reinterpret_cast<const char *>(turbogpt_revision), turbogpt_revision_size);
}
