#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace aegis::client {

using Digest32 = std::array<uint8_t, 32>;

// HMAC-SHA256 via Windows BCrypt -- same primitive and API pattern already
// used server-side in plugins/cs2/src/salt.cpp, not a new dependency.
bool hmac_sha256(const std::string& key, const std::string& msg, Digest32& out);

// SHA-256 of a file's bytes -- the module_hash probe's actual measurement.
// Returns false if the file cannot be opened/read.
bool sha256_file(const std::wstring& path, Digest32& out);

// SHA-256 of an in-memory buffer.
bool sha256_bytes(const uint8_t* data, size_t len, Digest32& out);

std::string to_hex(const uint8_t* data, size_t len);
std::string to_hex(const Digest32& d);
std::string from_hex(const std::string& hex);

}  // namespace aegis::client
