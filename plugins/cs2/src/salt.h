#pragma once

#include <cstdint>
#include <string>

namespace aegis {

// Stable salted account id. The detection stack must never see raw SteamIDs.
// Uses BLAKE2-ish? Keep it simple: HMAC-SHA256 truncated hex via Windows BCrypt
// / OpenSSL when available; otherwise a portable FNV-1a mix of salt||id for
// the scaffold (replace before production).
std::string salt_account_id(const std::string& salt, uint64_t steamid64);

}  // namespace aegis
