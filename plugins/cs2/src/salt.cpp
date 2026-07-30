#include "salt.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace aegis {

// Scaffold hash: FNV-1a 64 over salt || steamid. Replace with HMAC-SHA256
// before any production deployment — this exists so the wire format and
// adapter path can be exercised end-to-end without crypto deps.
static uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string salt_account_id(const std::string& salt, uint64_t steamid64) {
    std::string raw = salt;
    raw.push_back('|');
    char num[32];
    std::snprintf(num, sizeof(num), "%llu", static_cast<unsigned long long>(steamid64));
    raw += num;
    const uint64_t h = fnv1a64(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return std::string(out);
}

}  // namespace aegis
