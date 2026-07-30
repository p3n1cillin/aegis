#include "salt.h"

#include <cstdint>
#include <cstdio>
#include <string>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <bcrypt.h>
#  pragma comment(lib, "bcrypt.lib")
#endif

namespace aegis {

static uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::string hex16(const uint8_t* dig, size_t n) {
    // Truncate to 8 bytes → 16 hex chars (matches prior FNV width).
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.resize(16);
    for (size_t i = 0; i < 8; ++i) {
        const uint8_t b = i < n ? dig[i] : 0;
        out[i * 2] = kHex[b >> 4];
        out[i * 2 + 1] = kHex[b & 0xf];
    }
    return out;
}

#ifdef _WIN32
static bool hmac_sha256(const std::string& key, const std::string& msg, uint8_t out[32]) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD obj_len = 0, data_len = 0, hash_len = 0;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr,
                                    BCRYPT_ALG_HANDLE_HMAC_FLAG) != 0)
        return false;
    if (BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&obj_len),
                          sizeof(obj_len), &data_len, 0) != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }
    if (BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len),
                          sizeof(hash_len), &data_len, 0) != 0 || hash_len < 32) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }
    std::string obj;
    obj.resize(obj_len);
    NTSTATUS st = BCryptCreateHash(
        alg, &hash, reinterpret_cast<PUCHAR>(obj.data()), obj_len,
        reinterpret_cast<PUCHAR>(const_cast<char*>(key.data())),
        ULONG(key.size()), 0);
    if (st != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }
    st = BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(msg.data())),
                        ULONG(msg.size()), 0);
    if (st == 0) st = BCryptFinishHash(hash, out, 32, 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    return st == 0;
}
#endif

std::string salt_account_id(const std::string& salt, uint64_t steamid64) {
    char num[32];
    std::snprintf(num, sizeof(num), "%llu", static_cast<unsigned long long>(steamid64));
    const std::string msg = std::string(num);

#ifdef _WIN32
    uint8_t dig[32];
    if (!salt.empty() && salt != "CHANGE_ME_SERVER_SALT" && hmac_sha256(salt, msg, dig)) {
        return hex16(dig, 32);
    }
#endif
    // Fallback for tools / unsalted lab builds — never ship this.
    std::string raw = salt;
    raw.push_back('|');
    raw += msg;
    const uint64_t h = fnv1a64(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
    char out[17];
    std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(h));
    return std::string(out);
}

}  // namespace aegis
