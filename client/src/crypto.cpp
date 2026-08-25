#include "crypto.h"

#include <cstdio>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

namespace aegis::client {

namespace {

// Shared scaffolding for both the plain-hash and HMAC providers -- same
// BCrypt call sequence, keyed or not.
bool bcrypt_digest(LPCWSTR alg_id, ULONG alg_flags, const uint8_t* key, size_t key_len,
                    const uint8_t* msg, size_t msg_len, Digest32& out) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD obj_len = 0, data_len = 0, hash_len = 0;
    if (BCryptOpenAlgorithmProvider(&alg, alg_id, nullptr, alg_flags) != 0) return false;
    bool ok = true;
    ok = ok && BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&obj_len),
                                  sizeof(obj_len), &data_len, 0) == 0;
    ok = ok && BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len),
                                  sizeof(hash_len), &data_len, 0) == 0 && hash_len == 32;
    if (!ok) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }
    std::vector<uint8_t> obj(obj_len);
    NTSTATUS st = BCryptCreateHash(
        alg, &hash, obj.data(), obj_len,
        const_cast<PUCHAR>(key), static_cast<ULONG>(key_len), 0);
    if (st == 0) {
        st = BCryptHashData(hash, const_cast<PUCHAR>(msg), static_cast<ULONG>(msg_len), 0);
    }
    if (st == 0) {
        st = BCryptFinishHash(hash, out.data(), 32, 0);
    }
    if (hash) BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    return st == 0;
}

}  // namespace

bool hmac_sha256(const std::string& key, const std::string& msg, Digest32& out) {
    return bcrypt_digest(BCRYPT_SHA256_ALGORITHM, BCRYPT_ALG_HANDLE_HMAC_FLAG,
                         reinterpret_cast<const uint8_t*>(key.data()), key.size(),
                         reinterpret_cast<const uint8_t*>(msg.data()), msg.size(), out);
}

bool sha256_bytes(const uint8_t* data, size_t len, Digest32& out) {
    return bcrypt_digest(BCRYPT_SHA256_ALGORITHM, 0, nullptr, 0, data, len, out);
}

bool sha256_file(const std::wstring& path, Digest32& out) {
    // Streamed so a multi-hundred-MB game binary doesn't need to fit in one
    // buffer; BCrypt's hash object accumulates state across calls.
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD obj_len = 0, data_len = 0;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return false;
    if (BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&obj_len),
                          sizeof(obj_len), &data_len, 0) != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }
    std::vector<uint8_t> obj(obj_len);
    if (BCryptCreateHash(alg, &hash, obj.data(), obj_len, nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return false;
    }

    FILE* f = _wfopen(path.c_str(), L"rb");
    bool ok = f != nullptr;
    if (ok) {
        uint8_t buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            if (BCryptHashData(hash, buf, static_cast<ULONG>(n), 0) != 0) {
                ok = false;
                break;
            }
        }
        fclose(f);
    }
    if (ok) ok = BCryptFinishHash(hash, out.data(), 32, 0) == 0;
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    return ok;
}

std::string to_hex(const uint8_t* data, size_t len) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[i * 2] = kHex[data[i] >> 4];
        out[i * 2 + 1] = kHex[data[i] & 0xf];
    }
    return out;
}

std::string to_hex(const Digest32& d) { return to_hex(d.data(), d.size()); }

std::string from_hex(const std::string& hex) {
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = nibble(hex[i]), lo = nibble(hex[i + 1]);
        if (hi < 0 || lo < 0) break;
        out.push_back(static_cast<char>((hi << 4) | lo));
    }
    return out;
}

}  // namespace aegis::client
