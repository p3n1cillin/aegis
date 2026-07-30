#include "encode.h"

#include <cstring>

namespace aegis {

void MsgpackWriter::clear() { buf_.clear(); }
void MsgpackWriter::put(uint8_t b) { buf_.push_back(b); }
void MsgpackWriter::put_raw(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    buf_.insert(buf_.end(), b, b + n);
}
void MsgpackWriter::put_be16(uint16_t v) {
    put(uint8_t(v >> 8));
    put(uint8_t(v));
}
void MsgpackWriter::put_be32(uint32_t v) {
    put(uint8_t(v >> 24));
    put(uint8_t(v >> 16));
    put(uint8_t(v >> 8));
    put(uint8_t(v));
}
void MsgpackWriter::put_be64(uint64_t v) {
    for (int i = 7; i >= 0; --i) put(uint8_t(v >> (8 * i)));
}

void MsgpackWriter::map(size_t n) {
    if (n < 16) {
        put(uint8_t(0x80 | n));
    } else if (n < 0x10000) {
        put(0xde);
        put_be16(uint16_t(n));
    } else {
        put(0xdf);
        put_be32(uint32_t(n));
    }
}

void MsgpackWriter::array(size_t n) {
    if (n < 16) {
        put(uint8_t(0x90 | n));
    } else if (n < 0x10000) {
        put(0xdc);
        put_be16(uint16_t(n));
    } else {
        put(0xdd);
        put_be32(uint32_t(n));
    }
}

void MsgpackWriter::str(const char* s) {
    const size_t n = std::strlen(s);
    if (n < 32) {
        put(uint8_t(0xa0 | n));
    } else if (n < 0x100) {
        put(0xd9);
        put(uint8_t(n));
    } else if (n < 0x10000) {
        put(0xda);
        put_be16(uint16_t(n));
    } else {
        put(0xdb);
        put_be32(uint32_t(n));
    }
    put_raw(s, n);
}

void MsgpackWriter::str(const std::string& s) { str(s.c_str()); }
void MsgpackWriter::key(const char* k) { str(k); }

void MsgpackWriter::boolean(bool v) { put(v ? 0xc3 : 0xc2); }
void MsgpackWriter::nil() { put(0xc0); }

void MsgpackWriter::i64(int64_t v) {
    if (v >= 0) {
        u64(uint64_t(v));
        return;
    }
    if (v >= -32) {
        put(uint8_t(v));
        return;
    }
    if (v >= -128) {
        put(0xd0);
        put(uint8_t(v));
        return;
    }
    if (v >= -32768) {
        put(0xd1);
        put_be16(uint16_t(v));
        return;
    }
    if (v >= -2147483648LL) {
        put(0xd2);
        put_be32(uint32_t(v));
        return;
    }
    put(0xd3);
    put_be64(uint64_t(v));
}

void MsgpackWriter::u64(uint64_t v) {
    if (v < 128) {
        put(uint8_t(v));
    } else if (v < 0x100) {
        put(0xcc);
        put(uint8_t(v));
    } else if (v < 0x10000) {
        put(0xcd);
        put_be16(uint16_t(v));
    } else if (v < 0x100000000ULL) {
        put(0xce);
        put_be32(uint32_t(v));
    } else {
        put(0xcf);
        put_be64(v);
    }
}

void MsgpackWriter::f64(double v) {
    put(0xcb);
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(v), "f64 size");
    std::memcpy(&bits, &v, sizeof(bits));
    put_be64(bits);
}

}  // namespace aegis
