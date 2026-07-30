#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace aegis {

// Minimal MessagePack writer for the maps we emit. Avoids pulling a third-party
// library into the game process; frames stay small and schema-stable.
class MsgpackWriter {
public:
    void clear();
    const std::vector<uint8_t>& bytes() const { return buf_; }

    void map(size_t n);
    void array(size_t n);
    void key(const char* k);
    void str(const char* s);
    void str(const std::string& s);
    void boolean(bool v);
    void nil();
    void i64(int64_t v);
    void u64(uint64_t v);
    void f64(double v);

private:
    std::vector<uint8_t> buf_;
    void put(uint8_t b);
    void put_raw(const void* p, size_t n);
    void put_be16(uint16_t v);
    void put_be32(uint32_t v);
    void put_be64(uint64_t v);
};

}  // namespace aegis
