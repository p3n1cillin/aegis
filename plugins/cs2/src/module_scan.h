#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aegis {
namespace mod {

struct Section {
    std::string name;
    uint8_t* base = nullptr;
    size_t size = 0;
};

// Lightweight PE module view for RTTI vtable discovery (Windows).
class Module {
public:
    // Prefer from_address when multiple DLLs share the name "server.dll"
    // (MetaMod ships a loader stub also named server.dll).
    explicit Module(const char* module_name);
    static Module from_address(const void* addr);

    bool valid() const { return base_ != nullptr && size_ > 0; }
    void* base() const { return base_; }
    size_t size() const { return size_; }

    // Returns pointer to the first function entry of the primary vtable, or null.
    void* find_virtual_table(const std::string& class_name) const;

private:
    Module() = default;
    void init_from_handle(void* handle);

    void* base_ = nullptr;
    size_t size_ = 0;
    std::vector<Section> sections_;

    const Section* section(const char* name) const;
};

}  // namespace mod
}  // namespace aegis
