#pragma once

#include <cstdint>

struct ISchemaSystem;

namespace aegis {
namespace schema {

void init(ISchemaSystem* sys);
bool ready();

// Resolve field offset via server.dll schema. Walks base classes.
// Returns -1 if missing.
int offset_of(const char* class_name, const char* field_name);

template <typename T>
T* field(void* obj, const char* class_name, const char* field_name) {
    if (!obj) return nullptr;
    const int off = offset_of(class_name, field_name);
    if (off < 0) return nullptr;
    return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(obj) +
                                static_cast<uintptr_t>(off));
}

template <typename T>
T read(void* obj, const char* class_name, const char* field_name, T fallback = T{}) {
    T* p = field<T>(obj, class_name, field_name);
    return p ? *p : fallback;
}

}  // namespace schema
}  // namespace aegis
