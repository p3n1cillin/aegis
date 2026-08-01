#include "schema_offset.h"

#ifdef AEGIS_WITH_METAMOD
#  include <schemasystem/schemasystem.h>
#  include <schemasystem/schematypes.h>
#endif

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace aegis {
namespace schema {

#ifdef AEGIS_WITH_METAMOD

static ISchemaSystem* g_schema = nullptr;
static std::mutex g_mu;
static std::unordered_map<std::string, int> g_cache;

static CSchemaClassInfo* find_class(const char* class_name) {
    if (!g_schema || !class_name) return nullptr;
    CSchemaSystemTypeScope* scope = g_schema->FindTypeScopeForModule("server.dll");
    if (!scope) scope = g_schema->FindTypeScopeForModule("server");
    if (!scope) return nullptr;
    if (CSchemaClassInfo* raw = scope->FindRawClassBinding(class_name)) return raw;
    return scope->FindDeclaredClass(class_name).Get();
}

static int find_field_offset(CSchemaClassInfo* cls, const char* field_name) {
    if (!cls || !field_name) return -1;
    for (uint16_t i = 0; i < cls->m_nFieldCount; ++i) {
        const SchemaClassFieldData_t& f = cls->m_pFields[i];
        if (f.m_pszName && std::strcmp(f.m_pszName, field_name) == 0) {
            return f.m_nSingleInheritanceOffset;
        }
    }
    for (uint8_t b = 0; b < cls->m_nBaseClassCount; ++b) {
        CSchemaClassInfo* base = cls->m_pBaseClasses[b].m_pClass;
        const int off = find_field_offset(base, field_name);
        if (off >= 0) {
            return static_cast<int>(cls->m_pBaseClasses[b].m_nOffset) + off;
        }
    }
    return -1;
}

void init(ISchemaSystem* sys) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_schema = sys;
    g_cache.clear();
}

bool ready() { return g_schema != nullptr; }

int offset_of(const char* class_name, const char* field_name) {
    if (!class_name || !field_name || !g_schema) return -1;
    const std::string key = std::string(class_name) + "::" + field_name;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_cache.find(key);
        if (it != g_cache.end()) return it->second;
    }
    CSchemaClassInfo* cls = find_class(class_name);
    const int off = find_field_offset(cls, field_name);
    std::lock_guard<std::mutex> lock(g_mu);
    g_cache[key] = off;
    return off;
}

#else

void init(ISchemaSystem*) {}
bool ready() { return false; }
int offset_of(const char*, const char*) { return -1; }

#endif

}  // namespace schema
}  // namespace aegis
