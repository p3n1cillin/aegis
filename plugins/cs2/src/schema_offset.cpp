#include "schema_offset.h"

#ifdef AEGIS_WITH_METAMOD
#  include <schemasystem/schemasystem.h>
#  include <schemasystem/schematypes.h>
#endif

#include <cstdio>
#include <cstdlib>
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

static CSchemaClassInfo* find_in_scope(CSchemaSystemTypeScope* scope, const char* class_name) {
    if (!scope) return nullptr;
    if (CSchemaClassInfo* raw = scope->FindRawClassBinding(class_name)) return raw;
    return scope->FindDeclaredClass(class_name).Get();
}

// Game-specific classes (CCSPlayerPawn, weapon services, ...) live in the
// per-module scope, but engine-level classes (CGameSceneNode, CEntityInstance,
// ...) are registered once in the schema system's global scope and are never
// found by searching server.dll alone. Try both, module-local first since a
// title can shadow a global type with its own layout.
static CSchemaClassInfo* find_class(const char* class_name) {
    if (!g_schema || !class_name) return nullptr;
    CSchemaSystemTypeScope* mod_scope = g_schema->FindTypeScopeForModule("server.dll");
    if (!mod_scope) mod_scope = g_schema->FindTypeScopeForModule("server");
    if (CSchemaClassInfo* cls = find_in_scope(mod_scope, class_name)) return cls;
    return find_in_scope(g_schema->GlobalTypeScope(), class_name);
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

void dump_class(const char* class_name) {
    const char* path = std::getenv("AEGIS_LOG_FILE");
    if (!path) return;
    FILE* f = nullptr;
    if (fopen_s(&f, path, "a") != 0 || !f) return;
    CSchemaClassInfo* cls = find_class(class_name);
    if (!cls) {
        std::fprintf(f, "DUMP %s: class not found\n", class_name);
        std::fclose(f);
        return;
    }
    std::fprintf(f, "DUMP %s (fields=%u bases=%u):\n", class_name, cls->m_nFieldCount,
                 cls->m_nBaseClassCount);
    for (uint16_t i = 0; i < cls->m_nFieldCount; ++i) {
        const SchemaClassFieldData_t& fld = cls->m_pFields[i];
        std::fprintf(f, "  field %-32s off=%d\n", fld.m_pszName ? fld.m_pszName : "?",
                     fld.m_nSingleInheritanceOffset);
    }
    for (uint8_t b = 0; b < cls->m_nBaseClassCount; ++b) {
        CSchemaClassInfo* base = cls->m_pBaseClasses[b].m_pClass;
        std::fprintf(f, "  base %s off=%d\n", base ? base->m_pszName : "?",
                     static_cast<int>(cls->m_pBaseClasses[b].m_nOffset));
    }
    std::fflush(f);
    std::fclose(f);
}

#else

void init(ISchemaSystem*) {}
bool ready() { return false; }
int offset_of(const char*, const char*) { return -1; }
void dump_class(const char*) {}

#endif

}  // namespace schema
}  // namespace aegis
