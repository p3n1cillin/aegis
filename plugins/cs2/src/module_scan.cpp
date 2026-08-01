#include "module_scan.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <Windows.h>
#  include <Psapi.h>
#endif

#include <cstring>

namespace aegis {
namespace mod {

namespace {

#ifdef _WIN32

uint8_t* find_bytes(uint8_t* hay, size_t hay_len, const uint8_t* needle, size_t needle_len,
                    size_t start) {
    if (!hay || !needle || needle_len == 0 || hay_len < needle_len) return nullptr;
    for (size_t i = start; i + needle_len <= hay_len; ++i) {
        if (std::memcmp(hay + i, needle, needle_len) == 0) return hay + i;
    }
    return nullptr;
}

struct ScanInput {
    uint8_t* image;
    size_t image_size;
    uint8_t* data;
    size_t data_size;
    uint8_t* rdata;
    size_t rdata_size;
    const char* decorated;
    size_t decorated_len;
};

// No C++ objects with destructors here — the SEH wrapper below requires it.
void* scan_vtable_inner(const ScanInput& in) {
    // MSVC keeps RTTI type descriptor names in .data; fall back to .rdata.
    uint8_t* name_ptr = find_bytes(in.data, in.data_size,
                                   reinterpret_cast<const uint8_t*>(in.decorated),
                                   in.decorated_len + 1, 0);
    if (!name_ptr) {
        name_ptr = find_bytes(in.rdata, in.rdata_size,
                              reinterpret_cast<const uint8_t*>(in.decorated),
                              in.decorated_len + 1, 0);
    }
    if (!name_ptr) return nullptr;

    uint8_t* type_desc = name_ptr - 0x10;
    const uint32_t rtti_rva = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(type_desc) - reinterpret_cast<uintptr_t>(in.image));

    size_t search = 0;
    while (search < in.rdata_size) {
        uint8_t* col_mid = find_bytes(in.rdata, in.rdata_size,
                                      reinterpret_cast<const uint8_t*>(&rtti_rva),
                                      sizeof(rtti_rva), search);
        if (!col_mid) break;
        uint8_t* col_header = col_mid - 0xC;
        if (col_header >= in.rdata && *reinterpret_cast<int32_t*>(col_header) == 1 &&
            *reinterpret_cast<int32_t*>(col_mid - 0x8) == 0) {
            uint8_t* vtable_ref = find_bytes(in.rdata, in.rdata_size,
                                             reinterpret_cast<const uint8_t*>(&col_header),
                                             sizeof(void*), 0);
            if (vtable_ref) return vtable_ref + sizeof(void*);
        }
        search = static_cast<size_t>(col_mid - in.rdata) + 1;
    }
    return nullptr;
}

void* scan_vtable_guarded(const ScanInput& in) {
    __try {
        return scan_vtable_inner(in);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return nullptr;
    }
}

#endif  // _WIN32

}  // namespace

void Module::init_from_handle(void* handle) {
#ifdef _WIN32
    if (!handle) return;
    MODULEINFO info{};
    if (!GetModuleInformation(GetCurrentProcess(), static_cast<HMODULE>(handle), &info,
                              sizeof(info))) {
        return;
    }
    base_ = info.lpBaseOfDll;
    size_ = info.SizeOfImage;

    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base_);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(
        reinterpret_cast<uint8_t*>(base_) + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    auto* sec = IMAGE_FIRST_SECTION(nt);
    for (unsigned i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        Section s;
        char name[9] = {};
        std::memcpy(name, sec[i].Name, 8);
        s.name = name;
        s.base = reinterpret_cast<uint8_t*>(base_) + sec[i].VirtualAddress;
        s.size = sec[i].Misc.VirtualSize ? sec[i].Misc.VirtualSize : sec[i].SizeOfRawData;
        sections_.push_back(s);
    }
#else
    (void)handle;
#endif
}

Module::Module(const char* module_name) {
#ifdef _WIN32
    init_from_handle(GetModuleHandleA(module_name));
#else
    (void)module_name;
#endif
}

Module Module::from_address(const void* addr) {
    Module m;
#ifdef _WIN32
    HMODULE h = nullptr;
    if (addr && GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(addr), &h)) {
        m.init_from_handle(h);
    }
#else
    (void)addr;
#endif
    return m;
}

const Section* Module::section(const char* name) const {
    for (const auto& s : sections_) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

void* Module::find_virtual_table(const std::string& class_name) const {
#ifdef _WIN32
    if (!base_ || size_ == 0) return nullptr;
    const Section* data = section(".data");
    const Section* rdata = section(".rdata");
    if (!data || !rdata) return nullptr;

    const std::string av = ".?AV" + class_name + "@@";
    const std::string au = ".?AU" + class_name + "@@";

    ScanInput in{};
    in.image = static_cast<uint8_t*>(base_);
    in.image_size = size_;
    in.data = data->base;
    in.data_size = data->size;
    in.rdata = rdata->base;
    in.rdata_size = rdata->size;

    in.decorated = av.c_str();
    in.decorated_len = av.size();
    if (void* vt = scan_vtable_guarded(in)) return vt;

    in.decorated = au.c_str();
    in.decorated_len = au.size();
    return scan_vtable_guarded(in);
#else
    (void)class_name;
    return nullptr;
#endif
}

}  // namespace mod
}  // namespace aegis
