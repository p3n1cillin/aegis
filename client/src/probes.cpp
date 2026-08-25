#include "probes.h"

#include <algorithm>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace aegis::client {

namespace {

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

std::wstring self_exe_path() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::wstring(buf, n);
}

}  // namespace

bool probe_module_hash(Digest32& out) {
    return sha256_file(self_exe_path(), out);
}

bool probe_input_stack_ok() {
    if (IsDebuggerPresent()) return false;
    BOOL remote = FALSE;
    if (CheckRemoteDebuggerPresent(GetCurrentProcess(), &remote) && remote) return false;
    return true;
}

ProbeResults run_probes(const std::vector<std::string>& requested) {
    ProbeResults r;
    if (contains(requested, "module_hash")) {
        r.module_hash_ok = probe_module_hash(r.module_hash);
    }
    if (contains(requested, "input_stack")) {
        r.input_stack_ok = probe_input_stack_ok();
    }
    return r;
}

std::string build_measurement(const ProbeResults& r, const std::vector<std::string>& requested) {
    std::string out;
    if (contains(requested, "module_hash")) {
        out.append(reinterpret_cast<const char*>(r.module_hash.data()), r.module_hash.size());
    }
    if (contains(requested, "input_stack")) {
        out.push_back(r.input_stack_ok ? '\x01' : '\x00');
    }
    return out;
}

std::map<std::string, bool> probe_ok_map(const ProbeResults& r,
                                          const std::vector<std::string>& requested) {
    std::map<std::string, bool> out;
    if (contains(requested, "module_hash")) out["module_hash"] = r.module_hash_ok;
    if (contains(requested, "input_stack")) out["input_stack"] = r.input_stack_ok;
    return out;
}

}  // namespace aegis::client
