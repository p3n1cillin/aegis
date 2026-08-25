#pragma once

#include <map>
#include <string>
#include <vector>

#include "crypto.h"

namespace aegis::client {

struct ProbeResults {
    Digest32 module_hash{};
    bool module_hash_ok = false;
    bool input_stack_ok = false;
};

// Runs the probes the server's Challenge asked for (from the "probes" list
// in the /v1/attest/issue response) and nothing else -- an honest client
// only measures what it was actually challenged on.
ProbeResults run_probes(const std::vector<std::string>& requested);

// Real, functioning, and modest -- this is a reference client demonstrating
// the challenge-response protocol end-to-end, not a claim to rival
// kernel-level anti-cheat detection. See client/README.md for exactly what
// each probe does and does not catch.
//
//   module_hash:  SHA-256 of this client's own executable on disk. Detects a
//                 patched binary; does not detect in-memory-only tampering
//                 after launch (would need a kernel driver or a
//                 continuously-re-verified in-memory hash, neither of which
//                 exists here).
//   input_stack:  IsDebuggerPresent()/CheckRemoteDebuggerPresent() only.
//                 Real and standard, but a well-known, easily-defeated
//                 signal on its own -- do not read "true" as "definitely not
//                 tampered."
bool probe_module_hash(Digest32& out);
bool probe_input_stack_ok();

// The bytes sent to the server as `client_measurement` -- module hash
// followed by a single status byte, in the order `probes` was iterated.
std::string build_measurement(const ProbeResults& r, const std::vector<std::string>& requested);

// Whether every requested probe actually passed (module_hash_ok / this
// specific binary's own file could be read and hashed; input_stack_ok / no
// debugger detected). Sent to the server as `probe_ok` per probe name.
std::map<std::string, bool> probe_ok_map(const ProbeResults& r,
                                          const std::vector<std::string>& requested);

}  // namespace aegis::client
