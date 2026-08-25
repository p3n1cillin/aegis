#pragma once

#include <string>
#include <vector>

namespace aegis::client::json {

// Deliberately not a general-purpose JSON library: the wire contract here is
// a handful of known, flat fields (AttestIssueRequest/Response,
// AttestVerifyRequest/Response in aegis/service/__init__.py), and pulling in
// a JSON dependency for that is more risk than it saves. Every extractor
// below fails safe (empty/false/0) on anything unexpected rather than
// throwing, appropriate for parsing a response from a service this client
// itself is asking to vouch for its integrity -- a malformed response should
// not crash the attestor.

// Builds a compact flat JSON object from alternating key/value string pairs,
// e.g. object({"session_id", esc(sid), "player_id", esc(pid)}).
std::string object(const std::vector<std::string>& kv_pairs_pre_escaped);

// Wraps a string value in quotes with minimal escaping (backslash, quote,
// control chars) -- use before passing string values into object().
std::string quote(const std::string& s);

std::string extract_string(const std::string& j, const std::string& key);
bool extract_bool(const std::string& j, const std::string& key, bool fallback = false);
double extract_number(const std::string& j, const std::string& key, double fallback = 0.0);
std::vector<std::string> extract_string_array(const std::string& j, const std::string& key);

}  // namespace aegis::client::json
