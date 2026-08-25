#include "json_lite.h"

#include <cctype>
#include <cstdlib>

namespace aegis::client::json {

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) break;  // drop other control chars
                out += c;
        }
    }
    out += "\"";
    return out;
}

std::string object(const std::vector<std::string>& kv) {
    std::string out = "{";
    for (size_t i = 0; i + 1 < kv.size(); i += 2) {
        if (i) out += ",";
        out += quote(kv[i]);
        out += ":";
        out += kv[i + 1];
    }
    out += "}";
    return out;
}

namespace {

// Position just after `"key"` followed by optional whitespace and `:`, or
// std::string::npos if the key isn't present as an object member.
size_t find_value_start(const std::string& j, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = j.find(needle);
    if (pos == std::string::npos) return std::string::npos;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t' || j[pos] == '\n')) ++pos;
    if (pos >= j.size() || j[pos] != ':') return std::string::npos;
    ++pos;
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t' || j[pos] == '\n')) ++pos;
    return pos;
}

}  // namespace

std::string extract_string(const std::string& j, const std::string& key) {
    size_t pos = find_value_start(j, key);
    if (pos == std::string::npos || pos >= j.size() || j[pos] != '"') return {};
    ++pos;
    std::string out;
    while (pos < j.size() && j[pos] != '"') {
        if (j[pos] == '\\' && pos + 1 < j.size()) {
            ++pos;
            switch (j[pos]) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                default: out += j[pos];
            }
        } else {
            out += j[pos];
        }
        ++pos;
    }
    return out;
}

bool extract_bool(const std::string& j, const std::string& key, bool fallback) {
    size_t pos = find_value_start(j, key);
    if (pos == std::string::npos) return fallback;
    if (j.compare(pos, 4, "true") == 0) return true;
    if (j.compare(pos, 5, "false") == 0) return false;
    return fallback;
}

double extract_number(const std::string& j, const std::string& key, double fallback) {
    size_t pos = find_value_start(j, key);
    if (pos == std::string::npos) return fallback;
    size_t end = pos;
    while (end < j.size() &&
           (std::isdigit(static_cast<unsigned char>(j[end])) || j[end] == '-' ||
            j[end] == '+' || j[end] == '.' || j[end] == 'e' || j[end] == 'E')) {
        ++end;
    }
    if (end == pos) return fallback;
    return std::strtod(j.substr(pos, end - pos).c_str(), nullptr);
}

std::vector<std::string> extract_string_array(const std::string& j, const std::string& key) {
    std::vector<std::string> out;
    size_t pos = find_value_start(j, key);
    if (pos == std::string::npos || pos >= j.size() || j[pos] != '[') return out;
    ++pos;
    while (pos < j.size() && j[pos] != ']') {
        while (pos < j.size() && (j[pos] == ' ' || j[pos] == ',' || j[pos] == '\n')) ++pos;
        if (pos >= j.size() || j[pos] != '"') break;
        ++pos;
        std::string item;
        while (pos < j.size() && j[pos] != '"') {
            item += j[pos];
            ++pos;
        }
        out.push_back(item);
        ++pos;  // closing quote
    }
    return out;
}

}  // namespace aegis::client::json
