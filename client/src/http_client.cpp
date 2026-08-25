#include "http_client.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

#include <vector>

namespace aegis::client {

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

}  // namespace

HttpResponse http_post_json(const std::string& host, uint16_t port, const std::string& path,
                             const std::string& json_body, const std::string& api_key,
                             bool use_tls) {
    HttpResponse res;
    HINTERNET session = WinHttpOpen(L"aegis-attest-client/0.1", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) {
        res.error = "WinHttpOpen failed";
        return res;
    }
    HINTERNET connect = WinHttpConnect(session, widen(host).c_str(), port, 0);
    if (!connect) {
        res.error = "WinHttpConnect failed";
        WinHttpCloseHandle(session);
        return res;
    }
    DWORD flags = use_tls ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET request = WinHttpOpenRequest(connect, L"POST", widen(path).c_str(), nullptr,
                                            WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!request) {
        res.error = "WinHttpOpenRequest failed";
        WinHttpCloseHandle(connect);
        WinHttpCloseHandle(session);
        return res;
    }

    std::wstring headers = L"Content-Type: application/json\r\n";
    if (!api_key.empty()) {
        headers += L"X-Aegis-Api-Key: " + widen(api_key) + L"\r\n";
    }
    WinHttpAddRequestHeaders(request, headers.c_str(), static_cast<DWORD>(-1),
                             WINHTTP_ADDREQ_FLAG_ADD);

    BOOL sent = WinHttpSendRequest(
        request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        const_cast<void*>(static_cast<const void*>(json_body.data())),
        static_cast<DWORD>(json_body.size()), static_cast<DWORD>(json_body.size()), 0);
    if (sent) sent = WinHttpReceiveResponse(request, nullptr);

    if (sent) {
        DWORD status_code = 0, size = sizeof(status_code);
        WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                            WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &size, WINHTTP_NO_HEADER_INDEX);
        res.status = static_cast<int>(status_code);

        std::vector<char> buf;
        for (;;) {
            DWORD avail = 0;
            if (!WinHttpQueryDataAvailable(request, &avail) || avail == 0) break;
            size_t start = buf.size();
            buf.resize(start + avail);
            DWORD read = 0;
            if (!WinHttpReadData(request, buf.data() + start, avail, &read)) break;
            buf.resize(start + read);
            if (read == 0) break;
        }
        res.body.assign(buf.begin(), buf.end());
        res.ok = true;
    } else {
        res.error = "request failed, GetLastError=" + std::to_string(GetLastError());
    }

    WinHttpCloseHandle(request);
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return res;
}

}  // namespace aegis::client
