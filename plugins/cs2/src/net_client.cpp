#include "net_client.h"

#include <chrono>
#include <string>
#include <thread>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t kInvalid = INVALID_SOCKET;
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <netdb.h>
#  include <sys/socket.h>
#  include <unistd.h>
using socket_t = int;
static constexpr socket_t kInvalid = -1;
#endif

namespace aegis {

NetClient::NetClient() = default;

NetClient::~NetClient() { stop(); }

void NetClient::configure(std::string host, int port) {
    host_ = std::move(host);
    port_ = port;
}

void NetClient::start() {
    if (running_.exchange(true)) return;
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    thr_ = std::thread([this] { loop(); });
}

void NetClient::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (thr_.joinable()) thr_.join();
    if (sock_ != kInvalid) {
#ifdef _WIN32
        closesocket(static_cast<SOCKET>(sock_));
        WSACleanup();
#else
        ::close(sock_);
#endif
        sock_ = kInvalid;
    }
    connected_ = false;
}

bool NetClient::push(std::vector<uint8_t> packet) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.size() >= max_queue_) {
        queue_.erase(queue_.begin());
        dropped_++;
    }
    queue_.push_back(std::move(packet));
    cv_.notify_one();
    return true;
}

bool NetClient::ensure_connected() {
    if (sock_ != kInvalid && connected_) return true;
    if (sock_ != kInvalid) {
#ifdef _WIN32
        closesocket(static_cast<SOCKET>(sock_));
#else
        ::close(sock_);
#endif
        sock_ = kInvalid;
        connected_ = false;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string port = std::to_string(port_);
    if (getaddrinfo(host_.c_str(), port.c_str(), &hints, &res) != 0) return false;

    socket_t s = kInvalid;
    for (auto* p = res; p; p = p->ai_next) {
        s = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (s == kInvalid) continue;
        if (::connect(s, p->ai_addr, int(p->ai_addrlen)) == 0) break;
#ifdef _WIN32
        closesocket(s);
#else
        ::close(s);
#endif
        s = kInvalid;
    }
    freeaddrinfo(res);
    if (s == kInvalid) return false;
    sock_ = s;
    connected_ = true;
    return true;
}

bool NetClient::send_all(const uint8_t* data, size_t n) {
    size_t sent = 0;
    while (sent < n) {
#ifdef _WIN32
        const int r = ::send(static_cast<SOCKET>(sock_),
                             reinterpret_cast<const char*>(data + sent),
                             int(n - sent), 0);
#else
        const ssize_t r = ::send(sock_, data + sent, n - sent, 0);
#endif
        if (r <= 0) {
            connected_ = false;
            return false;
        }
        sent += size_t(r);
    }
    return true;
}

void NetClient::loop() {
    while (running_) {
        std::vector<uint8_t> pkt;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(200), [&] {
                return !queue_.empty() || !running_;
            });
            if (!running_ && queue_.empty()) break;
            if (queue_.empty()) continue;
            pkt = std::move(queue_.front());
            queue_.erase(queue_.begin());
        }
        if (!ensure_connected()) {
            // Re-queue at front so we don't silently lose a hdr.
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() < max_queue_) queue_.insert(queue_.begin(), std::move(pkt));
            else dropped_++;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }
        if (!send_all(pkt.data(), pkt.size())) {
            std::lock_guard<std::mutex> lock(mu_);
            if (queue_.size() < max_queue_) queue_.insert(queue_.begin(), std::move(pkt));
            else dropped_++;
        }
    }
}

}  // namespace aegis
