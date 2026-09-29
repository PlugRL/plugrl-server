// E44: a C++ program with no third-party libraries that trains a policy on
// its own environment.
//
// Everything up to the WebSocket class is plugrl-protocol's
// examples/plugrl_client.cpp at 3a59528 (the zero-dependency client of E2 and
// E7), unchanged except where marked "E44". After it: gymnasium's Pendulum-v1
// written out in C++, and a training loop that does what plugrl-env-client's
// rollout loop does with one env.
//
// Build:  g++ -std=c++17 -O2 -o pendulum_client pendulum_client.cpp
// Train:  ./pendulum_client HOST PORT SEED
// Check:  ./pendulum_client --check THETA THETA_DOT < actions   (no network)

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------- SHA-1
// RFC 3174. Needed only to verify the server's Sec-WebSocket-Accept.

class Sha1 {
 public:
  Sha1() { reset(); }

  void update(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buffer_[buffer_len_++] = data[i];
      if (buffer_len_ == 64) {
        transform(buffer_);
        total_ += 64;
        buffer_len_ = 0;
      }
    }
  }

  std::vector<uint8_t> digest() {
    uint64_t total_bits = (total_ + buffer_len_) * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    uint8_t zero = 0x00;
    while (buffer_len_ != 56) update(&zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[7 - i] = (total_bits >> (8 * i)) & 0xff;
    update(len_be, 8);

    std::vector<uint8_t> out(20);
    for (int i = 0; i < 5; ++i) {
      out[i * 4 + 0] = (h_[i] >> 24) & 0xff;
      out[i * 4 + 1] = (h_[i] >> 16) & 0xff;
      out[i * 4 + 2] = (h_[i] >> 8) & 0xff;
      out[i * 4 + 3] = h_[i] & 0xff;
    }
    return out;
  }

 private:
  void reset() {
    h_[0] = 0x67452301; h_[1] = 0xEFCDAB89; h_[2] = 0x98BADCFE;
    h_[3] = 0x10325476; h_[4] = 0xC3D2E1F0;
    buffer_len_ = 0; total_ = 0;
  }

  static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }

  void transform(const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (block[i * 4] << 24) | (block[i * 4 + 1] << 16) |
             (block[i * 4 + 2] << 8) | block[i * 4 + 3];
    for (int i = 16; i < 80; ++i)
      w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5A827999; }
      else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
      else             { f = b ^ c ^ d;                   k = 0xCA62C1D6; }
      uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e;
  }

  uint32_t h_[5];
  uint8_t buffer_[64];
  size_t buffer_len_;
  uint64_t total_;
};

std::string base64_encode(const uint8_t* data, size_t len) {
  static const char* tbl =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = data[i] << 16;
    if (i + 1 < len) n |= data[i + 1] << 8;
    if (i + 2 < len) n |= data[i + 2];
    out += tbl[(n >> 18) & 63];
    out += tbl[(n >> 12) & 63];
    out += (i + 1 < len) ? tbl[(n >> 6) & 63] : '=';
    out += (i + 2 < len) ? tbl[n & 63] : '=';
  }
  return out;
}

// ---------------------------------------------------------------- msgpack
//
// Only what the protocol uses: maps, strings, binary, arrays, integers,
// floats and booleans. No extension types - the wire format does not use any.

class Packer {
 public:
  void map(size_t n) {
    if (n < 16) put_u8(0x80 | static_cast<uint8_t>(n));
    else if (n < 65536) { put_u8(0xde); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdf); put_be32(static_cast<uint32_t>(n)); }
  }

  void array(size_t n) {
    if (n < 16) put_u8(0x90 | static_cast<uint8_t>(n));
    else if (n < 65536) { put_u8(0xdc); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdd); put_be32(static_cast<uint32_t>(n)); }
  }

  void str(const std::string& s) {
    size_t n = s.size();
    if (n < 32) put_u8(0xa0 | static_cast<uint8_t>(n));
    else if (n < 256) { put_u8(0xd9); put_u8(static_cast<uint8_t>(n)); }
    else if (n < 65536) { put_u8(0xda); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xdb); put_be32(static_cast<uint32_t>(n)); }
    buf_.append(s);
  }

  void bin(const uint8_t* data, size_t n) {
    if (n < 256) { put_u8(0xc4); put_u8(static_cast<uint8_t>(n)); }
    else if (n < 65536) { put_u8(0xc5); put_be16(static_cast<uint16_t>(n)); }
    else { put_u8(0xc6); put_be32(static_cast<uint32_t>(n)); }
    buf_.append(reinterpret_cast<const char*>(data), n);
  }

  void bin(const std::string& s) {
    bin(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }

  void integer(int64_t v) {
    if (v >= 0 && v < 128) { put_u8(static_cast<uint8_t>(v)); return; }
    if (v < 0 && v >= -32) { put_u8(static_cast<uint8_t>(0xe0 | (v + 32))); return; }
    put_u8(0xd3);
    for (int i = 7; i >= 0; --i) put_u8((static_cast<uint64_t>(v) >> (8 * i)) & 0xff);
  }

  void boolean(bool v) { put_u8(v ? 0xc3 : 0xc2); }

  const std::string& data() const { return buf_; }
  void clear() { buf_.clear(); }

 private:
  void put_u8(uint8_t v) { buf_.push_back(static_cast<char>(v)); }
  void put_be16(uint16_t v) { put_u8(v >> 8); put_u8(v & 0xff); }
  void put_be32(uint32_t v) {
    put_u8(v >> 24); put_u8((v >> 16) & 0xff); put_u8((v >> 8) & 0xff); put_u8(v & 0xff);
  }
  std::string buf_;
};

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
  enum class Kind { Nil, Bool, Int, UInt, Float, Str, Bin, Array, Map };
  Kind kind = Kind::Nil;
  bool b = false;
  int64_t i = 0;
  uint64_t u = 0;
  double f = 0;
  std::string s;              // Str and Bin both land here
  std::vector<ValuePtr> arr;
  std::vector<std::pair<ValuePtr, ValuePtr>> map;

  const ValuePtr find(const std::string& key) const {
    for (const auto& kv : map)
      if ((kv.first->kind == Kind::Str || kv.first->kind == Kind::Bin) &&
          kv.first->s == key)
        return kv.second;
    return nullptr;
  }
};

class Unpacker {
 public:
  Unpacker(const uint8_t* data, size_t len) : p_(data), end_(data + len) {}

  ValuePtr parse() {
    auto v = std::make_shared<Value>();
    uint8_t t = take_u8();

    if (t <= 0x7f) { v->kind = Value::Kind::UInt; v->u = t; return v; }
    if (t >= 0xe0) { v->kind = Value::Kind::Int;
                     v->i = static_cast<int8_t>(t); return v; }
    if ((t & 0xf0) == 0x80) return parse_map(v, t & 0x0f);
    if ((t & 0xf0) == 0x90) return parse_array(v, t & 0x0f);
    if ((t & 0xe0) == 0xa0) return parse_str(v, t & 0x1f);

    switch (t) {
      case 0xc0: v->kind = Value::Kind::Nil; return v;
      case 0xc2: v->kind = Value::Kind::Bool; v->b = false; return v;
      case 0xc3: v->kind = Value::Kind::Bool; v->b = true; return v;
      case 0xc4: return parse_bin(v, take_u8());
      case 0xc5: return parse_bin(v, take_be16());
      case 0xc6: return parse_bin(v, take_be32());
      case 0xca: { v->kind = Value::Kind::Float;
                   uint32_t r = take_be32(); float g;
                   std::memcpy(&g, &r, 4); v->f = g; return v; }
      case 0xcb: { v->kind = Value::Kind::Float;
                   uint64_t r = take_be64(); double g;
                   std::memcpy(&g, &r, 8); v->f = g; return v; }
      case 0xcc: v->kind = Value::Kind::UInt; v->u = take_u8(); return v;
      case 0xcd: v->kind = Value::Kind::UInt; v->u = take_be16(); return v;
      case 0xce: v->kind = Value::Kind::UInt; v->u = take_be32(); return v;
      case 0xcf: v->kind = Value::Kind::UInt; v->u = take_be64(); return v;
      case 0xd0: v->kind = Value::Kind::Int;
                 v->i = static_cast<int8_t>(take_u8()); return v;
      case 0xd1: v->kind = Value::Kind::Int;
                 v->i = static_cast<int16_t>(take_be16()); return v;
      case 0xd2: v->kind = Value::Kind::Int;
                 v->i = static_cast<int32_t>(take_be32()); return v;
      case 0xd3: v->kind = Value::Kind::Int;
                 v->i = static_cast<int64_t>(take_be64()); return v;
      case 0xd9: return parse_str(v, take_u8());
      case 0xda: return parse_str(v, take_be16());
      case 0xdb: return parse_str(v, take_be32());
      case 0xdc: return parse_array(v, take_be16());
      case 0xdd: return parse_array(v, take_be32());
      case 0xde: return parse_map(v, take_be16());
      case 0xdf: return parse_map(v, take_be32());
      default:
        throw std::runtime_error("unsupported msgpack type 0x" +
                                 std::to_string(static_cast<int>(t)));
    }
  }

 private:
  ValuePtr parse_str(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Str; v->s.assign(take(n), n); return v;
  }
  ValuePtr parse_bin(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Bin; v->s.assign(take(n), n); return v;
  }
  ValuePtr parse_array(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Array;
    for (size_t k = 0; k < n; ++k) v->arr.push_back(parse());
    return v;
  }
  ValuePtr parse_map(ValuePtr v, size_t n) {
    v->kind = Value::Kind::Map;
    for (size_t k = 0; k < n; ++k) {
      auto key = parse();
      auto val = parse();
      v->map.emplace_back(key, val);
    }
    return v;
  }

  const char* take(size_t n) {
    if (p_ + n > end_) throw std::runtime_error("msgpack: truncated");
    const char* r = reinterpret_cast<const char*>(p_);
    p_ += n;
    return r;
  }
  uint8_t take_u8() { return static_cast<uint8_t>(*take(1)); }
  uint16_t take_be16() { uint16_t a = take_u8(); return (a << 8) | take_u8(); }
  uint32_t take_be32() { uint32_t a = take_be16(); return (a << 16) | take_be16(); }
  uint64_t take_be64() { uint64_t a = take_be32(); return (a << 32) | take_be32(); }

  const uint8_t* p_;
  const uint8_t* end_;
};

// ---------------------------------------------------------------- ndarray
//
// {b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [...]}
//
// dtype is a numpy typestr: byte order, kind, item size. The only piece of
// numpy vocabulary in the protocol, and the reason this function exists.

void pack_ndarray(Packer& p, const std::string& raw, const std::string& dtype,
                  const std::vector<int64_t>& shape) {
  p.map(4);
  p.bin(std::string("__ndarray__")); p.boolean(true);
  p.bin(std::string("data"));        p.bin(raw);
  p.bin(std::string("dtype"));       p.str(dtype);
  p.bin(std::string("shape"));
  p.array(shape.size());
  for (int64_t d : shape) p.integer(d);
}

// A numpy typestr: byte order, kind, decimal item size in bytes.
// SPEC section 3.3.

struct TypeStr {
  char order;  // '<' or '>' after normalisation
  char kind;   // 'b', 'u', 'i', 'f', 'U'
  int size;    // bytes per element
};

TypeStr parse_typestr(const std::string& s) {
  if (s.size() < 3) throw std::runtime_error("malformed typestr: " + s);
  TypeStr t;
  t.order = (s[0] == '|') ? '<' : s[0];  // '|' means single-byte, so moot
  t.kind = s[1];
  t.size = std::stoi(s.substr(2));
  if (t.size <= 0) throw std::runtime_error("malformed typestr: " + s);
  return t;
}

// Read one element as a double, whatever integer or float type it is on the
// wire. The point is not generality for its own sake: SPEC section 8 asks a
// client to parse the typestr rather than assume a dtype, because the action
// dtype is the environment's and is never renegotiated.
double read_element(const std::string& raw, size_t index, const TypeStr& t) {
  size_t off = index * static_cast<size_t>(t.size);
  if (off + static_cast<size_t>(t.size) > raw.size())
    throw std::runtime_error("ndarray data shorter than shape implies");

  uint64_t bits = 0;
  for (int i = 0; i < t.size; ++i) {
    // Little-endian: byte i is the i-th least significant.
    int byte = (t.order == '<') ? i : (t.size - 1 - i);
    bits |= static_cast<uint64_t>(static_cast<uint8_t>(raw[off + byte]))
            << (8 * i);
  }

  switch (t.kind) {
    case 'f':
      if (t.size == 4) {
        float f;
        uint32_t narrow = static_cast<uint32_t>(bits);
        std::memcpy(&f, &narrow, 4);
        return f;
      }
      if (t.size == 8) {
        double d;
        std::memcpy(&d, &bits, 8);
        return d;
      }
      break;
    case 'b':
      return bits ? 1.0 : 0.0;
    case 'u':
      return static_cast<double>(bits);
    case 'i': {
      // Sign-extend from the declared width.
      int shift = 64 - 8 * t.size;
      return static_cast<double>(static_cast<int64_t>(bits << shift) >> shift);
    }
    default:
      break;
  }
  throw std::runtime_error("unsupported dtype kind/size in typestr");
}

// The typestr says "<f8", "<f4" and "<i8", so the bytes must be
// little-endian whatever this machine is. A memcpy from the host layout is
// right on x86 and silently wrong on a big-endian controller - and the
// receiver cannot tell, because the declared byte order still says little.

void put_le(std::string& out, uint64_t bits, int bytes) {
  for (int i = 0; i < bytes; ++i)
    out.push_back(static_cast<char>((bits >> (8 * i)) & 0xff));
}

std::string pack_f8(const std::vector<double>& v) {
  std::string out;
  out.reserve(v.size() * 8);
  for (double x : v) {
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    put_le(out, bits, 8);
  }
  return out;
}

std::string pack_f4(const std::vector<float>& v) {
  std::string out;
  out.reserve(v.size() * 4);
  for (float x : v) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    put_le(out, bits, 4);
  }
  return out;
}

std::string pack_i8(const std::vector<int64_t>& v) {
  std::string out;
  out.reserve(v.size() * 8);
  for (int64_t x : v) put_le(out, static_cast<uint64_t>(x), 8);
  return out;
}

// E44: the server ends a run by closing with a reason - "plugrl-server-stop"
// when training is over, "plugrl-server-resync" when the client should
// reconnect and ask again - so a close has to carry it.
struct ServerClosed : std::runtime_error {
  explicit ServerClosed(const std::string& r)
      : std::runtime_error("server closed the connection: " + r), reason(r) {}
  std::string reason;
};

// ---------------------------------------------------------------- websocket

class WebSocket {
 public:
  void connect(const std::string& host, int port) {
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0)
      throw std::runtime_error("cannot resolve " + host);
    fd_ = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd_ < 0 || ::connect(fd_, res->ai_addr, res->ai_addrlen) != 0) {
      freeaddrinfo(res);
      throw std::runtime_error("cannot connect to " + host);
    }
    freeaddrinfo(res);
    int one = 1;
    setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    handshake(host, port);
  }

  // RFC 6455 section 5.5.1: say goodbye before dropping the socket. Without
  // this the server sees the connection vanish and reports
  // `ConnectionClosedError: no close frame received or sent` - which is what
  // it did on all 45 runs of E7, where the client finishes first. It only
  // went unnoticed before because in every earlier test the *server* ran out
  // of steps first and closed the connection itself.
  void close_cleanly() {
    if (fd_ < 0) return;
    std::string frame;
    frame.push_back(static_cast<char>(0x88));  // FIN + close opcode
    frame.push_back(static_cast<char>(0x80 | 2));  // masked, 2-byte payload
    uint8_t key[4];
    for (int i = 0; i < 4; ++i) key[i] = static_cast<uint8_t>(rng_() & 0xff);
    frame.append(reinterpret_cast<char*>(key), 4);
    uint8_t status[2] = {0x03, 0xe8};  // 1000, normal closure
    for (int i = 0; i < 2; ++i)
      frame.push_back(static_cast<char>(status[i] ^ key[i % 4]));
    try {
      write_all(frame.data(), frame.size());
    } catch (const std::exception&) {
      // Already gone. Nothing useful to do on the way out.
    }
    ::close(fd_);
    fd_ = -1;
  }

  ~WebSocket() {
    close_cleanly();
    if (fd_ >= 0) ::close(fd_);
  }

  void send_binary(const std::string& payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x82));  // FIN + binary opcode
    size_t n = payload.size();
    uint8_t mask_bit = 0x80;                   // clients must mask
    if (n < 126) {
      frame.push_back(static_cast<char>(mask_bit | n));
    } else if (n < 65536) {
      frame.push_back(static_cast<char>(mask_bit | 126));
      frame.push_back(static_cast<char>((n >> 8) & 0xff));
      frame.push_back(static_cast<char>(n & 0xff));
    } else {
      frame.push_back(static_cast<char>(mask_bit | 127));
      for (int i = 7; i >= 0; --i)
        frame.push_back(static_cast<char>((n >> (8 * i)) & 0xff));
    }
    uint8_t key[4];
    for (int i = 0; i < 4; ++i) key[i] = static_cast<uint8_t>(rng_() & 0xff);
    frame.append(reinterpret_cast<char*>(key), 4);
    size_t off = frame.size();
    frame.append(payload);
    for (size_t i = 0; i < n; ++i) frame[off + i] ^= key[i % 4];
    write_all(frame.data(), frame.size());
  }

  // The largest message this client will accept. Observations with two
  // cameras run to a few hundred KiB; 256 MiB is far above anything the
  // protocol produces and far below what a hostile length field could ask
  // us to allocate.
  static constexpr uint64_t kMaxMessageBytes = 256ull << 20;

  std::string recv_message() {
    std::string message;
    bool first_data_frame = true;
    for (;;) {
      uint8_t h[2];
      read_all(h, 2);
      bool fin = h[0] & 0x80;
      uint8_t opcode = h[0] & 0x0f;
      bool masked = h[1] & 0x80;
      uint64_t len = h[1] & 0x7f;
      if (len == 126) {
        uint8_t e[2]; read_all(e, 2);
        len = (static_cast<uint64_t>(e[0]) << 8) | e[1];
      } else if (len == 127) {
        uint8_t e[8]; read_all(e, 8);
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
      }
      if (len > kMaxMessageBytes || message.size() + len > kMaxMessageBytes)
        throw std::runtime_error("frame larger than this client will accept");
      uint8_t key[4] = {0, 0, 0, 0};
      if (masked) read_all(key, 4);

      std::string chunk(len, '\0');
      if (len) read_all(reinterpret_cast<uint8_t*>(&chunk[0]), len);
      if (masked)
        for (uint64_t i = 0; i < len; ++i) chunk[i] ^= key[i % 4];

      // E44: RFC 6455 section 5.5.1 - a 2-byte status, then the reason.
      if (opcode == 0x8) throw ServerClosed(chunk.size() > 2 ? chunk.substr(2) : std::string());
      if (opcode == 0x9) { send_pong(chunk); continue; }
      if (opcode == 0xa) continue;

      // SPEC section 7.4: every protocol message is binary. A text frame
      // where one was expected means the peer is reporting an error, not
      // speaking the protocol - openpi servers send a traceback this way.
      // Unpacking it as msgpack would turn a legible error into "unsupported
      // msgpack type 0x47".
      if (first_data_frame) {
        if (opcode == 0x1)
          throw std::runtime_error("server sent a text frame: " +
                                   chunk.substr(0, 2048));
        if (opcode != 0x2)
          throw std::runtime_error("unexpected websocket opcode " +
                                   std::to_string(static_cast<int>(opcode)));
        first_data_frame = false;
      }

      message += chunk;
      if (fin) return message;
    }
  }

 private:
  void handshake(const std::string& host, int port) {
    uint8_t nonce[16];
    for (int i = 0; i < 16; ++i) nonce[i] = static_cast<uint8_t>(rng_() & 0xff);
    std::string key = base64_encode(nonce, 16);

    std::string req =
        "GET / HTTP/1.1\r\n"
        "Host: " + host + ":" + std::to_string(port) + "\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: " + key + "\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n";
    write_all(req.data(), req.size());

    std::string resp;
    while (resp.find("\r\n\r\n") == std::string::npos) {
      char c;
      read_all(reinterpret_cast<uint8_t*>(&c), 1);
      resp.push_back(c);
      if (resp.size() > 8192) throw std::runtime_error("handshake response too long");
    }
    if (resp.find("101") == std::string::npos)
      throw std::runtime_error("server refused the upgrade:\n" + resp);

    // Verify Sec-WebSocket-Accept: base64(sha1(key + GUID)).
    const std::string guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    Sha1 sha;
    std::string concat = key + guid;
    sha.update(reinterpret_cast<const uint8_t*>(concat.data()), concat.size());
    auto d = sha.digest();
    std::string expect = base64_encode(d.data(), d.size());
    if (resp.find(expect) == std::string::npos)
      throw std::runtime_error("Sec-WebSocket-Accept mismatch; expected " + expect);
  }

  void send_pong(const std::string& payload) {
    std::string frame;
    frame.push_back(static_cast<char>(0x8a));
    frame.push_back(static_cast<char>(0x80 | payload.size()));
    uint8_t key[4] = {0, 0, 0, 0};
    frame.append(reinterpret_cast<char*>(key), 4);
    frame.append(payload);
    write_all(frame.data(), frame.size());
  }

  void write_all(const char* data, size_t n) {
    size_t sent = 0;
    while (sent < n) {
      ssize_t k = ::send(fd_, data + sent, n - sent, 0);
      if (k <= 0) throw std::runtime_error("send failed");
      sent += static_cast<size_t>(k);
    }
  }

  void read_all(uint8_t* data, size_t n) {
    size_t got = 0;
    while (got < n) {
      ssize_t k = ::recv(fd_, data + got, n - got, 0);
      if (k <= 0) throw std::runtime_error("connection closed while reading");
      got += static_cast<size_t>(k);
    }
  }

  int fd_ = -1;
  std::mt19937 rng_{std::random_device{}()};
};

// ---------------------------------------------------------------- pendulum
//
// gymnasium's Pendulum-v1 (classic_control/pendulum.py, gymnasium 1.3.0),
// written out: the same constants, the same order of operations, and the same
// precision. The torque arrives as float32, and under NumPy 2 a Python float
// times a float32 stays float32, so u**2 * 0.001 and 3.0 * u are float32
// products; everything else is float64. The 200-step limit is the TimeLimit
// wrapper `gym.make` adds.

constexpr double kPi = 3.14159265358979323846;

// NumPy's %, whose result takes the divisor's sign - not C's fmod.
double angle_normalize(double x) {
  const double two_pi = 2 * kPi;
  double r = std::fmod(x + kPi, two_pi);
  if (r != 0 && r < 0) r += two_pi;
  return r - kPi;
}

struct Pendulum {
  static constexpr double kMaxSpeed = 8.0;
  static constexpr double kDt = 0.05;
  static constexpr double kG = 10.0;
  static constexpr double kM = 1.0;
  static constexpr double kL = 1.0;
  static constexpr float kMaxTorque = 2.0f;
  static constexpr int kMaxSteps = 200;

  double th = 0.0;
  double thdot = 0.0;
  int t = 0;

  // gymnasium draws theta from U(-pi, pi) and its rate from U(-1, 1). The
  // generator is this program's own; the distribution is gymnasium's.
  void reset(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> angle(-kPi, kPi), rate(-1.0, 1.0);
    th = angle(rng);
    thdot = rate(rng);
    t = 0;
  }

  std::vector<float> obs() const {
    return {static_cast<float>(std::cos(th)), static_cast<float>(std::sin(th)),
            static_cast<float>(thdot)};
  }

  // One step. Returns the reward; sets `truncated` when the time limit ends
  // the episode. Pendulum never terminates.
  double step(float action, bool& truncated) {
    float u = std::min(std::max(action, -kMaxTorque), kMaxTorque);
    float u_cost = 0.001f * (u * u);
    double an = angle_normalize(th);
    double costs = an * an + 0.1 * (thdot * thdot) + static_cast<double>(u_cost);
    float torque = static_cast<float>(3.0 / (kM * kL * kL)) * u;
    double newthdot =
        thdot + (3 * kG / (2 * kL) * std::sin(th) + static_cast<double>(torque)) * kDt;
    newthdot = std::min(std::max(newthdot, -kMaxSpeed), kMaxSpeed);
    th = th + newthdot * kDt;
    thdot = newthdot;
    ++t;
    truncated = t >= kMaxSteps;
    return -costs;
  }
};

// ---------------------------------------------------------------- client

const char* kStopReason = "plugrl-server-stop";

// The Observation dataclass on the wire, with Pendulum's state under the key
// gaussian-policy reads by default.
void pack_obs(Packer& p, const std::vector<float>& o) {
  p.map(3);
  p.str("images");
  p.map(0);
  p.str("states");
  p.map(1);
  p.str("obs");
  pack_ndarray(p, pack_f4(o), "<f4", {1, static_cast<int64_t>(o.size())});
  p.str("text");
  p.array(1);
  p.str("Pendulum-v1");
}

std::unique_ptr<WebSocket> open(const std::string& host, int port) {
  for (int attempt = 0;; ++attempt) {
    try {
      auto ws = std::make_unique<WebSocket>();
      ws->connect(host, port);
      auto raw = ws->recv_message();
      Unpacker up(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
      auto meta = up.parse();
      auto mt = meta->find("message_type");
      if (!mt || mt->s != "metadata") throw std::runtime_error("expected metadata");
      return ws;
    } catch (const std::exception& e) {
      if (attempt >= 60) throw;
      std::cerr << "waiting for the server: " << e.what() << "\n";
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  }
}

// The same exchange as plugrl-env-client's rollout loop with one env and
// replan 1: infer on the current observation, step with the first action,
// feed back the observation the step returned (before any reset), with the
// step's reward and flags; the step id counts from 0 within an episode.
int train(const std::string& host, int port, uint64_t seed) {
  std::mt19937_64 rng(seed);
  Pendulum env;
  env.reset(rng);
  const std::vector<int64_t> env_idx{0};
  int64_t step_id = 0;
  long long steps = 0;
  double episode_return = 0.0;
  std::vector<double> returns;

  auto ws = open(host, port);
  std::cout << "connected to ws://" << host << ":" << port << ", seed " << seed << "\n";

  for (;;) {
    Packer p;
    p.map(4);
    p.str("message_type"); p.str("infer");
    p.str("data");         pack_obs(p, env.obs());
    p.str("env_indices");  pack_ndarray(p, pack_i8(env_idx), "<i8", {1});
    p.str("step_ids");     pack_ndarray(p, pack_i8({step_id}), "<i8", {1});

    std::string reply;
    try {
      ws->send_binary(p.data());
      reply = ws->recv_message();
    } catch (const ServerClosed& e) {
      if (e.reason == kStopReason) break;  // the run is over
      // A resync, or any other orderly close: reconnect and ask again, as
      // plugrl-env-client does.
      ws = open(host, port);
      continue;
    }

    Unpacker up(reinterpret_cast<const uint8_t*>(reply.data()), reply.size());
    auto msg = up.parse();
    auto type = msg->find("message_type");
    if (!type || type->s != "action") throw std::runtime_error("expected an action");
    auto data = msg->find("data");
    auto action = data ? data->find("action") : nullptr;
    if (!action) throw std::runtime_error("no action field");
    auto dtype = action->find("dtype");
    auto blob = action->find("data");
    if (!dtype || !blob) throw std::runtime_error("malformed action ndarray");
    // [H, n, 1]: the first step of the chunk, for the one env.
    float a = static_cast<float>(read_element(blob->s, 0, parse_typestr(dtype->s)));

    bool truncated = false;
    double r = env.step(a, truncated);
    episode_return += r;
    ++steps;

    Packer fb;
    fb.map(4);
    fb.str("message_type"); fb.str("feedback");
    fb.str("env_indices");  pack_ndarray(fb, pack_i8(env_idx), "<i8", {1});
    fb.str("step_ids");     pack_ndarray(fb, pack_i8({step_id}), "<i8", {1});
    fb.str("data");
    fb.map(5);
    fb.str("obs");        pack_obs(fb, env.obs());
    fb.str("rewards");    pack_ndarray(fb, pack_f4({static_cast<float>(r)}), "<f4", {1});
    fb.str("terminated"); pack_ndarray(fb, std::string(1, '\0'), "|b1", {1});
    fb.str("truncated");  pack_ndarray(fb, std::string(1, truncated ? '\1' : '\0'), "|b1", {1});
    fb.str("info");
    if (truncated) {
      // The episode's return and length, as plugrl-env-client's
      // VectorEpisodeStatsWrapper reports them in info["episode"] - the
      // server's episode metrics read nothing else. SPEC.md does not
      // document this; E44's pilot found it (a spec-only client reported no
      // episodes at all). Pendulum defines no success, so s is false.
      fb.map(1);
      fb.str("episode");
      fb.map(4);
      fb.str("r");    pack_ndarray(fb, pack_f4({static_cast<float>(episode_return)}), "<f4", {1});
      fb.str("l");    pack_ndarray(fb, pack_i8({static_cast<int64_t>(env.t)}), "<i8", {1});
      fb.str("s");    pack_ndarray(fb, std::string(1, '\0'), "|b1", {1});
      fb.str("mask"); pack_ndarray(fb, std::string(1, '\1'), "|b1", {1});
    } else {
      fb.map(0);
    }
    ws->send_binary(fb.data());
    ++step_id;

    if (truncated) {
      returns.push_back(episode_return);
      if (returns.size() % 100 == 0)
        std::cout << "episode " << returns.size() << "  return " << episode_return << "\n";
      episode_return = 0.0;
      env.reset(rng);
      step_id = 0;
    }
  }

  auto mean = [](std::vector<double>::const_iterator a, std::vector<double>::const_iterator b) {
    return b > a ? std::accumulate(a, b, 0.0) / static_cast<double>(b - a) : 0.0;
  };
  size_t n = returns.size();
  size_t k = std::min<size_t>(10, n);
  std::cout << "server stopped the run after " << steps << " steps, " << n << " episodes\n";
  std::cout << "mean return, first " << k << " episodes: " << mean(returns.begin(), returns.begin() + k)
            << "  last " << k << ": " << mean(returns.end() - k, returns.end()) << "\n";
  return 0;
}

// No network: start from a given state, apply actions read from stdin, and
// print each step at full precision, for comparison with gymnasium.
int check(double th, double thdot) {
  Pendulum env;
  env.th = th;
  env.thdot = thdot;
  float a;
  std::cout << std::setprecision(17);
  while (std::cin >> a) {
    bool truncated = false;
    double r = env.step(a, truncated);
    auto o = env.obs();
    std::cout << o[0] << " " << o[1] << " " << o[2] << " " << r << " " << truncated << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  try {
    if (argc == 4 && std::string(argv[1]) == "--check")
      return check(std::stod(argv[2]), std::stod(argv[3]));
    if (argc != 4) {
      std::cerr << "usage: pendulum_client HOST PORT SEED\n"
                << "       pendulum_client --check THETA THETA_DOT < actions\n";
      return 2;
    }
    return train(argv[1], std::stoi(argv[2]), std::stoull(argv[3]));
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
