// Shared front-end utilities: bump arena, identifier interner, source buffer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <string_view>
#include <vector>

namespace epq {

// ---------------------------------------------------------------------------
// Arena: bump allocator. Everything allocated here lives until the arena dies;
// destructors are NOT run, so only trivially-destructible types go in it.
// ---------------------------------------------------------------------------
class Arena {
public:
    Arena() = default;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() {
        for (char* c : chunks_) ::operator delete(c);
    }

    void* alloc(size_t size, size_t align = alignof(std::max_align_t)) {
        uintptr_t p = (reinterpret_cast<uintptr_t>(cur_) + (align - 1)) & ~(uintptr_t)(align - 1);
        if (p + size > reinterpret_cast<uintptr_t>(end_)) {
            grow(size + align);
            p = (reinterpret_cast<uintptr_t>(cur_) + (align - 1)) & ~(uintptr_t)(align - 1);
        }
        cur_ = reinterpret_cast<char*>(p + size);
        return reinterpret_cast<void*>(p);
    }

    template <class T>
    T* make() {
        static_assert(std::is_trivially_destructible_v<T>);
        return new (alloc(sizeof(T), alignof(T))) T();
    }

    template <class T>
    T* array(size_t n) {
        static_assert(std::is_trivially_destructible_v<T>);
        if (n == 0) return nullptr;
        T* p = static_cast<T*>(alloc(sizeof(T) * n, alignof(T)));
        for (size_t i = 0; i < n; ++i) new (p + i) T();
        return p;
    }

private:
    void grow(size_t need) {
        size_t sz = next_size_;
        if (sz < need) sz = need;
        if (next_size_ < (size_t(1) << 24)) next_size_ *= 2;
        char* c = static_cast<char*>(::operator new(sz));
        chunks_.push_back(c);
        cur_ = c;
        end_ = c + sz;
    }
    std::vector<char*> chunks_;
    char* cur_ = nullptr;
    char* end_ = nullptr;
    size_t next_size_ = 4 * 1024;
};

// ---------------------------------------------------------------------------
// Interned identifiers. Sym is a dense index (0..count-1) into the table, so
// later stages can use plain vectors keyed by Sym.
// ---------------------------------------------------------------------------
using Sym = uint32_t;
constexpr Sym kNoSym = 0xffffffffu;

class Interner {
public:
    Interner() { table_.assign(256, kNoSym); }

    Sym intern(std::string_view s) {
        uint32_t h = hash(s);
        size_t mask = table_.size() - 1;
        size_t i = h & mask;
        for (;;) {
            Sym e = table_[i];
            if (e == kNoSym) break;
            if (hashes_[e] == h && names_[e] == s) return e;
            i = (i + 1) & mask;
        }
        // Copy the characters into our own storage so the view stays valid.
        char* mem = static_cast<char*>(arena_.alloc(s.size() + 1, 1));
        std::memcpy(mem, s.data(), s.size());
        mem[s.size()] = 0;
        Sym id = (Sym)names_.size();
        names_.emplace_back(mem, s.size());
        hashes_.push_back(h);
        table_[i] = id;
        if (names_.size() * 2 > table_.size()) rehash();
        return id;
    }

    std::string_view name(Sym s) const { return names_[s]; }
    size_t size() const { return names_.size(); }

private:
    static uint32_t hash(std::string_view s) {
        uint32_t h = 2166136261u;  // FNV-1a
        for (unsigned char c : s) h = (h ^ c) * 16777619u;
        return h;
    }
    void rehash() {
        std::vector<Sym> t(table_.size() * 2, kNoSym);
        size_t mask = t.size() - 1;
        for (Sym id = 0; id < names_.size(); ++id) {
            size_t i = hashes_[id] & mask;
            while (t[i] != kNoSym) i = (i + 1) & mask;
            t[i] = id;
        }
        table_.swap(t);
    }
    Arena arena_;
    std::vector<std::string_view> names_;
    std::vector<uint32_t> hashes_;
    std::vector<Sym> table_;
};

// Source position, 1-based, as bison's yy::position counts them (a tab is one column).
struct Pos {
    uint32_t line = 1;
    uint32_t col = 1;
};

// A location in bison's sense: [begin, end) where end is one past the last column.
struct Loc {
    Pos begin, end;
};

// Formats a location exactly like bison 2.4's `operator<<(ostream&, const location&)`:
// "FILE:L.C", "FILE:L.C-C2" or "FILE:L.C-L2.C2" (C2 = end column - 1, clamped to 1).
std::string format_loc(std::string_view file, const Loc& loc);

}  // namespace epq
