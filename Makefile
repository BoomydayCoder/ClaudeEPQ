# ClaudeEPQ build. Plain make, Apple clang, no dependencies.
#
#   make            optimised build        -> build/epq
#   make debug      ASan + UBSan build     -> build/epq-asan
#   make check      quick self-test (examples, conformance, core; needs nothing else)
#   make test-frontend   parser/resolver differential tests against the baseline
#   make clean
#
# Every src/*.cpp and src/*/*.cpp is compiled in, so new modules (inference,
# TIR, back ends) only need to drop their files under src/. Replace
# runtime/*.c is compiled into the binary as well (shared runtime).

CXX      := clang++
CXXSTD   := -std=c++20
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
OPT      := -O2
CXXFLAGS ?=
LDFLAGS  ?=
# Drop unreferenced code: fewer pages to map / fault at start-up. The host
# runtime (runtime/epq_rt_host.c) marks its entry points `used`, so code that
# reaches them only by address or by name still finds them.
LDOPT    := -Wl,-dead_strip
# Start-up (measured, docs/DESIGN.md §1). (1) No weak exports: a weak
# definition exported by the executable (libc++'s std::piecewise_construct)
# makes dyld coalesce it against every loaded image, ~0.25 ms per run.
# (2) Bind directly to the libSystem sub-libraries the binary uses: a symbol
# reached through libSystem's ~40 re-exports costs dyld a re-export search per
# bind, ~4 M instructions (~0.2 ms) for our ~260 imports; same dylibs, same
# shared-cache code, only the lookup path changes. Likewise libc++abi.
SDKLIB   := $(shell xcrun --show-sdk-path 2>/dev/null)/usr/lib
SUBLIBS  := $(wildcard $(addprefix $(SDKLIB)/system/lib,$(addsuffix .tbd,system_c system_kernel system_pthread \
              system_malloc system_platform dyld system_info compiler_rt macho system_m unwind)))
LDOPT    += -Wl,-unexported_symbol,'__ZNSt3__1*' $(SUBLIBS) $(wildcard $(SDKLIB)/libc++abi.tbd)

SRCS     := $(wildcard src/*.cpp src/*/*.cpp)
CSRCS    := $(wildcard runtime/*.c)
OBJDIR   := build/obj
ASANDIR  := build/obj-asan
OBJS     := $(SRCS:%.cpp=$(OBJDIR)/%.o) $(CSRCS:%.c=$(OBJDIR)/%.o)
ASANOBJS := $(SRCS:%.cpp=$(ASANDIR)/%.o) $(CSRCS:%.c=$(ASANDIR)/%.o)
GENDIR   := build/gen
# The runtime source, embedded into the C backend's output.
RTINC    := $(GENDIR)/epq_rt_src.inc
DEPS     := $(OBJS:.o=.d) $(ASANOBJS:.o=.d)

SANFLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined

.PHONY: all debug clean check test-frontend
all: build/epq

debug: build/epq-asan

build/epq: $(OBJS) Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(OPT) $(LDOPT) $(LDFLAGS) -o $@ $(OBJS)

build/epq-asan: $(ASANOBJS)
	@mkdir -p $(dir $@)
	$(CXX) -g $(SANFLAGS) $(LDFLAGS) -o $@ $(ASANOBJS)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) -I$(GENDIR) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(ASANDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) -O1 -g $(SANFLAGS) $(WARN) -I$(GENDIR) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	clang -std=c11 $(OPT) -Wall -Wextra -MMD -MP -c $< -o $@

$(ASANDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	clang -std=c11 -O1 -g $(SANFLAGS) -Wall -Wextra -MMD -MP -c $< -o $@

$(RTINC): runtime/epq_rt.h
	@mkdir -p $(dir $@)
	cd runtime && xxd -i epq_rt.h > ../$@.tmp && mv ../$@.tmp ../$@ && sed -i '' 's/^unsigned/static const unsigned/' ../$@

$(OBJDIR)/src/cgen.o $(ASANDIR)/src/cgen.o: $(RTINC)

check: build/epq
	@python3 tests/check.py

test-frontend: build/epq
	python3 tests/frontend/run_tests.py

clean:
	rm -rf build/obj build/obj-asan build/gen build/epq build/epq-asan

-include $(DEPS)
