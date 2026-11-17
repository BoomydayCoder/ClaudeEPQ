# ClaudeEPQ build. Plain make, Apple clang, no dependencies.
#
#   make            optimised build        -> build/epq
#   make debug      ASan + UBSan build     -> build/epq-asan
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

.PHONY: all debug clean test-frontend
all: build/epq

debug: build/epq-asan

build/epq: $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(OPT) $(LDFLAGS) -o $@ $(OBJS)

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

test-frontend: build/epq
	python3 tests/frontend/run_tests.py

clean:
	rm -rf build/obj build/obj-asan build/gen build/epq build/epq-asan

-include $(DEPS)
