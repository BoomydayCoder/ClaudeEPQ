# ClaudeEPQ build. Plain make, Apple clang, no dependencies.
#
#   make            optimised build        -> build/epq
#   make debug      ASan + UBSan build     -> build/epq-asan
#   make test-frontend   parser/resolver differential tests against the baseline
#   make clean
#
# Every src/*.cpp and src/*/*.cpp is compiled in, so new modules (inference,
# TIR, back ends) only need to drop their files under src/. Replace
# src/backend_stub.cpp with the real run_backend().

CXX      := clang++
CXXSTD   := -std=c++20
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
OPT      := -O2
CXXFLAGS ?=
LDFLAGS  ?=

SRCS     := $(wildcard src/*.cpp src/*/*.cpp)
OBJDIR   := build/obj
ASANDIR  := build/obj-asan
OBJS     := $(SRCS:%.cpp=$(OBJDIR)/%.o)
ASANOBJS := $(SRCS:%.cpp=$(ASANDIR)/%.o)
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
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(ASANDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) -O1 -g $(SANFLAGS) $(WARN) $(CXXFLAGS) -MMD -MP -c $< -o $@

test-frontend: build/epq
	python3 tests/frontend/run_tests.py

clean:
	rm -rf build/obj build/obj-asan build/epq build/epq-asan

-include $(DEPS)
