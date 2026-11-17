// Pass registry and pass manager (DESIGN.md §5). The pipeline, in order:
//   definite-assignment, constants, demote-globals   global-variable passes (§2.6)
//   fold                                              constant folding (tir.cpp)
//   region-info                                       analysis (§5.2 inputs)
//   simplify, inline, pairs, ifconv, listopt, simplify2, reclaim   optimisation passes
// --no-pass=a,b / EPQ_NO_PASS=a,b disable passes (any of the above);
// --only-pass=a,b runs only the named optimisation passes (the infrastructure
// passes above them still run unless disabled). --verify-tir / EPQ_VERIFY_TIR=1
// runs tir::verify after TIR construction and after every pass;
// --dump-tir-after=NAME dumps the module after that pass and stops.
#include <cstdio>
#include <cstring>
#include <iterator>

#include "passes.hpp"

namespace epq::tir {

namespace {

const Pass kPipeline[] = {
    {"definite-assignment", "drop Undefined-global checks proven unnecessary (§2.6)", pass_definite_assignment},
    {"constants", "single-literal globals become constants (§2.6)", pass_constants},
    {"demote-globals", "globals used only by main become locals of main (§2.6)", pass_demote_globals},
    {"fold", "constant folding of integer arithmetic", pass_fold},
    {"region-info", "analysis: header-mutation summaries, unique lists (§5.2)", pass_region_info},
    // ---- optimisation passes (§5) ----
    {"simplify", "local simplification, early (§5.6)", pass_simplify},
    {"litcmp", "comparisons against list literals without allocating them", pass_litcmp},
    {"inline", "inline small instances as InlineBlocks (§5.1)", pass_inline},
    {"recur", "recursion: purity, self tail calls -> loops, pure-call CSE, self-unrolling", pass_recur},
    {"pairs", "pair unboxing (§5.4)", pass_pairs},
    {"ifconv", "if-conversion to Select (§5.5)", pass_ifconv},
    {"appendopt", "append loops: capacity hints (Reserve) and bulk copies (AppendRange)", pass_appendopt},
    {"listopt", "header caching and check elision (§5.2/§5.3)", pass_listopt},
    {"simplify2", "local simplification, late (§5.6; same implementation as simplify)", pass_simplify},
    {"reclaim", "static reclamation: Free dead lists (§5.7)", pass_reclaim},
};
constexpr size_t kFirstOpt = 5;

// Default-off passes (integration): each entry names the reason and the saved
// failing case under tests/integration/known/. Enabled by --enable-pass=NAME /
// EPQ_ENABLE_PASS=NAME, or by naming the pass in --only-pass.
const char* const kDefaultOff[] = {
    nullptr,  // (none at the moment)
};

bool listed(const std::vector<std::string>& v, const char* name) {
    for (auto& s : v)
        if (s == name) return true;
    return false;
}

}  // namespace

const std::vector<Pass>& passes() {
    static const std::vector<Pass> v(std::begin(kPipeline), std::end(kPipeline));
    return v;
}

bool pass_is_optimisation(const char* name) {
    for (size_t i = kFirstOpt; i < std::size(kPipeline); ++i)
        if (!std::strcmp(kPipeline[i].name, name)) return true;
    return false;
}

bool pass_default_off(const char* name) {
    for (const char* n : kDefaultOff)
        if (n && !std::strcmp(n, name)) return true;
    return false;
}

bool run_passes(Module& m, const PassOptions& o) {
    for (size_t i = 0; i < std::size(kPipeline); ++i) {
        const Pass& p = kPipeline[i];
        if (listed(o.disabled, p.name)) continue;
        if (pass_default_off(p.name) && !listed(o.enabled, p.name) && !listed(o.only, p.name)) continue;
        if (i >= kFirstOpt && !o.only.empty() && !listed(o.only, p.name)) continue;
        if (o.verbose) std::fprintf(stderr, "pass %s\n", p.name);
        p.run(m);
        if (o.verify) verify(m, p.name);
        if (!o.dump_after.empty() && o.dump_after == p.name) {
            std::string s;
            dump(s, m);
            std::fwrite(s.data(), 1, s.size(), stdout);
            std::fflush(stdout);
            return true;
        }
    }
    return false;
}

void run_passes(Module& m, const std::vector<std::string>& disabled, bool verbose) {
    PassOptions o;
    o.disabled = disabled;
    o.verbose = verbose;
    run_passes(m, o);
}

}  // namespace epq::tir
