#!/usr/bin/env python3
# Generates a batch of operator x operand-kind programs (error precedence & side-effect order).
import itertools, sys
pre = ('f := (z) => { => z; }; g := (z, w) => { => 1; }; '
       'p := (t, v) => { >> t; => v; }; '
       'A := [0, [], f, 3, [1, [2]], g, -5, [7]]; ')
# name -> (static expr, side-effecting expr)
kinds = {
    'I0': ('0', 'p(1, 0)'),
    'I3': ('3', 'p(1, 3)'),
    'L': ('[1, 2]', 'p(1, [1, 2])'),
    'E': ('[]', 'p(1, [])'),
    'F': ('f', 'p(1, f)'),
    'a0': ('A[0]', 'p(1, A[0])'),
    'aE': ('A[1]', 'p(1, A[1])'),
    'aF': ('A[2]', 'p(1, A[2])'),
}
ops = ['+', '-', '*', '/', '=', '!=', '<', '>', '<=', '>=', ':+', '&', '|', 'IDX', 'CALL']
mode = sys.argv[1] if len(sys.argv) > 1 else 'se'
out = []
for op in ops:
    for (ka, va), (kb, vb) in itertools.product(kinds.items(), repeat=2):
        a = va[1].replace('p(1', 'p(11') if mode == 'se' else va[0]
        b = vb[1].replace('p(1', 'p(22') if mode == 'se' else vb[0]
        if op == 'IDX':
            e = '%s[%s]' % (a if mode != 'se' else '(' + a + ')', b)
        elif op == 'CALL':
            e = '(%s)(%s)' % (a, b)
        else:
            e = '(%s) %s (%s)' % (a, op, b)
        nm = 'op_%s_%s_%s_%s' % (mode, {'+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '=': 'eq', '!=': 'ne', '<': 'lt', '>': 'gt', '<=': 'le', '>=': 'ge', ':+': 'app', '&': 'and', '|': 'or', 'IDX': 'idx', 'CALL': 'call'}[op], ka, kb)
        # printing functions is outside the subset: compare via = with a marker instead
        body = 'r := %s; >> 9; >> r = f; >> r = A[2];' % e
        if op == ':+':
            body = 'r := %s; >> 9; >> |r|; >> r[|r| - 1] = f;' % e
        elif op in ('&', '|', 'IDX'):
            body += ' (r = f | r = A[2] | r = g) ? >> 0 : >> r;'
        else:
            body += ' >> r;'
        out.append('### %s\n%s%s' % (nm, pre, body))
print('\n'.join(out))
