#!/usr/bin/env python3
# unary ops, indexed assignment, for-each, conditions, calls x operand kinds
import itertools
pre = ('f := (z) => { => z; }; g := (z, w) => { => 1; }; '
       'p := (t, v) => { >> t; => v; }; '
       'A := [0, [], f, 3, [1, [2]], g, -5, [7], [4, 5, 6]]; ')
kinds = {
    'I0': ('0', 'p(1, 0)'),
    'I1': ('1', 'p(1, 1)'),
    'Im': ('-1', 'p(1, -1)'),
    'L': ('[4, 5]', 'p(1, [4, 5])'),
    'E': ('[]', 'p(1, [])'),
    'F': ('f', 'p(1, f)'),
    'a0': ('A[0]', 'p(1, A[0])'),
    'a3': ('A[3]', 'p(1, A[3])'),
    'aE': ('A[1]', 'p(1, A[1])'),
    'aL': ('A[8]', 'p(1, A[8])'),
    'aF': ('A[2]', 'p(1, A[2])'),
}
out = []
def add(nm, body):
    out.append('### %s\n%s%s' % (nm, pre, body))
for mode in ('st', 'se'):
    def K(k, tag):
        s, e = kinds[k]
        return s if mode == 'st' else e.replace('p(1', 'p(' + tag)
    for k in kinds:
        a = K(k, '11')
        add('u_%s_neg_%s' % (mode, k), 'r := -(%s); >> 9; (r = f | r = A[2]) ? >> 0 : >> r;' % a)
        add('u_%s_abs_%s' % (mode, k), 'r := |%s|; >> 9; >> r;' % a)
        add('u_%s_not_%s' % (mode, k), 'r := !(%s); >> 9; >> r;' % a)
        add('c_%s_if_%s' % (mode, k), '(%s) ? >> 1 : >> 2; >> 3;' % a)
        add('c_%s_ifand_%s' % (mode, k), '(%s & p(5, 0)) ? >> 1 : >> 2; ((%s) | p(6, 0)) ? >> 3 : >> 4;' % (a, K(k, '12')))
        add('c_%s_while_%s' % (mode, k), 'n := 0; (n < 2 & %s) @ n := n + 1; >> n;' % a)
        add('fe_%s_%s' % (mode, k), 'x := 7; (x : %s) @ { >> 9; (x = f | x = A[2]) ? >> 0 : >> x; } >> x = f;' % a)
        add('call0_%s_%s' % (mode, k), 'r := (%s)(p(22, 1), p(33, 2)); >> 9; >> r;' % a)
    ek = ['I1', 'F', 'aF', 'L']
    for kl, ki, ke in itertools.product(['L', 'E', 'I0', 'F', 'aL', 'aE', 'a0', 'aF'], ['I0', 'I1', 'Im', 'L', 'F', 'a0', 'a3', 'aE', 'aF'], ek):
        l, i, e = K(kl, '11'), K(ki, '22'), K(ke, '33')
        add('si_%s_%s_%s_%s' % (mode, kl, ki, ke), 'r := (%s)[%s] := %s; >> 9; (r = f | r = A[2]) ? >> 0 : >> r;' % (l, i, e))
print('\n'.join(out))
