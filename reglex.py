# ------------------------------------------------------------
# reglex.py
#
# tokenizer for a simple regular expression evaluator for
# symbols and +,*
# ------------------------------------------------------------
import ply.lex as lex
import ply.yacc as yacc
import sys

# Global variable definitions
# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# List of token names.   This is always required
tokens = (
   'SYMBOL',
   'PLUS',
   'ASTERISK',
   'LPAREN',
   'RPAREN',
)

# Regular expression rules for simple tokens
t_PLUS       = r'\+'
t_ASTERISK   = r'\*'
t_LPAREN     = r'\('
t_RPAREN     = r'\)'


# Lex functions
# A regular expression rule with some action code
def t_SYMBOL(t):
    r"""[0-9a-zA-Z] | \(\)"""
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r"""\n+"""
    t.lexer.lineno += len(t.value)


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Yacc functions
# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")


def p_expression_plus(p):
    """expression : expression PLUS expression"""

    p[0] = (p[2], p[1], p[3])


def p_expression_asterisk(p):
    """expression : expression ASTERISK"""
    p[0] = (p[2], p[1])


def p_expression_group(p):
    """expression : LPAREN expression RPAREN"""
    p[0] = p[2]


def p_expression_symbol(p):
    """expression : SYMBOL"""
    p[0] = ('symbol', p[1])


def p_expression_concat(p):
    """expression : expression expression"""
    p[0] = ('.', p[1], p[2])


class ENfa:
    state = []
    symbol = []
    func_dict = {}
    initial = []
    final = []
    todo_queue = []
    state_converting = []
    state_aggregating = []
    func_dict_converting = {}  # state_converting and func_dict_converting should have same length, same order.
    func_dict_aggregating = {}
    indistinguishable = []
    distinguishable = []
    belong_dict = {}

    '''
    def __init__(self, state, symbol, func_string_list, initial, final):
        self.state = state
        self.symbol = symbol
        for q in state:
            self.func_dict[q] = {}
        for func_string in func_string_list:
            func_string_split = func_string.split(',')
            if func_string_split[1] not in self.func_dict[func_string_split[0]].keys():
                self.func_dict[func_string_split[0]][func_string_split[1]] = []
            self.func_dict[func_string_split[0]][func_string_split[1]].append(func_string_split[2])
        self.initial = list(initial)
        self.final = final
        self.todo_queue.append(self.e_closure(self.initial))
        self.state_converting = list(self.todo_queue)
    '''

    def transition(self, from_state, input_symbol):
        if input_symbol in self.func_dict[from_state]:
            return self.func_dict[from_state][input_symbol]
        return False

    def e_closure(self, substate):
        e_closure_result = set()
        for ss in substate:
            e_closure_result = e_closure_result | self.dfs(ss)
        return list(e_closure_result)

    def dfs(self, start):
        visited, stack = set(), [start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                transition = self.transition(start, '()')
                if transition:
                    stack.extend(set(self.transition(start, '()')) - visited)
        return visited

    def shift(self, shift):
        func_dict_temp = {}
        for state in self.state:
            func_dict_temp[shifted_state(state, shift)] = {}
            for sym in self.symbol:
                func_dict_temp[shifted_state(state, shift)][sym] = []
                for to_state in self.func_dict[state][sym]:
                    func_dict_temp[shifted_state(state, shift)][sym].append(shifted_state(to_state, shift))
        self.func_dict = dict(func_dict_temp)
        self.state = [shifted_state(x, shift) for x in self.state]
        self.initial = [shifted_state(x, shift) for x in self.initial]
        self.final = [shifted_state(x, shift) for x in self.final]
        self.todo_queue = []
        self.todo_queue.append(self.e_closure(self.initial))
        self.state_converting = list(self.todo_queue)

    def rename_converting(self):
        # renaming
        self.state = []
        for i in range(len(self.state_converting)):
            self.state.append('q'+str(i))
        self.func_dict = {}
        self.func_dict_converting = list(self.func_dict_converting.items())
        for j in range(len(self.func_dict_converting)):
            self.func_dict_converting[j] = list(self.func_dict_converting[j])
            self.func_dict_converting[j][0] = self.state[self.state_converting.index(list(self.func_dict_converting[j][0]))]
            self.func_dict_converting[j][1] = list(self.func_dict_converting[j][1].items())
            for k in range(len(self.func_dict_converting[j][1])):
                self.func_dict_converting[j][1][k] = list(self.func_dict_converting[j][1][k])
                self.func_dict_converting[j][1][k][1] = self.state[self.state_converting.index(self.func_dict_converting[j][1][k][1])]
        self.func_dict = list(self.func_dict_converting)
        for i in range(len(self.func_dict)):
            self.func_dict[i][1] = dict(self.func_dict[i][1])

        self.func_dict = dict(self.func_dict_converting)
        final_copy = list(self.final)
        self.final = []
        for s_list in self.state_converting:
            if self.initial[0] in s_list:
                self.initial = []
                self.initial.append(self.state[self.state_converting.index(s_list)])
            for s_elem in s_list:
                if s_elem in final_copy:
                    self.final.append(self.state[self.state_converting.index(s_list)])

    def rename_aggregating(self):
        # renaming
        self.state = []
        for i in range(len(self.state_aggregating)):
            self.state.append('q'+str(i))
        self.func_dict = {}
        self.func_dict_aggregating = list(self.func_dict_aggregating.items())
        for j in range(len(self.func_dict_aggregating)):
            self.func_dict_aggregating[j] = list(self.func_dict_aggregating[j])
            self.func_dict_aggregating[j][0] = self.state[self.state_aggregating.index(list(self.func_dict_aggregating[j][0]))]
            self.func_dict_aggregating[j][1] = list(self.func_dict_aggregating[j][1].items())
            for k in range(len(self.func_dict_aggregating[j][1])):
                self.func_dict_aggregating[j][1][k] = list(self.func_dict_aggregating[j][1][k])
                self.func_dict_aggregating[j][1][k][1] = self.state[self.state_aggregating.index(self.func_dict_aggregating[j][1][k][1])]
        self.func_dict = list(self.func_dict_aggregating)
        for i in range(len(self.func_dict)):
            self.func_dict[i][1] = dict(self.func_dict[i][1])

        self.func_dict = dict(self.func_dict_aggregating)
        final_copy = list(self.final)
        self.final = []
        for s_list in self.state_aggregating:
            if self.initial[0] in s_list:
                self.initial = []
                self.initial.append(self.state[self.state_aggregating.index(s_list)])
            for s in s_list:
                if s in final_copy and self.state[self.state_aggregating.index(s_list)] not in self.final:
                    self.final.append(self.state[self.state_aggregating.index(s_list)])

    def convert_to_dfa(self):
        # converting
        while self.todo_queue:
            from_substate = self.todo_queue[0]
            for sym in self.symbol:
                to_substate = []
                for fs in from_substate:
                    transition = self.transition(fs, sym)
                    if not transition:
                        pass
                    elif isinstance(transition, list):
                        to_substate = list(set(to_substate) | set(transition))
                    else:
                        if transition not in to_substate:
                            to_substate.append(transition)
                if to_substate:
                    to_substate = self.e_closure(to_substate)
                if to_substate and (to_substate not in self.state_converting):
                    self.state_converting.append(list(to_substate))
                    if to_substate not in self.todo_queue:
                        self.todo_queue.append(list(to_substate))
                if self.func_dict_converting.get(tuple(from_substate)) is None:
                    self.func_dict_converting[tuple(from_substate)] = {}
                if to_substate:
                    self.func_dict_converting[tuple(from_substate)][sym] = list(to_substate)
            self.todo_queue.pop(0)

        self.rename_converting()

    def minimize(self):
        end_flag = True
        if not self.distinguishable:
            for s1 in self.state:
                for s2 in self.state:
                    if (s1 not in self.final and s2 in self.final) or (s1 in self.final and s2 not in self.final):
                        if my_sorted([s1, s2]) not in self.distinguishable:
                            self.distinguishable.append(my_sorted([s1, s2]))
                            end_flag = False
        else:
            for s1 in self.state:
                for s2 in self.state:
                    if my_sorted([s1, s2]) not in self.distinguishable:
                        if self.is_distinguishable([s1, s2]):
                            self.distinguishable.append(my_sorted([s1, s2]))
                            end_flag = False
        if not end_flag:
            self.minimize()
        else:
            for s1 in self.state:
                for s2 in self.state:
                    if my_sorted([s1, s2]) not in self.distinguishable and my_sorted([s1, s2]) not in self.indistinguishable:
                        self.indistinguishable.append(my_sorted([s1, s2]))
            self.aggregate()

    def find_intersection(self, m_list):
        for i, v in enumerate(m_list):
            for j, k in enumerate(m_list[i+1:], i+1):
                if v & k:
                    self.state_aggregating[i] = v.union(m_list.pop(j))
                    return self.find_intersection(m_list)
        return m_list

    def aggregate(self):
        self.state_aggregating = [set(i) for i in self.indistinguishable if i]
        self.find_intersection(self.state_aggregating)
        self.state_aggregating = [list(i) for i in self.state_aggregating if i]
        for s in self.state:
            if s not in sum(self.indistinguishable, []):
                self.state_aggregating.append([s])
        for substate in self.state_aggregating:
            for s in substate:
                self.belong_dict[s] = substate
        for substate in self.state_aggregating:
            self.func_dict_aggregating[tuple(substate)] = {}
            for sym in self.symbol:
                if self.transition(substate[0], sym):
                    self.func_dict_aggregating[tuple(substate)][sym] = self.belong_dict[self.transition(substate[0], sym)]
        self.rename_aggregating()

    def is_distinguishable(self, pair):
        for sym in self.symbol:
            transition1 = self.transition(pair[0], sym)
            transition2 = self.transition(pair[1], sym)
            if xor(transition1, transition2) or my_sorted([transition1, transition2]) in self.distinguishable:
                return True
        return False

    def print_self(self):
        print('State')
        print(','.join(self.state))
        print('Input symbol')
        print(','.join(self.symbol))
        print('State transition function')
        for q in my_sorted(list(self.func_dict.keys())):
            for sym in sorted(list(self.func_dict[q])):
                print(q + ',' + sym + ',' + self.transition(q, sym))
        print('Initial state')
        print(','.join(self.initial))
        print('Final state')
        print(','.join(self.final))

    def print_self_enfa(self):
        print('State')
        print(','.join(self.state))
        print('Input symbol')
        print(','.join(self.symbol))
        print('State transition function')
        for q in my_sorted(list(self.func_dict.keys())):
            for sym in sorted(list(self.func_dict[q])):
                for to_state in self.transition(q, sym):
                    print(q + ',' + sym + ',' + to_state)
        print('Initial state')
        print(','.join(self.initial))
        print('Final state')
        print(','.join(self.final))

    def print_self_file(self):
        f = open("output_"+sys.argv[1], 'w')
        f.write('State')
        f.write('\n')
        f.write(','.join(self.state))
        f.write('\n')
        f.write('Input symbol')
        f.write('\n')
        f.write(','.join(self.symbol))
        f.write('\n')
        f.write('State transition function')
        f.write('\n')
        for q in my_sorted(list(self.func_dict.keys())):
            for sym in sorted(list(self.func_dict[q])):
                f.write(q + ',' + sym + ',' + self.transition(q, sym))
                f.write('\n')
        f.write('Initial state')
        f.write('\n')
        f.write(','.join(self.initial))
        f.write('\n')
        f.write('Final state')
        f.write('\n')
        f.write(','.join(self.final))
        f.close()


class Dfa(ENfa):
    def __init__(self, state, symbol, func_string_list, initial, final):
        ENfa.__init__(self, state, symbol, func_string_list, initial, final)
        for key in list(self.func_dict.keys()):
            del self.func_dict[key]['()']


def my_sorted(state_list):
    # state_list_stored = list(state_list)
    if False in state_list:
        return state_list
    for i in range(len(state_list)):
        # state_list_stored[i] = int(state_list[i][1:])
        state_list[i] = int(state_list[i][1:])
    state_list = sorted(state_list)
    for i in range(len(state_list)):
        state_list[i] = 'q' + str(state_list[i])
    return state_list


def xor(b1, b2):
    return (b1 and not b2) or (not b1 and b2)


def shifted_state(q, shift):
    return 'q'+str(int(q[1:])+shift)


def make_enfa(ast):
    self = ENfa()
    if ast[0] == 'symbol':
        self.state = ['q0', 'q1']
        self.symbol = [ast[1]]
        self.func_dict['q0'] = {}
        self.func_dict['q1'] = {}
        self.func_dict['q0'][ast[1]] = ['q1']
        self.func_dict['q1'][ast[1]] = []
        self.initial = ['q0']
        self.final = ['q1']
    elif ast[0] == '*':
        sub_enfa = make_enfa(ast[1])
        sub_enfa.shift(1)
        state_len = len(sub_enfa.state)
        self.initial = ['q0']
        self.final = ['q'+str(state_len+1)]
        self.state = sub_enfa.state + [self.initial[0], self.final[0]]
        self.state = [self.initial[0]] + sub_enfa.state + [self.final[0]]
        if '()' in sub_enfa.symbol:
            self.symbol = list(sub_enfa.symbol)
        else:
            self.symbol = list(sub_enfa.symbol) + ['()']
        self.func_dict = dict(sub_enfa.func_dict)
        self.func_dict[self.initial[0]] = {}
        self.func_dict[self.final[0]] = {}
        '''
        for sym in self.symbol:
            self.func_dict[self.initial[0]][sym] = []
            self.func_dict[self.final[0]][sym] = []
        '''
        self.func_dict[self.initial[0]]['()'] = [sub_enfa.initial[0], self.final[0]]
        self.func_dict[sub_enfa.final[0]]['()'] = [sub_enfa.initial[0], self.final[0]]
        for state in self.state:
            for sym in self.symbol:
                if sym not in self.func_dict[state].keys():
                    self.func_dict[state][sym] = []
    elif ast[0] == '.':
        lhs_enfa = make_enfa(ast[1])
        lhs_enfa.func_dict = dict(lhs_enfa.func_dict)  # ????????????????????????????
        rhs_enfa = make_enfa(ast[2])
        rhs_enfa.func_dict = dict(rhs_enfa.func_dict)
        rhs_enfa.shift(len(lhs_enfa.state))
        self.initial = list(lhs_enfa.initial)
        self.final = list(rhs_enfa.final)
        self.state = list(lhs_enfa.state) + list(rhs_enfa.state)
        self.symbol = list(set(lhs_enfa.symbol) | set(rhs_enfa.symbol))
        self.symbol = sorted(self.symbol)
        if '()' not in self.symbol:
            self.symbol = self.symbol + ['()']
        lhs_enfa.func_dict[lhs_enfa.final[0]]['()'] = [rhs_enfa.initial[0]]
        self.func_dict = {**lhs_enfa.func_dict, **rhs_enfa.func_dict}
        for state in self.state:
            for sym in self.symbol:
                if sym not in self.func_dict[state].keys():
                    self.func_dict[state][sym] = []
    elif ast[0] == '+':
        lhs_enfa = make_enfa(ast[1])
        lhs_enfa.func_dict = dict(lhs_enfa.func_dict)  # ????????????????????????????
        lhs_enfa.shift(1)
        rhs_enfa = make_enfa(ast[2])
        rhs_enfa.shift(1+len(lhs_enfa.state))
        self.func_dict = {**lhs_enfa.func_dict, **rhs_enfa.func_dict}
        self.initial = ['q0']
        self.final = ['q'+str(len(lhs_enfa.state)+len(rhs_enfa.state)+1)]
        self.state = list(self.initial) + list(lhs_enfa.state) + list(rhs_enfa.state) + list(self.final)
        self.symbol = list(set(lhs_enfa.symbol) | set(rhs_enfa.symbol))
        if '()' not in self.symbol:
            self.symbol = self.symbol + ['()']
        '''
        if '()' not in lhs_enfa.symbol:
            for state in lhs_enfa.state:
                lhs_enfa.func_dict[state]['()'] = []
        if '()' not in rhs_enfa.symbol:
            for state in rhs_enfa.state:
        '''
        self.func_dict[lhs_enfa.final[0]]['()'] = [self.final[0]]
        self.func_dict[rhs_enfa.final[0]]['()'] = [self.final[0]]
        self.func_dict[self.initial[0]] = {}
        self.func_dict[self.initial[0]]['()'] = [lhs_enfa.initial[0], rhs_enfa.initial[0]]
        self.func_dict[self.final[0]] = {}
        for state in self.state:
            for sym in self.symbol:
                if sym not in self.func_dict[state].keys():
                    self.func_dict[state][sym] = []
    return self


# Build the lexer
lexer = lex.lex()

# Build the parser
parser = yacc.yacc()

# Take user input
regex_input = input('regex > ')

# Give the lexer some input. print lexed result.
lexer.input(regex_input)
s = ''.join([x.value for x in lexer])
print(regex_input)

# Parse the lexed string to make AST.
result = parser.parse(regex_input)
print(result)

result_enfa = make_enfa(result)
# key value of which is []. remove it.
for gstate in result_enfa.state:
    for gsym in result_enfa.symbol:
        if not result_enfa.func_dict[gstate][gsym]:
            del result_enfa.func_dict[gstate][gsym]
result_enfa.symbol.remove('()')
result_enfa.todo_queue = []
result_enfa.todo_queue.append(result_enfa.e_closure(result_enfa.initial))
result_enfa.state_converting = list(result_enfa.todo_queue)
print('ENFA')
print(result_enfa.func_dict)
result_enfa.print_self_enfa()
result_enfa.convert_to_dfa()
print('DFA')
print(result_enfa.func_dict)
result_enfa.print_self()
result_enfa.minimize()
print('MDFA')
print(result_enfa.func_dict)
result_enfa.print_self()

