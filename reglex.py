# ------------------------------------------------------------
# reglex.py
#
# tokenizer for a simple regular expression evaluator for
# symbols and +,*
# ------------------------------------------------------------
import ply.lex as lex
import ply.yacc as yacc

# List of token names.   This is always required
tokens = (
   'SYMBOL',
   'PLUS',
   'ASTERISK',
   'LPAREN',
   'RPAREN',
   'CONCAT',
)

# Regular expression rules for simple tokens
t_PLUS       = r'\+'
t_ASTERISK   = r'\*'
t_LPAREN     = r'\('
t_RPAREN     = r'\)'
# t_CONCAT     = r'^$'
t_CONCAT     = r'\.'


# A regular expression rule with some action code
def t_SYMBOL(t):
    r'[0-9a-zA-Z] | \(\)'
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Test it out
data = '''
ba*(a+b)
'''

# Give the lexer some input
lexer.input(data)

print('using iteration protocol')
for tok in lexer:
    print(tok.value, tok.type)


# Yacc example
# Get the token map from the lexer.  This is required.
# Error rule for syntax errors
def p_error(p):
    print("Syntax error in input!")


def p_expression_plus(p):
    '''expression : expression PLUS expression'''

    p[0] = (p[2], p[1], p[3])


def p_expression_asterisk(p):
    '''expression : expression ASTERISK'''
    p[0] = (p[2], p[1])


def p_expression_group(p):
    'expression : LPAREN expression RPAREN'
    p[0] = p[2]


def p_expression_symbol(p):
    'expression : SYMBOL'
    p[0] = ('symbol', p[1])


def p_expression_concat(p):
    'expression : expression expression'
    p[0] = ('.', p[1], p[2])


'''
def make_enfa(s):
    if s[0] == 'symbol':
'''


# Build the parser
parser = yacc.yacc()

while True:
    try:
        s = input('regex > ')
        lexer.input(s)
        s = ''.join([x.value for x in lexer])
        print(s)
    except EOFError:
        break
    if not s:
        continue
    result = parser.parse(s)
    print(result)
