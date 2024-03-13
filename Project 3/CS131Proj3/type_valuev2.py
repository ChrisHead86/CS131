from enum import Enum
from intbase import InterpreterBase


# Enumerated type for our different language data types
class Type(Enum):
    INT = 1
    BOOL = 2
    STRING = 3
    NIL = 4
    FUNC = 5
    LAMBDA = 6


# Represents a value, which has a type and its value
class Value:
    def __init__(self, type, value=None):
        self.t = type
        self.v = value
        self.is_ref = False
        self.ref_name = None 
        self.is_lam = None
        self.lam_num = None

    def value(self):
        return self.v

    def type(self):
        return self.t
    
    def get_ref(self):
        return self.is_ref
    
    def set_ref(self, change):
        self.is_ref = change
    
    def set_ref_name(self, name):
        self.ref_name = name

    def get_ref_name(self):
        return self.ref_name
    
    def get_is_lam(self):
        return self.is_lam
    
    def set_is_lam(self, change):
        self.is_lam = change

    def set_lam_num(self, num):
        self.lam_num = num

    def get_lam_num(self):
        return self.lam_num
    



def create_value(val):
    if val == InterpreterBase.TRUE_DEF:
        return Value(Type.BOOL, True)
    elif val == InterpreterBase.FALSE_DEF:
        return Value(Type.BOOL, False)
    elif val == InterpreterBase.NIL_DEF:
        return Value(Type.NIL, None)
    elif isinstance(val, str):
        return Value(Type.STRING, val)
    elif isinstance(val, int):
        return Value(Type.INT, val)
    else:
        raise ValueError("Unknown value type")


def get_printable(val):
    if val.type() == Type.INT:
        return str(val.value())
    if val.type() == Type.STRING:
        return val.value()
    if val.type() == Type.BOOL:
        if val.value() is True:
            return "true"
        return "false"
    return None
