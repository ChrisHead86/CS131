from env_v1 import EnvironmentManager
from type_valuev1 import Type, Value, create_value, get_printable
from intbase import InterpreterBase, ErrorType
from brewparse import parse_program
from collections import deque
import copy


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    BIN_OPS = {"+", "-", "*", "/", "<", ">", "<=", ">=", "&&", "||"}
    UN_OPS = {"neg", "!"}
    COMP_OPS = {"==", "!="}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.stack = []

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        main_func = self.__get_func_by_name("main")[0]
        self.env = EnvironmentManager()
        var_vals = {}
        self.stack.append((main_func, var_vals))
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        #5 lines written by copilot
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = []
            self.func_name_to_ast[func_name].append(func_def)

    def __get_func_by_name(self, name):
        if name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {name} not found")
        return self.func_name_to_ast[name]

    def __run_statements(self, statements):
        
        # all statements of a function are held in arg3 of the function AST node
        #make any loop/function exit to calling function when return is reached
        for statement in statements:
            if "return" in self.stack[-1][1]:
                return self.stack[-1][1]["return"]
            if self.trace_output:
                print(statement)
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == "if":
                self.__run_if(statement)
            elif statement.elem_type == "while":
                self.__run_while(statement)
            elif statement.elem_type == "return": 
                if statement.get("expression") is None:
                    return Interpreter.NIL_VALUE
                else:
                    self.stack[-1][1]["return"] = self.__eval_expr(statement.get("expression"))
                    return self.stack[-1][1]["return"]

            
                    
                
                
            
    def __run_while(self, while_ast):
        cond_ition = self.__eval_expr(while_ast.get("condition"))
        if cond_ition.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR, "Condition must be a boolean expression"
            )   
        while (cond_ition.value()):
            if "return" in self.stack[-1][1]:
                return self.stack[-1][1]["return"]
            self.__run_statements(while_ast.get("statements"))
            cond_ition = self.__eval_expr(while_ast.get("condition"))
        
        
            
                

    

        




    def __call_func(self, call_node):
        func_name = call_node.get("name")
        num_args = len(call_node.get("args"))
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)
        
        if func_name not in self.func_name_to_ast:
            super().error(ErrorType.NAME_ERROR, f"Function {func_name} not found")
        else:
            #next 11 lines written by copilot
            vars = []
            for x in call_node.get("args"):
                y = self.__eval_expr(x)
                vars.append(y)
            matches = self.func_name_to_ast[func_name]
            func_def = None
            for match in matches:
                if len(match.get("args")) == num_args:
                    func_def = match
            if func_def == None:
                super().error(ErrorType.NAME_ERROR, f"Function {func_name} not found")
            next_step = {}
             # add the arguments to the environment
            for i in range(len(vars)): 
                arg_name = func_def.get("args")[i].get("name")
                next_step[arg_name] = vars[i]
            self.stack.append((func_def, next_step))
            
             # run the function

            #4 lines written by copilot
            f = self.stack[-1]
            result = self.__run_statements(f[0].get("statements"))
            if self.stack[-1] == f:
                self.stack.pop()
                if result is None:
                    return Interpreter.NIL_VALUE
                else:
                    return result
            



            
            
            
            return result




        
        

            # add code here later to call other functions
            super().error(ErrorType.NAME_ERROR, f"Function {func_name} not found")
            return Interpreter.NIL_VALUE


    

    def __call_print(self, call_ast):
        output = ""
        for arg in call_ast.get("args"):
            result = self.__eval_expr(arg)  # result is a Value object
            output = output + get_printable(result)
        super().output(output)
        return Interpreter.NIL_VALUE

    def __call_input(self, call_ast):
        args = call_ast.get("args")
        if args is not None and len(args) == 1:
            result = self.__eval_expr(args[0])
            super().output(get_printable(result))
        elif args is not None and len(args) > 1:
            super().error(
                ErrorType.NAME_ERROR, "No inputi() function that takes > 1 parameter"
            )
        inp = super().get_input()
        if call_ast.get("name") == "inputi":
            return Value(Type.INT, int(inp))
        if call_ast.get("name") == "inputs":
            return Value(Type.STRING, inp)
        # we can support inputs here later

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        #3 lines written by copilot
        if var_name not in self.stack[-1][1] and var_name in self.stack[0][1]:
            self.stack[0][1][var_name] = value_obj
            
        self.stack[-1][1][var_name] = value_obj




    def __run_if(self, if_ast):
        cond_ition = self.__eval_expr(if_ast.get("condition"))
        if cond_ition.type() != Type.BOOL:
            super().error(
                ErrorType.TYPE_ERROR, "Condition must be a boolean expression"
            )
        if (cond_ition.value()):
            if "return" in self.stack[-1][1]:
                return self.stack[-1][1]["return"]
            self.__run_statements(if_ast.get("statements"))
        elif if_ast.get("else_statements") is not None:
            if "return" in self.stack[-1][1]:
                return self.stack[-1][1]["return"]
            self.__run_statements(if_ast.get("else_statements"))



    def __eval_expr(self, expr_ast):
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            return Value(Type.NIL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            #for loop written by copilot
            for x in range (len(self.stack)-1, -1, -1):
                if var_name in self.stack[x][1]:
                    val = self.stack[x][1][var_name]
                    break
                else:
                    val = None
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.UN_OPS:
            return self.__eval_un_op(expr_ast)
        if expr_ast.elem_type in Interpreter.COMP_OPS:
            return self.__eval_comp_op(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
    
    def __eval_comp_op(self, comp_op_ast):
        left_value_obj = self.__eval_expr(comp_op_ast.get("op1"))
        right_value_obj = self.__eval_expr(comp_op_ast.get("op2"))
        if comp_op_ast.elem_type not in self.op_to_lambda[Type.NIL]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {comp_op_ast.elem_type} for type {right_value_obj.type()}",
            )
        f = self.op_to_lambda[Type.NIL][comp_op_ast.elem_type]
        return f(left_value_obj, right_value_obj)

    def __eval_un_op(self, un_op_ast):
        value_obj = self.__eval_expr(un_op_ast.get("op1"))
        if un_op_ast.elem_type not in self.un_op_to_lambda[value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {un_op_ast.elem_type} for type {value_obj.type()}",
            )
        f = self.un_op_to_lambda[value_obj.type()][un_op_ast.elem_type]
        return f(value_obj)

    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if left_value_obj.type() != right_value_obj.type():
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {right_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        return f(left_value_obj, right_value_obj)

    def __setup_ops(self):
        self.op_to_lambda = {}
        self.un_op_to_lambda = {}
        # set up operations on integers
        self.un_op_to_lambda[Type.INT] = {}
        self.un_op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), x.value() - y.value()
        )
        # add other operators here later for int, string, bool, etc
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), x.value() * y.value()
        )
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            Type.INT, x.value() // y.value()
        )
        self.op_to_lambda[Type.INT]["<"] = lambda x, y: Value(
            Type.BOOL, x.value() < y.value()
        )
        self.op_to_lambda[Type.INT]["<="] = lambda x, y: Value(
            Type.BOOL, x.value() <= y.value()
        )
        self.op_to_lambda[Type.INT][">"] = lambda x, y: Value(
            Type.BOOL, x.value() > y.value()
        )
        self.op_to_lambda[Type.INT][">="] = lambda x, y: Value(
            Type.BOOL, x.value() >= y.value()
        )

        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            Type.STRING, x.value() + y.value()
        )
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            Type.BOOL, x.value() and y.value()
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            Type.BOOL, x.value() or y.value()
        )
        self.un_op_to_lambda[Type.BOOL]["!"] = lambda x: Value(
            Type.BOOL, not x.value()
        )
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        self.un_op_to_lambda[Type.INT]["neg"] = lambda x: Value(
            Type.INT, -x.value()
        )


