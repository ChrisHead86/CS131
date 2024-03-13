import copy
from enum import Enum

from brewparse import parse_program
from env_v2 import EnvironmentManager
from intbase import InterpreterBase, ErrorType
from type_valuev2 import Type, Value, create_value, get_printable





class ExecStatus(Enum):
    CONTINUE = 1
    RETURN = 2


# Main interpreter class
class Interpreter(InterpreterBase):
    # constants
    NIL_VALUE = create_value(InterpreterBase.NIL_DEF)
    TRUE_VALUE = create_value(InterpreterBase.TRUE_DEF)
    BIN_OPS = {"+", "-", "*", "/", "==", "!=", ">", ">=", "<", "<=", "||", "&&"}

    # methods
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)
        self.trace_output = trace_output
        self.__setup_ops()
        self.is_main = True

    # run a program that's provided in a string
    # usese the provided Parser found in brewparse.py to parse the program
    # into an abstract syntax tree (ast)
    def run(self, program):
        ast = parse_program(program)
        self.__set_up_function_table(ast)
        self.env = EnvironmentManager()
        main_func = self.__get_func_by_name("main", 0)
        self.__run_statements(main_func.get("statements"))

    def __set_up_function_table(self, ast):
        self.func_name_to_ast = {}
        for func_def in ast.get("functions"):
            func_name = func_def.get("name")
            num_params = len(func_def.get("args"))
            if func_name not in self.func_name_to_ast:
                self.func_name_to_ast[func_name] = {}
            self.func_name_to_ast[func_name][num_params] = func_def

    def __get_func_by_name(self, name, num_params):
        if name in self.func_name_to_ast:
            candidate_funcs = self.func_name_to_ast[name]
            if num_params not in candidate_funcs:
                super().error(
                    ErrorType.NAME_ERROR,
                    f"Function {name} taking {num_params} params not found",
                )
            return candidate_funcs[num_params]
        else:
            is_var_assign = self.env.get(name)
            if is_var_assign.type() == Type.LAMBDA:
                lam_env = is_var_assign.get_lam_num()
                return (is_var_assign.value(), lam_env)
            if is_var_assign.type() != Type.FUNC:
                super().error(ErrorType.TYPE_ERROR, f"{name} is not a function")
            if is_var_assign.value().get("name") in self.func_name_to_ast:
                num_param_need = len(is_var_assign.value().get("args")) 
                if num_param_need != num_params:
                    super().error(ErrorType.TYPE_ERROR, f"Function {name} taking {num_params} params not found")
                return is_var_assign.value()
            else:
                super().error(ErrorType.NAME_ERROR, f"Function {name} not found")


    def __run_statements(self, statements):
        self.env.push()
        for statement in statements:
            if self.trace_output:
                print(statement)
            status = ExecStatus.CONTINUE
            if statement.elem_type == InterpreterBase.FCALL_DEF:
                self.is_main = False
                self.__call_func(statement)
            elif statement.elem_type == "=":
                self.__assign(statement)
            elif statement.elem_type == InterpreterBase.RETURN_DEF:
                status, return_val = self.__do_return(statement)
            elif statement.elem_type == Interpreter.IF_DEF:
                status, return_val = self.__do_if(statement)
            elif statement.elem_type == Interpreter.WHILE_DEF:
                status, return_val = self.__do_while(statement)

            if status == ExecStatus.RETURN:
                self.env.pop()
                return (status, return_val)

        self.env.pop()
        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __call_func(self, call_node):
        func_name = call_node.get("name")
        if func_name == "print":
            return self.__call_print(call_node)
        if func_name == "inputi":
            return self.__call_input(call_node)
        if func_name == "inputs":
            return self.__call_input(call_node)

        actual_args = call_node.get("args")
        func_ast = self.__get_func_by_name(func_name, len(actual_args))
        if isinstance(func_ast, tuple):
            func_ast, env = func_ast
            env_to_push = env
        formal_args = func_ast.get("args")
        if len(actual_args) != len(formal_args):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Function {func_ast.get('name')} with {len(actual_args)} args not found",
            )
        if "env_to_push" in locals():
            env_num_to_push = self.env.get(func_name).get_lam_num()
            self.env.push_with_env(self.env.get_full_env_stack()[env_num_to_push])
        else:
            self.env.push()
        for formal_ast, actual_ast in zip(formal_args, actual_args):
            result = (copy.deepcopy(self.__eval_expr(actual_ast)))
            if formal_ast.elem_type == "refarg":
                #print(formal_ast.get("name"))
                #print(actual_ast.get("name"))
                result.set_ref(True)
                result.set_ref_name(actual_ast.get("name"))
            arg_name = formal_ast.get("name")
            self.env.create(arg_name, result)
        _, return_val = self.__run_statements(func_ast.get("statements"))
        if "env_to_push" in locals() and self.is_main:
            self.env.edit_lambda_env(env_num_to_push, self.env.get_full_env_stack()[-1])
        self.env.pop()
        self.is_main = True
        return return_val

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

    def __assign(self, assign_ast):
        var_name = assign_ast.get("name")
        value_obj = self.__eval_expr(assign_ast.get("expression"))
        if self.env.get(var_name) is not None:
            if self.env.get(var_name).get_ref():
                ref_name = self.env.get(var_name).get_ref_name()

                count = 3
                
                to_break = True
                while to_break:
                    if count >= self.env.get_stack_size():
                        break
                    check_stack = self.env.get_env(-count)
                    while len(check_stack) == 0:
                        count += 1 
                        check_stack = self.env.get_env(-count)


                    if check_stack.get(ref_name).get_ref():
                        value_obj.set_ref(True)
                        value_obj.set_ref_name(check_stack.get(ref_name).get_ref_name())
                        self.env.set_env_stack(-count, ref_name, value_obj)
                        ref_name = check_stack.get(ref_name).get_ref_name()
                        count +=1 
                    else:
                        self.env.set_env_stack(-count, ref_name, value_obj)                  
                        to_break = False
                
                value_obj.set_ref(True)      
                value_obj.set_ref_name(self.env.get(var_name).get_ref_name())

        self.env.set(var_name, value_obj)
    

    def __eval_expr(self, expr_ast):
        # print("here expr")
        # print("type: " + str(expr_ast.elem_type))
        if expr_ast.elem_type == InterpreterBase.NIL_DEF:
            # print("getting as nil")
            return Interpreter.NIL_VALUE
        if expr_ast.elem_type == InterpreterBase.INT_DEF:
            return Value(Type.INT, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.STRING_DEF:
            # print("getting as str")
            return Value(Type.STRING, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.BOOL_DEF:
            return Value(Type.BOOL, expr_ast.get("val"))
        if expr_ast.elem_type == InterpreterBase.VAR_DEF:
            var_name = expr_ast.get("name")
            if var_name in self.func_name_to_ast:
                    list_of_funcs = list(self.func_name_to_ast[var_name].items())
                    if len(list_of_funcs) > 1:
                        super().error(
                            ErrorType.NAME_ERROR, "Can not use overloaded function as var"
                        )
                    
                    num_params = list_of_funcs[0]
                    return Value(Type.FUNC, self.__get_func_by_name(var_name, num_params[0]))
            val = self.env.get(var_name)
            if val is None:
                super().error(ErrorType.NAME_ERROR, f"Variable {var_name} not found")
            return val
        if expr_ast.elem_type == InterpreterBase.FCALL_DEF:
            return self.__call_func(expr_ast)
        if expr_ast.elem_type in Interpreter.BIN_OPS:
            return self.__eval_op(expr_ast)
        if expr_ast.elem_type == Interpreter.NEG_DEF:
            return self.__eval_unary(expr_ast, Type.INT, lambda x: -1 * x)
        if expr_ast.elem_type == Interpreter.NOT_DEF:
            return self.__eval_unary(expr_ast, Type.BOOL, lambda x: not x)
        if expr_ast.elem_type == InterpreterBase.LAMBDA_DEF:
            cur_env = copy.deepcopy(self.env.get_cur_env())
            to_ret = Value(Type.LAMBDA, expr_ast)
            to_ret.set_is_lam(cur_env)
            lam_num = self.env.get_stack_size() - 1
            to_ret.set_lam_num(lam_num)
            return to_ret
            


    def __eval_op(self, arith_ast):
        left_value_obj = self.__eval_expr(arith_ast.get("op1"))
        right_value_obj = self.__eval_expr(arith_ast.get("op2"))
        if not self.__compatible_types(
            arith_ast.elem_type, left_value_obj, right_value_obj
        ):
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible types for {arith_ast.elem_type} operation",
            )
        if arith_ast.elem_type not in self.op_to_lambda[left_value_obj.type()]:
            super().error(
                ErrorType.TYPE_ERROR,
                f"Incompatible operator {arith_ast.elem_type} for type {left_value_obj.type()}",
            )
        f = self.op_to_lambda[left_value_obj.type()][arith_ast.elem_type]
        # print("here eval")
        # print(arith_ast)
        #print("evaluating " + str(left_value_obj.type()) + " " + str(arith_ast.elem_type) + " " + str(right_value_obj.type()))
        #print("obj left: " + str(left_value_obj.value()))
        #print("obj right: " + str(right_value_obj.value()))
        #print(f(left_value_obj, right_value_obj).value())
        return f(left_value_obj, right_value_obj)

    def __compatible_types(self, oper, obj1, obj2):
        # DOCUMENT: allow comparisons ==/!= of anything against anything
        if oper in ["==", "!="]:
            return True
        if oper in ["&&", "||"] and ((obj1.type() == Type.BOOL and obj2.type() == Type.INT) or (obj1.type() == Type.INT and obj2.type() == Type.BOOL) or (obj1.type() == Type.INT and obj2.type() == Type.INT)):
            return True 
        if oper in ["+", "-", "*", "/"] and ((obj1.type() == Type.BOOL and obj2.type() == Type.INT) or (obj1.type() == Type.INT and obj2.type() == Type.BOOL) or (obj1.type() == Type.INT and obj2.type() == Type.INT)):
            return True
        return obj1.type() == obj2.type()

    def __eval_unary(self, arith_ast, t, f):
        value_obj = self.__eval_expr(arith_ast.get("op1"))
        if value_obj.type() != t:
            if arith_ast.elem_type == Interpreter.NOT_DEF and value_obj.type() == Type.INT:
                if value_obj.value() == 0:
                    return Value(Type.BOOL, True)
                else:
                    return Value(Type.BOOL, False)
            else:
                super().error(
                    ErrorType.TYPE_ERROR,
                    f"Incompatible type for {arith_ast.elem_type} operation",
                )
        return Value(t, f(value_obj.value()))

    def __setup_ops(self):
        self.op_to_lambda = {}
        # set up operations on integers
        self.op_to_lambda[Type.INT] = {}
        self.op_to_lambda[Type.INT]["+"] = lambda x, y: Value(
            x.type(), (x.value() + y.value()) if y.type() == Type.INT else (
            x.value() + 1 if (y.type() == Type.BOOL and y.value() == True) else (
            x.value() 
        )))
        self.op_to_lambda[Type.INT]["-"] = lambda x, y: Value(
            x.type(), (x.value() - y.value()) if y.type() == Type.INT else (
            x.value() - 1 if (y.type() == Type.BOOL and y.value() == True) else (
            x.value() 
        )))
        self.op_to_lambda[Type.INT]["*"] = lambda x, y: Value(
            x.type(), (x.value() * y.value()) if y.type() == Type.INT else (
            x.value() * 1 if (y.type() == Type.BOOL and y.value() == True) else (
            x.value() * 0
        )))
        self.op_to_lambda[Type.INT]["/"] = lambda x, y: Value(
            x.type(), (x.value() // y.value()) if y.type() == Type.INT else (
            x.value() // 1 if (y.type() == Type.BOOL and y.value() == True) else (
            x.value() // 0
        )))
        self.op_to_lambda[Type.INT]["=="] = lambda x, y: Value(
            Type.BOOL, (x.value() == y.value() if y.type() == Type.INT else (
            False if y.type() != Type.BOOL else (
                (False == y.value()) if x.value() == 0 else (True == y.value())
            )
        )
            ))
        self.op_to_lambda[Type.INT]["!="] = lambda x, y: Value(
            Type.BOOL, (x.value() != y.value()) if y.type() == Type.INT else (
            True if y.type() != Type.BOOL else (
                (False != y.value()) if x.value() == 0 else (True != y.value())
            )
            )
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
        self.op_to_lambda[Type.INT]["&&"] = lambda x, y: Value(
            Type.BOOL, (x.value() != 0) and (y.value() != 0)) if y.type() == Type.INT else (
            (x.value() != 0) and y.value()
            )
        
        self.op_to_lambda[Type.INT]["||"] = lambda x, y: Value(
            Type.BOOL, (x.value() != 0) and (y.value() != 0) if y.type() == Type.INT else (
            (x.value() != 0) or y.value()
            )
            )
        
        #  set up operations on strings
        self.op_to_lambda[Type.STRING] = {}
        self.op_to_lambda[Type.STRING]["+"] = lambda x, y: Value(
            x.type(), x.value() + y.value()
        )
        self.op_to_lambda[Type.STRING]["=="] = lambda x, y: Value(
            Type.BOOL, x.value() == y.value()
        )
        self.op_to_lambda[Type.STRING]["!="] = lambda x, y: Value(
            Type.BOOL, x.value() != y.value()
        )
        #  set up operations on bools
        self.op_to_lambda[Type.BOOL] = {}
        self.op_to_lambda[Type.BOOL]["&&"] = lambda x, y: Value(
            Type.BOOL, (x.value() and y.value()) if y.type() == Type.BOOL else (
            False if y.type() != Type.INT else (
                (False and x.value()) if y.value() == 0 else (True and x.value())
            )
            )
        )
        self.op_to_lambda[Type.BOOL]["||"] = lambda x, y: Value(
            Type.BOOL, (x.value() or y.value()) if y.type() == Type.BOOL else (
            True if y.type() != Type.INT else (
                (False or x.value()) if y.value() == 0 else (True or x.value())
            )
            )
        )
        self.op_to_lambda[Type.BOOL]["=="] = lambda x, y: Value(
            Type.BOOL, (x.value() == y.value()) if y.type() == Type.BOOL else (
            False if y.type() != Type.INT else (
                (False == x.value()) if y.value() == 0 else (True == x.value())
            )
            )
        )
        self.op_to_lambda[Type.BOOL]["!="] = lambda x, y: Value(
            Type.BOOL, (x.value() != y.value()) if y.type() == Type.BOOL else (
            True if y.type() != Type.INT else (
                (False != x.value()) if y.value() == 0 else (True != x.value())
            )
            )
        )
        self.op_to_lambda[Type.BOOL]["+"] = lambda x, y: Value(
            Type.INT, (1 + y.value()) if (y.type() == Type.INT and x.value() == True) else (
            y.value() if (y.type() == Type.INT and x.value() == False) else (
            1 + 1 if (y.type() == Type.BOOL and x.value() == True and y.value() == True) else (
            1 if (y.type() == Type.BOOL and ((x.value() == True and y.value() == False) or (x.value() == False and y.value() == True))) else ( 
            0)))))
        self.op_to_lambda[Type.BOOL]["-"] = lambda x, y: Value(
            Type.INT, (1 - y.value()) if (y.type() == Type.INT and x.value() == True) else (
            0 - y.value() if (y.type() == Type.INT and x.value() == False) else (
            1 - 1 if (y.type() == Type.BOOL and x.value() == True and y.value() == True) else (
            1 if (y.type() == Type.BOOL and ((x.value() == True and y.value() == False))) else ( 
            -1 if (x.value() == False and y.value() == True) else (
            0))))))
        self.op_to_lambda[Type.BOOL]["*"] = lambda x, y: Value(
            Type.INT, (1 * y.value()) if (y.type() == Type.INT and x.value() == True) else (
            0 * y.value() if (y.type() == Type.INT and x.value() == False) else (
            1 * 1 if (y.type() == Type.BOOL and x.value() == True and y.value() == True) else (
            1 * 0 if (y.type() == Type.BOOL and ((x.value() == True and y.value() == False) or (x.value() == False and y.value() == True))) else ( 
            0)))))
        self.op_to_lambda[Type.BOOL]["/"] = lambda x, y: Value(
            Type.INT, (1 // y.value()) if (y.type() == Type.INT and x.value() == True) else (
            0 // y.value() if (y.type() == Type.INT and x.value() == False) else (
            1 // 1 if (y.type() == Type.BOOL and x.value() == True and y.value() == True) else (
            1 // 0 if (y.type() == Type.BOOL and (x.value() == True and y.value() == False))  else ( 
            0 // 1 if (y.type() == Type.BOOL and (x.value() == False and y.value() == True)) else (
            0 // 0))))))
        

        #  set up operations on nil
        self.op_to_lambda[Type.NIL] = {}
        self.op_to_lambda[Type.NIL]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.NIL]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        # set up ops on functions
        self.op_to_lambda[Type.FUNC] = {}
        self.op_to_lambda[Type.FUNC]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.FUNC]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

        # set up ops on lambdas
        self.op_to_lambda[Type.LAMBDA] = {}
        self.op_to_lambda[Type.LAMBDA]["=="] = lambda x, y: Value(
            Type.BOOL, x.type() == y.type() and x.value() == y.value()
        )
        self.op_to_lambda[Type.LAMBDA]["!="] = lambda x, y: Value(
            Type.BOOL, x.type() != y.type() or x.value() != y.value()
        )

    def __do_if(self, if_ast):
        cond_ast = if_ast.get("condition")
        result = self.__eval_expr(cond_ast)
        if result.type() != Type.BOOL and result.type() != Type.INT:
            super().error(
                ErrorType.TYPE_ERROR,
                "Incompatible type for if condition",
            )
        if (result.type() == Type.BOOL and result.value()) or (result.type() == Type.INT and result.value() != 0):
            statements = if_ast.get("statements")
            status, return_val = self.__run_statements(statements)
            return (status, return_val)
        else:
            else_statements = if_ast.get("else_statements")
            if else_statements is not None:
                status, return_val = self.__run_statements(else_statements)
                return (status, return_val)

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_while(self, while_ast):
        cond_ast = while_ast.get("condition")
        run_while = Interpreter.TRUE_VALUE
        while run_while.value():
            run_while = self.__eval_expr(cond_ast)
            if run_while.type() != Type.BOOL and run_while.type() != Type.INT:
                super().error(
                    ErrorType.TYPE_ERROR,
                    "Incompatible type for while condition",
                )
            if (run_while.type() == Type.BOOL and run_while.value()) or (run_while.type() == Type.INT and run_while.value() != 0):
                statements = while_ast.get("statements")
                status, return_val = self.__run_statements(statements)
                if status == ExecStatus.RETURN:
                    return status, return_val

        return (ExecStatus.CONTINUE, Interpreter.NIL_VALUE)

    def __do_return(self, return_ast):
        expr_ast = return_ast.get("expression")
        if expr_ast is None:
            return (ExecStatus.RETURN, Interpreter.NIL_VALUE)
        value_obj = copy.deepcopy(self.__eval_expr(expr_ast))
        return (ExecStatus.RETURN, value_obj)
    

def main():
    interpreter = Interpreter()
    program1 = """

func foo(f1, ref f2) {
  f1(); /* prints 1 */
  f2(); /* prints 1 */
}

func main() {
  x = 0;
  lam1 = lambda() { x = x + 1; print(x); };
  lam2 = lambda() { x = x + 1; print(x); };
  foo(lam1, lam2);
  lam1(); /* prints 1 */
  lam2(); /* prints 2 */
}

    """
    interpreter.run(program1)


if __name__ == "__main__":
    main()


