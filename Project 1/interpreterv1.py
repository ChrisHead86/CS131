from brewparse import parse_program
from intbase import InterpreterBase, ErrorType



class Interpreter(InterpreterBase):
    def __init__(self, console_output=True, inp=None, trace_output=False):
        super().__init__(console_output, inp)  
        self.variable_name_to_value = {}  

    def run(self, program):
        ast = parse_program(program)
        main_func_node = self.get_main_func_node(ast)
        self.run_func(main_func_node)
        if (main_func_node.dict["name"] != "main"):
            super().error(ErrorType.NAME_ERROR, "No main function")
    
    def get_main_func_node(self, ast):
        return ast.dict["functions"][0]
    
    def run_func(self, func_node):
        for statement_node in func_node.dict["statements"]:
            self.run_statement(statement_node)

    def run_statement(self, statement_node):
        if statement_node.elem_type == "=":
            self.do_assignment(statement_node)
        elif statement_node.elem_type == "fcall":
            self.do_func_call(statement_node)
    
    def do_assignment(self, assignment_node):
        target_var_name = assignment_node.dict["name"]
        source_node = assignment_node.dict["expression"]
        resulting_value = self.evaluate_expression(source_node)
        self.variable_name_to_value[target_var_name] = resulting_value
        
    
    def evaluate_expression(self, expression_node):
        if expression_node.elem_type == "int":
            return expression_node.dict["val"]
        elif expression_node.elem_type == "string":
            return expression_node.dict["val"]
        elif expression_node.elem_type == "+":
            return self.evaluate_addition(expression_node)
        elif expression_node.elem_type == "-":
            return self.evaluate_subtraction(expression_node)
        #the following 4 lines written by CoPilot
        elif expression_node.elem_type == "var":
            if (expression_node.dict["name"] not in self.variable_name_to_value):
                super().error(ErrorType.NAME_ERROR, "Variable not defined")
            return self.variable_name_to_value[expression_node.dict["name"]]
        elif expression_node.elem_type == "fcall":
            return self.do_func_call(expression_node)
    
    def evaluate_addition(self, addition_node):
        left_v = self.evaluate_expression(addition_node.dict["op1"])
        right_v = self.evaluate_expression(addition_node.dict["op2"])
        #following two lines written by CoPilot
        if (type(left_v) != type (right_v)):
            super().error(ErrorType.TYPE_ERROR, "Cannot add values of different types")
        return (left_v + right_v)

    def evaluate_subtraction(self, subtraction_node):
        left_v = self.evaluate_expression(subtraction_node.dict["op1"])
        right_v = self.evaluate_expression(subtraction_node.dict["op2"])
        if (type(left_v) != type (right_v)):
            super().error(ErrorType.TYPE_ERROR, "Cannot subtract values of different types")
        return left_v - right_v

    def do_func_call(self, func_call_node):
        output = ""
        if func_call_node.dict["name"] == "print":
            for arg in func_call_node.dict["args"]:
                output += str(self.evaluate_expression(arg))
            super().output(output)
        elif func_call_node.dict["name"] == "inputi":
            #following line written by CoPilot
            #print(str(self.evaluate_expression(func_call_node.dict["args"][0])))
            return super().get_input()
        else:
            super().error(
                ErrorType.NAME_ERROR, 
                "Function not found"
                )
