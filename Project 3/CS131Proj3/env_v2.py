# The EnvironmentManager class keeps a mapping between each variable name (aka symbol)
# in a brewin program and the Value object, which stores a type, and a value.
class EnvironmentManager:
    def __init__(self):
        self.environment = [{}]

    # returns a VariableDef object
    def get(self, symbol):
        for env in reversed(self.environment):
            if symbol in env:
                return env[symbol]

        return None

    def set(self, symbol, value):
        for env in reversed(self.environment):
            if symbol in env:
                env[symbol] = value
                return

        # symbol not found anywhere in the environment
        self.environment[-1][symbol] = value

    # create a new symbol in the top-most environment, regardless of whether that symbol exists
    # in a lower environment
    def create(self, symbol, value):
        self.environment[-1][symbol] = value

    # used when we enter a nested block to create a new environment for that block
    def push(self):
        self.environment.append({})  # [{}] -> [{}, {}]

    def push_with_env(self, env):
        self.environment.append(env)

    # used when we exit a nested block to discard the environment for that block
    def pop(self):
        self.environment.pop()

    def get_cur_env(self):
        return self.environment[-1]
    
    def get_env(self, stack_num):
        return self.environment[stack_num]
    
    def get_full_env_stack(self):
        return self.environment
    
    def set_env_stack(self, stack_num, symbol, value):
        self.environment[stack_num][symbol] = value

    def get_env_stack(self, stack_num, symbol):
        if symbol in self.environment[stack_num]:
            return self.environment[stack_num][symbol]
        return None
    
    def get_stack_size(self):
        return len(self.environment)
    
    def edit_lambda_env(self, stack_num, env):
        self.environment[stack_num] = env

    def pop_lambda(self):
        return self.environment.pop()
    
