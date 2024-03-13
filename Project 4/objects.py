
class Object:
    def __init__(self):
        self.methods = {}
        self.fields = {}
        self.prototype = None

    def get_method(self, name): 
        object = self
        while object is not None:
            if name in object.methods:
                return object.methods[name]
            object = object.prototype
        return None

    
    def get_field(self, name):
        object = self
        while object is not None:
            if name in object.fields:
                return object.fields[name]
            object = object.prototype
        return None
    
    def set_method(self, name, method):
        self.methods[name] = method

    def set_field(self, name, field):
        self.fields[name] = field


            
