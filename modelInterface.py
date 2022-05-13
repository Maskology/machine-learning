# CELL 49
class ModelInterface:
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model = self.get_model()
        self.compile_params = self.get_compile_params()
        self.callbacks = self.get_callbacks()

    def get_model(self):
        raise NotImplementedError

    def get_compile_params(self):
        raise NotImplementedError

    def get_callbacks(self):
        return None
