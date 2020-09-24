class FeroError(Exception):
    def __init__(self, *args):
        self.message = args[0] if len(args) > 0 else None
        super().__init__(*args)
