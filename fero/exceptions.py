"""This module defines an error to be used for fero-specific exceptions."""


class FeroError(Exception):
    """A base error for fero-specific exceptions."""

    def __init__(self, *args):
        """
        Create a `FeroError` with arbitrary positional arguments.

        :param args: an argument list of arbitrary length.
            The first argument, if present, is the message of the error.
        """
        self.message = args[0] if len(args) > 0 else None
        super().__init__(*args)
