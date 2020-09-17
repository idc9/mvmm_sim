import os
from pprint import pformat, pprint
import numbers


class ResultsWriter(object):
    def __init__(self, fpath=None, delete_if_exists=False,
                 to_print=True, to_write=True, newlines=2):
        self.fpath = fpath
        self.delete_if_exists = delete_if_exists
        self.to_print = to_print
        self.to_write = to_write
        self.newlines = int(newlines)

        if fpath is None:
            self.to_write = False
        else:
            if os.path.exists(self.fpath):
                if self.delete_if_exists:
                    os.remove(self.fpath)
                else:
                    raise ValueError('Warning {} already exists'.
                                     format(self.fpath))

    def write(self, text=None, name=None):

        if self.to_print:
            if name is not None:
                print(name)

            if text is not None:
                pprint(text)
            else:
                print()

            if self.newlines:
                for _ in range(self.newlines):
                    print()

        if self.to_write:
            with open(self.fpath, "a") as log_file:
                if name is not None:
                    log_file.write(name)
                    log_file.write(':\n')

                if text is not None:
                    log_file.write(pformat(text))

                if self.newlines:
                    log_file.write('\n' * self.newlines)
