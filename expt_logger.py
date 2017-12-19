from __future__ import print_function
import argparse
import ConfigParser
import datetime
import errno
import os
import shutil
import sys

class Logger(object):
    def __init__(self, filename="Expt_output.log"):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(filename, "a")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stderr = sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def stop_log(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.terminal = open(os.devnull, 'w')
        
    def close_log(self, log_dir=''):
        self.log.close()
        if log_dir != '':
            shutil.move(self.filename,
                        os.path.join(log_dir, self.filename))
        print ("Saving log file to {}".format(os.path.join(log_dir, self.filename)))
