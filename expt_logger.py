from __future__ import print_function
import argparse
import datetime
import errno
import os
import shutil
import sys
import tempfile

class Logger(object):
    def __init__(self, filename="Expt_output.log"):
        self.filename = filename
        self.terminal = sys.stdout
        self.suffix = '_v'
        ctr = 0
        self.log = tempfile.TemporaryFile(mode='r+')
        #print("Std err redirected to ", self.log)


        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def switch_log_file(self, log_dir=''):
        temp = self.log
        new_log = os.path.join(log_dir,self.filename)
        self.log = open(new_log, 'w')
        temp.seek(0)
        shutil.copyfileobj(temp, self.log)
        temp.close()


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
        print ("Saving log file to {}".format(os.path.join(log_dir, self.filename)))

        # Create file indicating expt ran to completion
        with open(os.path.join(log_dir, "EXPT.COMPLETED"), 'w') as f:
            f.write("Done!")
        
