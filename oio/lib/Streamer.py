#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sep 24, 2015 15:32
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Stream data to buffer
"""
import logging
import multiprocessing as mp
import signal
from time import time, sleep
from oio.lib.SharedBuffer import SharedBuffer
from queue import Empty


logger = logging.getLogger('Streamer')
TIMEOUT = 5


class Streamer(mp.Process):
    def __init__(self, queue, raw, update_interval=0.02):
        super(Streamer, self).__init__()

        # #### WARNING #####
        # On Windows, a logging.logger can't be added to the class
        # as loggers can't be pickled. There probably is a way around
        # using configuration dicts, but I'll stay away from that...
        # self.logger = logging.getLogger(__name__)
        # self.logger = mp.log_to_stderr()
        logger.debug('{} process initializing'.format(self.name))

        # # Queue Interface
        self.commands = {'stop': self.stop,
                         'offset': self.reposition,
                         'heartbeat': self.heartbeat}
        self.queue = queue
        self.alive = True
        self.update_interval = update_interval
        self.last_update = time()

        # Shared Buffer
        self.raw = raw
        self.buffer = SharedBuffer()

        # # Data specifics
        self.offset = None

    def run(self):
        """Main streaming loop."""
        # ignore CTRL+C, runs daemonic, will stop with parent or on heartbeat timeout
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.debug('Running...')

        self.buffer.initialize_from_raw(self.raw)
        self.last_update = time()

        while self.alive:
            if time() - self.last_update > 5:
                logger.error('Streamer exit on timeout: {}'.format(time()-self.last_update))
                self.alive = False
                break

            # Grab all messages currently in the queue
            instructions = self.__get_instructions()
            stop = 'stop' in [instr[0] for instr in instructions]

            # only use the last issued position update
            last_pos_instr = [instr for instr in instructions if instr[0] == 'offset']
            last_pos_instr = last_pos_instr[-1] if len(last_pos_instr) else []
            sub_instr = [instr for instr in instructions if instr[0] != 'offset']

            if len(last_pos_instr):
                sub_instr.append(last_pos_instr)

            for instr in sub_instr:
                if stop:
                    self.stop(None)
                    break
                self.__execute_instruction(instr)
            else:
                for instr in [instr for instr in instructions if instr[0] != 'heartbeat']:
                    logger.debug('Instructions: {}'.format(instr))
            sleep(self.update_interval)

    def heartbeat(self, t):
        self.last_update = t

    def stop(self, _):
        logger.debug('Received Stop Signal')
        self.alive = False

    def reposition(self, offset):
        pass

    def __get_instructions(self):
        cmdlets = []
        while self.queue.qsize():
            try:
                cmdlets.append(self.queue.get(False))
            except Empty:
                # logger.debug('Queue empty!')
                break

        return cmdlets

    def __execute_instruction(self, instruction):
        if len(instruction) == 2 and instruction[0] in self.commands:
            self.commands[instruction[0]](instruction[1])
            # logger.warning('Unable to execute {} because: {}'.format(instruction, e))
        else:
            logger.warning('Ignoring invalid instruction {}'.format(instruction))

    def __add_command(self, command, func):
        self.commands[command] = func


if __name__ == "__main__":
    pass
