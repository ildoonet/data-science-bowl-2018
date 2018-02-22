import threading
import logging
import sys
from contextlib import contextmanager

import tensorflow as tf

from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated


logger = logging.getLogger('DataFlowToQueue')


class DataFlowToQueue(threading.Thread):
    def __init__(self, ds, placeholders, queue_size=5):
        super(DataFlowToQueue).__init__()
        self.daemon = True

        self.ds = ds
        self.placeholders = placeholders
        self.queue = tf.FIFOQueue(queue_size, [ph.dtype for ph in placeholders if ph is not None],
                                  shapes=[ph.get_shape() for ph in placeholders if ph is not None])
        self.op = self.queue.enqueue(placeholders)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._coord = None
        self._sess = None

        self.last_dp = None

    @contextmanager
    def default_sess(self):
        if self._sess:
            with self._sess.as_default():
                yield
        else:
            logger.warning("DataFlowToQueue {} wasn't under a default session!".format(self.name))
            yield

    def size(self):
        return self.queue.size()

    def start(self):
        self._sess = tf.get_default_session()
        super(DataFlowToQueue).start()

    def set_coordinator(self, coord):
        self._coord = coord

    def run(self):
        with self.default_sess():
            try:
                while not self._coord.should_stop():
                    try:
                        self.ds.reset_state()
                        while True:
                            for dp in self.ds.get_data():
                                feed = {ph: ds for ph, ds in zip(self.placeholders, dp) if ph is not None}
                                self.op.run(feed_dict=feed)
                                self.last_dp = dp
                    except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                        logger.error('err type1, placeholders={}'.format(self.placeholders))
                        sys.exit(-1)
                    except Exception as e:
                        logger.error('err type2, err={}, placeholders={}'.format(str(e), self.placeholders))
                        if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                            pass
                        else:
                            logger.exception("Exception in {}:{}".format(self.name, str(e)))
                        sys.exit(-1)
            except Exception as e:
                logger.exception("Exception in {}:{}".format(self.name, str(e)))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def dequeue(self):
        return self.queue.dequeue()