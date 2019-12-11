""" Abstract parallel interface - suitable for sequential programs

    Uses either mpi4py or pypar for parallism if installed.
    Otherwise define a rudimentary interface for sequential execution.

    mpi4py wrap added 20130503 by Roberto Vidmar rvidmar@inogs.it
"""
import sys
import os
import time
import numpy as np

try:
  if (os.environ.has_key('ANUGA_MPI4PY') and
    os.environ['ANUGA_MPI4PY'].startswith('y')):
    raise ImportError
  # Try first pypar, this is the default behavior
  import pypar

# Roberto Vidmar, 20130415: the following exception imports mpi4py
except ImportError:
  try:
    from mpi4py import MPI

    # Wrap pypar calls and constants
    comm = MPI.COMM_WORLD
    get_processor_name = MPI.Get_processor_name
    finalize = MPI.Finalize
    barrier = comm.barrier
    default_tag = 1
    MAX = MPI.MAX
    MIN = MPI.MIN
    SUM = MPI.SUM
    PROD = MPI.PROD
    LAND = MPI.LAND
    BAND = MPI.BAND
    LOR = MPI.LOR
    BOR = MPI.BOR
    LXOR = MPI.LXOR
    BXOR = MPI.BXOR
    MAXLOC = MPI.MAXLOC
    MINLOC = MPI.MINLOC
    mpiWrapper = 'mpi4py'
    MAX_COMBUF = 10

    class Status(object):
      """ Simulate pypar return_status object
      """
      def __init__(self, status, buf):
        self.source = status.Get_source()
        self.tag = status.Get_tag()
        self.error = status.Get_error()
        self.length = len(buf)
        if isinstance(buf, np.ndarray):
          self.size = buf.dtype.itemsize
        else:
          self.size = 1

      def __repr__(self):
        s = "Par Status Object:\n"
        s += "  source=%d\n" % self.source
        s += "  tag=%d\n" % self.tag
        s += "  error=%d\n" % self.error
        s += "  length=%d\n" % self.length
        s += "  size=%d\n" % self.size
        return s

      def bytes(self):
        """ Number of bytes transmitted (excl control info)
        """
        return self.length * self.size

    def abort():
      comm.Abort()

    def allreduce(sendbuf, op, buffer=None, vanilla=0, bypass=False):
      return comm.Allreduce(sendbuf, buffer, op=op)

    def broadcast(buffer, root, vanilla=False, bypass=False):
      """ Uses numpy array Bcast if bypass is True
      """
      if bypass:
        return comm.Bcast(buffer, root)
      else:
        return comm.bcast(buffer, root)

    def gather(x, root, buffer=None, vanilla=0):
      """ Simulate pypar gather, vanilla is unused
      """
      if isinstance(x, str):
        recvmsg = comm.gather(x, root)
        if recvmsg is not None:
          return ''.join(recvmsg)
      elif isinstance(x, np.ndarray):
        if buffer is None:
          # Create a suitable buffer
          if x.ndim == 1:
            buffer = np.empty(x.size * comm.size, dtype = x.dtype)
          elif x.ndim == 2:
            buffer = np.empty((x.shape[0] * comm.size, x.shape[1]),
              dtype = x.dtype)
          else:
            raise NotImplementedError("Dimension of %d is not implemented" %
              x.ndim)
        comm.Gather(x, buffer, root)
        return buffer

    def print0(*args):
      """ Print arguments only if our rank is 0
      """
      if comm.rank == 0:
        for a in args:
          print a,
        print

    def rank():
      return comm.rank

    def receive(source, buffer=None, vanilla=False, tag=1, return_status=False,
      bypass=False):
      """ This wrap uses Recv for receiving if bypass is True, else recv
      """
      s = MPI.Status()
      if bypass:
        if not isinstance(buffer, np.ndarray):
          print "receive: This should NEVER happen!"
        comm.Recv(buffer, source=source, tag=tag, status=s)
      else:
        if buffer is None:
          buffer = comm.recv(source=source, tag=tag, status=s)
        else:
          buf = comm.recv(source=source, tag=tag, status=s)
          if isinstance(buf, np.ndarray):
            buffer[:] = buf.reshape(buffer.shape)
          else:
            buffer = buf

      if return_status:
        rs = Status(s, buffer)
        return buffer, rs
      else:
        return buffer

    def reduce(x, op, root, buffer=None, vanilla=0, bypass=False):
      """ Simulate pypar reduce, vanilla and bypass are not used
      """
      return comm.Reduce(x, buffer, op=op, root=root)

    def scatter(x, root, buffer=None, vanilla=False):
      """ Simulate pypar scatter, vanilla is not used
      """
      if isinstance(x, str):
        scatterer = comm.scatter
        l = len(x)
        n = l / comm.size
        sendobj = [x[i: i + n] for i in range(0, l, n)]
      elif isinstance(x, np.ndarray):
        scatterer = comm.Scatter
        sendobj = x
        if buffer is None:
          buffer = np.empty(x.size / comm.size, dtype=x.dtype)
      else:
        raise NotImplementedError(
          'Can only scatter strings and numpy arrays')

      if comm.rank:
        sendobj = None

      recvmsg = scatterer(sendobj, buffer, root=root)
      if recvmsg is None:
        return buffer
      else:
        return recvmsg

    def send(x, destination, use_buffer=False, vanilla=False, tag=1,
      bypass=False):
      """ This wrap uses Send for sendig if bypass is True, else send
      """
      if bypass:
        comm.Send(np.ascontiguousarray(x), dest=destination, tag=tag)
      else:
        comm.send(x, dest=destination, tag=tag)

    def size():
      return comm.size

    def send_recv_via_dicts(sendDict, recvDict):
      """ This wrap uses Irecv and Isend for exchanging numpy arrays stored
          in dicts.
      """
      if len(recvDict) > MAX_COMBUF:
        raise ValueError(
          "send_recv_via_dicts: Number of recv communication buffers > %d" %
          MAX_COMBUF)
      if len(sendDict) > MAX_COMBUF:
        raise ValueError(
          "send_recv_via_dicts: Number of send communication buffers > %d" %
          MAX_COMBUF)
      requests = []
      if len(recvDict) != len(sendDict):
        raise NotImplementedError("len(recvDict) != len(sendDict): %d %d" %
        len(recvDict), len(sendDict))

      skeys = sendDict.keys()
      skeys.sort()
      rkeys = recvDict.keys()
      rkeys.sort()
      assert rkeys == skeys
      # Keys are integers, indexes of processes
      for key in rkeys:
        recvBuf = recvDict[key][2]
        sendBuf = sendDict[key][2]
        comm.Sendrecv(np.ascontiguousarray(sendBuf), key, 123,
          recvBuf, key, 123)

    numprocs = size()
    myid = rank()

  except ImportError:
    print ('WARNING: Could not import neither pypar nor mpi4py - '
      'defining sequential interface')

    def size():
        return 1

    def rank():
        return 0

    def get_processor_name():
        try:
            hostname = os.environ['HOST']
        except:
            try:
                hostname = os.environ['HOSTNAME']
            except:
                hostname = 'Unknown'

        return hostname

    def abort():
        sys.exit()

    def finalize():
        pass

    def barrier():
        pass

    def time():
        return time.time()

    def send(*args, **kwargs):
        pass

    def print0(*args):
      """ Print arguments
      """
      for a in args:
        print a,
      print

    def receive(*args, **kwargs):
        pass

    def reduce(*args, **kwargs):
        pass

    def allreduce(*args, **kwargs):
        pass

    def send_recv_via_dicts(*args, **kwargs):
        pass

    MIN = None

    pypar_available = False
    mpiWrapper = None
  else:
    # Ok, mpi4py imported
    print0('WARNING: will use mpi4py instead of pypar')
    pypar_available = True
else:
  # Ok, pypar imported
  from pypar import *
  import anuga_parallel.pypar_extras as pypar_extras

  def print0(*args):
    """ Print arguments only if our rank is 0
    """
    if rank() == 0:
      for a in args:
        print a,
      print

  def allreduce(*args, **kargs):
    return pypar_extras.allreduce(*args, **kargs)

  def send_recv_via_dicts(sendDict, recvDict):
    pypar_extras.mpiextras.send_recv_via_dicts(
      sendDict, recvDict)

  numprocs = size()
  myid = rank()

  sys.stdout = sys.__stdout__
  pypar_available = True
  mpiWrapper = 'pypar'
