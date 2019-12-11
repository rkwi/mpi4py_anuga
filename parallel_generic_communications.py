"""
Generic implementation of update_timestep and update_ghosts for
parallel domains (eg shallow_water or advection)

Ole Nielsen, Stephen Roberts, Duncan Gray, Christopher Zoppou
Geoscience Australia, 2004-2010

Roberto Vidmar, rvidmar@inogs.it 20130508 replaced pypar with par and
                removed pypar_extras dependence

"""

import numpy as num

import anuga_parallel.parallel_abstraction as par




def setup_buffers(domain):
    """Buffers for synchronisation of timesteps
    """

    domain.local_timestep = num.zeros(1, num.float)
    domain.global_timestep = num.zeros(1, num.float)

    domain.local_timesteps = num.zeros(domain.numproc, num.float)

    domain.communication_time = 0.0
    domain.communication_reduce_time = 0.0
    domain.communication_broadcast_time = 0.0


def communicate_flux_timestep(domain, yieldstep, finaltime):
    """Calculate local timestep
    """

    import time

    #Compute minimal timestep across all processes
    domain.local_timestep[0] = domain.flux_timestep
    t0 = time.time()


    par.allreduce(domain.local_timestep, par.MIN,
      buffer=domain.global_timestep,
      bypass=True)

    domain.communication_reduce_time += time.time()-t0




#    par.reduce(domain.local_timestep, par.MIN, 0,
#                      buffer=domain.global_timestep,
#                      bypass=True)
#
#
#    domain.communication_reduce_time += time.time()-t0


    #Broadcast minimal timestep to all processors
    t0 = time.time()
    #par.broadcast(domain.global_timestep, 0)#, bypass=True)

    domain.communication_broadcast_time += time.time()-t0

    #old_flux_timestep = domain.flux_timestep
    domain.flux_timestep = domain.global_timestep[0]



def communicate_ghosts_blocking(domain):

    # We must send the information from the full cells and
    # receive the information for the ghost cells
    # We have a dictionary of lists with ghosts expecting updates from
    # the separate processors

    import numpy as num
    import time
    t0 = time.time()

    # update of non-local ghost cells
    for iproc in range(domain.numproc):
        if iproc == domain.processor:
            #Send data from iproc processor to other processors
            for send_proc in domain.full_send_dict:
                if send_proc != iproc:

                    Idf  = domain.full_send_dict[send_proc][0]
                    Xout = domain.full_send_dict[send_proc][2]

                    for i, q in enumerate(domain.conserved_quantities):
                        #print 'Send',i,q
                        Q_cv =  domain.quantities[q].centroid_values
                        Xout[:,i] = num.take(Q_cv, Idf)

                    par.send(Xout, int(send_proc), use_buffer=True, bypass=True)


        else:
            #Receive data from the iproc processor
            if  domain.ghost_recv_dict.has_key(iproc):

                Idg = domain.ghost_recv_dict[iproc][0]
                X   = domain.ghost_recv_dict[iproc][2]

                X = par.receive(int(iproc), buffer=X, bypass=True)

                for i, q in enumerate(domain.conserved_quantities):
                    #print 'Receive',i,q
                    Q_cv =  domain.quantities[q].centroid_values
                    num.put(Q_cv, Idg, X[:,i])

    #local update of ghost cells
    iproc = domain.processor
    if domain.full_send_dict.has_key(iproc):

        # LINDA:
        # now store full as local id, global id, value
        Idf  = domain.full_send_dict[iproc][0]

        # LINDA:
        # now store ghost as local id, global id, value
        Idg = domain.ghost_recv_dict[iproc][0]

        for i, q in enumerate(domain.conserved_quantities):
            #print 'LOCAL SEND RECEIVE',i,q
            Q_cv =  domain.quantities[q].centroid_values
            num.put(Q_cv, Idg, num.take(Q_cv, Idf))

    domain.communication_time += time.time()-t0



def communicate_ghosts_asynchronous(domain):

    # We must send the information from the full cells and
    # receive the information for the ghost cells
    # We have a dictionary of lists with ghosts expecting updates from
    # the separate processors
    # Using isend and irecv

    import numpy as num
    import time
    t0 = time.time()

    # update of non-local ghost cells by copying full cell data into the
    # Xout buffer arrays

    #iproc == domain.processor

    #Setup send buffer arrays for sending full data to other processors
    for send_proc in domain.full_send_dict:
        Idf  = domain.full_send_dict[send_proc][0]
        Xout = domain.full_send_dict[send_proc][2]

        for i, q in enumerate(domain.conserved_quantities):
            #print 'Store send data',i,q
            Q_cv =  domain.quantities[q].centroid_values
            Xout[:,i] = num.take(Q_cv, Idf)



#    from pprint import pprint
#
#    if par.rank() == 0:
#        print 'Before commun 0'
#        pprint(domain.full_send_dict)
#
#    if par.rank() == 1:
#        print 'Before commun 1'
#        pprint(domain.full_send_dict)


    # Do all the comuunication using isend/irecv via the buffers in the
    # full_send_dict and ghost_recv_dict
    par.send_recv_via_dicts(domain.full_send_dict, domain.ghost_recv_dict)

#
#    if par.rank() == 0:
#        print 'After commun 0'
#        pprint(domain.ghost_recv_dict)
#
#    if par.rank() == 1:
#        print 'After commun 1'
#        pprint(domain.ghost_recv_dict)

    # Now copy data from receive buffers to the domain
    for recv_proc in domain.ghost_recv_dict:
        Idg  = domain.ghost_recv_dict[recv_proc][0]
        X    = domain.ghost_recv_dict[recv_proc][2]

        #print recv_proc
        #print X

        for i, q in enumerate(domain.conserved_quantities):
            #print 'Read receive data',i,q
            Q_cv =  domain.quantities[q].centroid_values
            num.put(Q_cv, Idg, X[:,i])


    domain.communication_time += time.time()-t0


