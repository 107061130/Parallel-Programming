# PP HW 4 Report Template
> - Please include both brief and detailed answers.
> - The report should be based on the UCX code.
> - Describe the code using the 'permalink' from [GitHub repository](https://github.com/NTHU-LSALAB/UCX-lsalab).

## 1. Overview
> In conjunction with the UCP architecture mentioned in the lecture, please read [ucp_hello_world.c](https://github.com/NTHU-LSALAB/UCX-lsalab/blob/ce5c5ee4b70a88ce7c15d2fe8acff2131a44aa4a/examples/ucp_hello_world.c)
1. Identify how UCP Objects (`ucp_context`, `ucp_worker`, `ucp_ep`) interact through the API, including at least the following functions:
    ###  ucp_hello_world.c - main function
    **1. Initialization and Variable Declarations:**
    * Initializes various UCX-related variables (**ucp_params**, **worker_attr**, **worker_params**).
    * Initializes error handling options (**err_handling_opt**).
    * Initializes global variables related to test modes, port number.

    **2. Parsing Command-Line Arguments:**
    * Calls the **parse_cmd** function to parse command-line arguments.
    * Parses options such as test mode, emulation of unexpected failures, server name, port.

    **3. UCX Initialization:**
    * Reads the UCX configuration using **ucp_config_read**.
    * Initializes the UCX context (**ucp_context**) using **ucp_init**.
    * Creates a UCX worker (**ucp_worker**) using **ucp_worker_create**.
    * Queries worker attributes to obtain the local address.

    **4. Out-of-Band (OOB) Connection Establishment:**
    * Handles out-of-band (OOB) connection establishment based on whether the program is running as a client or server.
    * For a client, connects to the server and receives the server's UCX address.
    * For a server, waits for a client connection and sends its UCX address.

    **5. Client or Server Logic Execution:**
    * If the program is running as a client (**client_target_name** is not NULL), calls **run_ucx_client** function.
        * Call **ucp_ep_create** to create ucp endpoint(**client_ep**, **ep_params**) handle communication
        * Sends the client's UCX address to the server.
        * Receives a test string from the server.

    * If the program is running as a server, calls **run_ucx_server** function.
        * Call **ucp_ep_create** to create ucp endpoint(**server_ep**, **ep_params**) handle communication
        * Waits for the client's UCX address.
        * Sends a test string to the client.

    **6. Barrier for Synchronization:**
    * If there are no errors and the failure mode is not set, performs a synchronization barrier using **barrier** function.

    **7. Cleanup:**
    * Closes the out-of-band (OOB) socket.
    * Frees allocated memory for the peer address.

    **8. Worker and Context Cleanup:**
    * Releases the worker's UCX address using **ucp_worker_release_address**.
    * Destroys the UCX worker using **ucp_worker_destroy**.
    * Cleans up the UCX context using **ucp_cleanup**.

    **9. Return Statement:**
    * Returns an exit code (**0** for success, **-1** for failure).

2. What is the specific significance of the division of UCP Objects in the program? What important information do they carry?
    - `ucp_context`
    - `ucp_worker`
    - `ucp_ep`
    
    **1. ucp_context:**
    * Configuration settings: Features, request size, thread mode.
    * Handles to memory resources, such as memory pools.
    * Information needed for global UCP operations.

    **2. ucp_worker:**
    * Communication resources: Endpoint handles, listener handles.
    * Handles to memory resources used for communication.
    * Worker-specific configuration settings.

    **3. ucp_ep (ucp endpoint):**
    * Endpoint-specific configuration settings.
    * State information for communication with a remote peer.
    * Handles to memory resources associated with the specific endpoint.

    In summary, the division of UCP objects follows a hierarchical structure where the **ucp_context** is the global context, the **ucp_worker** represents a specific worker thread handling communication, and **ucp_ep** represents communication endpoints with remote peers. 
    
    Each level addresses specific concerns and responsibilities. This separation of concerns
    * simplifies the code, making it more readable and maintainable.
    * enables efficient resource management.
    * suitable for handling complex communication scenarios involving multiple workers and endpoints.
    
3. Based on the description in HW4, where do you think the following information is loaded/created?
    - `UCX_TLS`
    in **ucp_worker** layer, because it handle communication resources
    - TLS selected by UCX
    in **ucp_worker** layer, because communication strategy of each ep need to be pre-determined.

## 2. Implementation

> Describe how you implemented the two special features of HW4.
1. Which files did you modify, and where did you choose to print Line 1 and Line 2?
* `ucp/core/ucp_worker.c`, `ucs/config/parser.c`, `ucs/config/types.h`
* Line1 in `cp/core/ucp_worker.c` - `ucp_worker_print_used_tls()`
https://github.com/NTHU-LSALAB/UCX-lsalab/blob/7c0362c97c8fe9cbeaacaac90271dde0210ac529/src/ucp/core/ucp_worker.c#L1763
* Line2 in `ucs/config/parser.c` - `ucs_config_parser_print_opts()` 
https://github.com/NTHU-LSALAB/UCX-lsalab/blob/7c0362c97c8fe9cbeaacaac90271dde0210ac529/src/ucs/config/parser.c#L1853
2. How do the functions in these files call each other? Why is it designed this way?
`ucp_worker_print_used_tls()` --> `ucp_config_print()` --> `ucs_config_parser_print_opts()` 

3. Observe when Line 1 and 2 are printed during the call of which UCP API?
* Line1 in `cp/core/ucp_worker.c` - `ucp_worker_print_used_tls()`
https://github.com/NTHU-LSALAB/UCX-lsalab/blob/7c0362c97c8fe9cbeaacaac90271dde0210ac529/src/ucp/core/ucp_worker.c#L1763
* Line2 in `ucs/config/parser.c` - `ucs_config_parser_print_opts()` 
https://github.com/NTHU-LSALAB/UCX-lsalab/blob/7c0362c97c8fe9cbeaacaac90271dde0210ac529/src/ucs/config/parser.c#L1853

4. Does it match your expectations for questions **1-3**? Why?
Both `UCX_TLS` and TLS selected by UCX is in ucp_worker layer, because it handles End points communication. Match my expection.

5. In implementing the features, we see variables like lanes, tl_rsc, tl_name, tl_device, bitmap, iface, etc., used to store different Layer's protocol information. Please explain what information each of them stores.
* **lanes:**
    * Refer to communication channels or paths for data transmission. These channels represent parallel communication paths, allowing for concurrent data transfers to enhance performance.
    
* **tl_rsc:**
    * Stand for "Transport Layer Resource." This variable represent resources associated with a specific transport layer in UCX, potentially indicating the availability or configuration of resources for communication.

* **tl_name:**
    * Stands for "Transport Layer Name." It could represent the name or identifier of a particular transport layer within UCX. Transport layers in UCX may correspond to different communication protocols or methods.

* **tl_device:**
    * Refer to the device or hardware associated with a specific transport layer in UCX. This is a reference to the physical or virtual communication interface used by the UCX framework.

* **bitmap:**
    * A data structure to represent certain properties or settings. Managing and representing information using a series of bits.

* **iface:**
    * Communication interface, such as a network interface or a specific component that allows communication between different entities within the framework.

## 3. Optimize System 
1. Below are the current configurations for OpenMPI and UCX in the system. Based on your learning, what methods can you use to optimize single-node performance by setting UCX environment variables?

```
-------------------------------------------------------------------
/opt/modulefiles/openmpi/4.1.5:

module-whatis   {Sets up environment for OpenMPI located in /opt/openmpi}
conflict        mpi
module          load ucx
setenv          OPENMPI_HOME /opt/openmpi
prepend-path    PATH /opt/openmpi/bin
prepend-path    LD_LIBRARY_PATH /opt/openmpi/lib
prepend-path    CPATH /opt/openmpi/include
setenv          UCX_TLS ud_verbs
setenv          UCX_NET_DEVICES ibp3s0:1
-------------------------------------------------------------------
```

> Please use the following commands to test different data sizes for latency and bandwidth, to verify your ideas:
```bash
module load openmpi/4.1.5
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/standard/osu_latency
mpiucx -n 2 $HOME/UCX-lsalab/test/mpi/osu/pt2pt/standard/osu_bw
```
* **Origin**
    * use offset ud_verbs(infini-band)
    ```
    UCX_TLS=ud_verbs
    0x55f940f99dc0 self cfg#0 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x55d4aad93e50 self cfg#0 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x55d4aad93e50 intra-node cfg#1 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x55f940f99dc0 intra-node cfg#1 tag(ud_verbs/ibp3s0:1)
    # OSU MPI Latency Test v7.3
    # Size          Latency (us)
    # Datatype: MPI_CHAR.
    1                       1.63
    2                       1.54
    4                       1.74
    8                       1.52
    16                      1.65
    32                      1.90
    64                      1.88
    128                     1.94
    256                     3.09
    512                     3.42
    1024                    4.13
    2048                    5.61
    4096                    8.71
    8192                   10.62
    16384                  14.42
    32768                  22.33
    65536                  36.08
    131072                 62.28
    262144                123.91
    524288                233.76
    1048576               451.30
    2097152               888.97
    4194304              1785.03
    ```
    ```
    UCX_TLS=ud_verbs
    0x562f7748ee10 self cfg#0 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x55d5204d2e10 self cfg#0 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x562f7748ee10 intra-node cfg#1 tag(ud_verbs/ibp3s0:1)
    UCX_TLS=ud_verbs
    0x55d5204d2e10 intra-node cfg#1 tag(ud_verbs/ibp3s0:1)
    # OSU MPI Bandwidth Test v7.3
    # Size      Bandwidth (MB/s)
    # Datatype: MPI_CHAR.
    1                       2.26
    2                       5.77
    4                      11.85
    8                      23.67
    16                     45.12
    32                     90.37
    64                    171.45
    128                   308.88
    256                   401.63
    512                   715.14
    1024                 1177.03
    2048                 1703.45
    4096                 1730.86
    8192                 1999.59
    16384                2266.95
    32768                2350.21
    65536                2430.44
    131072               2430.26
    262144               2459.22
    524288               2489.06
    1048576              2482.73
    2097152              2465.35
    4194304              2407.16
    ```
    
* **optimized**
    * use `-x UCX_TLS=all` to list all communication strategies and select the best. 
    * For self communication, it select **self/memory** **cma/memor**y
    * For intra-node communication, it select **sysv/memory** **cma/memor**y
    * It can be obesrved that we get lower lantecy and higher bandwidth
    ```
    UCX_TLS=all
    0x564f8bb51e50 self cfg#0 tag(self/memory cma/memory)
    UCX_TLS=all
    0x555ae797edc0 self cfg#0 tag(self/memory cma/memory)
    UCX_TLS=all
    0x564f8bb51e50 intra-node cfg#1 tag(sysv/memory cma/memory)
    UCX_TLS=all
    0x555ae797edc0 intra-node cfg#1 tag(sysv/memory cma/memory)
    # OSU MPI Latency Test v7.3
    # Size          Latency (us)
    # Datatype: MPI_CHAR.
    1                       0.22
    2                       0.22
    4                       0.22
    8                       0.22
    16                      0.22
    32                      0.25
    64                      0.25
    128                     0.39
    256                     0.42
    512                     0.45
    1024                    0.53
    2048                    0.69
    4096                    1.01
    8192                    1.71
    16384                   3.04
    32768                   4.93
    65536                   8.60
    131072                 17.26
    262144                 35.38
    524288                 66.66
    1048576               131.79
    2097152               271.03
    4194304              1052.77
    ```
    ```
    UCX_TLS=all
    0x5567c2c0eea0 self cfg#0 tag(self/memory cma/memory)
    UCX_TLS=all
    0x5602974bfe10 self cfg#0 tag(self/memory cma/memory)
    UCX_TLS=all
    0x5602974bfe10 intra-node cfg#1 tag(sysv/memory cma/memory)
    UCX_TLS=all
    0x5567c2c0eea0 intra-node cfg#1 tag(sysv/memory cma/memory)
    # OSU MPI Bandwidth Test v7.3
    # Size      Bandwidth (MB/s)
    # Datatype: MPI_CHAR.
    1                       9.25
    2                      18.65
    4                      37.51
    8                      75.63
    16                    128.82
    32                    293.90
    64                    501.43
    128                   609.18
    256                  1199.14
    512                  2197.58
    1024                 3720.08
    2048                 5506.86
    4096                 7942.07
    8192                 9889.92
    16384                5060.00
    32768                7101.55
    65536                8348.90
    131072               8805.06
    262144               8189.27
    524288               8244.04
    1048576              8105.20
    2097152              8097.84
    4194304              7319.83
    ```
    
### Advanced Challenge: Multi-Node Testing

This challenge involves testing the performance across multiple nodes. You can accomplish this by utilizing the sbatch script provided below. The task includes creating tables and providing explanations based on your findings. Notably, Writing a comprehensive report on this exercise can earn you up to 5 additional points.

- For information on sbatch, refer to the documentation at [Slurm's sbatch page](https://slurm.schedmd.com/sbatch.html).
- To conduct multi-node testing, use the following command:
```
cd ~/UCX-lsalab/test/
sbatch run.batch
```
```
# OSU MPI Latency Test v7.3
# Size          Latency (us)
# Datatype: MPI_CHAR.
0x556ba2112060 self cfg#0 tag(self/memory cma/memory rc_verbs/ibp3s0:1)
0x55d2740a35c0 inter-node cfg#1 tag(rc_verbs/ibp3s0:1 tcp/ibp3s0)
1                       1.86
2                       1.84
4                       1.82
8                       1.83
16                      1.83
32                      1.84
64                      1.99
128                     3.15
256                     3.32
512                     3.57
1024                    4.14
2048                    5.22
4096                    7.40
8192                    9.46
16384                  12.77
32768                  18.49
65536                  29.83
131072                 52.84
262144                 94.56
524288                180.26
1048576               352.90
2097152               698.23
4194304              1389.74
0x556ba2112060 inter-node cfg#1 tag(rc_verbs/ibp3s0:1 tcp/ibp3s0)
```

## 4. Experience & Conclusion
1. What have you learned from this homework?
UXC Framework
