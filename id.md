我有一个论文的idea，请帮我整理一下：

1. 现在大模型服务部署缺少一个即插即用的配置优化器，用户对自己拥有的设备缺少一个先验的配置，导致大模型服务的性能无法得到最大化的利用
   1. 为了解决这个问题，我们提出了一个面向大模型服务的配置优化器，这个优化器可以：
      1. 根据用户当前的硬件条件和服务需求，自动调整服务的配置，使得服务的性能得到最大化的利用
      2. 支持在实例允许的时候动态更改配置，使得服务的性能可以随着服务负载的变化而变化
      3. 支持多种硬件，包括GPU和昇腾NPU
2. 所优化的配置包括:
   1. 服务模式：
      1. 默认模式，当queue中存在还没有进行prefill的请求时，接下来的step会被prefill请求占用，直到queue中没有prefill请求为止开始进行decode请求
      2. chunked prefill模式，当queue中存在还没有进行prefill的请求时，接下来的step会被prefill和decode请求组成的batch占用，这个batch被称为chunk，chunk的大小由`chunk_size`参数控制
      3. Disaggregation 模式，将prefill和decode放在不同的GPU上进行处理
   2. 超参数，包括 `max_num_seqs`和`chunk_size`


2. 相关工作面向llm 服务优化的 Simulator 是存在的，包括Apex和Vidur，但是它们更多的考虑到了对并行策略的挖掘以及对少数参数的调整，而没有考虑到对服务模式的调整以及动态调整配置的问题


3. 三种模式的优点和缺点：
   1. 三种模式的优点和缺点：
      1. 默认模式：
         1. 优点：简单，throughput 较高，ttft较低, 在离线场景超长输出任务下throughput最高
         2. 缺点：当queue中存在prefill请求时，decode请求会被阻塞，导致TPOT较低
      2. chunked prefill模式：
         1. 优点：throughput最高，很好的平衡了计算密集型（prefill）和访存密集型（decode）的请求
         2. 缺点：对于输入内容较长的请求，ttft较高
      3. Disaggregation 模式：
           1. 优点：ttft和tpot都较低，在给定SLO的情况下，可以得到满足最多qps的配置
           2. 缺点：存在KV Cache传输的开销，同时更适合适合于GPU集群的情况，对于少量GPU的情况（2或4），难以在生成内容较长的情况下得到最佳配置

4. 动态调整配置：
    1. 检测过去一段时间内的服务负载，根据负载的变化，调整服务的配置
   2. 案例：
      3. 过去一段时间内，request的输出长度比预期的要长，说明decode的时间较长，将Disaggregation模式切换为chunked prefill模式
      4. 过去一段时间内，queue中prefill/decode的时间比例变大，说明prefill的时间较长，将chunk_size调大或者切换为Disaggregation模式
      5. 过去一段时间内，queue中prefill/decode的时间比例变小，说明decode的时间较长，将chunk_size调小或者调大`max_num_seqs`