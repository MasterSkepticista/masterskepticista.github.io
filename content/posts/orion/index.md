---
title: "Data Parallelism using standard Ethernet"
date: 2024-08-05
summary: "Bag-of-tricks for multi-node training for the GPU Poor."
tags: ["data parallelism", "multi-node", "jax", "tensorflow", "pytorch"]
---

{{<katex>}}

{{< lead >}}
Big model ask big money.
{{< /lead >}}

The limiting factor on the size of models that can be trained, can be summarized to data movement speeds - either within the GPU (HBM bandwidth) or across GPUs (collective ops bandwidth). A big part of model scaling with the number of accelerators is achieved by reducing communication overhead across them. This is why protocols like Infiniband/NVLink exist.

But can we get away without spending a fortune on 100G/400G NICs for training models across nodes? Turns out, under the right assumptions, we can.

## Infrastructure

We had four[^1] server blades each with the following spec:
* Dual socket Xeon 6258R (28C/56T per socket)
* 512GB DDR4 Memory
* One RTX-3090 GPU
* Intel X540 Dual-port 10GbE, one of the ports was utilized for Internet via a 1GbE link.

## Goals

We had to consolidate these servers into a multi-node training cluster:
* With above **90%** scaling factor.
* With support for medium sized models (think ResNet-50/101 or ViT-S/B) up to 100M params.
* Using 10G Ethernet only. Any other NIC would require a new switch and a new card.

The constraint: spend $0.

## Single GPU Training: Baseline

Our focus was on data-parallel training since we could fit all our models on a single GPU (modulo tuning the batch size). 

Take ResNet50 for example. Training a ResNet50 on 90 epochs of ImageNet-1k on a single blade takes 48h in the default `TensorFloat32` precision. Since Ampere+ architectures support `bfloat16`, we could reduce data movement within the GPU, and saturate the Tensor Cores (Tesla architecture supports `float16`, but that requires gradient scaling in the training loop to avoid overflows. I won't cover that here given there exist plenty of guides online on how to use `float16`).

Here is a summary of changes:
```python
# JAX
conv = nn.Conv(..., dtype=jnp.bfloat16)
dense = nn.Dense(..., dtype=jnp.bfloat16)
...

# TensorFlow (keras)
tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

# Torch
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
  ...
```

We wrote our training pipelines in TensorFlow + JAX for two main reasons:
* Fine-grained data ETL tuning with `tf.data`, and 
* Higher GPU throughput via XLA. This frees us from low-level compiler optimizations. (If you are a PyTorch user, `torch.compile` will do this for you)

Half-precision matmuls got us down to 25h while hitting the same Top-1 score of 76.3%. We clocked a per-step time \\(T_s=230\text{ms}\\) (i.e. time taken per forward/backward pass of a batch). This was a reasonable single GPU baseline.

At this point we can extrapolate to our "ideal" cluster: it would train a ResNet50 just **below 6.3h with 100% scaling** across four nodes.

### Bandwidth Calculation

Training across nodes requires taking a global average of gradients across all GPUs on each backward pass. So each GPU would have to send and receive a full set of `gradient_size` data at each step.

$$
\text{gradient_size} = \frac{#\text{params}}{1e^6} \times \text{bytes_per_param}
$$

For a ResNet50 with 25M parameters, `gradient_size` is roughly 100MB per step, per GPU. Since each GPU needs a full copy of globally averaged gradients - a naive algorithm would require the lead host to `fetch` and `broadcast` 100MB data to/from each GPU. This would create a massive bottleneck on the main host, since the communication time would grow linearly on the number of GPUs.

Lucky for us, most implementations[^nccl] of collectives today use the `RingAllReduce` algorithm, which amortizes the amount of transfers as number of GPUs increase, by communicating 'chunked' gradients. In other words: data communicated per GPU reaches an asymptotic limit, independent of the number of GPUs in the cluster.

$$
\text{data_per_gpu} = 2 (N - 1) \frac{\text{gradient_size}}{N}
$$

If you are interested in the proof, Gibiansky has a great article explaining the [`RingAllReduce`](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) algorithm.

{{< alert "circle-info" >}}
In complex topologies spanning thousands of GPUs, a [`HierarchicalAllReduce`](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/) algorithm scales better.
{{< /alert >}}

On our 4-node cluster with 10GbE bi-directional links, time spent in communication would be

$$
T_c = \frac{\text{bytes}}{\text{bandwidth}} = \frac{150}{1.25 \times 1024} = 0.11 \text{s.}
$$

So we would pay a fixed cost of 110ms each time, to synchronize gradients.

## Multi GPU Training: Baseline

Lets start with a simple baseline that connects all 4 blades through a 10G switch. We can measure the time spent in computation and communication using Tensorboard profiler.

{{< figure
    src="/posts/orion/tb_prof_overview.png"
    alt="Step profile of ResNet50"
    caption="Step profile of ResNet50"
>}}

With a total forward + backward pass time \\(T_s = 234\text{ms}\\), we spend an additional \\(T_c = 106\text{ms}\\) in communication ("NCCL"), in line with our estimate above. Note that we already save time prefetching batches to the GPU by overlapping it with computation ("Copy" step).
With this information, we can calculate the scaling efficiency \\(\eta\\) of our cluster.

$$
\eta = \frac{T_s}{T_s + T_c} \approx 68.8\%
$$

Now, our ResNet50 takes 9.1h to train (i.e., a \\(2.74\times\\) speedup over single GPU baseline). It is a sizeable jump, but notice that more than 1 GPU worth of our compute is spent idling.

{{< figure
    src="/posts/orion/baseline.png"
    alt="Scaling efficiency baseline"
    caption="Scaling efficiency baseline"
>}}

If we take a close look at the formulation of \\(\eta\\), we only have two ways to increase scaling efficiency from here:
1. Reduce communication (\\(T_c\\)), and/or
2. Defer communication (by increasing \\(T_s\\)).

We will now explore each optimization in detail.

## 1. Reducing Communication

Upto this point we communicate 25M `float32` values at the end of each step. One way to reduce communication could be by compressing gradients (lossy or otherwise). Here are our options:
1. Cast gradients to `bfloat16`: No risk of overflow, but lossy due to high machine \\(\epsilon\\)[^bf16eps].
2. Cast gradients to `float16`: Risk of overflow, but lossless if renormalized.
3. Use a more intelligent gradient compression schema (like sparsity?).

We'll stick with a simple technique that worked for us. We cast gradients to `bfloat16` during communication. We did not observe any loss in accuracy.

```python
# JAX
grads = jax.tree.map(lambda g: g.astype(jnp.bfloat16), grads)
grads = lax.pmean(grads, axis_name="batch")  # AllReduce
grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)

# Tensorflow (Keras)
def compressed_aggregate_gradients(grads_and_vars):
  """An override for `tf.optimizers.Optimizer.aggregate_gradients` method to 
  compress gradients before allreduce."""
  grads, vars = zip(*grads_and_vars)
  grads = [tf.cast(g, tf.float16) for g in grads]
  grads = tf.distribute.get_replica_context().all_reduce(
      tf.distribute.ReduceOp.SUM, grads)
  grads = [tf.cast(g, tf.float32) for g in grads]
  grads_and_vars = zip(grads, vars)
  return grads_and_vars

optimizer.aggregate_gradients = compressed_aggregate_gradients
```

Our scaling efficiency with halved communication time is:

$$
\eta = \frac{T_s}{T_s + 0.5 \times T_c} = \frac{234}{234 + 53} \approx 81.5\%
$$

{{< figure
    src="/posts/orion/gcompression.png"
    alt="Gradient compression results"
    caption="Gradient compression results"
>}}

...which is pretty neat! This brings down our training time from 9.1h to 7.7h.


## 2. Deferring Communication

Gradient synchronization is required at the end of each batch, and there are only so many samples we can fit in a single forward/backward pass per batch...

...or can we?

Gradient accumulation is a common technique to emulate large batch sizes on GPUs with limited memory. But this can also be seen as a way of deferring communication. If the maximum batch size supported on a forward/backward pass is 512, which was the case for us here, we could prepare a larger 1024-size batch, and sum over gradients within the GPU with two "micro" batches.

The only potential downside of this trick, is if a given model/optimizer does _not_ scale with batch size. This could be the case for small datasets (but then why would you need data parallel?).

Here is a simple implementation in JAX:

```python
def accumulate_gradient(value_and_grad_fn,
                        params: PyTree,
                        batch: PyTree,
                        accum_steps: int = 1) -> Tuple[jnp.ndarray, PyTree]:
  """Accumulates gradients over given steps.
  
  Args:
    value_and_grad_fn: Gradient function that does not return aux values.
    params: Parameters, passed as first argument to `value_and_grad_fn`.
    batch: Batch, passed as second argument to `value_and_grad_fn`.
    accum_steps: Number of micro batches to accumulate over. Defaults to 1,
      which means no gradients are accumulated.
  
  Returns:
    Tuple (loss, grads).
  """
  if accum_steps > 1:
    bs = next(iter(jax.tree.leaves(batch))).shape[0]
    assert bs % accum_steps == 0, (
        f"Invalid accum_steps {accum_steps} for batch size `{bs}")
    microbatch_size = bs // accum_steps
    logging.info("Accumulating with microbatch_size %d over %d steps.",
                 microbatch_size, accum_steps)

    def get_microbatch(batch, i):
      return jax.tree.map(
          lambda t: jnp.reshape(t, (accum_steps, -1) + (t.shape[1:]))[i], batch)

    # Initialize accumulator.
    l, g = value_and_grad_fn(params, get_microbatch(batch, 0))

    def accumulate(i, l_and_g):
      l, g = l_and_g
      l_i, g_i = value_and_grad_fn(params, get_microbatch(batch, i))
      return (l + l_i, jax.tree.map(jnp.add, g, g_i))

    # Average over accum_steps.
    loss, grads = jax.lax.fori_loop(1, accum_steps, accumulate, (l, g))
    return jax.tree.map(lambda x: x / accum_steps, (loss, grads))
  else:
    return value_and_grad_fn(params, batch)
```

In theory, you could go all-in with many accumulation steps, such that the communication time as a fraction of total step time tends to zero - giving you an \\(\eta \approx 99\%\\).

In our case, we used 2 accumulation steps to match the 4096 batch-size in [BiT: BigTransfer](https://arxiv.org/abs/1912.11370) paper. Plugging values back into our equation:

$$
  \frac{2 \times T_s}{2 \times T_s + 0.5 \times T_c} = \frac{468}{468 + 53} \approx 89.8\%
$$

{{< figure
    src="/posts/orion/gaccumulation.png"
    alt="Gradient accumulation results"
    caption="Gradient accumulation results"
>}}

Ouch, we were SO close to hit our \\(90\%\\) goal!

## 3. Faster Communication

Ok, no scam going on here. We did _not_ end up buying a faster NIC. Remember that our existing NIC had dual 10G ethernet ports - one of which was running on 1G for networking. We reconfigured all four servers to connect directly to the 10G switch, which in turn was connected to the Internet via a single 1G port.

On paper, we had 20G bandwidth to/from each node. The question was, did NCCL support multi-NIC? Absolutely it did! I will spare you the details of benchmarking, but these were the two flags we set for NCCL.

```python
NCCL_SOCKET_IFNAME=ens803f  # Includes ens803f0 and ens803f1, 10G each.
NCCL_SOCKET_NTHREADS=1      # May be different on your setup.
```

With communication speed doubled, we crunch the numbers again:

$$
\eta = \frac{2 \times T_s}{2 \times T_s + 0.25 \times T_c} = \frac{468}{468 + 27} \approx 94.5\%
$$

{{< figure
    src="/posts/orion/multinic.png"
    alt="Multi-NIC communication results"
    caption="Multi-NIC communication results"
>}}

## Result

This cluster achieved a throughput roughly 20% higher than a $16/hr V100 AWS instance. Our team saved ~$120k for close to a year of uptime.

[^1]: We actually had an odd number of nodes. I rounded all calculations assuming 4, for it is a nice number for hardware.
[^nccl]: [NCCL Bandwidth and Throughput Calculation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
[^bf16eps]: [Comparing `bfloat16` range and precision to other 16-bit numbers](https://www.johndcook.com/blog/2018/11/15/bfloat16/)