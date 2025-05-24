---
title: "PyTorch: I’m Fast, JAX: You Call That Fast?"
date: 2024-08-16
summary: "A recipe to train Object Detection Transformers (really) fast."
tags: ["jax", "pytorch", "detr", "object-detection", "transformers"]
---

{{<katex>}}

PyTorch is far from being concluded slow. But it is always a fun (and worthwhile) exercise to flex how fast you can _really_ go with compilers if you can lay out a computation in the right way.

This is my work log of building a Detection Transformer ([DETR](https://arxiv.org/abs/2005.12872)) training pipeline in JAX. I find this object detection architecture special for many reasons:

1. It predicts bounding boxes and class labels directly, instead of generating a gazillion region-proposals and relying on esoteric post-processing techniques. 
2. It is end-to-end differentiable and parallelizable.
3. It fits the 'spirit of deep learning' and borrows wisdom from Rich Sutton's [Bitter Lesson](http://incompleteideas.net/IncIdeas/BitterLesson.html) of AI research.

[DETR](https://arxiv.org/abs/2005.12872) is one of the well written papers out there, I recommend going through it once.

However, DETR is slow to train. While there have been successors to DETR that improve algorithmic convergence rates, like [Deformable-DETR](https://arxiv.org/abs/2010.04159), or [Conditional-DETR](https://arxiv.org/abs/2108.06152), none of these implementations focus on running 'efficiently' on the GPU. There is a great deal of efficiency to be had here, which was the objective of this project. I will walk through the techniques that helped me provide up to \\(30\\%\\) higher GPU utilization against a best-effort optimized [PyTorch implementation of DETR](https://github.com/facebookresearch/detr).

## The Bottleneck

{{< figure
    src="https://raw.githubusercontent.com/MasterSkepticista/detr/main/.github/detr.png"
    alt="DETR Architecture"
    caption="DETR Architecture"
>}}

DETR has three main components: a convolutional backbone (typically a ResNet), a stack of encoder-decoder transformer blocks, and a bipartite matcher. Of the three, bipartite matching (hungarian) algorithm runs on the CPU. In fact, the original DETR implementation calls [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) sequentially, for each input-target pair. This leaves the GPU idle. Part of the gains we will see later, are by reducing this idle time.

> Idle GPU is wasted GPU.

## Baseline

To improve 'something', we must 'measure' that something. Our bench this time is an 8-A6000 cluster. I made a couple of changes to ensure PyTorch version was 'as fast as possible'. Here is a summary of digressions:
* Use [Flash Attention](https://arxiv.org/abs/2205.14135) in `F.scaled_dot_product_attention`.
* Enable use of tensor cores by setting `torch.set_float32_matmul_precision(...)` to "medium" ("high" works just as good).
* Wrap forward pass using `torch.autocast` to `bfloat16`.

With these changes, it took 3 days (2.1 steps/s) to train a 300-epoch baseline on our cluster. I will skip the napkin math, but this is already faster than authors' numbers when normalized for per GPU FLOP throughput - notably from use of the new flash attention kernel that Ampere GPUs support.

{{< alert "circle-info" >}}
I tried `torch.compile` with various options on sub-parts of the model/training step. It either ended up giving the same throughput, or failed to compile. So 'it is what it is'.
{{< /alert >}}

## Refactor
I decided to implement DETR in JAX. You can think of JAX as a front-end language to write [XLA](https://openxla.org/xla) optimized programs. XLA is an open-source Machine Learning compiler that optimizes Linear Algebra operations. XLA generally outperforms the superset of all PyTorch optimizations _when done right, by a good margin_. One downside of working with XLA/JAX is that it is harder to debug `jit` compiled programs. PyTorch, on the other hand, dispatches CUDA kernels eagerly (except when wrapped with `torch.compile`), which makes it easiest to debug and work with. But when you consider the cost of few compile minutes over how long production training runs like these typically are, it is worth the tradeoff.

Luckily a dusty [re-implementation](https://github.com/google-research/scenic/tree/main/scenic/projects/baselines/detr) of DETR in JAX made for a good head-start. But it did not work out-of-the-box due to deprecated JAX and Flax APIs. To get the ball rolling, I made a minimal set of [changes](https://github.com/google-research/scenic/pull/1062), without any optimizations.

Scenic also [provides](https://github.com/google-research/scenic/blob/main/scenic/model_lib/matchers/hungarian_jax.py) GPU and TPU implementations of Hungarian matching. This is already significant work off-the-table.

This implementation takes 6.5 days to replicate the PyTorch baseline, at nearly 1 step/s. How fast can we go?

{{< figure
    src="/posts/detr/pt_baseline.png"
>}}

Now, the optimizations.

### 1. Disable Matching for padded objects

This is actually a bug-fix rather than an optimization. COCO dataset does not guarantee a fixed number of objects for each image. This means the bipartite matcher would have to map a fixed set of object queries (say 100) to a randomly varying number of target objects for each image, triggering an expensive retrace of the graph.

{{< alert "circle-info" >}}
XLA compiler can generate optimized graphs in part because memory allocation/deallocation is predictable, and constant-folding/fusion of operators is simpler when the entire computational graph layout is static. This is the price you pay for performance. You can read more [here](https://www.tensorflow.org/guide/function).
{{< /alert >}}

To prevent retracing, we add 'padding' objects and a boolean mask that allows us to filter dummy objects when computing loss.

```python
# Adding padded dimensions
# input_pipeline.py#L145
padded_shapes = {
    'inputs': [max_size, max_size, 3],
    'padding_mask': [max_size, max_size],
    'label': {
        'boxes': [max_boxes, 4],
        'area': [max_boxes,],
        'objects/id': [max_boxes,],
        'is_crowd': [max_boxes,],
        'labels': [max_boxes,],
        'image/id': [],
        'size': [2,],
        'orig_size': [2,],
    },
}
```

But this still computes bipartite matching on `padded` objects. We can remove
constants from the `cost` matrix as they do not affect the final matching.

```patch

-- cost = cost * mask + (1.0 - mask) * cost_upper_bound
++ cost = cost * mask
```

With this bug-fix, we are now 40% faster, i.e. \\(1.4\\) steps/s. It now takes 4.7 days to train the baseline.

{{< figure
    src="/posts/detr/disable_padded.png"
>}}

### 2. Mixed Precision MatMuls

Yes, there are no 'free-lunches', but I think we can make a strong case for the invention of `bfloat16` data type.
We migrate `float32` matmuls to `bfloat16`, without any loss in final AP scores. This is what we did in the PyTorch baseline.
In `flax`, this is the same as supplying `dtype=jnp.bfloat16` on supported modules.

```python
# Example conversion.
conv = nn.Conv(..., dtype=jnp.bfloat16)
dense = nn.Dense(..., dtype=jnp.bfloat16)
...
```

This gets us above \\(2.1\\) steps/s. We now have performance parity with PyTorch, with 3.1 days taken to train the baseline!

{{< figure
    src="/posts/detr/mixed_prec.png"
>}}

Huh! We should've called it a day... but let's keep going.

### 3. Parallel Bipartite Matching on Decoders

To achieve a high overall \\(\text{mAP}\\) score, DETR authors propose computing loss over each decoder output. DETR uses a sequential stack of 6 decoders, each emitting bounding-box and classifier predictions for 100 object queries.

```python
# models/detr_base_model.py#L377
# Computing matchings for each decoder head (auxiliary predictions)
# outputs = {
#   "pred_logits": ndarray, 
#   "pred_boxes": ndarray,
#   "aux_outputs": [
#     {"pred_logits": ndarray, "pred_boxes": ndarray},
#     {"pred_logits": ndarray, "pred_boxes": ndarray},
#     ...
#   ]
# }
if matches is None:
  cost, n_cols = self.compute_cost_matrix(outputs, batch['label'])
  matches = self.matcher(cost, n_cols)
  if 'aux_outputs' in outputs:
    matches = [matches]
    for aux_pred in outputs['aux_outputs']:
      cost, n_cols = self.compute_cost_matrix(aux_pred, batch['label'])
      matches.append(self.matcher(cost, n_cols))
```

Computing optimal matchings on these decoder outputs can actually be done in parallel using `vmap`.
```python
# models/detr_base_model.py#L377
# After vectorization
if matches is None:
  predictions = [{
    "pred_logits": outputs["pred_logits"],
    "pred_boxes": outputs["pred_boxes"]
  }]
  if 'aux_outputs' in outputs:
    predictions.extend(outputs["aux_outputs"])

  def _compute_matches(predictions, targets):
    cost, n_cols = self.compute_cost_matrix(predictions, targets)
    return self.matcher(cost, n_cols)

  # Stack list of pytrees.
  predictions = jax.tree.map(
    lambda *args: jnp.stack(args), *predictions)

  # Compute matches in parallel for all outputs.
  matches = jax.vmap(_compute_matches, (0, None))(
    predictions, batch["label"])
  matches = list(matches)
```

With this change, we are now stepping **10% faster** than PyTorch, at \\(2.4\\) steps/s, i.e. 2.7 days to train.

{{< figure
    src="/posts/detr/parallel_match.png"
>}}

### 4. Use Flash Attention

XLA did not use flash attention kernel all along. It was added only recently through [`jax.nn.dot_product_attention`](https://github.com/google/jax/pull/21371) for Ampere and later architectures. Perhaps future XLA versions might automatically recognize a dot-product attention signature during `jit`, without us having to explicitly call via SDPA API. But that is not the case today, so we will make-do with this custom function call.

```python
# models/detr.py#L261
if True:
  x = jax.nn.dot_product_attention(
    query, key, value, mask=mask, implementation="cudnn")
else:
  x = attention_layers.dot_product_attention(
      query,
      key,
      value,
      mask=mask,
      dropout_rate=self.dropout_rate,
      broadcast_dropout=self.broadcast_dropout,
      dropout_rng=self.make_rng('dropout') if train else None,
      deterministic=not train,
      capture_attention_weights=False)
```

{{< alert "circle-info" >}}
As of writing, [`jax.nn.dot_product_attention`](https://github.com/google/jax/pull/21371) does not support attention dropout. This is because JAX and cuDNN use different PRNG implementations. Speedup outweighs the regularization benefits of dropout, so we will live with it for now.
{{< /alert >}}

For now, let us be content with the _potential_ speedup. We are now at \\(3.0\\) steps/s, **33% faster** than PyTorch, taking 2 days to train.

{{< figure
    src="/posts/detr/flash_attn.png"
>}}

## Summary

Further gains are possible by replacing exact matching with an approximate matching. It may be a good reason to do so - just like how minibatch SGD is random by its very nature. It is arguably its strong suit on its nice convergence properties.

Why should a matching algorithm be exact, if we are spending ~0.5M steps to converge anyway? Are there gains to be had by having an 'approximate' matching? Yes, and one way to go about it is using a regularized solver like Sinkhorn algorithm. But that's for another day.

You can find the code for DETR with all above optimizations [here](https://github.com/masterskepticista/detr). Update: It also supports Sinkhorn algorithm now!