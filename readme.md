# CG's custom nodes: Randoms

[Like my nodes? Buy me a coffee?](https://www.buymeacoffee.com/chrisgoringe)

## Randoms

A set of custom nodes for doing things randomly in an isolated `SeedContext`. This means they save the state of the random number system, reseed, do their thing, and then restore the state. In practice this means that adding one of these nodes doesn't change the sequence of random numbers generated by other nodes, *so they don't break reproducability*.

You can also use these nodes to do things systematically (going through all options one at a time).

To install:
```
cd [path to ComfyUI]/custom_nodes
git clone https://github.com/chrisgoringe/cg-random.git
```

To update:
```
cd [path to ComfyUI]/custom_nodes/cg-random
git pull
```

## Random Numbers

- `Random Int`
- `Random Float` 
- `Systematic Int`
- `Systematic Float`

All these nodes take a `minimum` and a `maximum`. 

`Random ...` generates a random value between `minimum` and `maximum` (inclusive of both ends). 

`Systematic ...` has a `step` value that is added each time (wrapping around when you get to the `max`), and a `reset` (which forces the value to the `minimum`). `step` can be negative (in which case it wraps when you get to `min`, and resets to `max`).

The `... Float` nodes also have an optional `decimal_places` to round the value to. Note that the rounding is applied last, so it is possible to get a value outside the range (`minimum`, `maximum`) due to rounding (if `maximum=0.899` and `decimal_places=1`, the value `0.88` can be generated and rounded to `0.9`)

## Random Checkpoint, LoRA, and Image Loaders

You can randomly, or systematically, load checkpoints, LoRAs, or images.

To use checkpoint loader (or LoRA loader) you need to create a directory in your checkpoints (LoRA) directory called `random`. So something like `models\Stable-diffusion\random`. Put the models you want to choose between into it and one of them will be loaded at random.

There is also a parameter `keep_for` which is the number of times a model will be used before a new random choice is made. Default is `1` (choose a random model every time) but if you are doing lots of runs (or doing permutations) you might want to save load time by using each random choice multiple times.

If you set the option to `systematic` instead of `random` it will loop through the options (keeping each one for `keep_for` runs then moving on). 

The random image loader also requires you to specify the `folder` to load from and the `extensions` to recognise as images.


