# NeRFNDC
### 

---

[Instant-NGP](https://github.com/NVlabs/instant-ngp) recently introduced a Multi-resolution Hash Encoding for neural graphics primitives like [NeRFs](https://www.matthewtancik.com/nerf). The original NVIDIA implementation mainly in C++/CUDA, based on [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), can train NeRFs upto 100x faster!

This project is a **pure PyTorch** implementation of [Instant-NGP](https://github.com/NVlabs/instant-ngp), built with the purpose of enabling AI Researchers to play around and innovate further upon this method.

This project is built on top of the super-useful [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch) implementation.

# Instructions
Download the nerf-synthetic dataset from here: [Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi).

To train a `chair` HashNeRF model:
```
python run_nerf.py --config configs/chair.txt --finest_res 512 --log2_hashmap_size 19 --lrate 0.01 --lrate_decay 10
```

To train for other objects like `ficus`/`hotdog`, replace `configs/chair.txt` with `configs/{object}.txt`:

![hotdog_ficus](https://user-images.githubusercontent.com/8559512/154066554-d3656d4a-1738-427c-982d-3ef4e4071969.gif)
