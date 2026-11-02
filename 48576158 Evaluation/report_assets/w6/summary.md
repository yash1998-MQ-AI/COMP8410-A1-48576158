## Quality and motion
| metric           |   n_items |   wan22 |   wan22_lo |   wan22_hi |   hunyuanvideo15 |   hunyuanvideo15_lo |   hunyuanvideo15_hi |   diff_A_minus_B |   diff_lo |   diff_hi |   A_higher |   B_higher | verdict             |
|:-----------------|----------:|--------:|-----------:|-----------:|-----------------:|--------------------:|--------------------:|-----------------:|----------:|----------:|-----------:|-----------:|:--------------------|
| clip_i_src_mean  |         6 |   0.96  |      0.913 |      0.987 |            0.913 |               0.882 |               0.94  |            0.047 |     0.006 |     0.092 |          5 |          1 | A higher            |
| dino_src_mean    |         6 |   0.962 |      0.924 |      0.986 |            0.939 |               0.917 |               0.955 |            0.023 |    -0.023 |     0.057 |          5 |          1 | no clear difference |
| dino_src_min     |         6 |   0.934 |      0.868 |      0.976 |            0.902 |               0.867 |               0.931 |            0.032 |    -0.045 |     0.095 |          5 |          1 | no clear difference |
| dino_consec      |         6 |   0.988 |      0.978 |      0.995 |            0.978 |               0.973 |               0.984 |            0.01  |    -0     |     0.018 |          5 |          1 | no clear difference |
| first_frame_ssim |         6 |   0.987 |      0.984 |      0.991 |            0.992 |               0.992 |               0.993 |           -0.005 |    -0.009 |    -0.002 |          1 |          5 | B higher            |
| clip_t           |         6 |   0.27  |      0.248 |      0.289 |            0.277 |               0.264 |               0.29  |           -0.007 |    -0.02  |     0.002 |          3 |          3 | no clear difference |
| motion_mad       |         6 |   0.643 |      0.353 |      0.94  |            2.682 |               1.776 |               3.687 |           -2.04  |    -3.005 |    -1.216 |          0 |          6 | B higher            |
| jitter_ratio     |         6 |   1.165 |      0.918 |      1.435 |            0.882 |               0.761 |               1.001 |            0.283 |     0.05  |     0.56  |          4 |          2 | A higher            |
| luma_drift       |         6 |   4.496 |      1.491 |      7.883 |           12.882 |               6.722 |              19.057 |           -8.386 |   -15.241 |     0.425 |          1 |          5 | no clear difference |
| dino_first_last  |         6 |   0.937 |      0.871 |      0.979 |            0.901 |               0.867 |               0.93  |            0.036 |    -0.035 |     0.097 |          5 |          1 | no clear difference |

## Run cost
| metric                | model          |   n |   mean |   median |    min |    max |
|:----------------------|:---------------|----:|-------:|---------:|-------:|-------:|
| latency_s             | wan22          |   6 | 279.64 |   279.65 | 277.4  | 281.73 |
| latency_s             | hunyuanvideo15 |   6 | 113.45 |   113.33 | 112.91 | 114.62 |
| latency_sample_s      | wan22          |   6 | 276.9  |   276.8  | 274.85 | 278.79 |
| latency_sample_s      | hunyuanvideo15 |   6 | 111.47 |   111.34 | 111    | 112.61 |
| peak_vram_gb          | wan22          |   6 |  22.82 |    22.82 |  22.82 |  22.82 |
| peak_vram_gb          | hunyuanvideo15 |   6 |  24.37 |    24.37 |  24.37 |  24.37 |
| peak_vram_reserved_gb | wan22          |   6 |  28.39 |    28.39 |  28.39 |  28.39 |
| peak_vram_reserved_gb | hunyuanvideo15 |   6 |  25.06 |    25.06 |  25.06 |  25.06 |
| file_mb               | wan22          |   6 |   5.94 |     5.86 |   2.7  |  10.01 |
| file_mb               | hunyuanvideo15 |   6 |   0.26 |     0.26 |   0.11 |   0.43 |
| gen_ok                | hunyuanvideo15 |   6 |   1    |   nan    | nan    | nan    |
| gen_ok                | wan22          |   6 |   1    |   nan    | nan    | nan    |
