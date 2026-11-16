### cifar10, init model, protocol `1step_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.926 ± 0.023 | 26.3 ± 4.5 | 0.006 ± 0.002 | 4 |
| peer, standard attack | mean | 0.763 ± 0.033 | 20.0 ± 1.1 | 0.029 ± 0.008 | 4 |
| peer, standard attack | trimmed_mean | 0.345 ± 0.040 | 12.2 ± 0.9 | 0.316 ± 0.088 | 4 |
| peer, standard attack | median | 0.290 ± 0.028 | 11.2 ± 0.6 | 0.449 ± 0.040 | 4 |
| peer, standard attack | topfed | 0.278 ± 0.026 | 11.5 ± 0.3 | 0.415 ± 0.055 | 4 |
| peer, standard attack | krum | 0.364 ± 0.051 | 12.9 ± 0.5 | 0.094 ± 0.065 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.636 ± 0.050 | 17.4 ± 1.2 | 0.051 ± 0.017 | 4 |
| peer, rule-aware attack | median | 0.462 ± 0.047 | 14.9 ± 0.9 | 0.095 ± 0.021 | 4 |
| peer, rule-aware attack | topfed | 0.345 ± 0.026 | 12.7 ± 0.5 | 0.187 ± 0.024 | 4 |
| peer, rule-aware attack | krum | 0.964 ± 0.026 | 32.0 ± 6.3 | 0.008 ± 0.005 | 4 |
| peer, rule-aware + true weights (oracle) | topfed | 0.412 ± 0.047 | 14.0 ± 0.8 | 0.124 ± 0.025 | 4 |
| calibration | flat_gray | 0.589 ± 0.097 | 17.5 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.060 | 12.7 ± 0.9 | — | 4 |

### cifar10, init model, protocol `1step_b1`, labels recovered (seeds [1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.924 ± 0.030 | 26.5 ± 5.4 | 0.005 ± 0.003 | 3 |
| peer, standard attack | mean | 0.494 ± 0.107 | 14.4 ± 2.0 | 0.121 ± 0.042 | 3 |
| peer, standard attack | trimmed_mean | 0.312 ± 0.043 | 12.0 ± 1.0 | 0.351 ± 0.079 | 3 |
| peer, standard attack | median | 0.295 ± 0.019 | 11.8 ± 0.4 | 0.446 ± 0.009 | 3 |
| peer, standard attack | topfed | 0.278 ± 0.011 | 11.9 ± 0.1 | 0.416 ± 0.036 | 3 |
| peer, standard attack | krum | 0.376 ± 0.036 | 13.0 ± 1.6 | 0.021 ± 0.002 | 3 |
| peer, rule-aware attack | trimmed_mean | 0.289 ± 0.066 | 11.4 ± 1.5 | 0.251 ± 0.127 | 3 |
| peer, rule-aware attack | median | 0.308 ± 0.016 | 12.0 ± 0.6 | 0.224 ± 0.046 | 3 |
| peer, rule-aware attack | topfed | 0.259 ± 0.011 | 11.0 ± 0.7 | 0.311 ± 0.040 | 3 |
| peer, rule-aware attack | krum | 0.961 ± 0.032 | 32.1 ± 7.8 | 0.008 ± 0.007 | 3 |
| peer, rule-aware + true weights (oracle) | topfed | 0.255 ± 0.035 | 10.9 ± 0.8 | 0.305 ± 0.024 | 3 |
| calibration | flat_gray | 0.603 ± 0.114 | 17.5 ± 0.4 | — | 3 |
| calibration | random_other_images | 0.349 ± 0.061 | 12.8 ± 1.1 | — | 3 |

### cifar10, init model, protocol `1step_b16` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.578 ± 0.026 | 16.4 ± 0.6 | 0.021 ± 0.008 | 4 |
| peer, standard attack | mean | 0.380 ± 0.020 | 13.9 ± 0.4 | 0.017 ± 0.007 | 4 |
| peer, standard attack | trimmed_mean | 0.322 ± 0.025 | 12.9 ± 0.5 | 0.129 ± 0.068 | 4 |
| peer, standard attack | median | 0.286 ± 0.020 | 12.2 ± 0.4 | 0.209 ± 0.083 | 4 |
| peer, standard attack | topfed | 0.272 ± 0.034 | 12.1 ± 0.7 | 0.164 ± 0.075 | 4 |
| peer, standard attack | krum | 0.337 ± 0.046 | 12.7 ± 1.1 | 0.035 ± 0.021 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.393 ± 0.016 | 14.1 ± 0.4 | 0.016 ± 0.005 | 4 |
| peer, rule-aware attack | median | 0.361 ± 0.023 | 13.6 ± 0.5 | 0.026 ± 0.007 | 4 |
| peer, rule-aware attack | topfed | 0.276 ± 0.021 | 12.2 ± 0.5 | 0.071 ± 0.024 | 4 |
| peer, rule-aware attack | krum | 0.428 ± 0.022 | 16.3 ± 0.7 | 0.134 ± 0.009 | 3 |
| peer, rule-aware + estimated weights | topfed | 0.352 | 13.7 | 0.023 | 1 |
| peer, rule-aware + true weights (oracle) | topfed | 0.338 ± 0.025 | 13.3 ± 0.6 | 0.033 ± 0.011 | 4 |
| calibration | flat_gray | 0.655 ± 0.065 | 20.8 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.357 ± 0.024 | 14.0 ± 0.6 | — | 4 |

### cifar10, init model, protocol `1step_b2` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.905 ± 0.032 | 25.5 ± 3.2 | 0.010 ± 0.005 | 4 |
| peer, standard attack | mean | 0.581 ± 0.044 | 16.5 ± 0.9 | 0.036 ± 0.007 | 4 |
| peer, standard attack | trimmed_mean | 0.327 ± 0.017 | 12.4 ± 0.4 | 0.198 ± 0.056 | 4 |
| peer, standard attack | median | 0.262 ± 0.016 | 11.2 ± 0.3 | 0.352 ± 0.052 | 4 |
| peer, standard attack | topfed | 0.284 ± 0.020 | 11.9 ± 0.8 | 0.289 ± 0.080 | 4 |
| peer, standard attack | krum | 0.370 ± 0.026 | 13.0 ± 0.2 | 0.048 ± 0.030 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.542 ± 0.026 | 15.9 ± 0.5 | 0.040 ± 0.007 | 4 |
| peer, rule-aware attack | median | 0.477 ± 0.036 | 15.1 ± 0.7 | 0.057 ± 0.009 | 4 |
| peer, rule-aware attack | topfed | 0.332 ± 0.063 | 12.5 ± 1.0 | 0.179 ± 0.077 | 4 |
| peer, rule-aware attack | krum | 0.945 ± 0.005 | 27.1 ± 2.0 | 0.009 ± 0.003 | 3 |
| peer, rule-aware + true weights (oracle) | topfed | 0.455 ± 0.025 | 14.7 ± 0.5 | 0.070 ± 0.019 | 4 |
| calibration | flat_gray | 0.569 ± 0.082 | 18.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.321 ± 0.023 | 12.4 ± 0.9 | — | 4 |

### cifar10, init model, protocol `1step_b4` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.875 ± 0.009 | 23.5 ± 1.2 | 0.014 ± 0.004 | 4 |
| peer, standard attack | mean | 0.439 ± 0.020 | 14.4 ± 0.3 | 0.040 ± 0.016 | 4 |
| peer, standard attack | trimmed_mean | 0.319 ± 0.019 | 12.7 ± 0.3 | 0.183 ± 0.103 | 4 |
| peer, standard attack | median | 0.281 ± 0.022 | 12.0 ± 0.5 | 0.285 ± 0.144 | 4 |
| peer, standard attack | topfed | 0.281 ± 0.032 | 12.1 ± 0.6 | 0.240 ± 0.146 | 4 |
| peer, standard attack | krum | 0.326 ± 0.014 | 12.5 ± 0.5 | 0.066 ± 0.015 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.489 ± 0.030 | 15.2 ± 0.6 | 0.030 ± 0.010 | 4 |
| peer, rule-aware attack | median | 0.438 ± 0.029 | 14.6 ± 0.5 | 0.044 ± 0.011 | 4 |
| peer, rule-aware attack | topfed | 0.292 ± 0.035 | 12.1 ± 0.7 | 0.143 ± 0.043 | 4 |
| peer, rule-aware attack | krum | 0.626 ± 0.049 | 18.0 ± 1.0 | 0.082 ± 0.020 | 3 |
| peer, rule-aware + estimated weights | topfed | 0.341 | 13.2 | 0.056 | 1 |
| peer, rule-aware + true weights (oracle) | topfed | 0.410 ± 0.034 | 14.0 ± 0.5 | 0.059 ± 0.019 | 4 |
| calibration | flat_gray | 0.569 ± 0.082 | 18.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.320 ± 0.021 | 12.5 ± 0.3 | — | 4 |

### cifar10, init model, protocol `1step_b8` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.752 ± 0.044 | 19.7 ± 1.1 | 0.019 ± 0.006 | 4 |
| peer, standard attack | mean | 0.388 ± 0.026 | 13.9 ± 0.6 | 0.027 ± 0.013 | 4 |
| peer, standard attack | trimmed_mean | 0.312 ± 0.027 | 12.7 ± 0.5 | 0.150 ± 0.078 | 4 |
| peer, standard attack | median | 0.279 ± 0.017 | 12.1 ± 0.4 | 0.242 ± 0.085 | 4 |
| peer, standard attack | topfed | 0.273 ± 0.024 | 12.1 ± 0.5 | 0.195 ± 0.062 | 4 |
| peer, standard attack | krum | 0.303 ± 0.020 | 12.2 ± 0.6 | 0.058 ± 0.005 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.424 ± 0.022 | 14.3 ± 0.4 | 0.024 ± 0.006 | 4 |
| peer, rule-aware attack | median | 0.385 ± 0.027 | 13.9 ± 0.6 | 0.033 ± 0.007 | 4 |
| peer, rule-aware attack | topfed | 0.272 ± 0.017 | 12.0 ± 0.4 | 0.098 ± 0.023 | 4 |
| peer, rule-aware attack | krum | 0.484 ± 0.060 | 16.1 ± 1.1 | 0.127 ± 0.025 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.330 | 13.3 | 0.060 | 1 |
| peer, rule-aware + true weights (oracle) | topfed | 0.357 ± 0.023 | 13.5 ± 0.6 | 0.047 ± 0.009 | 4 |
| calibration | flat_gray | 0.586 ± 0.112 | 19.4 ± 1.8 | — | 4 |
| calibration | random_other_images | 0.352 ± 0.032 | 13.3 ± 0.1 | — | 4 |

### cifar10, init model, protocol `5ep_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.547 ± 0.101 | 17.1 ± 2.4 | 0.105 ± 0.008 | 4 |
| peer, standard attack | mean | 0.313 ± 0.028 | 11.8 ± 0.3 | 0.181 ± 0.018 | 4 |
| peer, standard attack | trimmed_mean | 0.250 ± 0.026 | 10.8 ± 0.3 | 0.363 ± 0.036 | 4 |
| peer, standard attack | median | 0.235 ± 0.018 | 10.6 ± 0.3 | 0.444 ± 0.016 | 4 |
| peer, standard attack | topfed | 0.229 ± 0.012 | 10.6 ± 0.3 | 0.444 ± 0.020 | 4 |
| peer, standard attack | krum | 0.320 ± 0.031 | 12.3 ± 0.9 | 0.128 ± 0.052 | 4 |
| calibration | flat_gray | 0.589 ± 0.097 | 17.5 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.060 | 12.7 ± 0.9 | — | 4 |

### cifar10, init model, protocol `paper` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.345 ± 0.041 | 15.5 ± 1.4 | 0.244 ± 0.067 | 4 |
| peer, standard attack | mean | 0.305 ± 0.028 | 14.7 ± 0.7 | 0.305 ± 0.043 | 4 |
| peer, standard attack | trimmed_mean | 0.316 ± 0.034 | 14.7 ± 1.1 | 0.287 ± 0.078 | 4 |
| peer, standard attack | median | 0.320 ± 0.033 | 14.9 ± 1.0 | 0.301 ± 0.082 | 4 |
| peer, standard attack | topfed | 0.309 ± 0.022 | 14.6 ± 0.6 | 0.323 ± 0.088 | 4 |
| peer, standard attack | krum | 0.387 ± 0.025 | 17.5 ± 0.9 | 0.192 ± 0.044 | 3 (1 skipped: Krum chose the attacker) |
| calibration | flat_gray | 0.698 ± 0.067 | 24.4 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.419 ± 0.041 | 14.8 ± 0.6 | — | 4 |

### cifar10, warm model, protocol `1step_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.916 ± 0.019 | 25.7 ± 4.2 | 0.005 ± 0.004 | 4 |
| peer, standard attack | mean | 0.805 ± 0.042 | 21.8 ± 1.8 | 0.008 ± 0.002 | 4 |
| peer, standard attack | scoreweight | 0.698 ± 0.047 | 18.4 ± 1.0 | 0.029 ± 0.010 | 4 |
| peer, standard attack | hardfilter | 0.506 ± 0.015 | 14.6 ± 0.3 | 0.046 ± 0.022 | 4 (SSIM over included peers only: 0.679 ± 0.043; 5.5 of 9 peers included on average) |
| peer, standard attack | trimmed_mean | 0.271 ± 0.044 | 11.6 ± 0.7 | 0.161 ± 0.046 | 4 |
| peer, standard attack | median | 0.220 ± 0.028 | 10.8 ± 0.6 | 0.213 ± 0.035 | 4 |
| peer, standard attack | topfed | 0.209 ± 0.033 | 10.4 ± 0.5 | 0.235 ± 0.030 | 4 |
| peer, standard attack | krum | 0.314 ± 0.012 | 10.4 ± 0.7 | 0.015 ± 0.013 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.730 ± 0.059 | 19.7 ± 1.9 | 0.020 ± 0.006 | 4 |
| peer, rule-aware attack | median | 0.290 ± 0.066 | 11.6 ± 1.3 | 0.091 ± 0.028 | 4 |
| peer, rule-aware attack | topfed | 0.245 ± 0.030 | 11.0 ± 0.6 | 0.120 ± 0.034 | 4 |
| peer, rule-aware attack | krum | 0.922 ± 0.021 | 23.0 ± 0.7 | 0.007 ± 0.003 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.291 ± 0.057 | 11.7 ± 0.9 | 0.114 ± 0.036 | 4 |
| peer, rule-aware + true weights (oracle) | scoreweight | 0.768 ± 0.052 | 20.3 ± 1.5 | 0.015 ± 0.005 | 4 |
| peer, rule-aware + true weights (oracle) | hardfilter | 0.664 ± 0.035 | 19.1 ± 1.7 | 0.007 ± 0.003 | 4 (SSIM over included peers only: 0.852 ± 0.065; 5.5 of 9 peers included on average) |
| peer, rule-aware + true weights (oracle) | topfed | 0.287 ± 0.071 | 11.6 ± 0.9 | 0.105 ± 0.031 | 4 |
| calibration | flat_gray | 0.589 ± 0.097 | 17.5 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.060 | 12.7 ± 0.9 | — | 4 |

### cifar10, warm model, protocol `1step_b1`, labels recovered (seeds [1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.916 ± 0.029 | 26.0 ± 4.5 | 0.005 ± 0.003 | 3 |
| peer, standard attack | mean | 0.583 ± 0.129 | 16.7 ± 3.2 | 0.044 ± 0.014 | 3 |
| peer, standard attack | trimmed_mean | 0.262 ± 0.029 | 11.3 ± 0.3 | 0.211 ± 0.015 | 3 |
| peer, standard attack | median | 0.228 ± 0.022 | 10.7 ± 0.3 | 0.223 ± 0.034 | 3 |
| peer, standard attack | topfed | 0.215 ± 0.009 | 10.7 ± 0.2 | 0.255 ± 0.045 | 3 |
| peer, standard attack | krum | 0.344 ± 0.053 | 10.4 ± 0.3 | 0.003 ± 0.002 | 3 |
| peer, rule-aware attack | trimmed_mean | 0.291 ± 0.026 | 11.5 ± 0.5 | 0.125 ± 0.020 | 3 |
| peer, rule-aware attack | median | 0.258 ± 0.045 | 11.3 ± 1.2 | 0.118 ± 0.031 | 3 |
| peer, rule-aware attack | topfed | 0.229 ± 0.027 | 10.5 ± 0.7 | 0.157 ± 0.040 | 3 |
| peer, rule-aware attack | krum | 0.936 ± 0.036 | 23.9 ± 2.0 | 0.005 ± 0.002 | 3 |
| peer, rule-aware + true weights (oracle) | topfed | 0.236 ± 0.037 | 10.7 ± 0.9 | 0.154 ± 0.033 | 3 |
| calibration | flat_gray | 0.603 ± 0.114 | 17.5 ± 0.4 | — | 3 |
| calibration | random_other_images | 0.349 ± 0.061 | 12.8 ± 1.1 | — | 3 |

### cifar10, warm model, protocol `1step_b16` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.619 ± 0.104 | 17.7 ± 2.1 | 0.012 ± 0.003 | 4 |
| peer, standard attack | mean | 0.363 ± 0.040 | 13.2 ± 0.9 | 0.014 ± 0.003 | 4 |
| peer, standard attack | trimmed_mean | 0.303 ± 0.016 | 12.3 ± 0.6 | 0.041 ± 0.006 | 4 |
| peer, standard attack | median | 0.259 ± 0.012 | 11.5 ± 0.4 | 0.075 ± 0.008 | 4 |
| peer, standard attack | topfed | 0.240 ± 0.015 | 11.1 ± 0.5 | 0.076 ± 0.009 | 4 |
| peer, standard attack | krum | 0.316 ± 0.037 | 11.9 ± 1.0 | 0.020 ± 0.007 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.326 ± 0.038 | 12.7 ± 1.0 | 0.018 ± 0.004 | 4 |
| peer, rule-aware attack | median | 0.281 ± 0.016 | 11.9 ± 0.4 | 0.030 ± 0.005 | 4 |
| peer, rule-aware attack | topfed | 0.253 ± 0.009 | 11.5 ± 0.2 | 0.044 ± 0.006 | 4 |
| peer, rule-aware attack | krum | 0.351 ± 0.037 | 13.5 ± 1.3 | 0.143 ± 0.036 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.278 ± 0.008 | 11.9 ± 0.3 | 0.039 ± 0.007 | 4 (5 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.268 ± 0.008 | 11.7 ± 0.1 | 0.038 ± 0.006 | 4 |
| calibration | flat_gray | 0.655 ± 0.065 | 20.8 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.357 ± 0.024 | 14.0 ± 0.6 | — | 4 |

### cifar10, warm model, protocol `1step_b2` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.853 ± 0.064 | 23.8 ± 3.8 | 0.009 ± 0.005 | 4 |
| peer, standard attack | mean | 0.695 ± 0.028 | 18.9 ± 0.8 | 0.012 ± 0.005 | 4 |
| peer, standard attack | trimmed_mean | 0.255 ± 0.013 | 11.2 ± 0.2 | 0.119 ± 0.030 | 4 |
| peer, standard attack | median | 0.210 ± 0.009 | 10.5 ± 0.2 | 0.185 ± 0.049 | 4 |
| peer, standard attack | topfed | 0.211 ± 0.010 | 10.5 ± 0.2 | 0.167 ± 0.021 | 4 |
| peer, standard attack | krum | 0.309 ± 0.035 | 10.9 ± 0.8 | 0.034 ± 0.019 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.492 ± 0.100 | 14.8 ± 1.4 | 0.038 ± 0.023 | 4 |
| peer, rule-aware attack | median | 0.250 ± 0.013 | 11.2 ± 0.2 | 0.077 ± 0.020 | 4 |
| peer, rule-aware attack | topfed | 0.226 ± 0.019 | 10.7 ± 0.2 | 0.097 ± 0.021 | 4 |
| peer, rule-aware attack | krum | 0.927 ± 0.028 | 25.6 ± 2.0 | 0.007 ± 0.004 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.247 ± 0.023 | 11.1 ± 0.4 | 0.091 ± 0.016 | 4 |
| peer, rule-aware + true weights (oracle) | topfed | 0.251 ± 0.019 | 11.2 ± 0.4 | 0.086 ± 0.015 | 4 |
| calibration | flat_gray | 0.569 ± 0.082 | 18.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.321 ± 0.023 | 12.4 ± 0.9 | — | 4 |

### cifar10, warm model, protocol `1step_b4` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.861 ± 0.052 | 23.4 ± 2.1 | 0.008 ± 0.004 | 4 |
| peer, standard attack | mean | 0.498 ± 0.035 | 15.2 ± 0.7 | 0.020 ± 0.004 | 4 |
| peer, standard attack | trimmed_mean | 0.276 ± 0.026 | 11.7 ± 0.4 | 0.094 ± 0.037 | 4 |
| peer, standard attack | median | 0.227 ± 0.013 | 11.0 ± 0.3 | 0.154 ± 0.066 | 4 |
| peer, standard attack | topfed | 0.219 ± 0.012 | 10.8 ± 0.3 | 0.157 ± 0.064 | 4 |
| peer, standard attack | krum | 0.314 ± 0.022 | 11.8 ± 1.2 | 0.023 ± 0.004 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.339 ± 0.041 | 12.8 ± 0.6 | 0.040 ± 0.001 | 4 |
| peer, rule-aware attack | median | 0.251 ± 0.021 | 11.4 ± 0.4 | 0.064 ± 0.017 | 4 |
| peer, rule-aware attack | topfed | 0.231 ± 0.008 | 11.0 ± 0.2 | 0.086 ± 0.018 | 4 |
| peer, rule-aware attack | krum | 0.630 ± 0.064 | 18.1 ± 0.7 | 0.054 ± 0.033 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.257 ± 0.022 | 11.5 ± 0.4 | 0.078 ± 0.014 | 4 (5 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.245 ± 0.023 | 11.2 ± 0.4 | 0.077 ± 0.016 | 4 |
| calibration | flat_gray | 0.569 ± 0.082 | 18.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.320 ± 0.021 | 12.5 ± 0.3 | — | 4 |

### cifar10, warm model, protocol `1step_b8` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.723 ± 0.130 | 19.8 ± 3.6 | 0.011 ± 0.002 | 4 |
| peer, standard attack | mean | 0.412 ± 0.017 | 14.0 ± 0.4 | 0.017 ± 0.005 | 4 |
| peer, standard attack | trimmed_mean | 0.286 ± 0.022 | 12.1 ± 0.4 | 0.059 ± 0.011 | 4 |
| peer, standard attack | median | 0.240 ± 0.021 | 11.3 ± 0.5 | 0.100 ± 0.017 | 4 |
| peer, standard attack | topfed | 0.230 ± 0.009 | 11.2 ± 0.3 | 0.097 ± 0.016 | 4 |
| peer, standard attack | krum | 0.305 ± 0.013 | 11.5 ± 0.3 | 0.021 ± 0.005 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.330 ± 0.029 | 12.8 ± 0.6 | 0.026 ± 0.003 | 4 |
| peer, rule-aware attack | median | 0.270 ± 0.011 | 11.9 ± 0.3 | 0.042 ± 0.001 | 4 |
| peer, rule-aware attack | topfed | 0.245 ± 0.007 | 11.4 ± 0.3 | 0.064 ± 0.011 | 4 |
| peer, rule-aware attack | krum | 0.449 ± 0.120 | 14.4 ± 2.6 | 0.105 ± 0.076 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.266 ± 0.019 | 11.8 ± 0.4 | 0.054 ± 0.003 | 4 (5 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.264 ± 0.014 | 11.7 ± 0.3 | 0.051 ± 0.003 | 4 |
| calibration | flat_gray | 0.586 ± 0.112 | 19.4 ± 1.8 | — | 4 |
| calibration | random_other_images | 0.352 ± 0.032 | 13.3 ± 0.1 | — | 4 |

### cifar10, warm model, protocol `5ep_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.857 ± 0.076 | 22.6 ± 1.3 | 0.017 ± 0.011 | 4 |
| peer, standard attack | mean | 0.719 ± 0.073 | 19.3 ± 1.9 | 0.026 ± 0.012 | 4 |
| peer, standard attack | trimmed_mean | 0.283 ± 0.040 | 11.6 ± 0.6 | 0.158 ± 0.038 | 4 |
| peer, standard attack | median | 0.229 ± 0.035 | 10.9 ± 0.6 | 0.206 ± 0.023 | 4 |
| peer, standard attack | topfed | 0.224 ± 0.021 | 10.6 ± 0.5 | 0.225 ± 0.018 | 4 |
| peer, standard attack | krum | 0.315 ± 0.023 | 10.7 ± 0.9 | 0.012 ± 0.013 | 4 |
| calibration | flat_gray | 0.589 ± 0.097 | 17.5 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.060 | 12.7 ± 0.9 | — | 4 |

### cifar10, warm model, protocol `paper` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.248 ± 0.037 | 12.6 ± 0.7 | 0.356 ± 0.120 | 4 |
| peer, standard attack | mean | 0.265 ± 0.027 | 13.0 ± 1.0 | 0.418 ± 0.047 | 4 |
| peer, standard attack | trimmed_mean | 0.273 ± 0.013 | 13.1 ± 0.8 | 0.377 ± 0.036 | 4 |
| peer, standard attack | median | 0.259 ± 0.024 | 12.8 ± 0.8 | 0.394 ± 0.032 | 4 |
| peer, standard attack | topfed | 0.257 ± 0.016 | 13.1 ± 0.7 | 0.402 ± 0.021 | 4 |
| peer, standard attack | krum | 0.268 ± 0.030 | 13.4 ± 0.7 | 0.301 ± 0.108 | 3 (1 skipped: Krum chose the attacker) |
| calibration | flat_gray | 0.698 ± 0.067 | 24.4 ± 0.4 | — | 4 |
| calibration | random_other_images | 0.419 ± 0.041 | 14.8 ± 0.6 | — | 4 |

### cifar100, init model, protocol `1step_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.903 ± 0.058 | 24.8 ± 4.8 | 0.020 ± 0.010 | 4 |
| peer, standard attack | mean | 0.774 ± 0.052 | 20.6 ± 1.0 | 0.021 ± 0.002 | 4 |
| peer, standard attack | trimmed_mean | 0.300 ± 0.024 | 11.1 ± 0.2 | 0.295 ± 0.035 | 4 |
| peer, standard attack | median | 0.252 ± 0.033 | 10.4 ± 0.4 | 0.385 ± 0.026 | 4 |
| peer, standard attack | topfed | 0.239 ± 0.032 | 10.3 ± 0.5 | 0.418 ± 0.023 | 4 |
| peer, standard attack | krum | 0.320 ± 0.030 | 12.5 ± 0.4 | 0.138 ± 0.018 | 2 (2 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.593 ± 0.056 | 16.5 ± 1.0 | 0.067 ± 0.011 | 4 |
| peer, rule-aware attack | median | 0.459 ± 0.043 | 14.5 ± 0.8 | 0.117 ± 0.027 | 4 |
| peer, rule-aware attack | topfed | 0.353 ± 0.029 | 12.5 ± 0.5 | 0.229 ± 0.025 | 4 |
| peer, rule-aware attack | krum | 0.965 ± 0.002 | 28.2 ± 1.4 | 0.012 ± 0.005 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.430 ± 0.023 | 14.0 ± 0.6 | 0.128 ± 0.036 | 4 |
| calibration | flat_gray | 0.502 ± 0.181 | 15.2 ± 2.0 | — | 4 |
| calibration | random_other_images | 0.278 ± 0.012 | 11.0 ± 0.6 | — | 4 |

### cifar100, init model, protocol `1step_b1`, labels recovered (seeds [1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.886 ± 0.071 | 25.2 ± 6.0 | 0.019 ± 0.011 | 3 |
| peer, standard attack | mean | 0.765 ± 0.061 | 20.1 ± 1.2 | 0.023 ± 0.006 | 3 |
| peer, standard attack | trimmed_mean | 0.226 ± 0.034 | 10.1 ± 0.6 | 0.703 ± 0.075 | 3 |
| peer, standard attack | median | 0.223 ± 0.023 | 10.1 ± 0.5 | 0.693 ± 0.081 | 3 |
| peer, standard attack | topfed | 0.213 ± 0.027 | 10.0 ± 0.6 | 0.716 ± 0.062 | 3 |
| peer, standard attack | krum | 0.376 | 12.7 | 0.021 | 1 (2 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.248 ± 0.056 | 10.8 ± 0.9 | 0.554 ± 0.076 | 3 |
| peer, rule-aware attack | median | 0.225 ± 0.036 | 10.3 ± 0.6 | 0.527 ± 0.054 | 3 |
| peer, rule-aware attack | topfed | 0.233 ± 0.044 | 10.3 ± 0.7 | 0.582 ± 0.048 | 3 |
| peer, rule-aware attack | krum | 0.963 | 27.1 | 0.009 | 1 |
| peer, rule-aware + true weights (oracle) | topfed | 0.227 ± 0.054 | 10.2 ± 1.0 | 0.578 ± 0.035 | 3 |
| calibration | flat_gray | 0.504 ± 0.221 | 14.9 ± 2.4 | — | 3 |
| calibration | random_other_images | 0.277 ± 0.015 | 10.9 ± 0.7 | — | 3 |

### cifar100, init model, protocol `1step_b16` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.691 ± 0.031 | 19.1 ± 1.3 | 0.024 ± 0.004 | 4 |
| peer, standard attack | mean | 0.464 ± 0.011 | 14.8 ± 0.2 | 0.025 ± 0.002 | 4 |
| peer, standard attack | trimmed_mean | 0.286 ± 0.010 | 11.7 ± 0.1 | 0.175 ± 0.022 | 4 |
| peer, standard attack | median | 0.240 ± 0.005 | 11.0 ± 0.1 | 0.233 ± 0.018 | 4 |
| peer, standard attack | topfed | 0.232 ± 0.009 | 10.9 ± 0.2 | 0.226 ± 0.012 | 4 |
| peer, standard attack | krum | 0.308 ± 0.007 | 12.2 ± 0.1 | 0.072 ± 0.003 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.448 ± 0.021 | 14.5 ± 0.4 | 0.030 ± 0.003 | 4 |
| peer, rule-aware attack | median | 0.343 ± 0.013 | 12.8 ± 0.2 | 0.062 ± 0.005 | 4 |
| peer, rule-aware attack | topfed | 0.251 ± 0.007 | 11.3 ± 0.2 | 0.122 ± 0.010 | 4 |
| peer, rule-aware attack | krum | 0.606 ± 0.034 | 17.3 ± 0.2 | 0.037 ± 0.008 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.265 ± 0.006 | 11.6 ± 0.0 | 0.123 ± 0.009 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.308 ± 0.013 | 12.2 ± 0.3 | 0.081 ± 0.004 | 4 |
| calibration | flat_gray | 0.577 ± 0.088 | 19.5 ± 1.2 | — | 4 |
| calibration | random_other_images | 0.374 ± 0.033 | 13.8 ± 0.4 | — | 4 |

### cifar100, init model, protocol `1step_b2` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.896 ± 0.033 | 24.6 ± 1.8 | 0.015 ± 0.004 | 4 |
| peer, standard attack | mean | 0.657 ± 0.023 | 18.0 ± 0.5 | 0.025 ± 0.003 | 4 |
| peer, standard attack | trimmed_mean | 0.284 ± 0.019 | 11.1 ± 0.4 | 0.245 ± 0.028 | 4 |
| peer, standard attack | median | 0.244 ± 0.021 | 10.5 ± 0.5 | 0.322 ± 0.022 | 4 |
| peer, standard attack | topfed | 0.227 ± 0.016 | 10.4 ± 0.5 | 0.335 ± 0.025 | 4 |
| peer, standard attack | krum | 0.296 ± 0.029 | 12.0 ± 0.4 | 0.141 ± 0.007 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.540 ± 0.038 | 15.7 ± 0.8 | 0.054 ± 0.012 | 4 |
| peer, rule-aware attack | median | 0.435 ± 0.026 | 14.1 ± 0.6 | 0.096 ± 0.016 | 4 |
| peer, rule-aware attack | topfed | 0.329 ± 0.022 | 12.3 ± 0.5 | 0.214 ± 0.019 | 4 |
| peer, rule-aware attack | krum | 0.913 ± 0.040 | 26.8 ± 1.2 | 0.011 ± 0.002 | 3 |
| peer, rule-aware + true weights (oracle) | topfed | 0.410 ± 0.027 | 13.9 ± 0.6 | 0.114 ± 0.025 | 4 |
| calibration | flat_gray | 0.538 ± 0.121 | 16.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.024 | 12.7 ± 0.3 | — | 4 |

### cifar100, init model, protocol `1step_b4` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.870 ± 0.012 | 23.6 ± 1.3 | 0.017 ± 0.003 | 4 |
| peer, standard attack | mean | 0.570 ± 0.025 | 16.8 ± 0.4 | 0.026 ± 0.004 | 4 |
| peer, standard attack | trimmed_mean | 0.277 ± 0.024 | 11.4 ± 0.2 | 0.236 ± 0.028 | 4 |
| peer, standard attack | median | 0.237 ± 0.022 | 10.8 ± 0.2 | 0.296 ± 0.028 | 4 |
| peer, standard attack | topfed | 0.236 ± 0.019 | 10.8 ± 0.3 | 0.303 ± 0.022 | 4 |
| peer, standard attack | krum | 0.310 ± 0.015 | 12.3 ± 0.2 | 0.093 ± 0.027 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.504 ± 0.033 | 15.7 ± 0.6 | 0.044 ± 0.009 | 4 |
| peer, rule-aware attack | median | 0.409 ± 0.039 | 14.1 ± 0.7 | 0.081 ± 0.012 | 4 |
| peer, rule-aware attack | topfed | 0.304 ± 0.020 | 12.3 ± 0.4 | 0.187 ± 0.012 | 4 |
| peer, rule-aware attack | krum | 0.899 ± 0.033 | 26.6 ± 1.4 | 0.012 ± 0.003 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.324 ± 0.043 | 12.7 ± 0.8 | 0.176 ± 0.025 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.383 ± 0.021 | 13.6 ± 0.4 | 0.103 ± 0.018 | 4 |
| calibration | flat_gray | 0.555 ± 0.107 | 18.7 ± 1.5 | — | 4 |
| calibration | random_other_images | 0.322 ± 0.046 | 12.4 ± 0.6 | — | 4 |

### cifar100, init model, protocol `1step_b8` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.788 ± 0.019 | 21.1 ± 1.5 | 0.021 ± 0.003 | 4 |
| peer, standard attack | mean | 0.517 ± 0.018 | 15.9 ± 0.5 | 0.024 ± 0.003 | 4 |
| peer, standard attack | trimmed_mean | 0.271 ± 0.016 | 11.4 ± 0.1 | 0.203 ± 0.018 | 4 |
| peer, standard attack | median | 0.228 ± 0.011 | 10.7 ± 0.1 | 0.257 ± 0.019 | 4 |
| peer, standard attack | topfed | 0.222 ± 0.004 | 10.7 ± 0.1 | 0.259 ± 0.016 | 4 |
| peer, standard attack | krum | 0.308 ± 0.015 | 12.3 ± 0.3 | 0.089 ± 0.017 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.478 ± 0.029 | 15.2 ± 0.7 | 0.036 ± 0.006 | 4 |
| peer, rule-aware attack | median | 0.379 ± 0.024 | 13.6 ± 0.5 | 0.069 ± 0.007 | 4 |
| peer, rule-aware attack | topfed | 0.271 ± 0.018 | 11.7 ± 0.3 | 0.150 ± 0.018 | 4 |
| peer, rule-aware attack | krum | 0.744 ± 0.140 | 21.4 ± 3.0 | 0.030 ± 0.026 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.282 ± 0.023 | 12.1 ± 0.6 | 0.164 ± 0.008 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.338 ± 0.023 | 12.9 ± 0.4 | 0.094 ± 0.013 | 4 |
| calibration | flat_gray | 0.577 ± 0.088 | 19.5 ± 1.2 | — | 4 |
| calibration | random_other_images | 0.375 ± 0.026 | 13.3 ± 0.6 | — | 4 |

### cifar100, init model, protocol `5ep_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.431 ± 0.117 | 14.4 ± 1.7 | 0.149 ± 0.017 | 4 |
| peer, standard attack | mean | 0.285 ± 0.054 | 11.0 ± 0.9 | 0.248 ± 0.018 | 4 |
| peer, standard attack | trimmed_mean | 0.221 ± 0.035 | 10.0 ± 0.5 | 0.413 ± 0.013 | 4 |
| peer, standard attack | median | 0.223 ± 0.036 | 10.1 ± 0.5 | 0.451 ± 0.031 | 4 |
| peer, standard attack | topfed | 0.213 ± 0.034 | 9.9 ± 0.5 | 0.460 ± 0.031 | 4 |
| peer, standard attack | krum | 0.305 ± 0.029 | 12.2 ± 0.3 | 0.172 ± 0.016 | 2 (2 skipped: Krum chose the attacker) |
| calibration | flat_gray | 0.502 ± 0.181 | 15.2 ± 2.0 | — | 4 |
| calibration | random_other_images | 0.278 ± 0.012 | 11.0 ± 0.6 | — | 4 |

### cifar100, init model, protocol `paper` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.302 ± 0.021 | 14.2 ± 1.0 | 0.511 ± 0.048 | 4 |
| peer, standard attack | mean | 0.283 ± 0.019 | 13.0 ± 0.3 | 0.572 ± 0.021 | 4 |
| peer, standard attack | trimmed_mean | 0.296 ± 0.009 | 13.0 ± 0.3 | 0.571 ± 0.019 | 4 |
| peer, standard attack | median | 0.294 ± 0.021 | 13.2 ± 0.4 | 0.565 ± 0.023 | 4 |
| peer, standard attack | topfed | 0.293 ± 0.021 | 13.4 ± 0.5 | 0.567 ± 0.012 | 4 |
| peer, standard attack | krum | 0.319 ± 0.014 | 15.1 ± 0.6 | 0.476 ± 0.035 | 4 |
| calibration | flat_gray | 0.674 ± 0.093 | 24.7 ± 1.8 | — | 4 |
| calibration | random_other_images | 0.402 ± 0.012 | 15.6 ± 0.5 | — | 4 |

### cifar100, warm model, protocol `1step_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.820 ± 0.080 | 22.3 ± 3.6 | 0.002 ± 0.002 | 4 |
| peer, standard attack | mean | 0.831 ± 0.035 | 22.9 ± 1.1 | 0.004 ± 0.001 | 4 |
| peer, standard attack | scoreweight | 0.695 ± 0.074 | 18.9 ± 1.8 | 0.033 ± 0.017 | 4 |
| peer, standard attack | hardfilter | 0.603 ± 0.165 | 17.5 ± 3.6 | 0.064 ± 0.045 | 4 (SSIM over included peers only: 0.713 ± 0.107; 6.5 of 9 peers included on average) |
| peer, standard attack | trimmed_mean | 0.238 ± 0.043 | 9.7 ± 0.7 | 0.384 ± 0.033 | 4 |
| peer, standard attack | median | 0.190 ± 0.038 | 9.3 ± 0.6 | 0.469 ± 0.034 | 4 |
| peer, standard attack | topfed | 0.198 ± 0.036 | 9.7 ± 0.7 | 0.484 ± 0.054 | 4 |
| peer, standard attack | krum | 0.341 ± 0.031 | 12.2 ± 1.1 | 0.070 ± 0.045 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.826 ± 0.023 | 22.7 ± 0.5 | 0.013 ± 0.003 | 4 |
| peer, rule-aware attack | median | 0.435 ± 0.202 | 14.2 ± 4.0 | 0.095 ± 0.065 | 4 |
| peer, rule-aware attack | topfed | 0.235 ± 0.052 | 10.6 ± 1.0 | 0.212 ± 0.035 | 4 |
| peer, rule-aware attack | krum | 0.884 ± 0.046 | 25.4 ± 3.2 | 0.002 ± 0.002 | 3 |
| peer, rule-aware + estimated weights | topfed | 0.292 ± 0.065 | 11.6 ± 1.4 | 0.183 ± 0.011 | 4 |
| peer, rule-aware + true weights (oracle) | scoreweight | 0.817 ± 0.042 | 22.6 ± 1.2 | 0.004 ± 0.002 | 4 |
| peer, rule-aware + true weights (oracle) | hardfilter | 0.680 ± 0.133 | 19.6 ± 2.7 | 0.004 ± 0.002 | 4 (SSIM over included peers only: 0.832 ± 0.029; 6.5 of 9 peers included on average) |
| peer, rule-aware + true weights (oracle) | topfed | 0.296 ± 0.081 | 11.6 ± 1.5 | 0.157 ± 0.017 | 4 |
| calibration | flat_gray | 0.502 ± 0.181 | 15.2 ± 2.0 | — | 4 |
| calibration | random_other_images | 0.278 ± 0.012 | 11.0 ± 0.6 | — | 4 |

### cifar100, warm model, protocol `1step_b1`, labels recovered (seeds [1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.798 ± 0.081 | 22.5 ± 4.4 | 0.003 ± 0.002 | 3 |
| peer, standard attack | mean | 0.838 ± 0.035 | 23.1 ± 1.3 | 0.003 ± 0.001 | 3 |
| peer, standard attack | trimmed_mean | 0.200 ± 0.057 | 9.3 ± 0.5 | 0.624 ± 0.009 | 3 |
| peer, standard attack | median | 0.172 ± 0.052 | 8.6 ± 0.6 | 0.617 ± 0.013 | 3 |
| peer, standard attack | topfed | 0.187 ± 0.040 | 9.1 ± 0.2 | 0.635 ± 0.024 | 3 |
| peer, standard attack | krum | 0.339 ± 0.111 | 11.4 ± 4.2 | 0.004 ± 0.002 | 2 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.220 ± 0.028 | 9.3 ± 0.6 | 0.489 ± 0.004 | 3 |
| peer, rule-aware attack | median | 0.184 ± 0.022 | 8.9 ± 0.4 | 0.353 ± 0.079 | 3 |
| peer, rule-aware attack | topfed | 0.184 ± 0.036 | 9.2 ± 0.3 | 0.375 ± 0.097 | 3 |
| peer, rule-aware attack | krum | 0.880 ± 0.057 | 25.4 ± 4.6 | 0.002 ± 0.002 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.180 ± 0.022 | 9.1 ± 0.3 | 0.359 ± 0.101 | 3 |
| calibration | flat_gray | 0.504 ± 0.221 | 14.9 ± 2.4 | — | 3 |
| calibration | random_other_images | 0.277 ± 0.015 | 10.9 ± 0.7 | — | 3 |

### cifar100, warm model, protocol `1step_b16` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.836 ± 0.022 | 23.6 ± 0.5 | 0.004 ± 0.000 | 4 |
| peer, standard attack | mean | 0.640 ± 0.029 | 18.7 ± 0.4 | 0.011 ± 0.002 | 4 |
| peer, standard attack | trimmed_mean | 0.278 ± 0.016 | 11.5 ± 0.2 | 0.243 ± 0.060 | 4 |
| peer, standard attack | median | 0.246 ± 0.016 | 11.0 ± 0.4 | 0.313 ± 0.057 | 4 |
| peer, standard attack | topfed | 0.231 ± 0.020 | 10.7 ± 0.4 | 0.304 ± 0.056 | 4 |
| peer, standard attack | krum | 0.343 ± 0.015 | 12.7 ± 0.2 | 0.031 ± 0.005 | 3 (1 skipped: Krum chose the attacker) |
| peer, rule-aware attack | trimmed_mean | 0.573 ± 0.049 | 17.4 ± 0.9 | 0.025 ± 0.010 | 4 |
| peer, rule-aware attack | median | 0.560 ± 0.163 | 17.3 ± 3.2 | 0.032 ± 0.031 | 4 |
| peer, rule-aware attack | topfed | 0.257 ± 0.021 | 11.5 ± 0.4 | 0.140 ± 0.022 | 4 |
| peer, rule-aware attack | krum | 0.785 ± 0.034 | 22.0 ± 1.5 | 0.011 ± 0.007 | 3 |
| peer, rule-aware + estimated weights | topfed | 0.466 ± 0.130 | 15.4 ± 2.5 | 0.054 ± 0.044 | 4 (6 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.425 ± 0.173 | 14.6 ± 3.4 | 0.067 ± 0.056 | 4 |
| calibration | flat_gray | 0.577 ± 0.088 | 19.5 ± 1.2 | — | 4 |
| calibration | random_other_images | 0.374 ± 0.033 | 13.8 ± 0.4 | — | 4 |

### cifar100, warm model, protocol `1step_b2` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.818 ± 0.034 | 22.8 ± 1.4 | 0.003 ± 0.001 | 4 |
| peer, standard attack | mean | 0.801 ± 0.020 | 22.1 ± 0.3 | 0.004 ± 0.001 | 4 |
| peer, standard attack | trimmed_mean | 0.253 ± 0.021 | 10.3 ± 0.8 | 0.323 ± 0.045 | 4 |
| peer, standard attack | median | 0.188 ± 0.025 | 9.7 ± 0.8 | 0.402 ± 0.043 | 4 |
| peer, standard attack | topfed | 0.190 ± 0.025 | 9.8 ± 0.8 | 0.443 ± 0.054 | 4 |
| peer, standard attack | krum | 0.319 ± 0.028 | 12.2 ± 0.9 | 0.056 ± 0.019 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.813 ± 0.015 | 22.4 ± 0.2 | 0.010 ± 0.002 | 4 |
| peer, rule-aware attack | median | 0.558 ± 0.209 | 17.0 ± 4.3 | 0.053 ± 0.057 | 4 |
| peer, rule-aware attack | topfed | 0.268 ± 0.029 | 11.5 ± 0.8 | 0.213 ± 0.018 | 4 |
| peer, rule-aware attack | krum | 0.809 ± 0.077 | 23.5 ± 3.7 | 0.004 ± 0.003 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.393 ± 0.122 | 13.8 ± 2.5 | 0.116 ± 0.064 | 4 |
| peer, rule-aware + true weights (oracle) | topfed | 0.453 ± 0.157 | 14.9 ± 3.3 | 0.090 ± 0.069 | 4 |
| calibration | flat_gray | 0.538 ± 0.121 | 16.5 ± 1.7 | — | 4 |
| calibration | random_other_images | 0.332 ± 0.024 | 12.7 ± 0.3 | — | 4 |

### cifar100, warm model, protocol `1step_b4` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.830 ± 0.027 | 23.2 ± 1.0 | 0.003 ± 0.001 | 4 |
| peer, standard attack | mean | 0.782 ± 0.032 | 22.1 ± 0.7 | 0.005 ± 0.001 | 4 |
| peer, standard attack | trimmed_mean | 0.259 ± 0.029 | 10.8 ± 0.4 | 0.307 ± 0.033 | 4 |
| peer, standard attack | median | 0.208 ± 0.021 | 10.1 ± 0.5 | 0.382 ± 0.013 | 4 |
| peer, standard attack | topfed | 0.207 ± 0.016 | 10.2 ± 0.3 | 0.409 ± 0.015 | 4 |
| peer, standard attack | krum | 0.321 ± 0.038 | 12.4 ± 0.5 | 0.046 ± 0.015 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.759 ± 0.047 | 21.6 ± 1.0 | 0.012 ± 0.005 | 4 |
| peer, rule-aware attack | median | 0.682 ± 0.053 | 20.0 ± 1.1 | 0.016 ± 0.005 | 4 |
| peer, rule-aware attack | topfed | 0.331 ± 0.084 | 13.0 ± 1.6 | 0.174 ± 0.062 | 4 |
| peer, rule-aware attack | krum | 0.807 ± 0.058 | 23.0 ± 1.3 | 0.003 ± 0.001 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.532 ± 0.112 | 16.8 ± 2.2 | 0.063 ± 0.053 | 4 (6 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.597 ± 0.069 | 18.1 ± 1.4 | 0.025 ± 0.008 | 4 |
| calibration | flat_gray | 0.555 ± 0.107 | 18.7 ± 1.5 | — | 4 |
| calibration | random_other_images | 0.322 ± 0.046 | 12.4 ± 0.6 | — | 4 |

### cifar100, warm model, protocol `1step_b8` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.831 ± 0.013 | 23.1 ± 1.2 | 0.004 ± 0.002 | 4 |
| peer, standard attack | mean | 0.750 ± 0.021 | 21.3 ± 0.3 | 0.006 ± 0.001 | 4 |
| peer, standard attack | trimmed_mean | 0.271 ± 0.017 | 11.1 ± 0.1 | 0.269 ± 0.023 | 4 |
| peer, standard attack | median | 0.225 ± 0.017 | 10.4 ± 0.2 | 0.330 ± 0.023 | 4 |
| peer, standard attack | topfed | 0.214 ± 0.024 | 10.2 ± 0.3 | 0.347 ± 0.037 | 4 |
| peer, standard attack | krum | 0.327 ± 0.022 | 12.5 ± 0.4 | 0.039 ± 0.004 | 4 |
| peer, rule-aware attack | trimmed_mean | 0.699 ± 0.055 | 20.2 ± 0.9 | 0.014 ± 0.006 | 4 |
| peer, rule-aware attack | median | 0.724 ± 0.007 | 20.9 ± 0.2 | 0.012 ± 0.001 | 4 |
| peer, rule-aware attack | topfed | 0.326 ± 0.045 | 12.9 ± 0.9 | 0.137 ± 0.034 | 4 |
| peer, rule-aware attack | krum | 0.787 ± 0.139 | 22.9 ± 3.2 | 0.018 ± 0.031 | 4 |
| peer, rule-aware + estimated weights | topfed | 0.506 ± 0.096 | 16.4 ± 1.9 | 0.071 ± 0.043 | 4 (6 runs, averaged within seed) |
| peer, rule-aware + true weights (oracle) | topfed | 0.638 ± 0.041 | 19.1 ± 0.8 | 0.018 ± 0.005 | 4 |
| calibration | flat_gray | 0.577 ± 0.088 | 19.5 ± 1.2 | — | 4 |
| calibration | random_other_images | 0.375 ± 0.026 | 13.3 ± 0.6 | — | 4 |

### cifar100, warm model, protocol `5ep_b1` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.779 ± 0.126 | 21.6 ± 4.2 | 0.005 ± 0.008 | 4 |
| peer, standard attack | mean | 0.742 ± 0.076 | 20.3 ± 2.2 | 0.014 ± 0.011 | 4 |
| peer, standard attack | trimmed_mean | 0.242 ± 0.050 | 10.2 ± 1.2 | 0.391 ± 0.031 | 4 |
| peer, standard attack | median | 0.195 ± 0.027 | 9.6 ± 0.8 | 0.466 ± 0.045 | 4 |
| peer, standard attack | topfed | 0.200 ± 0.044 | 9.7 ± 1.0 | 0.506 ± 0.073 | 4 |
| peer, standard attack | krum | 0.313 ± 0.061 | 11.9 ± 1.5 | 0.095 ± 0.030 | 4 |
| calibration | flat_gray | 0.502 ± 0.181 | 15.2 ± 2.0 | — | 4 |
| calibration | random_other_images | 0.278 ± 0.012 | 11.0 ± 0.6 | — | 4 |

### cifar100, warm model, protocol `i16_e1_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.851 ± 0.007 | 23.5 ± 0.4 | 0.004 ± 0.001 | 2 |
| peer, standard attack | mean | 0.657 ± 0.008 | 18.9 ± 0.2 | 0.010 ± 0.001 | 2 |
| peer, standard attack | median | 0.250 ± 0.005 | 11.0 ± 0.2 | 0.279 ± 0.045 | 2 |
| peer, standard attack | topfed | 0.229 ± 0.005 | 10.6 ± 0.3 | 0.263 ± 0.031 | 2 |
| peer, rule-aware attack | median | 0.625 ± 0.051 | 18.6 ± 1.0 | 0.014 ± 0.005 | 2 |
| peer, rule-aware attack | topfed | 0.258 ± 0.003 | 11.6 ± 0.0 | 0.127 ± 0.023 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.565 ± 0.089 | 17.3 ± 1.7 | 0.018 ± 0.010 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.611 ± 0.025 | 18.2 ± 0.5 | 0.014 ± 0.004 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i16_e1_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.758 ± 0.018 | 19.7 ± 0.7 | 0.011 ± 0.001 | 2 |
| peer, standard attack | mean | 0.586 ± 0.013 | 17.1 ± 0.5 | 0.013 ± 0.002 | 2 |
| peer, standard attack | median | 0.240 ± 0.000 | 10.8 ± 0.1 | 0.296 ± 0.037 | 2 |
| peer, standard attack | topfed | 0.226 ± 0.000 | 10.7 ± 0.1 | 0.271 ± 0.028 | 2 |
| peer, rule-aware attack | median | 0.513 ± 0.055 | 16.0 ± 1.0 | 0.029 ± 0.010 | 2 |
| peer, rule-aware attack | topfed | 0.272 ± 0.007 | 11.8 ± 0.2 | 0.119 ± 0.014 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.310 ± 0.012 | 12.4 ± 0.3 | 0.106 ± 0.024 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.384 ± 0.086 | 13.7 ± 1.6 | 0.058 ± 0.024 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i16_e2_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.845 ± 0.010 | 23.1 ± 0.1 | 0.005 ± 0.002 | 2 |
| peer, standard attack | mean | 0.642 ± 0.004 | 18.6 ± 0.1 | 0.010 ± 0.001 | 2 |
| peer, standard attack | median | 0.245 ± 0.002 | 10.8 ± 0.2 | 0.289 ± 0.038 | 2 |
| peer, standard attack | topfed | 0.228 ± 0.008 | 10.7 ± 0.3 | 0.283 ± 0.039 | 2 |
| peer, rule-aware attack | median | 0.615 ± 0.021 | 18.4 ± 0.4 | 0.015 ± 0.001 | 2 |
| peer, rule-aware attack | topfed | 0.258 ± 0.001 | 11.5 ± 0.1 | 0.133 ± 0.025 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.441 ± 0.086 | 14.9 ± 1.7 | 0.072 ± 0.014 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.566 ± 0.028 | 17.4 ± 0.5 | 0.019 ± 0.007 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i16_e2_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.795 ± 0.022 | 21.4 ± 0.3 | 0.014 ± 0.000 | 2 |
| peer, standard attack | mean | 0.582 ± 0.002 | 17.3 ± 0.0 | 0.016 ± 0.000 | 2 |
| peer, standard attack | median | 0.246 ± 0.004 | 10.9 ± 0.3 | 0.294 ± 0.034 | 2 |
| peer, standard attack | topfed | 0.232 ± 0.007 | 10.7 ± 0.4 | 0.292 ± 0.037 | 2 |
| peer, rule-aware attack | median | 0.499 ± 0.034 | 16.0 ± 0.7 | 0.031 ± 0.000 | 2 |
| peer, rule-aware attack | topfed | 0.261 ± 0.003 | 11.5 ± 0.0 | 0.143 ± 0.023 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.345 ± 0.001 | 13.1 ± 0.1 | 0.101 ± 0.016 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.382 ± 0.151 | 13.7 ± 2.8 | 0.080 ± 0.066 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i16_e5_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.754 ± 0.032 | 20.1 ± 0.7 | 0.018 ± 0.004 | 2 |
| peer, standard attack | mean | 0.492 ± 0.007 | 15.7 ± 0.1 | 0.022 ± 0.001 | 2 |
| peer, standard attack | median | 0.251 ± 0.010 | 11.0 ± 0.4 | 0.267 ± 0.006 | 2 |
| peer, standard attack | topfed | 0.246 ± 0.017 | 10.9 ± 0.4 | 0.252 ± 0.040 | 2 |
| peer, rule-aware attack | median | 0.364 ± 0.023 | 13.4 ± 0.4 | 0.065 ± 0.013 | 2 |
| peer, rule-aware attack | topfed | 0.257 ± 0.003 | 11.4 ± 0.1 | 0.181 ± 0.014 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.303 ± 0.023 | 12.2 ± 0.4 | 0.118 ± 0.040 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.364 ± 0.054 | 13.2 ± 1.2 | 0.061 ± 0.001 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i16_e5_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.649 ± 0.055 | 17.9 ± 0.8 | 0.043 ± 0.010 | 2 |
| peer, standard attack | mean | 0.446 ± 0.003 | 14.8 ± 0.2 | 0.037 ± 0.001 | 2 |
| peer, standard attack | median | 0.247 ± 0.012 | 10.8 ± 0.5 | 0.274 ± 0.011 | 2 |
| peer, standard attack | topfed | 0.239 ± 0.025 | 10.7 ± 0.6 | 0.290 ± 0.026 | 2 |
| peer, rule-aware attack | median | 0.311 ± 0.000 | 12.3 ± 0.1 | 0.090 ± 0.003 | 2 |
| peer, rule-aware attack | topfed | 0.253 ± 0.003 | 11.3 ± 0.2 | 0.149 ± 0.002 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.288 ± 0.018 | 12.0 ± 0.3 | 0.122 ± 0.004 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.288 ± 0.034 | 11.9 ± 0.5 | 0.109 ± 0.024 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.370 ± 0.015 | 14.0 ± 0.5 | — | 2 |

### cifar100, warm model, protocol `i32_e1_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.806 ± 0.002 | 22.1 ± 0.6 | 0.004 ± 0.000 | 2 |
| peer, standard attack | mean | 0.504 ± 0.022 | 15.9 ± 0.3 | 0.017 ± 0.002 | 2 |
| peer, standard attack | median | 0.246 ± 0.004 | 11.1 ± 0.1 | 0.275 ± 0.051 | 2 |
| peer, standard attack | topfed | 0.234 ± 0.000 | 11.0 ± 0.2 | 0.249 ± 0.041 | 2 |
| peer, rule-aware attack | median | 0.267 ± 0.012 | 11.8 ± 0.0 | 0.072 ± 0.015 | 2 |
| peer, rule-aware attack | topfed | 0.250 ± 0.006 | 11.5 ± 0.0 | 0.108 ± 0.033 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.317 ± 0.045 | 12.7 ± 1.0 | 0.073 ± 0.009 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.263 ± 0.002 | 11.5 ± 0.1 | 0.075 ± 0.013 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i32_e1_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.661 ± 0.031 | 18.3 ± 0.9 | 0.018 ± 0.005 | 2 |
| peer, standard attack | mean | 0.457 ± 0.001 | 15.1 ± 0.2 | 0.021 ± 0.001 | 2 |
| peer, standard attack | median | 0.244 ± 0.002 | 11.1 ± 0.2 | 0.294 ± 0.050 | 2 |
| peer, standard attack | topfed | 0.229 ± 0.011 | 10.8 ± 0.5 | 0.274 ± 0.041 | 2 |
| peer, rule-aware attack | median | 0.262 ± 0.007 | 11.7 ± 0.0 | 0.081 ± 0.011 | 2 |
| peer, rule-aware attack | topfed | 0.239 ± 0.006 | 11.2 ± 0.1 | 0.117 ± 0.028 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.280 ± 0.015 | 12.0 ± 0.5 | 0.104 ± 0.009 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.254 ± 0.015 | 11.4 ± 0.1 | 0.095 ± 0.018 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i32_e2_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.804 ± 0.001 | 22.1 ± 0.6 | 0.005 ± 0.000 | 2 |
| peer, standard attack | mean | 0.506 ± 0.026 | 16.0 ± 0.4 | 0.016 ± 0.002 | 2 |
| peer, standard attack | median | 0.248 ± 0.001 | 11.2 ± 0.2 | 0.284 ± 0.047 | 2 |
| peer, standard attack | topfed | 0.232 ± 0.005 | 10.8 ± 0.3 | 0.259 ± 0.046 | 2 |
| peer, rule-aware attack | median | 0.267 ± 0.009 | 11.8 ± 0.0 | 0.076 ± 0.015 | 2 |
| peer, rule-aware attack | topfed | 0.243 ± 0.004 | 11.3 ± 0.1 | 0.115 ± 0.031 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.260 ± 0.025 | 11.5 ± 0.4 | 0.095 ± 0.019 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.265 ± 0.005 | 11.6 ± 0.3 | 0.075 ± 0.015 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i32_e2_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.579 ± 0.064 | 16.7 ± 2.0 | 0.049 ± 0.018 | 2 |
| peer, standard attack | mean | 0.405 ± 0.012 | 14.0 ± 0.6 | 0.037 ± 0.005 | 2 |
| peer, standard attack | median | 0.239 ± 0.004 | 10.9 ± 0.3 | 0.308 ± 0.012 | 2 |
| peer, standard attack | topfed | 0.214 ± 0.000 | 10.4 ± 0.3 | 0.306 ± 0.012 | 2 |
| peer, rule-aware attack | median | 0.247 ± 0.009 | 11.2 ± 0.5 | 0.098 ± 0.002 | 2 |
| peer, rule-aware attack | topfed | 0.226 ± 0.005 | 10.9 ± 0.5 | 0.137 ± 0.004 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.250 ± 0.023 | 11.2 ± 1.0 | 0.128 ± 0.003 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.226 ± 0.009 | 10.6 ± 0.3 | 0.117 ± 0.002 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i32_e5_b32` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.732 ± 0.001 | 20.0 ± 0.3 | 0.014 ± 0.001 | 2 |
| peer, standard attack | mean | 0.438 ± 0.008 | 14.9 ± 0.0 | 0.023 ± 0.000 | 2 |
| peer, standard attack | median | 0.245 ± 0.005 | 11.1 ± 0.3 | 0.274 ± 0.009 | 2 |
| peer, standard attack | topfed | 0.224 ± 0.006 | 10.7 ± 0.3 | 0.263 ± 0.009 | 2 |
| peer, rule-aware attack | median | 0.268 ± 0.008 | 11.8 ± 0.3 | 0.076 ± 0.003 | 2 |
| peer, rule-aware attack | topfed | 0.238 ± 0.005 | 11.3 ± 0.3 | 0.117 ± 0.005 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.257 ± 0.001 | 11.7 ± 0.3 | 0.092 ± 0.008 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.250 ± 0.006 | 11.4 ± 0.1 | 0.089 ± 0.006 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i32_e5_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.400 ± 0.039 | 13.0 ± 1.6 | 0.140 ± 0.009 | 2 |
| peer, standard attack | mean | 0.259 ± 0.004 | 10.6 ± 0.4 | 0.127 ± 0.004 | 2 |
| peer, standard attack | median | 0.211 ± 0.007 | 9.3 ± 0.4 | 0.360 ± 0.010 | 2 |
| peer, standard attack | topfed | 0.204 ± 0.012 | 9.3 ± 0.6 | 0.345 ± 0.019 | 2 |
| peer, rule-aware attack | median | 0.214 ± 0.005 | 9.9 ± 0.3 | 0.147 ± 0.003 | 2 |
| peer, rule-aware attack | topfed | 0.206 ± 0.002 | 10.1 ± 0.3 | 0.177 ± 0.009 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.241 ± 0.052 | 10.7 ± 1.4 | 0.202 ± 0.020 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.203 ± 0.005 | 9.8 ± 0.4 | 0.175 ± 0.014 | 2 |
| calibration | flat_gray | 0.540 ± 0.022 | 19.8 ± 1.0 | — | 2 |
| calibration | random_other_images | 0.338 ± 0.033 | 13.3 ± 0.3 | — | 2 |

### cifar100, warm model, protocol `i8_e1_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.833 ± 0.016 | 22.8 ± 0.0 | 0.003 ± 0.000 | 2 |
| peer, standard attack | mean | 0.753 ± 0.006 | 21.2 ± 0.1 | 0.005 ± 0.001 | 2 |
| peer, standard attack | median | 0.225 ± 0.010 | 10.3 ± 0.2 | 0.340 ± 0.027 | 2 |
| peer, standard attack | topfed | 0.215 ± 0.003 | 10.3 ± 0.1 | 0.356 ± 0.054 | 2 |
| peer, rule-aware attack | median | 0.735 ± 0.000 | 21.0 ± 0.2 | 0.011 ± 0.001 | 2 |
| peer, rule-aware attack | topfed | 0.340 ± 0.031 | 13.2 ± 0.8 | 0.123 ± 0.041 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.460 ± 0.060 | 15.5 ± 1.3 | 0.088 ± 0.033 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.673 ± 0.012 | 19.7 ± 0.2 | 0.014 ± 0.003 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.368 ± 0.023 | 13.3 ± 1.0 | — | 2 |

### cifar100, warm model, protocol `i8_e2_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.812 ± 0.019 | 22.0 ± 0.0 | 0.007 ± 0.001 | 2 |
| peer, standard attack | mean | 0.696 ± 0.002 | 19.8 ± 0.1 | 0.009 ± 0.002 | 2 |
| peer, standard attack | median | 0.229 ± 0.013 | 10.5 ± 0.2 | 0.341 ± 0.024 | 2 |
| peer, standard attack | topfed | 0.214 ± 0.008 | 10.3 ± 0.2 | 0.356 ± 0.050 | 2 |
| peer, rule-aware attack | median | 0.664 ± 0.012 | 19.4 ± 0.1 | 0.019 ± 0.003 | 2 |
| peer, rule-aware attack | topfed | 0.356 ± 0.006 | 13.4 ± 0.3 | 0.115 ± 0.023 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.479 ± 0.159 | 15.7 ± 3.1 | 0.074 ± 0.074 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.603 ± 0.021 | 18.2 ± 0.4 | 0.023 ± 0.002 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.368 ± 0.023 | 13.3 ± 1.0 | — | 2 |

### cifar100, warm model, protocol `i8_e5_b8` (seeds [0, 1])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.698 ± 0.074 | 19.0 ± 0.9 | 0.020 ± 0.009 | 2 |
| peer, standard attack | mean | 0.526 ± 0.007 | 16.2 ± 0.0 | 0.025 ± 0.003 | 2 |
| peer, standard attack | median | 0.230 ± 0.022 | 10.7 ± 0.3 | 0.302 ± 0.004 | 2 |
| peer, standard attack | topfed | 0.226 ± 0.018 | 10.6 ± 0.3 | 0.325 ± 0.013 | 2 |
| peer, rule-aware attack | median | 0.452 ± 0.016 | 15.1 ± 0.1 | 0.048 ± 0.004 | 2 |
| peer, rule-aware attack | topfed | 0.277 ± 0.013 | 11.9 ± 0.3 | 0.155 ± 0.012 | 2 |
| peer, rule-aware + estimated weights | topfed | 0.316 ± 0.015 | 12.6 ± 0.2 | 0.129 ± 0.009 | 2 |
| peer, rule-aware + true weights (oracle) | topfed | 0.397 ± 0.015 | 14.1 ± 0.1 | 0.063 ± 0.005 | 2 |
| calibration | flat_gray | 0.538 ± 0.019 | 18.6 ± 0.8 | — | 2 |
| calibration | random_other_images | 0.368 ± 0.023 | 13.3 ± 1.0 | — | 2 |

### cifar100, warm model, protocol `paper` (seeds [0, 1, 2, 3])

| observer | rule | SSIM | PSNR (dB) | match loss | n |
|---|---|---|---|---|---|
| server view (one client's raw update) | client | 0.298 ± 0.033 | 13.7 ± 1.5 | 0.712 ± 0.030 | 4 |
| peer, standard attack | mean | 0.260 ± 0.026 | 12.2 ± 1.4 | 0.714 ± 0.027 | 4 |
| peer, standard attack | trimmed_mean | 0.252 ± 0.032 | 12.0 ± 1.3 | 0.707 ± 0.015 | 4 |
| peer, standard attack | median | 0.244 ± 0.036 | 11.8 ± 1.3 | 0.722 ± 0.021 | 4 |
| peer, standard attack | topfed | 0.234 ± 0.027 | 11.7 ± 1.4 | 0.723 ± 0.026 | 4 |
| peer, standard attack | krum | 0.289 ± 0.039 | 13.6 ± 1.2 | 0.679 ± 0.041 | 4 |
| calibration | flat_gray | 0.674 ± 0.093 | 24.7 ± 1.8 | — | 4 |
| calibration | random_other_images | 0.402 ± 0.012 | 15.6 ± 0.5 | — | 4 |

