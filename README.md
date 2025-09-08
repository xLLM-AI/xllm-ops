<!-- Copyright 2022 JD Co.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this project except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

[English](./README.md) | [中文](./README_zh.md)

<div align="center">
<img src="docs/logo_with_llm.png" alt="xLLM" style="width:50%; height:auto;">
    
[![Document](https://img.shields.io/badge/Document-black?logo=html5&labelColor=grey&color=red)](https://xllm.readthedocs.io/zh-cn/latest/) [![Docker](https://img.shields.io/badge/Docker-black?logo=docker&labelColor=grey&color=%231E90FF)](https://hub.docker.com/r/xllm/xllm-ai) [![License](https://img.shields.io/badge/license-Apache%202.0-brightgreen?labelColor=grey)](https://opensource.org/licenses/Apache-2.0) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/jd-opensource/xllm) 
    
</div>

---------------------
<p align="center">
| <a href="https://xllm.readthedocs.io/zh-cn/latest/"><b>Documentation</b></a> | 
</p>

## 1. Overview

XLLM OPS is a scalable and high-performance operator library designed for large language models.

With the widespread application of large language models, developers face challenges such as low computational efficiency and high resource consumption during model training and inference. These issues have created an urgent need for an efficient operator library to enhance performance and reduce costs.

To address this, we designed XLLM OPS, a domestic chip operator library focused on performance optimization, aiming to provide faster computation speeds and lower resource consumption.


## 2. Quick Start
* Compile the operator library
```bash
bash build.sh
```

## 3. Test
```bash
cd test
export VCPKG_ROOT=/path/to/vcpkg
bash build.sh
./bin/group_gemm_gtest
```

## 4. Performance Results
![groupmatmul测试](docs/groupmatmul_performance.png)
* The optimized GroupMatmul operator shows significant advantages in computation time, especially when 
k=128 and m=64. As shown in the figure, the optimized operator's time consumption is reduced by 50%.
* After using the topKtopP fusion operator, in the qwen2-0.5B model, TTOT decreased by 37%, and TTFT improved by 10%.


## 5. Community Support
If you encounter any issues along the way, you are welcomed to submit reproducible steps and log snippets in the project's Issues area, or contact the xLLM Core team directly via your internal Slack. Moreover, we have established a WeChat user group. You can find our group chat QR code image [here](https://qr.link/JZaROS) or visit the following live QR code. Welcome to contact us!

<div align="center">
  <img src="docs/wechat_qrcode1.png" alt="qrcode1" width="30%" />
  <img src="docs/wechat_qrcode2.png" alt="qrcode2" width="30%" />
</div>

## 6. Acknowledgments
This project was made possible thanks to the following open-source projects: 
- [cann-ops-dev](https://gitee.com/ascend/cann-ops-adv) - Adopted the engineering construction in cann-ops-adv


## 7. License
[Apache License]( ./LICENSE.md)

#### xLLM OPS is provided by JD.com


