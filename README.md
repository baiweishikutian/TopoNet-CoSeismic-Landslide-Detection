# TopoNet: A Topography-Aware Deep Network with Gating Mechanisms for Co-Seismic Landslide Detection
 **Note:  
Model Architecture

TopoNet is a topography-aware multimodal neural network designed for co-seismic landslide mapping from medium-resolution remote sensing imagery. The network explicitly models spectral–topographic interactions and emphasizes boundary preservation and cross-regional generalization. The architecture consists of three core components:

Topography-Aware Gate (TAG) Module

The Topography-Aware Gate dynamically modulates spectral feature responses under the guidance of topographic information. By learning adaptive gating weights conditioned on terrain attributes, this module suppresses irrelevant or redundant spectral responses while enhancing landslide-sensitive features, enabling more effective multimodal fusion and improving robustness across heterogeneous geographic regions.

Boundary-Aware Convolutional Enhancement (BACE) Module

The Boundary-Aware Convolutional Enhancement module strengthens multi-scale contextual representation by integrating dilated and depthwise separable convolutions. It is specifically designed to enhance boundary sensitivity in complex textured areas, reducing edge blurring and improving the delineation of small and fragmented landslides commonly observed in medium-resolution imagery.

Improved Residual Block

The improved residual block preserves fine-grained surface textures while maintaining stable gradient propagation. By refining local feature representation without excessive smoothing, this block enhances boundary extraction and improves discrimination between landslides and spectrally similar background objects such as bare soil or roads.

Overall, TopoNet follows an encoder–decoder paradigm, where multimodal features are progressively encoded with topography-guided fusion and decoded to generate accurate landslide probability maps. The architecture is optimized for both detection accuracy and cross-regional transferability under data-scarce conditions.
<img width="9232" height="4320" alt="最终图-总体模型（图4）" src="https://github.com/user-attachments/assets/f1330e6e-ce59-47d3-a9c9-9e2fab87cfee" />

## Authors
**Jia Liu, Zihang Liu, Xing Gao, Yuming Wu, Zhitao Wei, Jiaye Yao**

