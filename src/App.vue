<!--<template>-->
<!--&lt;!&ndash;  <v-md-editor v-model="text" height="400px" :include-level="[1,2,3, 4]" :default-show-toc="true"></v-md-editor>&ndash;&gt;-->
<!--&lt;!&ndash;  <v-md-preview :text="text" height="400px" ></v-md-preview>&ndash;&gt;-->

<!--</template>-->

<!--<script>-->
<!--export default {-->
<!--  data() {-->
<!--    return {-->
<!--      text: 'hello',-->
<!--    };-->
<!--  },-->
<!--};-->
<!--</script>-->
<!--<template>-->
<!--  <v-md-preview :text="text"></v-md-preview>-->
<!--</template>-->

<!--<script>-->
<!--export default {-->
<!--  data() {-->
<!--    return {-->
<!--      text: 'sdf',-->
<!--    };-->
<!--  },-->
<!--};-->
<!--</script>-->


<template>
  <div>
    <div
        v-for="anchor in titles"
        :style="{ padding: `10px 0 10px ${anchor.indent * 20}px` }"
        @click="handleAnchorClick(anchor)"
    >
      <a style="cursor: pointer">{{ anchor.title }}</a>
    </div>
    <v-md-preview :text="text" ref="preview" />
  </div>
</template>

<script>
const text = `
# Balanced Multimodal Learning via On-the-fly Gradient Modulation

> 作者：Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, Di Hu
>
> 发表时间：2022
>
> 期刊：IEEE Conference on Computer Vision and Pattern Recognition（CVPR）
>
> 链接：https://arxiv.org/abs/2203.15332

## 背景与动机

与单模态数据相比，多模态数据通常提供更多的视图，因此使用多模态数据进行学习应该匹配或优于单模态情况。然而在某些情况下，使用联合训练策略优化所有模态统一学习目标的多模态模型可能不如单模态模型。这种现象违背了通过整合多模态信息来提高模型性能的意图。

然而作者进一步发现，即使多模态模型优于单模态模型，它们仍然不能充分利用多模态的潜力。

如下图，实验中仅使用audio数据和visual数据的模型，效果明显好于联合优化后单模态编码器。单模态模型比多模态模型表现更好，说明多模态模型中单模态被抑制。同时与音频案例相比，视觉模态的准确度下降得更明显，这与VGGSound数据集是一个更面相声音的数据集有关，声音模态占据主导地位，导致了优化不平衡现象。

![pPY6jPJ.png](https://z1.ax1x.com/2023/08/24/pPY6jPJ.png)
## 提出解决方法

- 从损失函数优化算法的角度分析了这种不平衡的现象。
- 提出了一种基于动态梯度调整的自适应控制策略，即通过监测不同模态对学习目标贡献的差异来动态调整其优化。
- 引入了一个动态变化的额外高斯噪声，避免梯度调整可能带来的泛化性能下降。

### 优化不平衡分析\t

数据集：$D=\\{x_{i},y_{i}\\}_{i=1,2\\ldots N}$其中$x_{i}=(x^{a}_{i},x^{v}_{i})$，$a$和$v$代表两种不同的模态。$y_{i}\\in{1,2,...,M}$，$M$代表最终类别总数。

使用两个编码器$\\varphi^{a}(\\theta^{a},\\cdot)$和$\\varphi^{v}(\\theta^{v},\\cdot)$来提取特征，其中$\\theta^{a}$和$\\theta^{v}$是编码器的参数。使用串联的方式将两个特征汇集，并令$M$和$b$作为最后一个线性分类器的参数。则多模态模型的预测输出如下:
$$
f(x_{i})=W[\\varphi^{a}(\\theta^{a},x_{i}^{a});\\varphi^{v}(\\theta^{v},x_{i}^{v})]+b \\tag{1}
$$
$W$可以表示为两个块的组合:[$W^a,W^v$]
$$
f(x_{i})=W^{a}\\cdot\\varphi^{a}(\\theta^{a},x_{i}^{a})+W^{v}\\cdot\\varphi^{v}(\\theta^{v},x_{i}^{v})+b \\tag{2}
$$
以交叉熵损失函数为目标优化函数
$$
L=-\\log\\frac{e^{f(x_{i})_{y_{i}}}}{\\sum_{k=1}^{M}e^{f(x_{i})_{k}}} \\tag{3}
$$
那么，$W^a$和$\\theta^{a}$梯度更新如下：
$$
\\begin{aligned}
W_{t+1}^{a}& =W_{t}^{a}-\\eta\\nabla_{W^{a}}L(W_{t}^{a})  \\\\
&=W_t^a-\\eta\\frac{1}{N}\\sum_{i=1}^{N}\\frac{\\partial L}{\\partial f(x_i)}\\varphi^a(\\theta^a,x_i^a)
\\end{aligned} \\tag{4}
$$

$$
\\begin{aligned}
\\theta_{t+1}^{a}& =\\theta_{t}^{a}-\\eta\\nabla_{\\theta^{a}}L(\\theta_{t}^{a})  \\\\
&=\\theta_t^a-\\eta\\frac{1}{N}\\sum_{i=1}^N\\frac{\\partial L}{\\partial f(x_i)}\\frac{\\partial(W_t^a\\cdot\\varphi_t^a(\\theta^a,x_i^a))}{\\partial\\theta_t^a}
\\end{aligned} \\tag{5}
$$

通过上诉两个公式可以看出，两个模态编码器参数的梯度更新除了与$\\frac{\\partial L}{\\partial f(x_i)_c}$相关，其他的地方没有任何相关性。其中
$$
\\frac{\\partial L}{\\partial f(x_i)_c}=\\frac{e^{(W^a\\cdot\\varphi_i^a+W^v\\cdot\\varphi_i^v+b)_c}}{\\sum_{k=1}^Me^{(W^a\\cdot\\varphi_i^a+W^v\\cdot\\varphi_i^v+b)_k}}-1_{c=yi} \\tag{6}
$$
所以可以推断出，当其中一个模态表现得更好时 ，它通过$W^u\\cdot\\varphi_i^u$对$\\frac{\\partial L}{\\partial f(x_i)_c}$贡献更大，从而降低整体损失。因此这也导致了当多模态模型的训练即将收敛时，另一模态仍然可能存在未优化的表示，需要进一步的训练。

### 动态梯度调制

![pPClxC8.png](https://s1.ax1x.com/2023/08/24/pPY6qVU.png)

根据优化不平衡分析，作者提出通过监测各模态对学习目标贡献的差异，自适应调节各模态的梯度。具体如下：
$$
s_i^a=\\sum_{k=1}^M1_{k=y_i}\\cdot\\text{softmax}(W_t^a\\cdot\\varphi_t^a(\\theta^a,x_i^a)+\\frac b2)_k \\tag{7}
$$

$$
s_i^v=\\sum_{k=1}^M1_{k=y_i}\\cdot\\mathrm{softmax}(W_t^v\\cdot\\varphi_t^v(\\theta^v,x_i^v)+\\frac b2)_k \\tag{8}
$$

$$
\\rho_t^v=\\frac{\\sum_{i\\in B_t}s_i^v}{\\sum_{i\\in B_t}s_i^a}  \\tag{9}
$$

使用$W_{i}^{u}\\cdot\\varphi_{i}(\\theta^{u},x_{i}^{u})+\\frac{b}{2}$作为模态u的近似预测来估计多模态模型的单模态性能，利用$\\rho_t^u$动态监测音频和视觉模态之间的贡献差异，我们可以

通过以下方式自适应调制梯度:
$$
k_t^u=\\left\\{\\begin{array}{cc}1-\\tanh(\\alpha\\cdot\\rho_t^u)&&\\rho_t^u>1\\\\1&&\\text{others}\\end{array}\\right. \\tag{10}
$$
其中$a$为超参数，将系数$k_t^u$乘到SGD优化方法中，$\\theta^{u}_t$迭代更新如下:
$$
\\theta_{t+1}^u=\\theta_t^u-\\eta\\cdot k_t^u\\tilde{g}(\\theta_t^u) \\tag{11}
$$
通过$k_t^u$，减轻了性能较好的($\\rho_t^u$ > 1)模态的优化，而另一个模态不受影响，能够摆脱有限的优化努力，获得足够的训练。

### 泛化增强

$$
\\theta_{t+1}^u=\\theta_t^u-\\eta\\tilde{g}(\\theta_t^u) \\tag{12}
$$

其中：$ \\tilde{g}(\\theta_{t}^{u})=\\frac{1}{m}\\sum_{x\\in B_{t}}\\nabla_{\\theta^{u}}\\ell(x;\\theta_{t}^{u})$，$B_t$是在第$t$步中选择的随机小批量，大小为$m$，$\\nabla_{\\theta^{u}}\\ell(x;\\theta_{t}^{u})$为该批次的梯度。由中心极限定理，当批大小m足够大时
$$
\\tilde{g}(\\theta_{t}^{u})\\sim\\mathcal{N}(\\nabla_{\\theta^{u}}L(\\theta_{t}^{u}),\\Sigma^{sgd}(\\theta_{t}^{u})) \\tag{13}
$$

$$
\\begin{aligned}
\\Sigma^{sgd}(\\theta_{t}^{u})& \\approx\\frac{1}{m}[\\frac{1}{N}\\sum_{i=1}^{N}\\nabla_{\\theta^{u}}\\ell(x_{i};\\theta_{t}^{u})\\nabla_{\\theta^{u}}\\ell(x_{i};\\theta_{t}^{u})^{\\mathrm{T}}  \\\\
&-\\nabla_{\\theta^{u}}L(\\theta_{t}^{u})\\nabla_{\\theta^{u}}L(\\theta_{t}^{u})^{\\mathrm{T}}].
\\end{aligned} \\tag{14}
$$

> 这个公式的含义是，我们通过计算单个训练样本上的梯度以及整个训练集上的真实梯度，来近似估计随机梯度估计$\\tilde{g}(\\theta_{t}^{u})$的协方差矩阵。
>
> $\\Sigma^{sgd}(\\theta_{t}^{u})$:表示在参数$\\theta_{t}^{u}$处估计的随机梯度估计协方差矩阵。
>
> $m$: 批大小，表示在每次计算梯度时使用的训练样本数量。
>
> $N$: 总训练样本数量。
>
> $\\nabla_{\\theta^{u}}\\ell(x_{i};\\theta_{t}^{u})$: 表示损失函数相对于参数$\\theta_{t}^{u}$在第$i$个训练样本上的$x_{i}$梯度。
>
> $\\nabla_{\\theta^{u}}L(x_{i};\\theta_{t}^{u})$: 表示损失函数相对于参数$\\theta^{u}$在整个训练集上的真实梯度。
>
> $\\ell(x_{i};\\theta_{t}^{u})$: 表示损失函数在第$i$个训练样本$x_{i}$上的值。
>
> $x_{i}$: 第$i$个训练样本。
>
> $T$: 表示转置操作。

可以将上述梯度下降算法更改为以下形式，$\\xi_{t}$是噪声项
$$
\\theta_{t+1}^{u}=\\theta_{t}^{u}-\\eta\\nabla_{\\theta^{u}}L(\\theta_{t}^{u})+\\eta\\xi_{t},\\xi_{t}\\sim\\mathcal{N}(0,\\Sigma^{sgd}(\\theta_{t}^{u})) \\tag{15}
$$

- SGD generalization ability：SGD中的噪声与其泛化能力密切相关，SGD噪声越大往往泛化效果越好。SGD噪声的协方差与学习率与批大小的比值成正比【Three factors influencing minima in sgd】

由以上定理可知，$\\xi_{t}$越高，泛化能力越好，当我们用系数$k_t^u$来调制梯度时，$\\theta_{t}^{u}$更新为:
$$
\\begin{aligned}\\theta_{t+1}^u&=\\theta_t^u-\\eta\\nabla_{\\theta^u}L'(\\theta_t^u)+\\eta\\xi_t',\\\\\\xi_t'&\\sim\\mathcal{N}(0,(k_t^u)^2\\cdot\\Sigma^{sgd}(\\theta_t^u)),\\end{aligned} \\tag{16}
$$
其中：$\\eta\\nabla_{\\theta^u}L^{\\prime}(\\theta_t^u)=k_t^u\\cdot\\eta\\nabla_{\\theta^u}L(\\theta_t^u)$

现在协方差比之前变得更小，降低了SGD优化方法的泛化能力。因此，需要开发一种控制SGD噪声的方法，恢复其泛化能力。

为了增强SGD噪声，作者引入了一种简单而有效的泛化增强(GE)方法，该方法在梯度中加入随机采样的高斯噪声$h(\\theta_{t}^{u})\\sim \\mathcal{N}(0,\\Sigma^{sgd}(\\theta_{t}^{u}))$，该方法与$\\tilde{g}(\\theta_{t}^{u})$具有相同的协方差，并根据当前迭代动态变化。然后，结合OGM方法，将$\\theta_{t}^{u}$更新为:
$$
\\begin{aligned}
\\theta_{t+1}^{u}& =\\theta_{t}^{u}-\\eta(k_{t}^{u}\\tilde{g}(\\theta_{t}^{u})+h(\\theta_{t}^{u}))  \\\\
&=\\theta_{t}^{u}-\\eta\\nabla_{\\theta^{u}}L^{\\prime}(\\theta_{t}^{u})+\\eta\\xi_{t}^{\\prime}+\\eta\\epsilon_{t}
\\end{aligned} \\tag{17}
$$

$$
$\\epsilon_{t}\\sim\\mathcal{N}(0,\\Sigma^{sgd}(\\theta_{t}^{u}))$
$$

因为$\\epsilon_{t}$和$\\xi_t$相互独立，所以上诉可以更新为
$$
\\begin{aligned}\\theta_{t+1}^u&=\\theta_t^u-\\eta\\nabla_{\\theta^u}L'(\\theta_t^u)+\\eta\\xi_t'',\\\\\\xi_t''&\\sim\\mathcal{N}(0,((k_t^u)^2+1)\\Sigma^{sgd}(\\theta_t^u)).\\end{aligned} \\tag{18}
$$
因此，SGD噪声的协方差得到恢复甚至增强，可以使多模态模型的优化更加均衡，保证每个模态的泛化能力。整体OGM-GE策略算法如下：

![pPClxC8.png](https://s1.ax1x.com/2023/08/24/pPY2eDP.png)

## 实验数据

作者首先将 OGM-GE 方法应用于几种常见的融合方法： baseline、concatenation 和 summation，以及专门设计的融合方法：FiLM 。性能如下表所示。从表中可以看出，两个模态性能不平衡，audio模态性能要明显优于 visual 模态。结合OGM-GE方法以后，可以看到模型的性能有显著提升。

![pPClxC8.png](https://s1.ax1x.com/2023/08/24/pPY213j.png)

  在CREMA-D和Kinetics-Sounds数据集上与其他调制策略比较

![image-20230830232248286](C:\\Users\\mr.chang\\AppData\\Roaming\\Typora\\typora-user-images\\image-20230830232248286.png)

在多模态模式下，调制后的音频和视觉模态性能都有所提高

![image-20230830232339151](C:\\Users\\mr.chang\\AppData\\Roaming\\Typora\\typora-user-images\\image-20230830232339151.png)

在训练过程中监测差异比的变化。可以看出，应用OGM-GE后，差异比$ρ_a$明显减小。

![image-20230830232400676](C:\\Users\\mr.chang\\AppData\\Roaming\\Typora\\typora-user-images\\image-20230830232400676.png)
`;

export default {
  data() {
    return {
      text,
      titles: [],
    };
  },
  mounted() {
    const anchors = this.$refs.preview.$el.querySelectorAll('h1,h2,h3,h4,h5,h6');
    const titles = Array.from(anchors).filter((title) => !!title.innerText.trim());

    if (!titles.length) {
      this.titles = [];
      return;
    }

    const hTags = Array.from(new Set(titles.map((title) => title.tagName))).sort();

    this.titles = titles.map((el) => ({
      title: el.innerText,
      lineIndex: el.getAttribute('data-v-md-line'),
      indent: hTags.indexOf(el.tagName),
    }));
  },
  methods: {
    handleAnchorClick(anchor) {
      const { preview } = this.$refs;
      const { lineIndex } = anchor;

      const heading = preview.$el.querySelector(`[data-v-md-line="${lineIndex}"]`);

      if (heading) {
        // 注意：如果你使用的是编辑组件的预览模式,则这里的方法名改为 previewScrollToTarget
        preview.scrollToTarget({
          target: heading,
          scrollContainer: window,
          top: 60,
        });
      }
    },
  },
};
</script>
