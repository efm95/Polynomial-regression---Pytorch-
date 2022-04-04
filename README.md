# Polynomial regression with PyTorch

Small PyTorch framework used to estimate the true weights of

$$p(x) = x^3+2x^2-4x-8 = \sum_{i=0}^{3}\mathbf{w}_i^*x^i$$.

Consider the polynomial $p$ given by

\begin{equation*}
    p(x)=x^3 +2x^2-4x-8 = \sum_{i=0}^{3}\mathbf{w}_i^{*}x^i,
\end{equation*}
where $\mathbf{w}^*=[-8,-4,2,1]^\top$. Consider also an i.i.d. dataset $\mathcal{D} = \{(x_i, y_i)\}^N_{i=1}$, where $y_i = p(x_i)+\epsilon_i$, and each $\epsilon_i \sim N(0,\sigma = 0.5)$.\newline

By assuming that $\textbf{w}^*$ is unknown, a polynomial liner regression approach could estimate it given the dataset $\mathcal{D}$. This would require to apply a feature map that transform the original dataset $\mathcal{D}$ into an expanded dataset $\mathcal{D}'= \{(\mathbf{X}_i, y_i)\}^N_{i=1}$, where $\mathbf{X}_i=[1,x_i,x^{2}_i,x^{3}_i]^\top$.\newline 
