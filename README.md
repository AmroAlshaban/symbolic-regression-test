# Wheel
Rebuilding Algorithms and Machine Learning functions

Let $AB = a, AD = b, AC = \lambda b, CP = t_1 CB, DP = t_2 DE$  where $a$ and $b$ are vectors and $0 < \lambda , \mu , t_1 , t_2 < 1$. We have $CP = t_1 \left(-\lambda b + a \right) = t_1 a - t_1 \lambda b$ so $AP = t_1 a + \left(\lambda - t_1 \lambda \right)b$. But we also have $DP = t_2 \left(\mu a - b \right) = t_2 \mu a - t_2 b$, so another form for $AP$ is $AP = t_2 \mu a + \left(1 - t_2 \right)b$. Since $AP$ is an angle bisector, then by the angle bisector theorem we have $\frac{AD}{AE} = \frac{b}{\mu a} = \frac{t_2}{1 - t_2}$ and $\frac{AC}{AB} = \frac{\lambda b}{a} = \frac{t_1}{1 - t_1}$. Thus we get the four equations $t_2 \mu = t_1 , 1 - t_2 = \lambda - \lambda t_1 , \left(\frac{b}{a} \right) \left(1 - t_2 \right) = t_2 \mu , \left(\frac{b}{a} \right) \left(1 - t_1 \right) = \frac{t_1}{\lambda}$. We wish to show that $\frac{1}{AB} + \frac{1}{AC} = \frac{1}{AE} + \frac{1}{AD}$, or written in the symbols above, $\frac{1}{a} + \frac{1}{\lambda b} = \frac{1}{\mu a} + \frac{1}{b}$. But we observe that $\frac{1}{a} + \frac{1}{\lambda b} = \frac{t_1}{\left(1 - t_2 \right) b} + \frac{1 - t_1}{\left(1 - t_2 \right)b} = \frac{1}{\left(1 - t_2 \right)b}$, and $\frac{1}{\mu a} + \frac{1}{b} = \frac{t_2}{\left(1 - t_2 \right)b} + \frac{1}{b} = \frac{1}{\left(1 - t_2 \right)b}$ which proves equality. 


The equation of simple harmonic motion is given by $x = A\sin(\omega t) + B\cos(\omega t)$, and adding a constant does not change the nature of the equation since it just shifts the function, and we can take the constant to the LHS and create a new variable. The only one that cannot be written in this form is B.


$$ \left( \begin{array}{ccc} 1 & 0 & 0 \\
1 & 1 & 0 \\ 
0 & 1 & 1 \end{array} \right) $$

Write $\tan^2(x) = \frac{\sin^2(x)}{\cos^2(x)}$ and remember that $\sin^2(x) = \frac{1}{2} \left(1 - \cos(2x) \right)$, now find a similar identity for $\cos^2(x)$ and you'll reach the result.

This is just another form of the triangle inequality (sometimes called reverse triangle inequality: $\left|a + b \right| \geq \left| |a| - |b| \right|$). As for $\left|z^2 - {z_0}^2 \right| = \left|z - z_0 \right| \left|z + z_0 \right|$, they probably meant to put equal instead, but the rest is correct. This becomes: $\left|z^2 - {z_0}^2 \right| = \left|z - z_0 \right| \left|z + z_0 \right| < \left|z + z_0 \right| = \left|z - z_0 + 2z_0 \right|$ then apply the standard triangle inequality ($z + z_0 = z - z_0 + 2z_0$, they just added an extra $z_0$ and subtracted one so the result didn't change, and the question says $\left|z - z_0 \right| < 1$).

Expand $\left|z_1 - z_2 \right|^2 + \left|z_2 - z_3 \right|^2 + \left|z_3 - z_4 \right|^2 + \left|z_4 - z_1 \right|^2 = \left(z_1 - z_2 \right) \left(\bar{z_1} - \bar{z_2}\right) + \left(z_2 - z_3 \right) \left(\bar{z_2} - \bar{z_3}\right) + \left(z_3 - z_4 \right) \left(\bar{z_3} - \bar{z_4}\right) + \left(z_4 - z_1 \right) \left(\bar{z_4} - \bar{z_1}\right)$ then use the equality $\left|z_1 \right|^2 +\left|z_2 \right|^2 + \left|z_3 \right|^2 + \left|z_4 \right|^2 = 1$ to simplify the expression into $2 - \left(z_1 \bar{z_2} + \bar{z_1}z_2 \right) - \left(z_1 \bar{z_4} + \bar{z_1}z_4 \right) - \left(z_2 \bar{z_3} + \bar{z_2}z_3 \right) - \left(z_3 \bar{z_4} + \bar{z_3}z_4 \right)$. Lastly, substitute $z_4 = -z_1 - z_2 - z_3$ to finally simplify into $\left|z_1 - z_2 \right|^2 + \left|z_2 - z_3 \right|^2 + \left|z_3 - z_4 \right|^2 + \left|z_4 - z_1 \right|^2 = 2\left(1 + \left(z_1 + z_3 \right)^2 \right) \geq 2$.


The basic idea here is that the two statements, the first regarding $X$ and the second regarding $Y$, are identical; you can transform each problem into the other. 
For instance, consider $\sum_{i=1}^{n} y_i = a - \frac{1}{2} n \left(n + 1 \right) \in \mathbb{N}$. Then, you can transform this problem into:
$a = \sum_{i=1}^{n} y_i + \frac{1}{2} n \left(n + 1 \right) = \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} i = \sum_{i=1}^{n} \left(y_i + i \right)$, and now simply rename $y_i + i = x_i$, and you can do the same thing in reverse. So essentially you're not changing the problem or the size of the solution set, you're just transforming the values. 


The equation of the regression line is given by $y = mx + \left(\bar{y} - m \bar{x} \right)$ where $m$ is the slope/gradient of the regression line, $\bar{x}$ is the mean of the years and $\bar{y}$ is the mean of the number of colonies (this is because the mean of the given sample must lie on the regression line, so the $y$-intercept is given by $\bar{y} - m \bar{x}$). Assuming you know what the slope is (you could use a formula or just approximate it from your graph), just substitute $x = 2040$ into the equation of the line.

Suppose $\tau$ is not the identity but satisfies the criterion $\tau \sigma = \sigma \tau$ for all $\sigma \in S_3$. Since $S_3$ is a non-Abelian group of order 6, it must have at least an element of order 2 and an element of order 3 (as both divide 6) by Cauchy's theorem (or we can construct these elements since we know what $S_3$ looks like), but it has no element of order 6 (else it is cyclic).

Let $\sigma_0$ be an element of order 2 and $\sigma_1$ an element of order 3. If $\tau$ has order 2, then $\tau \sigma_1 \neq \text{id}$ (else $\sigma_1 = \tau$, a contradiction), so $\tau \sigma_1$ must have order 2 or 3. But we find that $\left(\tau \sigma_1 \right)^2 = \tau^2 {\sigma_1}^2 = {\sigma_1}^{-1} \neq \text{id}$ and $\left(\tau \sigma_1 \right)^3 = \tau^3 {\sigma_1}^3 = \tau \neq \text{id}$, a contradiction. Similarily, if $\tau$ has order 3, then $\tau \sigma_0 \neq \text{id}$ and the rest follows as the first case with little adjustments. Thus $\tau$ cannot satisfy the property $\tau \sigma = \sigma \tau$ for all $\sigma \in S_3$ unless $\tau = \text{id}$.

Knowing that $a_i = ar^{i - 1} \rightarrow a_i^2 = a^2 \left(r^2 \right)^{i - 1}$, try to deduce why $\\{ {a_i}^2 \\}_{i \in \mathbb{N} }$ must be a geometric sequence.

We can add $\frac{1}{2}$ on both sides of our equation to get $x + 9xy + \frac{9}{2} y + \frac{1}{2} = 8$ which we can simplify to $(2x + 1)(9y + 1) = 16$. Here the question is basically giving us a hint that we simply need to try out integer pairs, so the possible pairs are $(1,\ 16),\ (2,\ 8),\ (4,\ 4),\ (8,\ 2),\ (16,\ 1),\ (-1,\ -16),\ (-2,\ -8),\ (-4,\ -4),\ (-8,\ -2),\ (-16,\ -1)$. 

For each choice, we must find $x$ and $y$ and then test the choices given and see if any is satisfied. Here, I'm not certain what they mean by "$x$ and $y$ both have a certain maximum value" but if they mean we should choose the positive choices instead of the negatives, then we find that $(4, 4)$ produces the maximum $xy$ with $x = \frac{3}{2}$ and $y = \frac{1}{3}$, and we get $x - y = \frac{7}{6}$ implying choice C is the answer (but again this depends on what they mean by "$x$ and $y$ both have a certain maximum value" which I don't know for sure).

The idea is correct but it's not necessary that $-2x + \frac{3}{2} y = mx$ and $-x - \frac{3}{2} y = my$, where $m$ is the slope. It's given by $-2x + \frac{3}{2} y = \alpha x$ and $-x - \frac{3}{2} y = \alpha y$ for some $\alpha$. But we don't really need to do this, just remember that $\binom{-2x + \frac{3}{2} y}{-x - \frac{3}{2} y}$ lines inside the line $y = mx$ so just substitute and solve for $m$.

Let $\binom{x}{y}$ be a point inside the line $y = mx$ and let's assume that this line is invariant under the transformation CAB (i.e. CAB multiplied by $\binom{x}{y}$ must ALSO lie inside the line $y = mx$). We have $CAB\binom{x}{y} = \binom{-2x + \frac{3}{2} y}{-x - \frac{3}{2} y}$. But as we agreed, we have $y = mx$ and the new output must also lie inside the line, so $-x - \frac{3}{2} y = m\left(-2x + \frac{3}{2} y \right)$. Finally replace $y$ with $mx$, so we get $-x - \frac{3}{2} (mx) = m\left(-2x + \frac{3}{2} (mx) \right)$. Now, you must find the values of $m$ so that this equation holds true.
