# Wheel
Rebuilding Algorithms and Machine Learning functions

Let $AB = a, AD = b, AC = \lambda b, CP = t_1 CB, DP = t_2 DE$  where $a$ and $b$ are vectors and $0 < \lambda , \mu , t_1 , t_2 < 1$. We have $CP = t_1 \left(-\lambda b + a \right) = t_1 a - t_1 \lambda b$ so $AP = t_1 a + \left(\lambda - t_1 \lambda \right)b$. But we also have $DP = t_2 \left(\mu a - b \right) = t_2 \mu a - t_2 b$, so another form for $AP$ is $AP = t_2 \mu a + \left(1 - t_2 \right)b$. Since $AP$ is an angle bisector, then by the angle bisector theorem we have $\frac{AD}{AE} = \frac{b}{\mu a} = \frac{t_2}{1 - t_2}$ and $\frac{AC}{AB} = \frac{\lambda b}{a} = \frac{t_1}{1 - t_1}$. Thus we get the four equations $t_2 \mu = t_1 , 1 - t_2 = \lambda - \lambda t_1 , \left(\frac{b}{a} \right) \left(1 - t_2 \right) = t_2 \mu , \left(\frac{b}{a} \right) \left(1 - t_1 \right) = \frac{t_1}{\lambda}$. We wish to show that $\frac{1}{AB} + \frac{1}{AC} = \frac{1}{AE} + \frac{1}{AD}$, or written in the symbols above, $\frac{1}{a} + \frac{1}{\lambda b} = \frac{1}{\mu a} + \frac{1}{b}$. But we observe that $\frac{1}{a} + \frac{1}{\lambda b} = \frac{t_1}{\left(1 - t_2 \right) b} + \frac{1 - t_1}{\left(1 - t_2 \right)b} = \frac{1}{\left(1 - t_2 \right)b}$, and $\frac{1}{\mu a} + \frac{1}{b} = \frac{t_2}{\left(1 - t_2 \right)b} + \frac{1}{b} = \frac{1}{\left(1 - t_2 \right)b}$ which proves equality. 


The equation of simple harmonic motion is given by $x = A\sin(\omega t) + B\cos(\omega t)$, and adding a constant does not change the nature of the equation since it just shifts the function, and we can take the constant to the LHS and create a new variable. The only one that cannot be written in this form is B.


$$ \left( \begin{array}{ccc} 1 & 0 & 0 \\
1 & 1 & 0 \\ 
0 & 1 & 1 \end{array} \right) $$
