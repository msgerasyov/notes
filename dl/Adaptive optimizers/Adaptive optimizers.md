Source: http://ruder.io/optimizing-gradient-descent/

### Momentum

SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. In these scenarios, SGD oscillates across the slopes of the ravine while only making hesitant progress along the bottom towards the local optimum.

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations. It does this by adding a fraction γ of the update vector of the past time step to the current update vector:

$$v_t = \gamma v_{t − 1} + \eta \nabla_\theta J(\theta)$$
$$\theta = \theta - v_t$$

Note: Some implementations exchange the signs in the equations. The momentum term $\gamma$ is usually set to 0.9 or a similar value.

```
alpha = 0.08 # learning rate. Please change.
mu = 0.9 # momentum. Please change.
v = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    v = mu * v + alpha * compute_grad(X[ind,:], y[ind], w)
    w = w - v
```

### Nesterov accelerated gradient

We know that we will use our momentum term $\gamma v_{t−1}$ to move the parameters $\theta$. Computing $\theta - \gamma v_{t-1}$ thus gives us an approximation of the next position of the parameters (the gradient is missing for the full update), a rough idea where our parameters are going to be. We can now effectively look ahead by calculating the gradient not w.r.t. to our current parameters $\theta$ but w.r.t. the approximate future position of our parameters:

$$v_t = \gamma v_{t−1} + \eta \nabla_\theta J(\theta - \gamma v_{t-1})$$
$$\theta = \theta - v_t$$

```
alpha = 0.1 # learning rate. Please change.
mu = 0.9 # momentum. Please change.
v = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    v = mu * v + alpha * compute_grad(X[ind,:], y[ind], w - mu * v)
    w = w - v
```

Now that we are able to adapt our updates to the slope of our error function and speed up SGD in turn, we would also like to adapt our updates to each individual parameter to perform larger or smaller updates depending on their importance.

### Adagrad

Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates (i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data. 

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

$G_t$ here is a diagonal matrix where each diagonal element $i,i$ is the sum of the squares of the gradients w.r.t. $\theta_i$ up to time step $t$. One of Adagrad's main benefits is that it eliminates the need to manually tune the learning rate. 

```
alpha = 0.7 # learning rate. Please change.
g_sum = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    g_sum += g ** 2
    w = w - (alpha / (g_sum + 1e-8)**(1/2)) * g
```

Adagrad's main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge. The following algorithms aim to resolve this flaw.

### Adadelta

Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size $w$. Instead of inefficiently storing $w$ previous squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients.

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g^2_t$$
$$E[\Delta \theta^2]_t = \gamma E[\Delta \theta^2]_{t-1} + (1-\gamma) \Delta \theta^2_t$$
$$\Delta \theta_t = - \frac{\sqrt{E[\Delta \theta^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}}g_t$$
$$\theta_{t+1} = \theta_t + \Delta \theta_t$$

The authors note that the units in this update (as well as in SGD, Momentum, or Adagrad) do not match, i.e. the update should have the same hypothetical units as the parameter.

```
gamma = 0.9 #moving average of gradient norm squared. Please change.
eps = 1e-2
g_rms = 0.0
delta_w = 0.0
delta_w_rms = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    g_rms = gamma * g_rms + (1 - gamma) * g**2
    delta_w = - ((delta_w_rms + eps)**(1/2)) / ((g_rms + eps)**(1/2)) * g
    w = w + delta_w
    delta_w_rms = gamma * delta_w_rms + (1 - gamma) * delta_w**2
```

With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the update rule.

### RMSprop

RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates. RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.

$$E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g^2_t$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}}g_t$$

```
w = np.array([0,0,0,0,0,1])

alpha = 0.07 # learning rate. Please change.
gamma = 0.9 #moving average of gradient norm squared. Please change.
g_rms = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)

    g = compute_grad(X[ind,:], y[ind], w)
    g_rms = gamma * g_rms + (1 - gamma) * g**2
    w = w - alpha * g / (g_rms + 1e-8)**(1/2)
```

### Adam

Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients $v_t$ like **Adadelta** and **RMSprop**, **Adam** also keeps an exponentially decaying average of past gradients $m_t$, similar to momentum.  We compute the decaying averages of past and past squared gradients $m_t$ and $v_t$ respectively as follows:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

As $m_t$ and $v_t$ are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small (i.e. $\beta_1$ and $\beta_2$ are close to 1). They counteract these biases by computing bias-corrected first and second moment estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta^t_1}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta^t_2}$$

They then use these to update the parameters just as we have seen in Adadelta and RMSprop, which yields the Adam update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$	

```
alpha = 0.1 # learning rate. Please change.
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-8
m = 0.0
v = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g**2
    m_hat = m / (1 - beta_1**(i+1))
    v_hat = v / (1 - beta_2**(i+1))
    w = w - alpha * m_hat / (v_hat**(1/2) + 1e-8)
```

### AdaMax

The $v_t$ factor in the Adam update rule scales the gradient inversely proportionally to the $l_2$ norm of the past gradients. We can generalize this update to the $l_p$ norm. Norms for large p values generally become numerically unstable, which is why $l_1$ and $l_2$ norms are most common in practice. However, $l_{\infty}$ also generally exhibits stable behavior. For this reason, the authors propose **AdaMax** (Kingma and Ba, 2015) and show that $v_t$ with $l_{\infty}$ converges to the following more stable value:

$$u_t = \beta_2^{\infty} v_{t-1} + (1 - \beta_2^{\infty})|g_t|^{\infty} = max(\beta_2 v_{t-1}, |g_t|)$$

We can now plug this into the Adam update equation by replacing $\sqrt{v_t + \epsilon}$ with $u_t$ to obtain the AdaMax update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{u_t} \hat{m}_t$$

Note that as $u_t$ relies on the max operation, it is not as suggestible to bias towards zero as $m_t$ and $v_t$ in Adam, which is why we do not need to compute a bias correction for $u_t$.

```
alpha = 0.3 # learning rate. Please change.
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-8
m = 0.0
u = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    m = beta_1 * m + (1 - beta_1) * g
    u = np.maximum(beta_2 * u, np.abs(g))
    m_hat = m / (1 - beta_1**(i+1))
    w = w - alpha * m_hat / u
```

### Nadam

As we have seen before, **Adam** can be viewed as a combination of **RMSprop** and **momentum**: RMSprop contributes the exponentially decaying average of past squared gradients $v_t$, while momentum accounts for the exponentially decaying average of past gradients $m_t$. We have also seen that **Nesterov accelerated gradient (NAG)** is superior to vanilla momentum.

Nadam (Nesterov-accelerated Adaptive Moment Estimation) thus combines Adam and NAG. In order to incorporate NAG into Adam, we need to modify its momentum term $m_t$.

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}(\beta_1 \hat{m}_t + \frac{(1 - \beta_1)g_t}{1 - \beta_1^t})$$

```
alpha = 0.1 # learning rate. Please change.
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-8
m = 0.0
v = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g**2
    m_hat = m / (1 - beta_1**(i+1))
    v_hat = v / (1 - beta_2**(i+1))
    w = w - alpha * (beta_1 * m_hat + (1 - beta_1) * g / (1 - beta_1**(i+1))) / (v_hat**(1/2) + 1e-8)
```

### AMSGrad

As adaptive learning rate methods have become the norm in training neural networks, practitioners noticed that in some cases, e.g. for object recognition or machine translation they fail to converge to an optimal solution and are outperformed by SGD with momentum.

Reddi et al. (2018) formalize this issue and pinpoint the exponential moving average of past squared gradients as a reason for the poor generalization behaviour of adaptive learning rate methods. In settings where Adam converges to a suboptimal solution, it has been observed that some minibatches provide large and informative gradients, but as these minibatches only occur rarely, exponential averaging diminishes their influence, which leads to poor convergence. 

To fix this behaviour, the authors propose a new algorithm, AMSGrad that uses the maximum of past squared gradients $v_t$ rather than the exponential average to update the parameters. 

Instead of using $v_t$ (or its bias-corrected version $\hat{v}_t$) directly, we now employ the previous $v_{t−1}$ if it is larger than the current one:

$$\hat{v}_t = max(\hat{v}_{t − 1}, v_t)$$

This way, AMSGrad results in a non-increasing step size, which avoids the problems suffered by Adam. For simplicity, the authors also remove the debiasing step that we have seen in Adam. The full AMSGrad update without bias-corrected estimates can be seen below:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{v}_t = max(\hat{v}_{t − 1}, v_t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}m_t$$

```
alpha = 0.1 # learning rate. Please change.
beta_1 = 0.9
beta_2 = 0.999
eps = 1e-8
m = 0.0
v = 0.0
v_hat = 0.0

for i in range(n_iter):
    ind = np.random.choice(X.shape[0], batch_size)
    
    g = compute_grad(X[ind,:], y[ind], w)
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * g**2
    v_hat = np.maximum(v_hat, v)
    w = w - alpha * m / (v_hat**(1/2) + 1e-8)
```


