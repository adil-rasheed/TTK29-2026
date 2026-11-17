# Where should three sensors measure a cooling rod?

## My starting idea

A metal rod cools down over time. I can afford only three temperature sensors.
From their noisy measurements I want to estimate the rod's thermal diffusivity.
Where should the sensors go so that the estimate is as accurate as possible,
and how much does placement actually matter?

## Course connections

Physics-based heat diffusion, inverse parameter estimation, noise and
uncertainty, and sensor placement. Sparse field reconstruction and parameter
identification are related but different tasks; I want to understand that
distinction.

## Possible comparison

Evenly spaced sensors, random positions and a smarter (e.g. sensitivity-based)
placement, all with the same number of sensors. Start with synthetic data from
a one-dimensional heat model. Use a simple estimator before anything complex.

## What I do not yet know

Has this been studied before? Which placement criterion fits a single unknown
parameter? How do noise, model errors and the initial condition change the
answer? What is a fair baseline?

## Constraints and expected outcome

A laptop, the course Python environment and a few weeks. I want an honest
comparison with figures and a short report whose claims I can explain.
