# Deepgroebner

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://0708andreas.github.io/Deepgroebner.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://0708andreas.github.io/Deepgroebner.jl/dev)
[![Build Status](https://github.com/0708andreas/Deepgroebner.jl/workflows/CI/badge.svg)](https://github.com/0708andreas/Deepgroebner.jl/actions) --->

This is a project to use deep reinforcement learning to improve the performance of Buchbergers Algorithm. It is based in this article: [https://arxiv.org/abs/2005.01917]. It is implemented in Julia using [Flux.jl](https://arxiv.org/abs/1811.01457).

# What is this about?
A central problem in algebraic geomtry is the computation of Gröbner bases. Given a set of generators of a polynomial ideal, a Gröbner basis is a more well-behaved set of generators of that ideal. Buchbergers algorithm is a widely used algorithm for computing Gröbner bases. On a high level, it works by considering all pairs of generators from the given generating set. From each pair, is computes their S-polynomial, and reduces that polynomial modulo the other generators. If this reduces to something non-zero, the remainder is added to the set of generators, and the procedure starts again.

A central choice is which S-polynomial to reduce next. Choosing the order in which S-polynomials are reduced can significantly improve the running time of the algorithm. Several human-developed heuristiccs exists. The purpose of this project is to use deep reinforcement learning to learn new and better strategies.

This problem is particularly difficult for reinforcement learning methods, because the running time of Buchbergers algorithm varies a lot from input to input. The worst case running time is $O(2^{2^n})$, but something this bad is rarely encountered. Thus, the reward function has a huge variance, which makes the training very unstable. The original paper utilized Generalized Advantage Estimation to combat this, which we demonstrate is necessary. Without it, the model fails to learn a better strategy than the existing ones.
