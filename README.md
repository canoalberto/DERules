# Learning classification rules with differential evolution for high-speed data stream mining on GPUs

High-speed data streams are potentially infinite sequences of rapidly arriving instances that may be subject to concept drift phenomenon. Hence, dedicated learning algorithms must be able to update themselves with new data and provide an accurate prediction in a limited amount of time. This requirement was considered as prohibitive for using evolutionary algorithms for high-speed data stream mining. This paper introduces a massively parallel implementation on GPUs of a differential evolution algorithm for learning classification rules in the presence of concept drift. The proposal based on the DE /rand - to - best/1/bin strategy takes advantage of up to four nested levels of parallelism to maximize the performance of the algorithm. Efficient GPU kernels parallelize the evolution of the populations, rules, conditional clauses, and evaluation on instances. The proposed method is evaluated on 25 data stream benchmarks considering different types of concept drifts. Results are compared with other publicly available streaming rule learners. Obtained results and their statistical analysis proves an excellent performance of the proposed classifier that offers improved predictive accuracy, model update time, decision time, and a compact rule set.

# Manuscript - 2018 IEEE Congress on Evolutionary Computation (CEC)

https://ieeexplore.ieee.org/document/8477961

# Citing DERules

> A. Cano and B. Krawczyk. Learning classification rules with differential evolution for high-speed data stream mining on GPUs. In IEEE Congress on Evolutionary Computation, 197-204, 2018.
