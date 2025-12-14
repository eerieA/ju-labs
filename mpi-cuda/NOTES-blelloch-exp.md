A simple Blelloch on paper, with 1 thread block containing 8 threads, doing prefix sum on a 8 element array.

<!-- TOC -->

- [Wrong but informative procedure](#wrong-but-informative-procedure)
- [Correct procedure](#correct-procedure)

<!-- /TOC -->

## Wrong but informative procedure

<img alt="" src="./assets/blelloch-on-paper-01.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-02.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-03.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-04.jpg" width="100%">

## Correct procedure

We should not execute this like it is sequential. Have to do it with simulating parallelism: all threads "run" until a sync_threads().

<img alt="" src="./assets/blelloch-on-paper-05.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-06.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-07.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-08.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-09.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-10.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-11.jpg" width="100%">

<img alt="" src="./assets/blelloch-on-paper-12.jpg" width="100%">

At the end we can see that we get exactly the result of a prefix sum on the original `data`.