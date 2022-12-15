# ricdam

RICDaM (Recommending Interoperable and Consistent Data Models) is a framework
that produces a ranked set of candidates to model an input dataset.

Those candidates are obtained from Content, Interoperability, and Consistency
scores that exploit a background Knowledge Graph built from existing RDF
datasets in the same or related domains.

This repository contains a demonstrator and [example datasets](datasets/).

## Running

To run the demonstrator using the ready-to-use Docker image:

    docker run --rm -p 8050:8050 ghcr.io/danielapoliveira/ricdam:v1.0.0

Then, point your browser to [`http://localhost:8050/ricdam/`](http://localhost:8050/ricdam/).

## Building

To build a Docker image for the demonstrator:

    docker buildx build -t ghcr.io/danielapoliveira/ricdam:v1.0.0 .

> **Note:** Ready-to-use images are available in the
> [GitHub Container Registry](https://github.com/users/danielapoliveira/packages/container/package/ricdam).
