# gsoup v0.0.6

## In a nutshell
A python library implementing various computational geometry / graphics algorithms with focus on clarity rather than performance.

## Long version
This library is the result of me getting tired of replicating pieces of utility code to do similar work across different projects. I decided to push any fundamental piece of code into this repository. Later on I also wanted a place I can implement any related algorithm and easily verify/test/modify them so I decided it would be usefull to add them to the same place. The focus is on clarity and concisness for all implementations, but here and there some immediate/easy performance gains are used.

The majority of the code uses numpy, but a significant effort has been made to also be compatible with pytorch for GPU computations, as many of my interests rely on a strong auto-differentiaion (and GPU enabled) package.

All the code is used for self-educational purposes and to facilitate faster research of concepts in computational geometry / graphics.

## Installation
`pip install gsoup`

## Usage
`import gsoup`

## Developers
`git clone https://github.com/yoterel/gsoup.git`

`cd gsoup`

`pip install -e .`

Feel free to submit pull requests (run the tests first).

## Build
`bumpver update --no-fetch --patch`

`python -m build`

`twine check dist/*`

`twine upload dist/*`
