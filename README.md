# gsoup

## In a nutshell
A python library implementing various geometry / graphics algorithms, with focus on clarity rather than performance.

## Long version
This library is the result of me getting tired of replicating pieces of utility code to do similar work across different projects. I decided to push any fundamental piece of code into this repository. The focus is on clarity and concisness for all implementations, but here and there some immediate/easy performance gains are used.

The majority of the code uses numpy, but some effort has been made to also be compatible with pytorch for GPU computations, as many of my interests rely on a strong auto-differentiaion (and GPU enabled) package.

All the code is used for self-educational purposes and to facilitate faster research of concepts in graphics, and specifically for applications in AR.

## Installation
`pip install gsoup`

## Usage
`import gsoup`

## Developers
`git clone https://github.com/yoterel/gsoup.git`

`cd gsoup`

`pip install -e .[dev,build]`

Feel free to submit pull requests (run the tests first).

## Build
`bumpver update --no-fetch --patch`

`python -m build`

`twine check dist/*`

`twine upload dist/*`

or if you defined a ~/.pypirc:

`twine upload --repository gsoup dist/*`

see .pypirc for an example file.


