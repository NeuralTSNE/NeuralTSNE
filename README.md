# NeuralTSNE

NeuralTSNE is a parametric t-SNE implementation that uses neural networks to learn the mapping from high-dimensional data to a low-dimensional space. It uses PyTorch for the neural network implementation and can be run on a GPU for faster computations. It also emloys Lightning library, which serves as a high-level wrapper for PyTorch, to simplify the training process. The package can be imported as Python module or used as a command-line tool.

## Features

- Neural t-SNE implementation
- CUDA support for faster computations
- Integration with PyTorch
- Comprehensive test coverage
- Documentation generated with Sphinx

## Installation

To install the package, run:

```sh
pip install NeuralTSNE
```

## Usage

Example usage was provided in the [examples](examples/) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](NeuralTSNE/LICENSE.txt) file for details.

## Acknowledgements

The project is supported by the National Science Center in Poland (Sonata 2021/43/D/ST4/00920, ``Statistical Learning of Slow Collective Variables from Atomistic Simulations'').
